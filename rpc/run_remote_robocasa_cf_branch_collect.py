"""Counterfactual-branch dataset collector for Cosmos Policy (T2, CF plan).

Copied from ``run_remote_robocasa_oracle_branch3.py`` (single official-
checkpoint ``CosmosPolicyClient`` for every query -- this corrects the
original plan's assumption that ``postfailure_gated_branch3.py`` was the
right base: that file queries a *second*, gated_action32-recovery-trained
client for its retries, which is not what a same-checkpoint counterfactual
benchmark needs).

Each query generates one 32-action plan and executes its first 16 actions.
Evaluator-only (Oracle) simulator state decides whether a clear local
interaction failed at t16. A confirmed failure starts ``branch_count``
deterministic sibling branches from the exact same simulator snapshot --
unlike the upstream evaluation runner, ALL branches are executed and
recorded (no early break on first recovery), each branch's imagined t32
endpoint and Cosmos value are kept (not deleted), and the committed episode
trajectory NEVER adopts a branch: after the branch loop, execution always
restores to the original (unbranched) continuation, exactly as if the
counterfactual branches had never happened. Every branch's 32-action plan is
executed in full (16 committed + 16 diagnostic-only) to derive both a t16
"local_label" and a t32 "target grasped/dropped" label. Privileged state is
never sent to the policy; it only produces labels and anchor selection.

Two anchor kinds exist. ``anchor_kind="post_failure"`` (the original T2
kind): anchors at the Oracle-detected local-failure t16 boundary.
``anchor_kind="in_distribution"`` (T17, added 2026-09-03 for the v2 plan):
anchors at the FIRST chunk in an
episode whose own execution newly achieves ``target_grasped`` (false at
the chunk's t0, true at its t_end/t16) -- i.e. a successful, in-distribution
grasp, not a failure. At most one in_distribution anchor per episode
(``in_dist_anchor_done``); its branches, snapshot, npz, and jsonl fields
are otherwise identical to post_failure's, with ``anchor_id`` prefixed
``indist-`` and ``anchor_kind`` recorded accordingly. Branch label
semantics (``local_label`` / ``t32_label``) are unchanged: still whether
each branch's OWN plan achieves the grasp predicate within 16/32 steps from
the anchor, independent of why the anchor was selected.

``anchor_kind="both"`` (2026-09-10) runs BOTH triggers on the same
trajectory: the same episode yields post_failure anchors at every
Oracle-detected local failure AND in_distribution anchors around its
first grasp. With ``--in-dist-pregrasp-chunks N`` the in_distribution
trigger anchors the last N chunks before the grasp-achieving one (each
recorded with ``pregrasp_offset`` 1..N) instead of only the immediately
preceding chunk. A t0 already collected as a post_failure anchor in the
same episode is skipped as an in_distribution source, so no state is
collected twice or counted in both pools.

``anchor_kind="pre_action"`` from the original plan (branching before the
grasp attempt even starts) was dropped by decision on 2026-09-03 -- no
existing trigger logic to build on in any runner in this repo, and Table 4
/ T15 (its downstream use) was cut from scope at the same time.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import math
import os
import pickle
import random
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, TextIO

import numpy as np
from PIL import Image

from .client import CosmosPolicyClient, RemoteInferenceError
from .run_remote_robocasa_collect import (
    CONTROLLER_CONFIGS_PATH,
    FUTURE_IMAGE_KEYS,
    TASK_MAX_STEPS,
    _atomic_write_json,
    _copy_observation,
    _default_run_id,
    _request_id,
    _set_seed_everywhere,
    _to_environment_action,
    _validate_single_response,
    prepare_observation,
)
from .run_remote_robocasa_collect_v2 import (
    _finite_or_none,
    _grasped_object_names,
    _gripper_aperture,
    _target_object_eef_distance,
    _task_state,
    _task_success,
    training_layout_style_pairs,
)
from .oracle_retry_predicates import (
    evaluate_initial_failure,
    evaluate_retry_recovery,
)
from .run_remote_robocasa_targeted_collect import (
    SimulatorSnapshot as PhysicsSimulatorSnapshot,
    _capture_simulator_snapshot as _capture_physics_snapshot,
)


VERDICTS = frozenset(("continue", "wait", "local_retry"))
CAMERA_IMAGE_KEYS = ("primary_image", "secondary_image", "wrist_image")
MAX_VERIFIER_RESPONSE_BYTES = 64 * 1024
MAX_SEED = 2**63 - 1
DEFAULT_TEST_LAYOUTS = "((1,1),(2,2),(4,4),(6,9),(7,10))"


@dataclass(frozen=True)
class ObjectStateSnapshot:
    target: Any
    values: Mapping[str, Any]


@dataclass(frozen=True)
class SimulatorSnapshot:
    physics: PhysicsSimulatorSnapshot
    physics_auxiliary: Mapping[str, np.ndarray]
    environment_values: Mapping[str, Any]
    controller_states: tuple[ObjectStateSnapshot, ...]

    @property
    def state_sha256(self) -> str:
        return self.physics.state_sha256


_UNSNAPSHOTTABLE = object()


def _clone_state_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple):
        copied = tuple(_clone_state_value(item) for item in value)
        return (
            _UNSNAPSHOTTABLE
            if any(item is _UNSNAPSHOTTABLE for item in copied)
            else copied
        )
    if isinstance(value, list):
        copied = [_clone_state_value(item) for item in value]
        return (
            _UNSNAPSHOTTABLE
            if any(item is _UNSNAPSHOTTABLE for item in copied)
            else copied
        )
    if isinstance(value, Mapping):
        copied = {key: _clone_state_value(item) for key, item in value.items()}
        return (
            _UNSNAPSHOTTABLE
            if any(item is _UNSNAPSHOTTABLE for item in copied.values())
            else copied
        )
    return _UNSNAPSHOTTABLE


def _snapshot_object(target: Any) -> ObjectStateSnapshot:
    values: dict[str, Any] = {}
    for name, value in vars(target).items():
        copied = _clone_state_value(value)
        if copied is not _UNSNAPSHOTTABLE:
            values[name] = copied
    return ObjectStateSnapshot(target=target, values=values)


def _controller_objects(env: Any) -> tuple[Any, ...]:
    objects: list[Any] = []
    seen: set[int] = set()
    for robot in getattr(env, "robots", ()):
        candidates: list[Any] = []
        composite_controller = getattr(robot, "composite_controller", None)
        if composite_controller is not None:
            candidates.append(composite_controller)
        controller = getattr(robot, "controller", None)
        if controller is not None:
            candidates.append(controller)
        part_controllers = getattr(robot, "part_controllers", None)
        if isinstance(part_controllers, Mapping):
            candidates.extend(part_controllers.values())
        for candidate in candidates:
            for target in (
                candidate,
                getattr(candidate, "interpolator_pos", None),
                getattr(candidate, "interpolator_ori", None),
            ):
                if target is None or id(target) in seen:
                    continue
                seen.add(id(target))
                objects.append(target)
        grippers = getattr(robot, "gripper", None)
        if isinstance(grippers, Mapping):
            for gripper in grippers.values():
                if gripper is not None and id(gripper) not in seen:
                    seen.add(id(gripper))
                    objects.append(gripper)
    return tuple(objects)


def _capture_simulator_snapshot(env: Any) -> SimulatorSnapshot:
    sim_data = getattr(env.sim, "data", None)
    environment_values = {
        name: _clone_state_value(getattr(env, name))
        for name in ("timestep", "cur_time", "done")
        if hasattr(env, name)
    }
    return SimulatorSnapshot(
        physics=_capture_physics_snapshot(env),
        physics_auxiliary={
            name: np.asarray(getattr(sim_data, name)).copy()
            for name in (
                "ctrl",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
                "mocap_pos",
                "mocap_quat",
                "userdata",
            )
            if sim_data is not None and hasattr(sim_data, name)
        },
        environment_values=environment_values,
        controller_states=tuple(
            _snapshot_object(target) for target in _controller_objects(env)
        ),
    )


def _restore_simulator_snapshot(
    env: Any,
    snapshot: SimulatorSnapshot,
    *,
    expected_observation: Mapping[str, np.ndarray],
    flip_images: bool,
) -> tuple[Mapping[str, Any], dict[str, np.ndarray], dict[str, bool]]:
    physics = snapshot.physics
    env.sim.set_state_from_flattened(physics.flattened_state)
    env.sim.forward()
    for name, value in snapshot.physics_auxiliary.items():
        destination = getattr(env.sim.data, name)
        destination[...] = value
    for name, value in snapshot.environment_values.items():
        setattr(env, name, _clone_state_value(value))
    for object_state in snapshot.controller_states:
        for name, value in object_state.values.items():
            setattr(object_state.target, name, _clone_state_value(value))
    np.random.set_state(physics.numpy_random_state)
    random.setstate(physics.python_random_state)

    restored_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
    state_hash_exact = _array_sha256(restored_state) == physics.state_sha256
    if not state_hash_exact:
        raise RuntimeError("restored flattened simulator state hash differs from anchor")
    getter = getattr(env, "_get_observations", None)
    if not callable(getter):
        raise RuntimeError("RoboCasa environment lacks _get_observations")
    try:
        raw = getter(force_update=True)
    except TypeError:
        raw = getter()
    resampled = prepare_observation(raw, flip_images)
    policy_observation = _copy_observation(expected_observation)
    checks: dict[str, bool] = {"simulator_state_hash": state_hash_exact}
    for key, expected in expected_observation.items():
        checks[f"policy_input_{key}"] = np.array_equal(
            np.asarray(expected), np.asarray(policy_observation[key])
        )
        checks[f"resampled_sensor_{key}"] = bool(
            key in resampled
            and np.array_equal(np.asarray(expected), np.asarray(resampled[key]))
        )
    if not all(value for key, value in checks.items() if key.startswith("policy_input_")):
        raise RuntimeError("frozen branch policy observation differs from anchor")
    return raw, policy_observation, checks


class InferenceClient(Protocol):
    def infer(
        self,
        observation: Mapping[str, np.ndarray],
        task_description: str,
        **kwargs: Any,
    ) -> Any: ...


class Verifier(Protocol):
    def verify(
        self,
        *,
        request_id: str,
        task_description: str,
        imagined: Mapping[str, np.ndarray],
        actual_t0: Mapping[str, np.ndarray],
        actual_t8: Mapping[str, np.ndarray],
        actual_t16: Mapping[str, np.ndarray],
    ) -> "VerifierResult": ...


@dataclass(frozen=True)
class RethinkEvalConfig:
    task_name: str
    seed: int
    num_trials_per_task: int
    server_url: str
    output_dir: Path
    run_id: str
    num_queries_best_of_n: int = 1
    chunk_size: int = 32
    num_open_loop_steps: int = 16
    num_denoising_steps_action: int = 5
    flip_images: bool = True
    env_img_res: int = 224
    robots: str = "PandaMobile"
    controllers: str = "OSC_POSE"
    obj_instance_split: str = "B"
    layout_and_style_ids: str = DEFAULT_TEST_LAYOUTS
    randomize_cameras: bool = False
    controller_configs_path: Path = CONTROLLER_CONFIGS_PATH
    decision_horizon: int = 16
    max_local_retry_queries: int = 3
    # N most recent pre-grasp chunks to anchor on when an in_distribution
    # trigger fires (2026-09-10). 1 = the original single-anchor behaviour.
    in_dist_pregrasp_chunks: int = 1
    # Also anchor on the grasp-achieving chunk's OWN t0 (pregrasp_offset=0) --
    # the pre-C-3 behaviour. 2026-09-10: re-added as an option because the
    # C-3 "one chunk earlier" fix was tuned on obj_instance_split A, and on
    # split B (test) it overshoots: measured 0.03/8 branch successes at
    # offset 1 and 0.00/8 at offsets 2-3, i.e. essentially every in_dist
    # anchor is all-false and useless for M1 (1.1% / 0.4% / 0.0% mixed).
    # Mixed needs branch success near 50%; offset 0 was 97.7% on the EASIER
    # split A, so on split B it is the candidate for that middle ground.
    in_dist_include_grasp_chunk: bool = False
    recursive_retry_from_retry_query: bool = False
    alternate_seed_offset: int = 1_000_000
    alternate_episode_stride: int = 10_000


@dataclass(frozen=True)
class VerifierResult:
    request_id: str
    verdict: str
    score: float | None
    latency_seconds: float
    probabilities: dict[str, float] | None = None
    raw_top1_label: str | None = None
    local_retry_threshold: float | None = None


@dataclass
class PendingExpectation:
    policy_request_id: str
    query_index: int
    origin_timestep: int
    policy_seed: int
    selected_seed: int
    execution_horizon: int
    mode: str
    anchor_id: str | None
    retry_ordinal: int | None
    task_state_t0: dict[str, Any]
    task_state_t8: dict[str, Any] | None
    planned_action_sha256: str
    planned_first16_sha256: str
    min_target_object_eef_distance: float | None
    min_control_eef_distance: float | None
    control_contact_seen: bool
    eef_position_t0: list[float]
    max_eef_displacement: float
    min_gripper_aperture: float
    max_gripper_aperture: float
    executed_actions: list[np.ndarray]

    @property
    def due_timestep(self) -> int:
        return self.origin_timestep + 16


@dataclass
class RethinkEpisodeResult:
    episode_index: int
    task_description: str
    success: bool
    episode_length: int
    environment_seed: int
    query_count: int
    oracle_calls: int
    eligible_failure_anchors: int
    recovery_round_1: int
    recovery_round_2: int
    recovery_round_3: int
    unrecovered_anchors: int
    retry_queries: int
    retry_triggers: int
    action_plan_differences: int
    action_plan_collisions: int
    censored_oracle_checks: int
    branch_state_restores: int
    original_terminal_restores: int
    original_terminal_restore_hash_matches: int
    simulated_branch_steps: int
    layout_id: int
    style_id: int


@dataclass
class RethinkEpisodeArtifacts:
    result: RethinkEpisodeResult
    primary_images: list[np.ndarray]
    secondary_images: list[np.ndarray]
    wrist_images: list[np.ndarray]


def _validate_http_url(value: str, name: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{name} must not contain embedded credentials")
    if parsed.fragment:
        raise ValueError(f"{name} must not contain a URL fragment")
    return value


def official_layout_style_pair(
    cfg: RethinkEvalConfig, episode_idx: int
) -> tuple[int, int]:
    """Return the exact scene pair selected by the official baseline evaluator."""
    if cfg.layout_and_style_ids == "TRAIN_NONTEST":
        import robocasa  # noqa: F401 - register RoboCasa environments first
        from robosuite.environments.base import REGISTERED_ENVS

        env_class = REGISTERED_ENVS[cfg.task_name]
        all_pairs = training_layout_style_pairs(
            getattr(env_class, "EXCLUDE_LAYOUTS", ())
        )
        pair = all_pairs[(cfg.seed + episode_idx) % len(all_pairs)]
        return int(pair[0]), int(pair[1])
    all_pairs = ast.literal_eval(cfg.layout_and_style_ids)
    if not isinstance(all_pairs, (tuple, list)) or not all_pairs:
        raise ValueError("layout_and_style_ids must contain at least one pair")
    pair = all_pairs[(episode_idx // 10) % len(all_pairs)]
    if (
        not isinstance(pair, (tuple, list))
        or len(pair) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in pair)
    ):
        raise ValueError("layout_and_style_ids contains an invalid pair")
    return int(pair[0]), int(pair[1])


def create_robocasa_env(
    cfg: RethinkEvalConfig,
    *,
    seed: int | None,
    episode_idx: int,
) -> Any:
    """Create the environment with the exact official baseline arguments."""
    try:
        import robocasa  # noqa: F401 - importing registers RoboCasa environments
        import robosuite
    except ImportError as exc:
        raise RuntimeError(
            "RoboCasa evaluation requires the pinned RoboCasa and robosuite packages"
        ) from exc

    layout_and_style_ids = (official_layout_style_pair(cfg, episode_idx),)
    try:
        with cfg.controller_configs_path.open("rb") as stream:
            controller_configs = pickle.load(stream)
    except OSError as exc:
        raise RuntimeError(
            f"cannot load official RoboCasa controller config: {cfg.controller_configs_path}"
        ) from exc

    return robosuite.make(
        env_name=cfg.task_name,
        robots=cfg.robots,
        controller_configs=controller_configs,
        camera_names=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        camera_widths=cfg.env_img_res,
        camera_heights=cfg.env_img_res,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=cfg.obj_instance_split,
        generative_textures=None,
        randomize_cameras=cfg.randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )


def _maximum_queries_per_episode(cfg: RethinkEvalConfig) -> int:
    chunks = math.ceil(TASK_MAX_STEPS[cfg.task_name] / cfg.num_open_loop_steps)
    return chunks * (1 + cfg.max_local_retry_queries) + 1


def validate_config(cfg: RethinkEvalConfig) -> None:
    if cfg.task_name not in TASK_MAX_STEPS:
        raise ValueError(f"unsupported official RoboCasa task: {cfg.task_name!r}")
    if cfg.seed < 0:
        raise ValueError("seed must be non-negative")
    if cfg.num_trials_per_task < 1:
        raise ValueError("num_trials_per_task must be positive")
    if cfg.num_queries_best_of_n != 1:
        raise ValueError("sequential Rethink requires num_queries=1")
    if cfg.chunk_size != 32:
        raise ValueError("Cosmos Rethink requires the model's fixed 32-action chunk")
    if cfg.num_open_loop_steps != 16:
        raise ValueError("normal Cosmos execution must use exactly 16 actions")
    if cfg.env_img_res != 224:
        raise ValueError("the current RPC protocol requires env_img_res=224")
    if not cfg.run_id or len(_request_id(cfg.run_id, 0, 0)) > 128:
        raise ValueError("run_id is empty or too long for auditable RPC request IDs")
    if cfg.decision_horizon != cfg.num_open_loop_steps:
        raise ValueError("local-retry decisions must occur at the t16 replan boundary")
    if cfg.max_local_retry_queries < 0:
        raise ValueError("max_local_retry_queries must be non-negative")
    if cfg.max_local_retry_queries < 1:
        raise ValueError("counterfactual branch collection requires at least one branch per anchor")
    if not 1 <= cfg.in_dist_pregrasp_chunks <= 8:
        raise ValueError("in_dist_pregrasp_chunks must be between 1 and 8")
    if cfg.recursive_retry_from_retry_query:
        raise ValueError("snapshot branches are siblings, not recursive retries")
    for episode_idx in range(cfg.num_trials_per_task):
        official_layout_style_pair(cfg, episode_idx)

    max_queries = _maximum_queries_per_episode(cfg)
    if cfg.alternate_seed_offset <= cfg.num_queries_best_of_n:
        raise ValueError("alternate_seed_offset must not overlap normal candidate seeds")
    if cfg.alternate_episode_stride <= max_queries + cfg.num_queries_best_of_n:
        raise ValueError(
            "alternate_episode_stride is too small to keep episode seed schedules disjoint"
        )
    maximum_alternate = (
        cfg.seed
        + cfg.alternate_seed_offset
        + (cfg.num_trials_per_task - 1) * cfg.alternate_episode_stride
        + max_queries
        + cfg.num_queries_best_of_n
    )
    if maximum_alternate > MAX_SEED:
        raise ValueError("deterministic alternate seed schedule exceeds int64 range")


def deterministic_policy_seed(
    cfg: RethinkEvalConfig,
    *,
    episode_idx: int,
    query_index: int,
    retry: bool,
) -> int:
    """Return the reproducible base seed for one Cosmos RPC query."""
    if episode_idx < 0 or episode_idx >= cfg.num_trials_per_task:
        raise ValueError("episode_idx is outside the configured evaluation range")
    if query_index < 0 or query_index >= _maximum_queries_per_episode(cfg):
        raise ValueError("query_index is outside the validated deterministic schedule")
    if not retry:
        return cfg.seed
    return (
        cfg.seed
        + cfg.alternate_seed_offset
        + episode_idx * cfg.alternate_episode_stride
        + query_index
    )


def _validate_rgb_image(image: Any, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.shape != (224, 224, 3):
        raise RuntimeError(f"{name} must be uint8 with shape (224, 224, 3)")
    return np.ascontiguousarray(array)


def _image_sha256(image: np.ndarray) -> str:
    array = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes())
    return digest.hexdigest()


def _privileged_snapshot(
    env: Any, task_name: str, observation: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    grasped = _grasped_object_names(env)
    eef_position = np.asarray(observation["proprio"], dtype=np.float64)[2:5]
    control_distance: float | None = None
    control_contact = False
    fixture = None
    button = None
    if task_name == "CoffeePressButton":
        fixture = getattr(env, "coffee_machine", None)
        button = "start_button"
    elif task_name in {"TurnOnMicrowave", "TurnOffMicrowave"}:
        fixture = getattr(env, "microwave", None)
        button = "start_button" if task_name == "TurnOnMicrowave" else "stop_button"
    if fixture is not None and button is not None:
        geom_name = f"{getattr(fixture, 'naming_prefix', '')}{button}"
        try:
            geom_id = env.sim.model.geom_name2id(geom_name)
            control_position = np.asarray(env.sim.data.geom_xpos[geom_id], dtype=np.float64)
            control_distance = float(np.linalg.norm(control_position - eef_position))
            control_contact = bool(
                env.check_contact(env.robots[0].gripper["right"], geom_name)
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            control_distance = None
            control_contact = False
    return {
        "task_state": _task_state(env, task_name),
        "task_success": _task_success(env),
        "grasped_objects": grasped,
        "target_grasped": "obj" in grasped,
        "target_object_eef_distance": _finite_or_none(
            _target_object_eef_distance(env, observation)
        ),
        "gripper_aperture": float(_gripper_aperture(observation)),
        "eef_position": [float(value) for value in eef_position],
        "control_eef_distance": control_distance,
        "control_contact": control_contact,
    }


def is_in_distribution_trigger(
    *,
    anchor_kind: str,
    in_dist_anchor_done: bool,
    grasped_before: bool,
    grasped_now: bool,
) -> bool:
    """Pure trigger condition for a T17 in_distribution anchor (2026-09-03):
    fire once per episode, on the first
    chunk whose own execution newly achieves the grasp -- not grasped
    entering the chunk (``t0["target_grasped"]``), grasped by its end
    (``t_end["target_grasped"]``). Factored out as a pure function so it is
    unit-testable without a real RoboCasa env (see
    tests/test_cf_branch_collect_contract.py).
    """
    return (
        anchor_kind in ("in_distribution", "both")
        and not in_dist_anchor_done
        and grasped_now
        and not grasped_before
    )


def should_create_cf_branch_anchor(
    *, anchor_kind: str, eligible_failure: bool, in_distribution_trigger: bool
) -> bool:
    """Pure anchor-creation gate combining both anchor kinds' triggers."""
    return (
        anchor_kind in ("post_failure", "both") and eligible_failure
    ) or in_distribution_trigger


def _attempt_evidence(expectation: PendingExpectation) -> dict[str, Any]:
    distance = expectation.min_target_object_eef_distance
    aperture_delta = (
        expectation.max_gripper_aperture - expectation.min_gripper_aperture
    )
    return {
        "min_target_object_eef_distance": distance,
        "gripper_aperture_delta": aperture_delta,
        "min_control_eef_distance": expectation.min_control_eef_distance,
        "interaction_action_delta": expectation.max_eef_displacement,
        "contact": expectation.control_contact_seen,
        "near_control_action": bool(
            expectation.min_control_eef_distance is not None
            and expectation.min_control_eef_distance < 0.05
            and expectation.max_eef_displacement >= 0.008
        ),
    }


def _encode_rgb_png(image: np.ndarray, name: str) -> dict[str, str]:
    array = _validate_rgb_image(image, name)
    stream = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(stream, format="PNG")
    return {
        "data": base64.b64encode(stream.getvalue()).decode("ascii"),
        "media_type": "image/png",
    }


def _camera_arrays(
    imagined: Mapping[str, np.ndarray],
    actual_t0: Mapping[str, np.ndarray],
    actual_t8: Mapping[str, np.ndarray],
    actual_t16: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    future_mapping = {
        "imagined_primary": "future_image",
        "imagined_secondary": "future_image2",
        "imagined_wrist": "future_wrist_image",
    }
    arrays: dict[str, np.ndarray] = {}
    for output_key, input_key in future_mapping.items():
        if input_key not in imagined:
            raise RuntimeError(f"imagined endpoint is missing {input_key!r}")
        arrays[output_key] = _validate_rgb_image(imagined[input_key], output_key)
    for offset, actual in ((0, actual_t0), (8, actual_t8), (16, actual_t16)):
        for camera, input_key in (
            ("primary", "primary_image"),
            ("secondary", "secondary_image"),
            ("wrist", "wrist_image"),
        ):
            output_key = f"actual_t{offset}_{camera}"
            if input_key not in actual:
                raise RuntimeError(f"actual t{offset} observation is missing {input_key!r}")
            arrays[output_key] = _validate_rgb_image(actual[input_key], output_key)
    return arrays


class VisualVerifierClient:
    """Strict HTTP client whose API cannot carry privileged simulator state."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: float,
        retries: int,
    ) -> None:
        self.endpoint = _validate_http_url(endpoint, "verifier endpoint")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if retries < 0:
            raise ValueError("retries must be non-negative")
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def verify(
        self,
        *,
        request_id: str,
        task_description: str,
        imagined: Mapping[str, np.ndarray],
        actual_t0: Mapping[str, np.ndarray],
        actual_t8: Mapping[str, np.ndarray],
        actual_t16: Mapping[str, np.ndarray],
    ) -> VerifierResult:
        if not request_id or len(request_id) > 128:
            raise ValueError("verifier request_id must contain 1-128 characters")
        if not task_description or len(task_description) > 512:
            raise ValueError("task_description must contain 1-512 characters")
        if any(ord(character) < 32 for character in task_description):
            raise ValueError("task_description must not contain control characters")
        arrays = _camera_arrays(imagined, actual_t0, actual_t8, actual_t16)
        request_document = {
            "request_id": request_id,
            "task": task_description,
            "imagined": [
                _encode_rgb_png(arrays["imagined_primary"], "imagined_primary"),
                _encode_rgb_png(arrays["imagined_secondary"], "imagined_secondary"),
                _encode_rgb_png(arrays["imagined_wrist"], "imagined_wrist"),
            ],
            "actual_t0": [
                _encode_rgb_png(arrays[f"actual_t0_{key}"], f"actual_t0_{key}")
                for key in ("primary", "secondary", "wrist")
            ],
            "actual_t8": [
                _encode_rgb_png(arrays[f"actual_t8_{key}"], f"actual_t8_{key}")
                for key in ("primary", "secondary", "wrist")
            ],
            "actual_t16": [
                _encode_rgb_png(arrays[f"actual_t16_{key}"], f"actual_t16_{key}")
                for key in ("primary", "secondary", "wrist")
            ],
        }
        payload = json.dumps(
            request_document,
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        started = time.monotonic()
        response_document = self._post(payload)
        latency = time.monotonic() - started
        response_request_id = response_document.get("request_id")
        if response_request_id != request_id:
            raise RuntimeError("verifier response request_id does not match request")
        if response_document.get("parsed") is not True:
            raise RuntimeError("verifier returned an unparsed or ambiguous prediction")
        verdict = response_document.get("label")
        if not isinstance(verdict, str) or verdict not in VERDICTS:
            raise RuntimeError("verifier label must be continue, wait, or local_retry")
        score_value = response_document.get("score")
        score: float | None = None
        if score_value is not None:
            if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
                raise RuntimeError("verifier score must be numeric or null")
            score = float(score_value)
            if not math.isfinite(score):
                raise RuntimeError("verifier score must be finite")
        if response_document.get("gate_applied") is not True:
            raise RuntimeError("verifier must apply the frozen local-retry gate")
        raw_top1_label = response_document.get("raw_top1_label")
        if raw_top1_label not in VERDICTS:
            raise RuntimeError("verifier raw_top1_label is invalid")
        threshold_value = response_document.get("local_retry_threshold")
        if isinstance(threshold_value, bool) or not isinstance(
            threshold_value, (int, float)
        ):
            raise RuntimeError("verifier local_retry_threshold must be numeric")
        threshold = float(threshold_value)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise RuntimeError("verifier local_retry_threshold must be in [0, 1]")
        probability_document = response_document.get("probabilities")
        if not isinstance(probability_document, Mapping) or set(
            probability_document
        ) != VERDICTS:
            raise RuntimeError("verifier probabilities have invalid labels")
        probabilities = {
            str(label): float(value)
            for label, value in probability_document.items()
        }
        if any(
            not math.isfinite(value) or value < 0.0
            for value in probabilities.values()
        ) or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6):
            raise RuntimeError("verifier probabilities are invalid")
        if score is None or not math.isclose(
            score, probabilities["local_retry"], abs_tol=1e-6
        ):
            raise RuntimeError("verifier score must equal p(local_retry)")
        return VerifierResult(
            request_id=request_id,
            verdict=verdict,
            score=score,
            latency_seconds=latency,
            probabilities=probabilities,
            raw_top1_label=str(raw_top1_label),
            local_retry_threshold=threshold,
        )

    def _post(self, payload: bytes) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        last_error: BaseException | None = None
        for attempt in range(self.retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            request = urllib.request.Request(
                self.endpoint,
                data=payload,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=remaining) as response:
                    content_type = response.headers.get_content_type()
                    body = response.read(MAX_VERIFIER_RESPONSE_BYTES + 1)
                if content_type != "application/json":
                    raise RuntimeError(
                        f"verifier returned unexpected content type {content_type!r}"
                    )
                if len(body) > MAX_VERIFIER_RESPONSE_BYTES:
                    raise RuntimeError("verifier response exceeds size limit")
                document = json.loads(body.decode("utf-8"))
                if not isinstance(document, dict):
                    raise RuntimeError("verifier response must be a JSON object")
                return document
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 502, 503, 504} or attempt >= self.retries:
                    body = exc.read(MAX_VERIFIER_RESPONSE_BYTES).decode(
                        "utf-8", errors="replace"
                    )
                    raise RuntimeError(
                        f"verifier returned HTTP {exc.code}: {body}"
                    ) from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"verifier returned invalid JSON: {exc}") from exc
            if attempt < self.retries:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.25 * (2**attempt), 2.0, remaining))
        raise RuntimeError(f"verifier request failed: {last_error}") from last_error


def _write_event(stream: TextIO, event: Mapping[str, Any]) -> None:
    stream.write(json.dumps(dict(event), sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def _observation_hashes(observation: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {
        key: _image_sha256(_validate_rgb_image(observation[key], key))
        for key in CAMERA_IMAGE_KEYS
    }


def _verifier_hashes(
    imagined: Mapping[str, np.ndarray],
    actual_t0: Mapping[str, np.ndarray],
    actual_t8: Mapping[str, np.ndarray],
    actual_t16: Mapping[str, np.ndarray],
) -> dict[str, str]:
    arrays = _camera_arrays(imagined, actual_t0, actual_t8, actual_t16)
    return {key: _image_sha256(value) for key, value in arrays.items()}


def _verifier_request_id(policy_request_id: str, actual_timestep: int) -> str:
    digest = hashlib.sha256(policy_request_id.encode("utf-8")).hexdigest()[:24]
    return f"rethink-{digest}-t{actual_timestep:04d}"


def run_episode(
    cfg: RethinkEvalConfig,
    env: Any,
    task_description: str,
    client: InferenceClient,
    episode_idx: int,
    event_log: TextIO,
    log_file: TextIO,
) -> RethinkEpisodeArtifacts:
    """Run one episode with privileged decisions isolated from policy inputs."""
    obs = None
    for _ in range(10):
        dummy_action = np.zeros(env.action_spec[0].shape)
        obs, _, _, _ = env.step(dummy_action)
    if obs is None:
        raise RuntimeError("RoboCasa stabilization produced no observation")

    action_queue: deque[np.ndarray] = deque()
    pending: list[PendingExpectation] = []
    primary_images: list[np.ndarray] = []
    secondary_images: list[np.ndarray] = []
    wrist_images: list[np.ndarray] = []
    query_count = 0
    oracle_calls = 0
    eligible_failure_anchors = 0
    recovery_counts = {1: 0, 2: 0, 3: 0}
    unrecovered_anchors = 0
    retry_queries = 0
    retry_triggers = 0
    action_plan_differences = 0
    action_plan_collisions = 0
    retry_next_query = False
    active_retry_trigger: str | None = None
    active_anchor: dict[str, Any] | None = None
    success = False
    episode_length = 0

    for timestep in range(TASK_MAX_STEPS[cfg.task_name]):
        observation = prepare_observation(obs, cfg.flip_images)
        primary_images.append(observation["primary_image"])
        secondary_images.append(observation["secondary_image"])
        wrist_images.append(observation["wrist_image"])

        snapshot = _privileged_snapshot(env, cfg.task_name, observation)
        for expectation in pending:
            distance = snapshot.get("target_object_eef_distance")
            if isinstance(distance, (int, float)) and not isinstance(distance, bool):
                expectation.min_target_object_eef_distance = (
                    float(distance)
                    if expectation.min_target_object_eef_distance is None
                    else min(expectation.min_target_object_eef_distance, float(distance))
                )
            aperture = float(snapshot["gripper_aperture"])
            expectation.min_gripper_aperture = min(
                expectation.min_gripper_aperture, aperture
            )
            expectation.max_gripper_aperture = max(
                expectation.max_gripper_aperture, aperture
            )
            control_distance = snapshot.get("control_eef_distance")
            if isinstance(control_distance, (int, float)) and not isinstance(
                control_distance, bool
            ):
                expectation.min_control_eef_distance = (
                    float(control_distance)
                    if expectation.min_control_eef_distance is None
                    else min(expectation.min_control_eef_distance, float(control_distance))
                )
            expectation.control_contact_seen = (
                expectation.control_contact_seen
                or snapshot.get("control_contact") is True
            )
            eef_position = np.asarray(snapshot["eef_position"], dtype=np.float64)
            expectation.max_eef_displacement = max(
                expectation.max_eef_displacement,
                float(
                    np.linalg.norm(
                        eef_position
                        - np.asarray(expectation.eef_position_t0, dtype=np.float64)
                    )
                ),
            )
            if expectation.origin_timestep + 8 == timestep:
                if expectation.task_state_t8 is not None:
                    raise RuntimeError("duplicate t8 state for pending policy query")
                expectation.task_state_t8 = dict(snapshot)

        overdue = [item for item in pending if item.due_timestep < timestep]
        if overdue:
            raise RuntimeError("a pending Oracle planning state missed its exact t16 decision")
        due = [item for item in pending if item.due_timestep == timestep]
        for expectation in due:
            if expectation.task_state_t8 is None:
                raise RuntimeError("pending Oracle item is missing its t8 evidence")
            evidence = _attempt_evidence(expectation)
            if expectation.mode == "normal":
                decision = evaluate_initial_failure(
                    cfg.task_name,
                    expectation.task_state_t0,
                    expectation.task_state_t8,
                    snapshot,
                    attempt_evidence=evidence,
                    task_description=task_description,
                )
            else:
                if active_anchor is None or expectation.anchor_id != active_anchor["id"]:
                    raise RuntimeError("retry query lost its Oracle recovery anchor")
                decision = evaluate_retry_recovery(
                    cfg.task_name,
                    expectation.task_state_t0,
                    expectation.task_state_t8,
                    snapshot,
                    attempt_evidence=evidence,
                    task_description=task_description,
                )
            oracle_calls += 1
            executed = np.asarray(expectation.executed_actions, dtype=np.float64)
            executed_hash = _array_sha256(executed)
            if executed.shape != (cfg.num_open_loop_steps, 7):
                raise RuntimeError("Oracle planning state did not execute exactly 16 actions")
            _write_event(
                event_log,
                {
                    "event": "oracle_decision",
                    "episode_index": episode_idx,
                    "actual_timestep": timestep,
                    "origin_timestep": expectation.origin_timestep,
                    "decision_horizon": 16,
                    "policy_request_id": expectation.policy_request_id,
                    "policy_seed": expectation.policy_seed,
                    "selected_seed": expectation.selected_seed,
                    "execution_horizon": expectation.execution_horizon,
                    "mode": expectation.mode,
                    "anchor_id": expectation.anchor_id,
                    "retry_ordinal": expectation.retry_ordinal,
                    "decision": decision,
                    "attempt_evidence": evidence,
                    "privileged_t0": expectation.task_state_t0,
                    "privileged_t8": expectation.task_state_t8,
                    "privileged_t16": snapshot,
                    "planned_first16_sha256": expectation.planned_first16_sha256,
                    "executed_first16_sha256": executed_hash,
                    "planned_equals_executed": (
                        expectation.planned_first16_sha256 == executed_hash
                    ),
                    "policy_input_contains_privileged_state": False,
                },
            )
            pending.remove(expectation)
            episode_retry_budget_available = (
                cfg.max_local_retry_queries == 0
                or retry_queries < cfg.max_local_retry_queries
            )
            trigger = False
            if expectation.mode == "normal" and decision["eligible_failure"]:
                eligible_failure_anchors += 1
                anchor_id = (
                    f"episode-{episode_idx:04d}/anchor-{eligible_failure_anchors:03d}"
                )
                if episode_retry_budget_available:
                    active_anchor = {
                        "id": anchor_id,
                        "source_request_id": expectation.policy_request_id,
                        "last_plan_sha256": expectation.planned_action_sha256,
                    }
                    trigger = True
                else:
                    unrecovered_anchors += 1
            elif expectation.mode == "oracle_retry":
                ordinal = expectation.retry_ordinal
                if not isinstance(ordinal, int) or ordinal not in recovery_counts:
                    raise RuntimeError("Oracle retry has invalid retry ordinal")
                if decision["recovered"]:
                    recovery_counts[ordinal] += 1
                    _write_event(
                        event_log,
                        {
                            "event": "oracle_recovery",
                            "episode_index": episode_idx,
                            "anchor_id": expectation.anchor_id,
                            "recovery_round": ordinal,
                            "policy_request_id": expectation.policy_request_id,
                            "reason": decision["reason"],
                        },
                    )
                    active_anchor = None
                elif episode_retry_budget_available:
                    # Once an explicit failure establishes an anchor, the
                    # privileged upper-bound retries until recovery or cap.
                    trigger = True
                else:
                    unrecovered_anchors += 1
                    active_anchor = None
            if trigger:
                cleared_action_count = len(action_queue)
                action_queue.clear()
                retry_triggers += 1
                retry_next_query = True
                active_retry_trigger = expectation.policy_request_id
                _write_event(
                    event_log,
                    {
                        "event": "oracle_retry_trigger",
                        "episode_index": episode_idx,
                        "timestep": timestep,
                        "trigger_policy_request_id": expectation.policy_request_id,
                        "anchor_id": active_anchor["id"] if active_anchor else None,
                        "cleared_action_count": cleared_action_count,
                        "retry_budget_remaining": (
                            None
                            if cfg.max_local_retry_queries == 0
                            else cfg.max_local_retry_queries - retry_queries
                        ),
                        "retry_queries_used": retry_queries,
                    },
                )

        if not action_queue:
            is_retry = retry_next_query
            execution_horizon = cfg.num_open_loop_steps
            policy_seed = deterministic_policy_seed(
                cfg,
                episode_idx=episode_idx,
                query_index=query_count,
                retry=is_retry,
            )
            request_id = _request_id(cfg.run_id, episode_idx, timestep)
            policy_task_description = task_description
            policy_started = time.monotonic()
            response = client.infer(
                observation,
                policy_task_description,
                seed=policy_seed,
                num_queries=cfg.num_queries_best_of_n,
                num_denoising_steps_action=cfg.num_denoising_steps_action,
                return_future_images=False,
                request_id=request_id,
            )
            policy_latency = time.monotonic() - policy_started
            seeds, actions, values, selected_index, selected_future = (
                _validate_single_response(
                    response,
                    request_id=request_id,
                    base_seed=policy_seed,
                    num_queries=cfg.num_queries_best_of_n,
                    chunk_size=cfg.chunk_size,
                    require_future_images=False,
                )
            )
            selected_plan = np.asarray(actions[selected_index], dtype=np.float64)
            selected_actions = selected_plan[:execution_horizon]
            plan_hash = _array_sha256(selected_plan)
            first16_hash = _array_sha256(selected_actions)
            action_queue.extend(
                np.array(selected_actions[index], copy=True)
                for index in range(execution_horizon)
            )
            anchor_id = active_anchor["id"] if is_retry and active_anchor else None
            retry_ordinal = retry_queries + 1 if is_retry else None
            if is_retry:
                if active_anchor is None:
                    raise RuntimeError("Oracle retry query has no active anchor")
                if plan_hash == active_anchor["last_plan_sha256"]:
                    action_plan_collisions += 1
                else:
                    action_plan_differences += 1
                active_anchor["last_plan_sha256"] = plan_hash
            distance = snapshot.get("target_object_eef_distance")
            pending.append(
                PendingExpectation(
                    policy_request_id=request_id,
                    query_index=query_count,
                    origin_timestep=timestep,
                    policy_seed=policy_seed,
                    selected_seed=int(seeds[selected_index]),
                    execution_horizon=execution_horizon,
                    mode="oracle_retry" if is_retry else "normal",
                    anchor_id=anchor_id,
                    retry_ordinal=retry_ordinal,
                    task_state_t0=dict(snapshot),
                    task_state_t8=None,
                    planned_action_sha256=plan_hash,
                    planned_first16_sha256=first16_hash,
                    min_target_object_eef_distance=(
                        float(distance)
                        if isinstance(distance, (int, float))
                        and not isinstance(distance, bool)
                        else None
                    ),
                    min_control_eef_distance=(
                        float(snapshot["control_eef_distance"])
                        if isinstance(snapshot.get("control_eef_distance"), (int, float))
                        and not isinstance(snapshot.get("control_eef_distance"), bool)
                        else None
                    ),
                    control_contact_seen=snapshot.get("control_contact") is True,
                    eef_position_t0=list(snapshot["eef_position"]),
                    max_eef_displacement=0.0,
                    min_gripper_aperture=float(snapshot["gripper_aperture"]),
                    max_gripper_aperture=float(snapshot["gripper_aperture"]),
                    executed_actions=[],
                )
            )
            query_event: dict[str, Any] = {
                "event": "policy_query",
                "episode_index": episode_idx,
                "query_index": query_count,
                "request_id": request_id,
                "timestep": timestep,
                "mode": "oracle_retry" if is_retry else "normal",
                "policy_task_description": policy_task_description,
                "policy_seed": policy_seed,
                "returned_seeds": [int(value) for value in seeds],
                "selected_index": selected_index,
                "selected_seed": int(seeds[selected_index]),
                "execution_horizon": execution_horizon,
                "retry_budget_remaining_before_query": (
                    None
                    if cfg.max_local_retry_queries == 0
                    else cfg.max_local_retry_queries - retry_queries
                ),
                "retry_trigger_policy_request_id": active_retry_trigger,
                "retry_ordinal": retry_queries + 1 if is_retry else None,
                "policy_latency_seconds": policy_latency,
                "observation_image_hashes": _observation_hashes(observation),
                "planned_action_sha256": plan_hash,
                "planned_first16_sha256": first16_hash,
                "policy_input_fields": ["observation", "task_description", "seed"],
                "policy_input_contains_privileged_state": False,
                "value_selection_used": False,
                "action_dtype": str(actions.dtype),
            }
            _write_event(event_log, query_event)
            log_file.write(
                f"t={timestep}: mode={'retry' if is_retry else 'normal'} "
                f"seed={policy_seed} selected={int(seeds[selected_index])} "
                f"horizon={execution_horizon}\n"
            )
            log_file.flush()
            query_count += 1
            if is_retry:
                retry_queries += 1
                retry_next_query = False
                active_retry_trigger = None

        policy_action = np.asarray(action_queue.popleft(), dtype=np.float64)
        if not pending:
            raise RuntimeError("executed action has no pending Oracle planning state")
        pending[-1].executed_actions.append(np.array(policy_action, copy=True))
        action = _to_environment_action(policy_action, int(env.action_dim))
        obs, _, _, _ = env.step(action)
        episode_length += 1
        # This is the untouched official episode termination/evaluation signal.
        # It is never included in policy input. Oracle decisions use separate
        # evaluator-only snapshots captured before policy requests.
        if env._check_success():
            success = True
            if active_anchor is not None and pending[-1].mode == "oracle_retry":
                ordinal = pending[-1].retry_ordinal
                if isinstance(ordinal, int) and ordinal in recovery_counts:
                    recovery_counts[ordinal] += 1
                    _write_event(
                        event_log,
                        {
                            "event": "oracle_recovery",
                            "episode_index": episode_idx,
                            "anchor_id": active_anchor["id"],
                            "recovery_round": ordinal,
                            "policy_request_id": pending[-1].policy_request_id,
                            "reason": "official_task_success_before_t16",
                        },
                    )
                    active_anchor = None
            log_file.write(f"Success detected at timestep {timestep}\n")
            log_file.flush()
            break

    if active_anchor is not None:
        unrecovered_anchors += 1
        active_anchor = None
    _write_event(
        event_log,
        {
            "event": "episode_end",
            "episode_index": episode_idx,
            "success": success,
            "episode_length": episode_length,
            "query_count": query_count,
            "oracle_calls": oracle_calls,
            "eligible_failure_anchors": eligible_failure_anchors,
            "recovery_round_1": recovery_counts[1],
            "recovery_round_2": recovery_counts[2],
            "recovery_round_3": recovery_counts[3],
            "unrecovered_anchors": unrecovered_anchors,
            "retry_triggers": retry_triggers,
            "retry_queries": retry_queries,
            "censored_oracle_checks": len(pending),
        },
    )
    return RethinkEpisodeArtifacts(
        result=RethinkEpisodeResult(
            episode_index=episode_idx,
            task_description=task_description,
            success=success,
            episode_length=episode_length,
            environment_seed=cfg.seed * episode_idx * 256,
            query_count=query_count,
            oracle_calls=oracle_calls,
            eligible_failure_anchors=eligible_failure_anchors,
            recovery_round_1=recovery_counts[1],
            recovery_round_2=recovery_counts[2],
            recovery_round_3=recovery_counts[3],
            unrecovered_anchors=unrecovered_anchors,
            retry_queries=retry_queries,
            retry_triggers=retry_triggers,
            action_plan_differences=action_plan_differences,
            action_plan_collisions=action_plan_collisions,
            censored_oracle_checks=len(pending),
            branch_state_restores=0,
            original_terminal_restores=0,
            original_terminal_restore_hash_matches=0,
            simulated_branch_steps=0,
            layout_id=official_layout_style_pair(cfg, episode_idx)[0],
            style_id=official_layout_style_pair(cfg, episode_idx)[1],
        ),
        primary_images=primary_images,
        secondary_images=secondary_images,
        wrist_images=wrist_images,
    )


def run_episode_snapshot_branching(
    cfg: RethinkEvalConfig,
    env: Any,
    task_description: str,
    client: InferenceClient,
    episode_idx: int,
    event_log: TextIO,
    log_file: TextIO,
) -> RethinkEpisodeArtifacts:
    """Run one committed trajectory with counterfactual snapshot branches."""
    obs = None
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))
    if obs is None:
        raise RuntimeError("RoboCasa stabilization produced no observation")

    primary_images: list[np.ndarray] = []
    secondary_images: list[np.ndarray] = []
    wrist_images: list[np.ndarray] = []
    query_count = oracle_calls = eligible_failure_anchors = 0
    retry_queries = retry_triggers = 0
    action_plan_differences = action_plan_collisions = 0
    branch_state_restores = original_terminal_restores = 0
    original_terminal_restore_hash_matches = simulated_branch_steps = 0
    recovery_counts = {1: 0, 2: 0, 3: 0}
    unrecovered_anchors = censored_oracle_checks = 0
    committed_timestep = 0
    success = False
    forced_observation: dict[str, np.ndarray] | None = None
    max_steps = TASK_MAX_STEPS[cfg.task_name]

    def simulator_hash() -> str:
        state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
        return _array_sha256(state)

    def update_tracker(
        tracker: PendingExpectation,
        privileged: Mapping[str, Any],
        relative_step: int,
    ) -> None:
        distance = privileged.get("target_object_eef_distance")
        if isinstance(distance, (int, float)) and not isinstance(distance, bool):
            tracker.min_target_object_eef_distance = (
                float(distance)
                if tracker.min_target_object_eef_distance is None
                else min(tracker.min_target_object_eef_distance, float(distance))
            )
        aperture = float(privileged["gripper_aperture"])
        tracker.min_gripper_aperture = min(tracker.min_gripper_aperture, aperture)
        tracker.max_gripper_aperture = max(tracker.max_gripper_aperture, aperture)
        control_distance = privileged.get("control_eef_distance")
        if isinstance(control_distance, (int, float)) and not isinstance(
            control_distance, bool
        ):
            tracker.min_control_eef_distance = (
                float(control_distance)
                if tracker.min_control_eef_distance is None
                else min(tracker.min_control_eef_distance, float(control_distance))
            )
        tracker.control_contact_seen = (
            tracker.control_contact_seen
            or privileged.get("control_contact") is True
        )
        eef = np.asarray(privileged["eef_position"], dtype=np.float64)
        tracker.max_eef_displacement = max(
            tracker.max_eef_displacement,
            float(np.linalg.norm(eef - np.asarray(tracker.eef_position_t0))),
        )
        if relative_step == 8:
            tracker.task_state_t8 = dict(privileged)

    def issue_query(
        mode: str,
        policy_observation: Mapping[str, np.ndarray],
        *,
        anchor_id: str | None = None,
        ordinal: int | None = None,
        trigger_request_id: str | None = None,
    ) -> dict[str, Any]:
        nonlocal query_count, retry_queries, retry_triggers
        nonlocal action_plan_differences, action_plan_collisions
        is_branch = mode == "oracle_branch"
        seed = deterministic_policy_seed(
            cfg,
            episode_idx=episode_idx,
            query_index=query_count,
            retry=is_branch,
        )
        base_request_id = _request_id(cfg.run_id, episode_idx, committed_timestep)
        request_id = (
            f"{base_request_id}-branch-{ordinal:02d}-q{query_count:04d}"
            if is_branch and ordinal is not None
            else base_request_id
        )
        # Branch queries carry the imagined t32 endpoint and Cosmos value all
        # the way into anchors.jsonl (that's the entire point of T2); normal
        # continuation queries don't need them, so skip the extra payload.
        started = time.monotonic()
        response = client.infer(
            policy_observation,
            task_description,
            seed=seed,
            num_queries=1,
            num_denoising_steps_action=cfg.num_denoising_steps_action,
            return_future_images=is_branch,
            request_id=request_id,
        )
        latency = time.monotonic() - started
        seeds, actions, values, selected_index, selected_future = (
            _validate_single_response(
                response,
                request_id=request_id,
                base_seed=seed,
                num_queries=1,
                chunk_size=cfg.chunk_size,
                require_future_images=is_branch,
            )
        )
        cosmos_value = float(values[selected_index])
        plan = np.asarray(actions[selected_index], dtype=np.float64)
        plan_hash = _array_sha256(plan)
        first16 = np.asarray(plan[: cfg.num_open_loop_steps], dtype=np.float64)
        first16_hash = _array_sha256(first16)
        _write_event(
            event_log,
            {
                "event": "policy_query",
                "episode_index": episode_idx,
                "query_index": query_count,
                "request_id": request_id,
                "timestep": committed_timestep,
                "mode": mode,
                "anchor_id": anchor_id,
                "branch_ordinal": ordinal,
                "policy_task_description": task_description,
                "policy_seed": seed,
                "returned_seeds": [int(value) for value in seeds],
                "selected_index": selected_index,
                "selected_seed": int(seeds[selected_index]),
                "execution_horizon": cfg.num_open_loop_steps,
                "retry_trigger_policy_request_id": trigger_request_id,
                "policy_latency_seconds": latency,
                "observation_image_hashes": _observation_hashes(policy_observation),
                "planned_action_sha256": plan_hash,
                "planned_first16_sha256": first16_hash,
                "policy_input_fields": ["observation", "task_description", "seed"],
                "policy_input_contains_privileged_state": False,
                "value_selection_used": False,
                "action_dtype": str(actions.dtype),
            },
        )
        query_count += 1
        if is_branch:
            retry_queries += 1
            retry_triggers += 1
        log_file.write(
            f"t={committed_timestep}: mode={mode} seed={seed} "
            f"selected={int(seeds[selected_index])}\n"
        )
        log_file.flush()
        return {
            "request_id": request_id,
            "seed": seed,
            "selected_seed": int(seeds[selected_index]),
            "plan": plan,
            "plan_hash": plan_hash,
            "first16": first16,
            "first16_hash": first16_hash,
            "cosmos_value": cosmos_value,
            # {} for non-branch queries (return_future_images was False).
            "imagined": dict(selected_future),
        }

    def execute_plan(
        query: Mapping[str, Any],
        start_observation: Mapping[str, np.ndarray],
        t0: Mapping[str, Any],
        action_limit: int,
    ) -> dict[str, Any]:
        distance = t0.get("target_object_eef_distance")
        tracker = PendingExpectation(
            policy_request_id=str(query["request_id"]),
            query_index=0,
            origin_timestep=committed_timestep,
            policy_seed=int(query["seed"]),
            selected_seed=int(query["selected_seed"]),
            execution_horizon=action_limit,
            mode="branch" if "branch" in str(query["request_id"]) else "normal",
            anchor_id=None,
            retry_ordinal=None,
            task_state_t0=dict(t0),
            task_state_t8=None,
            planned_action_sha256=str(query["plan_hash"]),
            planned_first16_sha256=str(query["first16_hash"]),
            min_target_object_eef_distance=(
                float(distance)
                if isinstance(distance, (int, float)) and not isinstance(distance, bool)
                else None
            ),
            min_control_eef_distance=(
                float(t0["control_eef_distance"])
                if isinstance(t0.get("control_eef_distance"), (int, float))
                and not isinstance(t0.get("control_eef_distance"), bool)
                else None
            ),
            control_contact_seen=t0.get("control_contact") is True,
            eef_position_t0=list(t0["eef_position"]),
            max_eef_displacement=0.0,
            min_gripper_aperture=float(t0["gripper_aperture"]),
            max_gripper_aperture=float(t0["gripper_aperture"]),
            executed_actions=[],
        )
        prepared = _copy_observation(start_observation)
        local_primary: list[np.ndarray] = []
        local_secondary: list[np.ndarray] = []
        local_wrist: list[np.ndarray] = []
        raw_observation: Mapping[str, Any] | None = None
        privileged = dict(t0)
        task_succeeded = False
        actions = np.asarray(query["first16"], dtype=np.float64)
        for index in range(action_limit):
            local_primary.append(np.array(prepared["primary_image"], copy=True))
            local_secondary.append(np.array(prepared["secondary_image"], copy=True))
            local_wrist.append(np.array(prepared["wrist_image"], copy=True))
            policy_action = np.asarray(actions[index], dtype=np.float64)
            tracker.executed_actions.append(np.array(policy_action, copy=True))
            raw_observation, _, _, _ = env.step(
                _to_environment_action(policy_action, int(env.action_dim))
            )
            prepared = prepare_observation(raw_observation, cfg.flip_images)
            privileged = _privileged_snapshot(env, cfg.task_name, prepared)
            update_tracker(tracker, privileged, index + 1)
            if env._check_success():
                task_succeeded = True
                break
        if raw_observation is None:
            raise RuntimeError("plan execution produced no observation")
        executed = np.asarray(tracker.executed_actions, dtype=np.float64)
        return {
            "raw_observation": raw_observation,
            "prepared_observation": prepared,
            "t8": tracker.task_state_t8,
            "t16": privileged,
            "evidence": _attempt_evidence(tracker),
            "steps": len(tracker.executed_actions),
            "task_success": task_succeeded,
            "executed_hash": _array_sha256(executed),
            "terminal_state_hash": simulator_hash(),
            "primary": local_primary,
            "secondary": local_secondary,
            "wrist": local_wrist,
        }

    while committed_timestep < max_steps:
        observation = (
            forced_observation
            if forced_observation is not None
            else prepare_observation(obs, cfg.flip_images)
        )
        forced_observation = None
        t0 = _privileged_snapshot(env, cfg.task_name, observation)
        anchor_snapshot: SimulatorSnapshot = _capture_simulator_snapshot(env)
        frozen_observation = _copy_observation(observation)
        original_query = issue_query("normal", frozen_observation)
        remaining = max_steps - committed_timestep
        action_limit = min(cfg.num_open_loop_steps, remaining)
        original = execute_plan(original_query, frozen_observation, t0, action_limit)
        original_terminal_snapshot = _capture_simulator_snapshot(env)

        if original["task_success"]:
            primary_images.extend(original["primary"])
            secondary_images.extend(original["secondary"])
            wrist_images.extend(original["wrist"])
            committed_timestep += int(original["steps"])
            obs = original["raw_observation"]
            success = True
            break
        if action_limit < cfg.num_open_loop_steps:
            primary_images.extend(original["primary"])
            secondary_images.extend(original["secondary"])
            wrist_images.extend(original["wrist"])
            committed_timestep += int(original["steps"])
            obs = original["raw_observation"]
            censored_oracle_checks += 1
            continue
        if original["t8"] is None:
            raise RuntimeError("normal chunk did not capture t8 evidence")

        decision = evaluate_initial_failure(
            cfg.task_name,
            t0,
            original["t8"],
            original["t16"],
            attempt_evidence=original["evidence"],
            task_description=task_description,
        )
        oracle_calls += 1
        _write_event(
            event_log,
            {
                "event": "oracle_decision",
                "episode_index": episode_idx,
                "actual_timestep": committed_timestep + cfg.num_open_loop_steps,
                "origin_timestep": committed_timestep,
                "decision_horizon": cfg.num_open_loop_steps,
                "policy_request_id": original_query["request_id"],
                "mode": "normal",
                "decision": decision,
                "attempt_evidence": original["evidence"],
                "planned_first16_sha256": original_query["first16_hash"],
                "executed_first16_sha256": original["executed_hash"],
                "planned_equals_executed": (
                    original_query["first16_hash"] == original["executed_hash"]
                ),
                "policy_input_contains_privileged_state": False,
            },
        )
        if not decision["eligible_failure"] or cfg.max_local_retry_queries == 0:
            primary_images.extend(original["primary"])
            secondary_images.extend(original["secondary"])
            wrist_images.extend(original["wrist"])
            committed_timestep += int(original["steps"])
            obs = original["raw_observation"]
            continue

        eligible_failure_anchors += 1
        anchor_id = f"episode-{episode_idx:04d}/anchor-{eligible_failure_anchors:03d}"
        _write_event(
            event_log,
            {
                "event": "oracle_branch_anchor",
                "episode_index": episode_idx,
                "anchor_id": anchor_id,
                "committed_timestep": committed_timestep,
                "snapshot_state_sha256": anchor_snapshot.state_sha256,
                "frozen_observation_hashes": _observation_hashes(frozen_observation),
                "original_plan_sha256": original_query["plan_hash"],
                "original_terminal_state_sha256": original["terminal_state_hash"],
            },
        )
        seen_plan_hashes = {str(original_query["plan_hash"])}
        selected_branch: dict[str, Any] | None = None
        selected_query: dict[str, Any] | None = None
        for ordinal in range(1, cfg.max_local_retry_queries + 1):
            restored_raw, branch_observation, checks = _restore_simulator_snapshot(
                env,
                anchor_snapshot,
                expected_observation=frozen_observation,
                flip_images=cfg.flip_images,
            )
            del restored_raw
            branch_state_restores += 1
            _write_event(
                event_log,
                {
                    "event": "oracle_branch_restore",
                    "episode_index": episode_idx,
                    "anchor_id": anchor_id,
                    "branch_ordinal": ordinal,
                    "snapshot_state_sha256": anchor_snapshot.state_sha256,
                    "checks": checks,
                },
            )
            branch_query = issue_query(
                "oracle_branch",
                branch_observation,
                anchor_id=anchor_id,
                ordinal=ordinal,
                trigger_request_id=str(original_query["request_id"]),
            )
            if branch_query["plan_hash"] in seen_plan_hashes:
                action_plan_collisions += 1
            else:
                action_plan_differences += 1
                seen_plan_hashes.add(str(branch_query["plan_hash"]))
            branch = execute_plan(
                branch_query,
                branch_observation,
                t0,
                cfg.num_open_loop_steps,
            )
            simulated_branch_steps += int(branch["steps"])
            recovered = bool(branch["task_success"])
            branch_decision: dict[str, Any] | None = None
            if not recovered:
                if branch["t8"] is None:
                    raise RuntimeError("Oracle branch did not capture t8 evidence")
                branch_decision = evaluate_retry_recovery(
                    cfg.task_name,
                    t0,
                    branch["t8"],
                    branch["t16"],
                    attempt_evidence=branch["evidence"],
                    task_description=task_description,
                )
                oracle_calls += 1
                recovered = bool(branch_decision["recovered"])
                _write_event(
                    event_log,
                    {
                        "event": "oracle_decision",
                        "episode_index": episode_idx,
                        "actual_timestep": committed_timestep + branch["steps"],
                        "origin_timestep": committed_timestep,
                        "decision_horizon": cfg.num_open_loop_steps,
                        "policy_request_id": branch_query["request_id"],
                        "mode": "oracle_branch",
                        "anchor_id": anchor_id,
                        "branch_ordinal": ordinal,
                        "decision": branch_decision,
                        "attempt_evidence": branch["evidence"],
                        "planned_first16_sha256": branch_query["first16_hash"],
                        "executed_first16_sha256": branch["executed_hash"],
                        "planned_equals_executed": (
                            branch_query["first16_hash"] == branch["executed_hash"]
                        ),
                        "policy_input_contains_privileged_state": False,
                    },
                )
            if recovered:
                recovery_counts[ordinal] += 1
                selected_branch, selected_query = branch, branch_query
                _write_event(
                    event_log,
                    {
                        "event": "oracle_branch_commit",
                        "episode_index": episode_idx,
                        "anchor_id": anchor_id,
                        "recovery_round": ordinal,
                        "policy_request_id": branch_query["request_id"],
                        "terminal_state_sha256": branch["terminal_state_hash"],
                        "reason": (
                            "official_task_success"
                            if branch["task_success"]
                            else branch_decision["reason"]
                        ),
                    },
                )
                break
            _write_event(
                event_log,
                {
                    "event": "oracle_branch_reject",
                    "episode_index": episode_idx,
                    "anchor_id": anchor_id,
                    "branch_ordinal": ordinal,
                    "policy_request_id": branch_query["request_id"],
                    "terminal_state_sha256": branch["terminal_state_hash"],
                },
            )

        if selected_branch is not None and selected_query is not None:
            primary_images.extend(selected_branch["primary"])
            secondary_images.extend(selected_branch["secondary"])
            wrist_images.extend(selected_branch["wrist"])
            committed_timestep += int(selected_branch["steps"])
            obs = selected_branch["raw_observation"]
            success = bool(selected_branch["task_success"])
            if success:
                break
            continue

        unrecovered_anchors += 1
        restored_raw, restored_observation, checks = _restore_simulator_snapshot(
            env,
            original_terminal_snapshot,
            expected_observation=original["prepared_observation"],
            flip_images=cfg.flip_images,
        )
        branch_state_restores += 1
        original_terminal_restores += 1
        restored_hash = simulator_hash()
        restore_matches = restored_hash == original["terminal_state_hash"]
        original_terminal_restore_hash_matches += int(restore_matches)
        _write_event(
            event_log,
            {
                "event": "oracle_original_terminal_restore",
                "episode_index": episode_idx,
                "anchor_id": anchor_id,
                "snapshot_state_sha256": original_terminal_snapshot.state_sha256,
                "restore_checks": checks,
                "expected_terminal_state_sha256": original["terminal_state_hash"],
                "actual_terminal_state_sha256": restored_hash,
                "terminal_hash_matches": restore_matches,
            },
        )
        if not restore_matches:
            raise RuntimeError("original terminal snapshot restore hash differs from baseline branch")
        primary_images.extend(original["primary"])
        secondary_images.extend(original["secondary"])
        wrist_images.extend(original["wrist"])
        committed_timestep += int(original["steps"])
        obs = restored_raw
        forced_observation = _copy_observation(restored_observation)

    _write_event(
        event_log,
        {
            "event": "episode_end",
            "episode_index": episode_idx,
            "success": success,
            "episode_length": committed_timestep,
            "query_count": query_count,
            "oracle_calls": oracle_calls,
            "eligible_failure_anchors": eligible_failure_anchors,
            "recovery_round_1": recovery_counts[1],
            "recovery_round_2": recovery_counts[2],
            "recovery_round_3": recovery_counts[3],
            "unrecovered_anchors": unrecovered_anchors,
            "retry_triggers": retry_triggers,
            "retry_queries": retry_queries,
            "censored_oracle_checks": censored_oracle_checks,
            "branch_state_restores": branch_state_restores,
            "original_terminal_restores": original_terminal_restores,
            "original_terminal_restore_hash_matches": original_terminal_restore_hash_matches,
            "simulated_branch_steps": simulated_branch_steps,
        },
    )
    return RethinkEpisodeArtifacts(
        result=RethinkEpisodeResult(
            episode_index=episode_idx,
            task_description=task_description,
            success=success,
            episode_length=committed_timestep,
            environment_seed=cfg.seed * episode_idx * 256,
            query_count=query_count,
            oracle_calls=oracle_calls,
            eligible_failure_anchors=eligible_failure_anchors,
            recovery_round_1=recovery_counts[1],
            recovery_round_2=recovery_counts[2],
            recovery_round_3=recovery_counts[3],
            unrecovered_anchors=unrecovered_anchors,
            retry_queries=retry_queries,
            retry_triggers=retry_triggers,
            action_plan_differences=action_plan_differences,
            action_plan_collisions=action_plan_collisions,
            censored_oracle_checks=censored_oracle_checks,
            branch_state_restores=branch_state_restores,
            original_terminal_restores=original_terminal_restores,
            original_terminal_restore_hash_matches=original_terminal_restore_hash_matches,
            simulated_branch_steps=simulated_branch_steps,
            layout_id=official_layout_style_pair(cfg, episode_idx)[0],
            style_id=official_layout_style_pair(cfg, episode_idx)[1],
        ),
        primary_images=primary_images,
        secondary_images=secondary_images,
        wrist_images=wrist_images,
    )


def _anchor_dir_name(anchor_id: str) -> str:
    """anchor_id is "episode-0004/anchor-002"; make it one filesystem-safe token."""
    return anchor_id.replace("/", "__")


def run_episode_cf_branch_collect(
    cfg: RethinkEvalConfig,
    env: Any,
    task_description: str,
    client: InferenceClient,
    episode_idx: int,
    event_log: TextIO,
    log_file: TextIO,
    *,
    anchor_log: TextIO,
    arrays_dir: Path,
    snapshots_dir: Path,
    anchor_kind: str = "post_failure",
) -> RethinkEpisodeArtifacts:
    """Committed trajectory identical to the normal (unbranched) baseline;
    every Oracle-eligible local failure additionally spawns
    ``cfg.max_local_retry_queries`` sibling counterfactual branches that are
    each executed in FULL (t0->t16->t32, all recorded, none early-stopped),
    then always discarded -- the committed trajectory never adopts a branch.
    This is the collector for T2; see the module docstring for what changed
    relative to the evaluation runner it was copied from.
    """
    # bare "cf_bench" (not "<repo-package>.cf_bench"): this
    # file's own PYTHONPATH convention puts ANCHOR_ROOT itself on sys.path
    # (see run_remote_robocasa_cf_branch_collect.sh), not its parent, so
    # cf_bench resolves directly -- matching how cf_bench/snapshot_io.py
    # imports its own "rpc.*" siblings the same way.
    from cf_bench.snapshot_io import dump_snapshot

    K = cfg.max_local_retry_queries

    obs = None
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))
    if obs is None:
        raise RuntimeError("RoboCasa stabilization produced no observation")

    primary_images: list[np.ndarray] = []
    secondary_images: list[np.ndarray] = []
    wrist_images: list[np.ndarray] = []
    query_count = oracle_calls = eligible_failure_anchors = 0
    retry_queries = retry_triggers = 0
    action_plan_differences = action_plan_collisions = 0
    branch_state_restores = original_terminal_restores = 0
    original_terminal_restore_hash_matches = simulated_branch_steps = 0
    recovery_counts = {ordinal: 0 for ordinal in range(1, K + 1)}
    unrecovered_anchors = censored_oracle_checks = 0
    committed_timestep = 0
    success = False
    forced_observation: dict[str, np.ndarray] | None = None
    # T17 (in_distribution anchors, 2026-09-03): at most one per
    # episode, fired the first time a chunk's own execution newly
    # achieves target_grasped (false at chunk start, true at chunk
    # end) -- an "in-distribution" anchor, unlike post_failure's
    # local-failure trigger.
    in_dist_anchor_done = False
    # C-3 escalation fix (2026-09-03, user-authorized "move the trigger one
    # chunk earlier"): round 1's real data showed anchoring at the START of
    # the chunk that achieves the grasp (the ORIGINAL, still-correct
    # semantics of in_dist_anchor_done's trigger) selects for pre-grasp
    # states the SAME checkpoint already found "easy" (97.7% branch
    # success, mixed_rate=3.7%, both far outside the >=25% target). Instead
    # of branching from the current chunk's own state, branch from the
    # PRIOR chunk's state -- a genuinely earlier, less-decided moment,
    # since the policy has not yet even begun the chunk that turned out to
    # succeed. `recent_chunk_states` holds the last N preceding iterations'
    # own (t0, anchor_snapshot, frozen_observation, committed_timestep)
    # bundles; it is short (or empty) only near the start of an episode, in
    # which case the missing offsets are simply skipped -- see
    # is_in_distribution_trigger()'s own contract: this can only happen
    # once per episode anyway, since grasped_before becomes True from the
    # next chunk onward regardless.
    # 2026-09-10: a rolling buffer of the last `cfg.in_dist_pregrasp_chunks`
    # chunk states (was a single `prior_chunk_state`), so an in_distribution
    # trigger can branch from the last N pre-grasp chunks instead of only the
    # immediately preceding one. N=1 reproduces the old behaviour exactly.
    recent_chunk_states: list[dict[str, Any]] = []
    # t0 timesteps already collected as post_failure anchors in this episode;
    # such a state is skipped as an in_distribution source (see the overlap
    # rule where branch_sources is built).
    post_failure_origin_timesteps: set[int] = set()
    max_steps = TASK_MAX_STEPS[cfg.task_name]

    def simulator_hash() -> str:
        state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
        return _array_sha256(state)

    def update_tracker(
        tracker: PendingExpectation, privileged: Mapping[str, Any], relative_step: int
    ) -> None:
        distance = privileged.get("target_object_eef_distance")
        if isinstance(distance, (int, float)) and not isinstance(distance, bool):
            tracker.min_target_object_eef_distance = (
                float(distance)
                if tracker.min_target_object_eef_distance is None
                else min(tracker.min_target_object_eef_distance, float(distance))
            )
        aperture = float(privileged["gripper_aperture"])
        tracker.min_gripper_aperture = min(tracker.min_gripper_aperture, aperture)
        tracker.max_gripper_aperture = max(tracker.max_gripper_aperture, aperture)
        control_distance = privileged.get("control_eef_distance")
        if isinstance(control_distance, (int, float)) and not isinstance(
            control_distance, bool
        ):
            tracker.min_control_eef_distance = (
                float(control_distance)
                if tracker.min_control_eef_distance is None
                else min(tracker.min_control_eef_distance, float(control_distance))
            )
        tracker.control_contact_seen = (
            tracker.control_contact_seen or privileged.get("control_contact") is True
        )
        eef = np.asarray(privileged["eef_position"], dtype=np.float64)
        tracker.max_eef_displacement = max(
            tracker.max_eef_displacement,
            float(np.linalg.norm(eef - np.asarray(tracker.eef_position_t0))),
        )
        if relative_step == 8:
            tracker.task_state_t8 = dict(privileged)

    def issue_query(
        mode: str,
        policy_observation: Mapping[str, np.ndarray],
        *,
        anchor_id: str | None = None,
        ordinal: int | None = None,
        trigger_request_id: str | None = None,
    ) -> dict[str, Any]:
        nonlocal query_count, retry_queries, retry_triggers
        nonlocal action_plan_differences, action_plan_collisions
        is_branch = mode == "cf_branch"
        seed = deterministic_policy_seed(
            cfg, episode_idx=episode_idx, query_index=query_count, retry=is_branch
        )
        base_request_id = _request_id(cfg.run_id, episode_idx, committed_timestep)
        request_id = (
            f"{base_request_id}-branch-{ordinal:02d}-q{query_count:04d}"
            if is_branch and ordinal is not None
            else base_request_id
        )
        started = time.monotonic()
        response = client.infer(
            policy_observation,
            task_description,
            seed=seed,
            num_queries=1,
            num_denoising_steps_action=cfg.num_denoising_steps_action,
            return_future_images=is_branch,
            request_id=request_id,
        )
        latency = time.monotonic() - started
        seeds, actions, values, selected_index, selected_future = (
            _validate_single_response(
                response,
                request_id=request_id,
                base_seed=seed,
                num_queries=1,
                chunk_size=cfg.chunk_size,
                require_future_images=is_branch,
            )
        )
        cosmos_value = float(values[selected_index])
        plan = np.asarray(actions[selected_index], dtype=np.float64)
        plan_hash = _array_sha256(plan)
        first16 = np.asarray(plan[: cfg.num_open_loop_steps], dtype=np.float64)
        first16_hash = _array_sha256(first16)
        _write_event(
            event_log,
            {
                "event": "policy_query",
                "episode_index": episode_idx,
                "query_index": query_count,
                "request_id": request_id,
                "timestep": committed_timestep,
                "mode": mode,
                "anchor_id": anchor_id,
                "branch_ordinal": ordinal,
                "policy_task_description": task_description,
                "policy_seed": seed,
                "returned_seeds": [int(value) for value in seeds],
                "selected_index": selected_index,
                "selected_seed": int(seeds[selected_index]),
                "execution_horizon": cfg.num_open_loop_steps,
                "retry_trigger_policy_request_id": trigger_request_id,
                "policy_latency_seconds": latency,
                "observation_image_hashes": _observation_hashes(policy_observation),
                "planned_action_sha256": plan_hash,
                "planned_first16_sha256": first16_hash,
                "policy_input_fields": ["observation", "task_description", "seed"],
                "policy_input_contains_privileged_state": False,
                "value_selection_used": False,
                "action_dtype": str(actions.dtype),
            },
        )
        query_count += 1
        if is_branch:
            retry_queries += 1
            retry_triggers += 1
        log_file.write(
            f"t={committed_timestep}: mode={mode} seed={seed} "
            f"selected={int(seeds[selected_index])}\n"
        )
        log_file.flush()
        return {
            "request_id": request_id,
            "seed": seed,
            "selected_seed": int(seeds[selected_index]),
            "plan": plan,
            "plan_hash": plan_hash,
            "first16": first16,
            "first16_hash": first16_hash,
            "cosmos_value": cosmos_value,
            "imagined": dict(selected_future),
        }

    def execute_steps(
        actions: np.ndarray,
        start_observation: Mapping[str, np.ndarray],
        t0: Mapping[str, Any],
        *,
        origin_timestep: int,
        stop_on_success: bool,
    ) -> dict[str, Any]:
        """Execute up to len(actions) environment steps from the current env
        state (env is NOT reset/restored by this helper -- caller controls
        that). Returns per-step tracking plus the final observation/state."""
        distance = t0.get("target_object_eef_distance")
        tracker = PendingExpectation(
            policy_request_id="",
            query_index=0,
            origin_timestep=origin_timestep,
            policy_seed=0,
            selected_seed=0,
            execution_horizon=len(actions),
            mode="cf_branch",
            anchor_id=None,
            retry_ordinal=None,
            task_state_t0=dict(t0),
            task_state_t8=None,
            planned_action_sha256="",
            planned_first16_sha256="",
            min_target_object_eef_distance=(
                float(distance)
                if isinstance(distance, (int, float)) and not isinstance(distance, bool)
                else None
            ),
            min_control_eef_distance=(
                float(t0["control_eef_distance"])
                if isinstance(t0.get("control_eef_distance"), (int, float))
                and not isinstance(t0.get("control_eef_distance"), bool)
                else None
            ),
            control_contact_seen=t0.get("control_contact") is True,
            eef_position_t0=list(t0["eef_position"]),
            max_eef_displacement=0.0,
            min_gripper_aperture=float(t0["gripper_aperture"]),
            max_gripper_aperture=float(t0["gripper_aperture"]),
            executed_actions=[],
        )
        prepared = _copy_observation(start_observation)
        raw_observation: Mapping[str, Any] | None = None
        privileged = dict(t0)
        task_succeeded = False
        for index in range(len(actions)):
            policy_action = np.asarray(actions[index], dtype=np.float64)
            tracker.executed_actions.append(np.array(policy_action, copy=True))
            raw_observation, _, _, _ = env.step(
                _to_environment_action(policy_action, int(env.action_dim))
            )
            prepared = prepare_observation(raw_observation, cfg.flip_images)
            privileged = _privileged_snapshot(env, cfg.task_name, prepared)
            update_tracker(tracker, privileged, index + 1)
            if env._check_success():
                task_succeeded = True
                if stop_on_success:
                    break
        if raw_observation is None:
            raise RuntimeError("plan execution produced no observation")
        executed = np.asarray(tracker.executed_actions, dtype=np.float64)
        return {
            "raw_observation": raw_observation,
            "prepared_observation": prepared,
            "t8": tracker.task_state_t8,
            "t_end": privileged,
            "evidence": _attempt_evidence(tracker),
            "steps": len(tracker.executed_actions),
            "task_success": task_succeeded,
            "executed_hash": _array_sha256(executed),
            "terminal_state_hash": simulator_hash(),
        }

    while committed_timestep < max_steps:
        observation = (
            forced_observation
            if forced_observation is not None
            else prepare_observation(obs, cfg.flip_images)
        )
        forced_observation = None
        t0 = _privileged_snapshot(env, cfg.task_name, observation)
        anchor_snapshot: SimulatorSnapshot = _capture_simulator_snapshot(env)
        frozen_observation = _copy_observation(observation)
        # T17 fix (2026-09-03): remember this chunk's own pre-execution state
        # so an in_distribution trigger (detected only after this chunk's
        # execution completes) can branch from the PRIOR chunk's state --
        # i.e. the state just before the grasp succeeded -- rather than from
        # this chunk's own state, which by construction already achieved
        # target_grasped and is therefore an "easy" pre-grasp state (this was
        # the root cause of C-3 round 1's 3.7% mixed_rate, far below the 25%
        # acceptance threshold: branching from the already-successful state
        # gives the branches almost nothing left to fail at).
        this_chunk_state = {
            "t0": t0,
            "anchor_snapshot": anchor_snapshot,
            "frozen_observation": frozen_observation,
            "origin_timestep": committed_timestep,
        }
        original_query = issue_query("normal", frozen_observation)
        remaining = max_steps - committed_timestep
        action_limit = min(cfg.num_open_loop_steps, remaining)
        original = execute_steps(
            original_query["first16"][:action_limit],
            frozen_observation,
            t0,
            origin_timestep=committed_timestep,
            stop_on_success=True,
        )
        original_terminal_snapshot = _capture_simulator_snapshot(env)

        if original["task_success"]:
            committed_timestep += int(original["steps"])
            obs = original["raw_observation"]
            success = True
            break
        if action_limit < cfg.num_open_loop_steps:
            committed_timestep += int(original["steps"])
            obs = original["raw_observation"]
            censored_oracle_checks += 1
            recent_chunk_states.append(this_chunk_state)
            del recent_chunk_states[:-max(1, int(cfg.in_dist_pregrasp_chunks))]
            continue
        if original["t8"] is None:
            raise RuntimeError("normal chunk did not capture t8 evidence")

        decision = evaluate_initial_failure(
            cfg.task_name,
            t0,
            original["t8"],
            original["t_end"],
            attempt_evidence=original["evidence"],
            task_description=task_description,
        )
        oracle_calls += 1
        _write_event(
            event_log,
            {
                "event": "oracle_decision",
                "episode_index": episode_idx,
                "actual_timestep": committed_timestep + cfg.num_open_loop_steps,
                "origin_timestep": committed_timestep,
                "decision_horizon": cfg.num_open_loop_steps,
                "policy_request_id": original_query["request_id"],
                "mode": "normal",
                "decision": decision,
                "attempt_evidence": original["evidence"],
                "planned_first16_sha256": original_query["first16_hash"],
                "executed_first16_sha256": original["executed_hash"],
                "planned_equals_executed": (
                    original_query["first16_hash"] == original["executed_hash"]
                ),
                "policy_input_contains_privileged_state": False,
            },
        )
        in_distribution_trigger = is_in_distribution_trigger(
            anchor_kind=anchor_kind,
            in_dist_anchor_done=in_dist_anchor_done,
            grasped_before=bool(t0["target_grasped"]),
            grasped_now=bool(original["t_end"]["target_grasped"]),
        )
        should_create_anchor = should_create_cf_branch_anchor(
            anchor_kind=anchor_kind,
            eligible_failure=decision["eligible_failure"],
            in_distribution_trigger=in_distribution_trigger,
        )
        if not should_create_anchor or K == 0:
            committed_timestep += int(original["steps"])
            obs = original["raw_observation"]
            recent_chunk_states.append(this_chunk_state)
            del recent_chunk_states[:-max(1, int(cfg.in_dist_pregrasp_chunks))]
            continue
        # 2026-09-10 (anchor_kind="both", --in-dist-pregrasp-chunks): one
        # chunk can now yield MORE than one anchor. post_failure still yields
        # exactly one (this chunk's own t0). An in_distribution trigger yields
        # up to `cfg.in_dist_pregrasp_chunks` anchors -- the t0 of the last N
        # chunks BEFORE the one that achieved the grasp (offset 1 = the
        # immediately preceding chunk, which is the original single-anchor
        # behaviour and stays the default). The chunk that achieved the grasp
        # is deliberately NOT a source: C-3 round 1 measured 97.7% branch
        # success from it, i.e. it produces almost no mixed anchors.
        #
        # Overlap rule: a t0 already collected as a post_failure anchor in
        # THIS episode is skipped as an in_distribution source rather than
        # collected twice or double-tagged. The two pools are meant to
        # contrast "Oracle flagged a local failure here" against "ordinary
        # state on the way to a grasp"; a state that is both belongs to the
        # former, and re-simulating it would also pay for K branches twice.
        if in_distribution_trigger:
            in_dist_anchor_done = True
        branch_sources: list[tuple[dict[str, Any], str, int]] = []  # (state, kind, pregrasp_offset)
        if in_distribution_trigger and cfg.in_dist_include_grasp_chunk:
            if this_chunk_state["origin_timestep"] not in post_failure_origin_timesteps:
                branch_sources.append((this_chunk_state, "in_distribution", 0))
        if in_distribution_trigger:
            for offset in range(1, int(cfg.in_dist_pregrasp_chunks) + 1):
                if offset > len(recent_chunk_states):
                    break
                candidate = recent_chunk_states[-offset]
                if candidate["origin_timestep"] in post_failure_origin_timesteps:
                    continue
                branch_sources.append((candidate, "in_distribution", offset))
        if decision["eligible_failure"] and anchor_kind in ("post_failure", "both"):
            branch_sources.append((this_chunk_state, "post_failure", 0))
        if not branch_sources:
            committed_timestep += int(original["steps"])
            obs = original["raw_observation"]
            recent_chunk_states.append(this_chunk_state)
            del recent_chunk_states[:-max(1, int(cfg.in_dist_pregrasp_chunks))]
            continue

        restored_raw = None
        restored_observation = None
        for branch_source, source_kind, pregrasp_offset in branch_sources:
            branch_t0 = branch_source["t0"]
            branch_anchor_snapshot = branch_source["anchor_snapshot"]
            branch_frozen_observation = branch_source["frozen_observation"]
            branch_origin_timestep = branch_source["origin_timestep"]
            if source_kind == "post_failure":
                post_failure_origin_timesteps.add(branch_origin_timestep)
            eligible_failure_anchors += 1
            anchor_id_prefix = "indist-" if source_kind == "in_distribution" else ""
            anchor_id = (
                f"episode-{episode_idx:04d}/{anchor_id_prefix}anchor-{eligible_failure_anchors:03d}"
            )
            anchor_dir_name = _anchor_dir_name(anchor_id)
            _write_event(
                event_log,
                {
                    "event": "cf_branch_anchor",
                    "episode_index": episode_idx,
                    "anchor_id": anchor_id,
                    "committed_timestep": branch_origin_timestep,
                    "snapshot_state_sha256": branch_anchor_snapshot.state_sha256,
                    "frozen_observation_hashes": _observation_hashes(branch_frozen_observation),
                    "original_plan_sha256": original_query["plan_hash"],
                    "original_terminal_state_sha256": original["terminal_state_hash"],
                },
            )
            dump_snapshot(branch_anchor_snapshot, snapshots_dir / anchor_dir_name)

            branch_records: list[dict[str, Any]] = []
            branch_plans: dict[str, np.ndarray] = {}
            anchor_arrays: dict[str, np.ndarray] = {
                "pre_primary": np.asarray(branch_frozen_observation["primary_image"]),
                "pre_secondary": np.asarray(branch_frozen_observation["secondary_image"]),
                "pre_wrist": np.asarray(branch_frozen_observation["wrist_image"]),
                "pre_proprio": np.asarray(branch_frozen_observation["proprio"], dtype=np.float64),
            }
            for ordinal in range(1, K + 1):
                restored_raw, branch_observation, checks = _restore_simulator_snapshot(
                    env,
                    branch_anchor_snapshot,
                    expected_observation=branch_frozen_observation,
                    flip_images=cfg.flip_images,
                )
                del restored_raw
                branch_state_restores += 1
                _write_event(
                    event_log,
                    {
                        "event": "cf_branch_restore",
                        "episode_index": episode_idx,
                        "anchor_id": anchor_id,
                        "branch_ordinal": ordinal,
                        "snapshot_state_sha256": branch_anchor_snapshot.state_sha256,
                        "checks": checks,
                    },
                )
                branch_query = issue_query(
                    "cf_branch",
                    branch_observation,
                    anchor_id=anchor_id,
                    ordinal=ordinal,
                    trigger_request_id=str(original_query["request_id"]),
                )
                plan_hash = str(branch_query["plan_hash"])
                branch_id = f"{ordinal:02d}"
                branch_plans[branch_id] = branch_query["plan"]
                if plan_hash in {str(original_query["plan_hash"]), *[
                    str(p["plan_sha256"]) for p in branch_records
                ]}:
                    action_plan_collisions += 1
                else:
                    action_plan_differences += 1

                branch16 = execute_steps(
                    branch_query["first16"],
                    branch_observation,
                    branch_t0,
                    origin_timestep=branch_origin_timestep,
                    stop_on_success=False,
                )
                simulated_branch_steps += int(branch16["steps"])

                local_label = bool(branch16["task_success"])
                branch_decision: dict[str, Any] | None = None
                if not local_label:
                    if branch16["t8"] is None:
                        raise RuntimeError("cf branch did not capture t8 evidence")
                    branch_decision = evaluate_retry_recovery(
                        cfg.task_name,
                        branch_t0,
                        branch16["t8"],
                        branch16["t_end"],
                        attempt_evidence=branch16["evidence"],
                        task_description=task_description,
                    )
                    oracle_calls += 1
                    local_label = bool(branch_decision["recovered"])

                # t32: always execute the second half of the SAME 32-action plan
                # (no new policy query -- diagnostic-only continuation, matching
                # the "actual_t32 is diagnostic, never paired supervision"
                # convention already used by recovery/local_recovery_dataset.py).
                remaining_for_t32 = min(
                    cfg.num_open_loop_steps, max_steps - branch_origin_timestep - branch16["steps"]
                )
                second_half = np.asarray(
                    branch_query["plan"][
                        cfg.num_open_loop_steps : cfg.num_open_loop_steps + remaining_for_t32
                    ],
                    dtype=np.float64,
                )
                if remaining_for_t32 > 0:
                    branch32 = execute_steps(
                        second_half,
                        branch16["prepared_observation"],
                        branch16["t_end"],
                        origin_timestep=branch_origin_timestep + branch16["steps"],
                        stop_on_success=True,
                    )
                    t32_label = bool(branch32["task_success"] or local_label)
                    actual_t32 = branch32["prepared_observation"]
                    actual_proprio_t32 = np.asarray(actual_t32["proprio"], dtype=np.float64)
                else:
                    # branch already reached the episode horizon at t16.
                    t32_label = local_label
                    actual_t32 = branch16["prepared_observation"]
                    actual_proprio_t32 = np.asarray(actual_t32["proprio"], dtype=np.float64)

                if local_label:
                    recovery_counts[ordinal] += 1

                record = {
                    "branch_id": branch_id,
                    "seed": int(branch_query["seed"]),
                    "plan_sha256": plan_hash,
                    "cosmos_value": float(branch_query["cosmos_value"]),
                    "returned_seeds": [int(branch_query["seed"])],
                    "local_label": local_label,
                    "t32_label": t32_label,
                    "downstream_label": None,
                }
                branch_records.append(record)
                actual_t16 = branch16["prepared_observation"]
                anchor_arrays[f"b{branch_id}_plan32"] = branch_query["plan"]
                anchor_arrays[f"b{branch_id}_imagined_primary_t32"] = branch_query["imagined"][
                    "future_image"
                ]
                anchor_arrays[f"b{branch_id}_imagined_secondary_t32"] = branch_query["imagined"][
                    "future_image2"
                ]
                anchor_arrays[f"b{branch_id}_imagined_wrist_t32"] = branch_query["imagined"][
                    "future_wrist_image"
                ]
                anchor_arrays[f"b{branch_id}_actual_primary_t16"] = np.asarray(
                    actual_t16["primary_image"]
                )
                anchor_arrays[f"b{branch_id}_actual_secondary_t16"] = np.asarray(
                    actual_t16["secondary_image"]
                )
                anchor_arrays[f"b{branch_id}_actual_wrist_t16"] = np.asarray(
                    actual_t16["wrist_image"]
                )
                anchor_arrays[f"b{branch_id}_actual_proprio_t16"] = np.asarray(
                    actual_t16["proprio"], dtype=np.float64
                )
                anchor_arrays[f"b{branch_id}_actual_primary_t32"] = np.asarray(
                    actual_t32["primary_image"]
                )
                anchor_arrays[f"b{branch_id}_actual_secondary_t32"] = np.asarray(
                    actual_t32["secondary_image"]
                )
                anchor_arrays[f"b{branch_id}_actual_wrist_t32"] = np.asarray(
                    actual_t32["wrist_image"]
                )
                anchor_arrays[f"b{branch_id}_actual_proprio_t32"] = actual_proprio_t32
                _write_event(
                    event_log,
                    {
                        "event": "cf_branch_labeled",
                        "episode_index": episode_idx,
                        "anchor_id": anchor_id,
                        "branch_ordinal": ordinal,
                        "policy_request_id": branch_query["request_id"],
                        "local_label": local_label,
                        "t32_label": t32_label,
                        "cosmos_value": record["cosmos_value"],
                    },
                )

            for record in branch_records:
                own_id = record["branch_id"]
                record["pairwise_l2_to_others"] = {
                    other_id: float(
                        np.linalg.norm(branch_plans[own_id] - branch_plans[other_id])
                    )
                    for other_id in branch_plans
                    if other_id != own_id
                }

            arrays_path = arrays_dir / f"{anchor_dir_name}.npz"
            np.savez_compressed(arrays_path, **anchor_arrays)
            _write_event(
                anchor_log,
                {
                    "anchor_id": anchor_id,
                    "task": cfg.task_name,
                    "env_seed": int(cfg.seed * episode_idx * 256),
                    "episode_index": episode_idx,
                    "query_timestep": branch_origin_timestep,
                    "anchor_kind": source_kind,
                    # forward-compatible list form (2026-09-10): one element
                    # today, because an overlapping state is collected once as
                    # post_failure rather than tagged as both.
                    "anchor_kinds": [source_kind],
                    # 0 for post_failure; 1..N = how many chunks BEFORE the
                    # grasp-achieving chunk this in_distribution anchor sits.
                    "pregrasp_offset": pregrasp_offset,
                    "task_text": task_description,
                    "layout_id": official_layout_style_pair(cfg, episode_idx)[0],
                    "style_id": official_layout_style_pair(cfg, episode_idx)[1],
                    "obj_instance_split": cfg.obj_instance_split,
                    "snapshot_dir": str((snapshots_dir / anchor_dir_name).resolve()),
                    "arrays_file": str(arrays_path.resolve()),
                    "baseline_continuation_label": None,
                    "branches": branch_records,
                },
            )

            # Never adopt a branch into the committed trajectory: always restore
            # the original (unbranched) terminal state and continue as if the
            # counterfactual branches had never been simulated.
            unrecovered_anchors += 1
            restored_raw, restored_observation, checks = _restore_simulator_snapshot(
                env,
                original_terminal_snapshot,
                expected_observation=original["prepared_observation"],
                flip_images=cfg.flip_images,
            )
            branch_state_restores += 1
            original_terminal_restores += 1
            restored_hash = simulator_hash()
            restore_matches = restored_hash == original["terminal_state_hash"]
            original_terminal_restore_hash_matches += int(restore_matches)
            _write_event(
                event_log,
                {
                    "event": "cf_original_terminal_restore",
                    "episode_index": episode_idx,
                    "anchor_id": anchor_id,
                    "snapshot_state_sha256": original_terminal_snapshot.state_sha256,
                    "restore_checks": checks,
                    "expected_terminal_state_sha256": original["terminal_state_hash"],
                    "actual_terminal_state_sha256": restored_hash,
                    "terminal_hash_matches": restore_matches,
                },
            )
            if not restore_matches:
                raise RuntimeError(
                    "original terminal snapshot restore hash differs from baseline branch"
                )
        committed_timestep += int(original["steps"])
        obs = restored_raw
        forced_observation = _copy_observation(restored_observation)
        recent_chunk_states.append(this_chunk_state)
        del recent_chunk_states[:-max(1, int(cfg.in_dist_pregrasp_chunks))]

    _write_event(
        event_log,
        {
            "event": "episode_end",
            "episode_index": episode_idx,
            "success": success,
            "episode_length": committed_timestep,
            "query_count": query_count,
            "oracle_calls": oracle_calls,
            "eligible_failure_anchors": eligible_failure_anchors,
            "unrecovered_anchors": unrecovered_anchors,
            "retry_triggers": retry_triggers,
            "retry_queries": retry_queries,
            "censored_oracle_checks": censored_oracle_checks,
            "branch_state_restores": branch_state_restores,
            "original_terminal_restores": original_terminal_restores,
            "original_terminal_restore_hash_matches": original_terminal_restore_hash_matches,
            "simulated_branch_steps": simulated_branch_steps,
        },
    )
    return RethinkEpisodeArtifacts(
        result=RethinkEpisodeResult(
            episode_index=episode_idx,
            task_description=task_description,
            success=success,
            episode_length=committed_timestep,
            environment_seed=cfg.seed * episode_idx * 256,
            query_count=query_count,
            oracle_calls=oracle_calls,
            eligible_failure_anchors=eligible_failure_anchors,
            recovery_round_1=recovery_counts.get(1, 0),
            recovery_round_2=recovery_counts.get(2, 0),
            recovery_round_3=recovery_counts.get(3, 0),
            unrecovered_anchors=unrecovered_anchors,
            retry_queries=retry_queries,
            retry_triggers=retry_triggers,
            action_plan_differences=action_plan_differences,
            action_plan_collisions=action_plan_collisions,
            censored_oracle_checks=censored_oracle_checks,
            branch_state_restores=branch_state_restores,
            original_terminal_restores=original_terminal_restores,
            original_terminal_restore_hash_matches=original_terminal_restore_hash_matches,
            simulated_branch_steps=simulated_branch_steps,
            layout_id=official_layout_style_pair(cfg, episode_idx)[0],
            style_id=official_layout_style_pair(cfg, episode_idx)[1],
        ),
        # The committed trajectory is byte-identical to what a plain N=1
        # baseline run would have produced (branches never touch it), so
        # this collector intentionally does not accumulate a per-step video
        # frame list -- _save_episode_video is skipped for this runner (see
        # run_evaluation). Anchor-level images live in anchor_arrays/*.npz.
        primary_images=[],
        secondary_images=[],
        wrist_images=[],
    )


def _save_episode_video(
    artifacts: RethinkEpisodeArtifacts,
    rollout_dir: Path,
    log_file: TextIO,
    video_saver: Callable[..., Any],
) -> None:
    result = artifacts.result
    video_saver(
        artifacts.primary_images,
        artifacts.secondary_images,
        artifacts.wrist_images,
        result.episode_index,
        success=result.success,
        task_description=result.task_description,
        rollout_data_dir=str(rollout_dir),
        log_file=log_file,
    )


def _result_document(
    cfg: RethinkEvalConfig,
    episodes: list[RethinkEpisodeResult],
    *,
    complete: bool,
) -> dict[str, Any]:
    successes = sum(int(item.success) for item in episodes)
    total = len(episodes)
    recovered_1 = sum(item.recovery_round_1 for item in episodes)
    recovered_2 = sum(item.recovery_round_2 for item in episodes)
    recovered_3 = sum(item.recovery_round_3 for item in episodes)
    eligible = sum(item.eligible_failure_anchors for item in episodes)
    return {
        "complete": complete and total == cfg.num_trials_per_task,
        "episodes": [asdict(item) for item in episodes],
        "num_queries_best_of_n": cfg.num_queries_best_of_n,
        "run_id": cfg.run_id,
        "seed": cfg.seed,
        "chunk_size": cfg.chunk_size,
        "num_open_loop_steps": cfg.num_open_loop_steps,
        "num_denoising_steps_action": cfg.num_denoising_steps_action,
        "flip_images": cfg.flip_images,
        "obj_instance_split": cfg.obj_instance_split,
        "layout_and_style_ids": cfg.layout_and_style_ids,
        "controller_configs_path": str(cfg.controller_configs_path),
        "success_rate": successes / total if total else 0.0,
        "task_name": cfg.task_name,
        "total_episodes": total,
        "total_successes": successes,
        "total_oracle_calls": sum(item.oracle_calls for item in episodes),
        "total_eligible_failure_anchors": eligible,
        "total_recovery_round_1": recovered_1,
        "total_recovery_round_2": recovered_2,
        "total_recovery_round_3": recovered_3,
        "recovery_at_1": recovered_1 / eligible if eligible else None,
        "recovery_at_2": (recovered_1 + recovered_2) / eligible if eligible else None,
        "recovery_at_3": (
            (recovered_1 + recovered_2 + recovered_3) / eligible
            if eligible
            else None
        ),
        "total_unrecovered_anchors": sum(item.unrecovered_anchors for item in episodes),
        "total_retry_queries": sum(item.retry_queries for item in episodes),
        "total_retry_triggers": sum(item.retry_triggers for item in episodes),
        "total_action_plan_differences": sum(
            item.action_plan_differences for item in episodes
        ),
        "total_action_plan_collisions": sum(
            item.action_plan_collisions for item in episodes
        ),
        "total_censored_oracle_checks": sum(
            item.censored_oracle_checks for item in episodes
        ),
        "total_branch_state_restores": sum(item.branch_state_restores for item in episodes),
        "total_original_terminal_restores": sum(item.original_terminal_restores for item in episodes),
        "total_original_terminal_restore_hash_matches": sum(
            item.original_terminal_restore_hash_matches for item in episodes
        ),
        "total_simulated_branch_steps": sum(item.simulated_branch_steps for item in episodes),
        "oracle_enabled": True,
        "oracle_kind": "privileged_snapshot_branch_upper_bound",
        "verifier_enabled": False,
        "verifier_url": None,
        "privileged_state_in_policy_input": False,
        "snapshot_rollback_used": True,
        "best_of_k_used": True,
        "failed_branches_consume_episode_horizon": False,
        "all_failed_fallback": "restore_saved_original_terminal_snapshot",
        "normal_horizon": cfg.num_open_loop_steps,
        "decision_horizon": cfg.decision_horizon,
        "max_local_retry_queries": cfg.max_local_retry_queries,
        "max_local_retry_queries_semantics": "per_anchor_snapshot_branch_cap_3",
        "retry_policy_instruction": "original_task_description",
        "retry_conditioning": "frozen_anchor_visual_state",
        "t8_is_evidence_only": True,
        "retry_is_single_query": True,
        "recursive_retry_from_retry_query": False,
        "alternate_seed_offset": cfg.alternate_seed_offset,
        "alternate_episode_stride": cfg.alternate_episode_stride,
        "validated": complete and total == cfg.num_trials_per_task,
    }


def run_evaluation(
    cfg: RethinkEvalConfig,
    *,
    client: InferenceClient,
    env_factory: Callable[..., Any] = create_robocasa_env,
    anchor_kind: str = "post_failure",
) -> dict[str, Any]:
    if anchor_kind not in ("post_failure", "in_distribution", "both"):
        raise NotImplementedError(
            f"anchor_kind={anchor_kind!r} is not implemented; see module docstring"
        )
    validate_config(cfg)
    _set_seed_everywhere(cfg.seed)
    # No video_saver: run_episode_cf_branch_collect does not accumulate a
    # per-step frame list (it's a collector, not an evaluation run whose
    # rollout video matters) -- see that function's docstring.

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = cfg.output_dir / "anchor_arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = cfg.output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    result_path = cfg.output_dir / "result.json"
    episodes: list[RethinkEpisodeResult] = []
    # 2026-09-11: episodes whose policy request the SERVER refused. Seen once
    # in production as a T5 text-embedding cache miss: an episode's language
    # instruction was absent from the precomputed cache, the server tried to
    # build the T5 encoder on the fly, and under HF_HUB_OFFLINE the tokenizer
    # files are not there ("TypeError: not a string" out of sentencepiece),
    # so that ONE request 500'd while the server kept serving every other
    # request normally. It is deterministic per (task, seed, episode) -- the
    # object sampling is seeded, so a retry of the whole task rebuilds the
    # same instruction and fails at the same episode -- which is why the
    # per-task retry cannot recover it and the remaining episodes were lost.
    # Skipping just that episode keeps the other 49.
    skipped_server_error: list[int] = []

    with (cfg.output_dir / "eval.log").open(
        "a", encoding="utf-8", buffering=1
    ) as log_file, (cfg.output_dir / "oracle_events.jsonl").open(
        "a", encoding="utf-8", buffering=1
    ) as event_log, (cfg.output_dir / "anchors.jsonl").open(
        "a", encoding="utf-8", buffering=1
    ) as anchor_log:
        for episode_idx in range(cfg.num_trials_per_task):
            environment_seed = cfg.seed * episode_idx * 256
            env = env_factory(cfg, seed=environment_seed, episode_idx=episode_idx)
            try:
                env.reset()
                task_description = env.get_ep_meta()["lang"]
                if not isinstance(task_description, str) or not task_description:
                    raise RuntimeError(
                        "RoboCasa returned an invalid post-reset language instruction"
                    )
                log_file.write(f"Starting episode {episode_idx}: {task_description}\n")
                artifacts = run_episode_cf_branch_collect(
                    cfg,
                    env,
                    task_description,
                    client,
                    episode_idx,
                    event_log,
                    log_file,
                    anchor_log=anchor_log,
                    arrays_dir=arrays_dir,
                    snapshots_dir=snapshots_dir,
                    anchor_kind=anchor_kind,
                )
                # No per-step video: run_episode_cf_branch_collect does not
                # accumulate frame lists (anchor-level images live in
                # anchor_arrays/*.npz instead; see its docstring).
                episodes.append(artifacts.result)
            except RemoteInferenceError as exc:
                # Only the server refusing THIS episode's request. Every other
                # RuntimeError here (an invalid language instruction, a
                # snapshot-restore hash mismatch) is a data-integrity failure
                # and must still abort the run -- RemoteInferenceError is
                # matched exactly, never its RuntimeError base class.
                skipped_server_error.append(episode_idx)
                log_file.write(
                    f"SKIP episode {episode_idx}: server refused the request "
                    f"({exc}); continuing with the next episode\n"
                )
                _write_event(
                    event_log,
                    {
                        "event": "episode_skipped_server_error",
                        "episode_index": episode_idx,
                        "error": str(exc)[:500],
                    },
                )
                continue
            finally:
                env.close()

            partial = _result_document(cfg, episodes, complete=False)
            _atomic_write_json(result_path, partial)
            successes = sum(int(item.success) for item in episodes)
            log_file.write(f"Success: {episodes[-1].success}\n")
            log_file.write(f"# episodes completed so far: {len(episodes)}\n")
            log_file.write(
                f"# successes: {successes} ({successes / len(episodes) * 100:.1f}%)\n"
            )

        # Inside the `with`: log_file is closed the moment the block exits.
        # 2026-09-11: this summary was first written after the block, which
        # raised "ValueError: I/O operation on closed file" on the only run
        # that actually skipped an episode -- the task had finished all 49 of
        # its remaining episodes (their anchors were already on disk, written
        # incrementally) but died before result.json(complete=True) and its
        # DONE marker, so it looked like a lost task in the audit.
        if skipped_server_error:
            log_file.write(
                f"# episodes skipped on server error: {len(skipped_server_error)} "
                f"{skipped_server_error}\n"
            )

    result = _result_document(cfg, episodes, complete=True)
    _atomic_write_json(result_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, dest="task_name")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--trials", required=True, type=int, dest="num_trials_per_task")
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--num-queries", type=int, default=1, dest="num_queries_best_of_n")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--num-open-loop-steps", type=int, default=16)
    parser.add_argument("--num-denoising-steps-action", type=int, default=5)
    parser.add_argument("--obj-instance-split", default="B", choices=("A", "B"))
    parser.add_argument("--layout-and-style-ids", default=DEFAULT_TEST_LAYOUTS)
    parser.add_argument("--no-flip-images", action="store_false", dest="flip_images")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--controller-configs-path", type=Path, default=CONTROLLER_CONFIGS_PATH)
    parser.add_argument("--decision-horizon", type=int, default=16)
    parser.add_argument(
        "--branch-count",
        type=int,
        default=4,
        dest="max_local_retry_queries",
        help="K: number of sibling counterfactual branches sampled per eligible anchor",
    )
    parser.add_argument(
        "--recursive-retry-from-retry-query",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="must remain false because counterfactual branches are siblings",
    )
    parser.add_argument(
        "--in-dist-include-grasp-chunk",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also anchor the grasp-achieving chunk's own t0 (pregrasp_offset=0)",
    )
    parser.add_argument(
        "--in-dist-pregrasp-chunks",
        type=int,
        default=1,
        help="N: anchor on the last N chunks before the grasp-achieving one "
        "when an in_distribution trigger fires (1 = original behaviour)",
    )
    parser.add_argument("--alternate-seed-offset", type=int, default=1_000_000)
    parser.add_argument("--alternate-episode-stride", type=int, default=10_000)
    parser.add_argument(
        "--anchor-kind",
        choices=("post_failure", "in_distribution", "both"),
        default="post_failure",
        help="pre_action was dropped 2026-09-03 -- see module docstring; "
        "in_distribution added 2026-09-03 for T17",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or _default_run_id(args.task_name, args.seed)
    cfg = RethinkEvalConfig(
        task_name=args.task_name,
        seed=args.seed,
        num_trials_per_task=args.num_trials_per_task,
        server_url=args.server_url,
        output_dir=args.output_dir,
        run_id=run_id,
        num_queries_best_of_n=args.num_queries_best_of_n,
        chunk_size=args.chunk_size,
        num_open_loop_steps=args.num_open_loop_steps,
        num_denoising_steps_action=args.num_denoising_steps_action,
        flip_images=args.flip_images,
        obj_instance_split=args.obj_instance_split,
        layout_and_style_ids=args.layout_and_style_ids,
        controller_configs_path=args.controller_configs_path,
        in_dist_pregrasp_chunks=args.in_dist_pregrasp_chunks,
        in_dist_include_grasp_chunk=args.in_dist_include_grasp_chunk,
        decision_horizon=args.decision_horizon,
        max_local_retry_queries=args.max_local_retry_queries,
        recursive_retry_from_retry_query=args.recursive_retry_from_retry_query,
        alternate_seed_offset=args.alternate_seed_offset,
        alternate_episode_stride=args.alternate_episode_stride,
    )
    validate_config(cfg)
    client = CosmosPolicyClient(
        cfg.server_url,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )
    health = client.health()
    if health.get("status") != "ok":
        raise RuntimeError(f"remote Cosmos Policy server is unhealthy: {health!r}")
    result = run_evaluation(cfg, client=client, anchor_kind=args.anchor_kind)
    print(f"Success rate: {result['success_rate']:.4f}")
    print(f"Total episodes: {result['total_episodes']}")
    print(f"Total successes: {result['total_successes']}")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
