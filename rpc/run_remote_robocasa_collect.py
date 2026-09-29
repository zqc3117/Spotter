"""Collect aligned Cosmos imagined endpoints and realized RoboCasa outcomes.

This is an isolated copy of the parity-verified remote evaluator. It never
writes into the official evaluation roots. Every model query is recorded with
its predicted t+32 endpoint, the actual t+16/t+32 observations, actions, and
offline-only simulator grasp labels for training a sensor-only verifier.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pickle
import random
import re
import sys
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, TextIO

import numpy as np

from .client import CosmosPolicyClient


CONTROLLER_CONFIGS_PATH = Path(
    "cosmos_policy/experiments/robot/robocasa/robocasa_controller_configs.pkl"
)

TASK_MAX_STEPS = {
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
}

FUTURE_IMAGE_KEYS = ("future_image", "future_image2", "future_wrist_image")


def _set_seed_everywhere(seed: int) -> None:
    """Mirror ``cosmos_policy.utils.utils.set_seed_everywhere`` locally."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class InferenceClient(Protocol):
    def infer(
        self,
        observation: Mapping[str, np.ndarray],
        task_description: str,
        **kwargs: Any,
    ) -> Any: ...


@dataclass(frozen=True)
class RemoteEvalConfig:
    task_name: str
    seed: int
    num_trials_per_task: int
    server_url: str
    output_dir: Path
    run_id: str
    num_queries_best_of_n: int = 1
    query_seed_stride: int = 0
    chunk_size: int = 32
    num_open_loop_steps: int = 16
    num_denoising_steps_action: int = 5
    request_future_images: bool = False
    flip_images: bool = True
    env_img_res: int = 224
    robots: str = "PandaMobile"
    controllers: str = "OSC_POSE"
    obj_instance_split: str = "A"
    layout_and_style_ids: str = "TRAIN_NONTEST"
    randomize_cameras: bool = False
    controller_configs_path: Path = CONTROLLER_CONFIGS_PATH


@dataclass
class EpisodeResult:
    episode_index: int
    task_description: str
    success: bool
    episode_length: int
    environment_seed: int
    query_count: int


@dataclass
class EpisodeArtifacts:
    result: EpisodeResult
    primary_images: list[np.ndarray]
    secondary_images: list[np.ndarray]
    wrist_images: list[np.ndarray]
    future_predictions: list[dict[str, np.ndarray]]


@dataclass
class PendingQuery:
    query_index: int
    request_id: str
    timestep: int
    selected_seed: int
    selected_value: float
    pre_observation: dict[str, np.ndarray]
    imagined_observation: dict[str, np.ndarray]
    planned_actions: np.ndarray
    grasped_0: list[str]
    min_target_object_eef_distance: float
    min_gripper_aperture: float
    max_gripper_aperture: float
    min_target_object_eef_distance_t0_t8: float
    min_gripper_aperture_t0_t8: float
    max_gripper_aperture_t0_t8: float
    min_target_object_eef_distance_t0_t16: float
    min_gripper_aperture_t0_t16: float
    max_gripper_aperture_t0_t16: float
    actual_8: dict[str, np.ndarray] | None = None
    grasped_8: list[str] | None = None
    actual_16: dict[str, np.ndarray] | None = None
    grasped_16: list[str] | None = None


TEST_LAYOUT_STYLE_PAIRS = {(1, 1), (2, 2), (4, 4), (6, 9), (7, 10)}


def training_layout_style_pairs(
    excluded_layouts: Sequence[int] = (),
) -> tuple[tuple[int, int], ...]:
    """Return every RoboCasa scene except the five official test scenes."""
    excluded = {int(layout) for layout in excluded_layouts}
    return tuple(
        (layout, style)
        for layout in range(10)
        for style in range(12)
        if layout not in excluded
        if (layout, style) not in TEST_LAYOUT_STYLE_PAIRS
    )


def validate_config(cfg: RemoteEvalConfig) -> None:
    if cfg.task_name not in TASK_MAX_STEPS:
        raise ValueError(f"unsupported official RoboCasa task: {cfg.task_name!r}")
    if cfg.num_trials_per_task < 1:
        raise ValueError("num_trials_per_task must be positive")
    if cfg.query_seed_stride < 0:
        raise ValueError("query_seed_stride must be non-negative")
    if cfg.num_queries_best_of_n != 1:
        raise ValueError("sequential Rethink collection requires num_queries=1")
    if cfg.chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not 1 <= cfg.num_open_loop_steps <= cfg.chunk_size:
        raise ValueError("num_open_loop_steps must be between 1 and chunk_size")
    if cfg.env_img_res != 224:
        raise ValueError("the current RPC protocol requires env_img_res=224")
    if not cfg.run_id or len(_request_id(cfg.run_id, 0, 0)) > 128:
        raise ValueError("run_id is empty or too long for auditable RPC request IDs")


def prepare_observation(obs: Mapping[str, Any], flip_images: bool = False) -> dict[str, np.ndarray]:
    """Mirror the pinned official evaluator's RoboCasa preprocessing."""
    camera_mapping = {
        "primary_image": "robot0_agentview_left_image",
        "secondary_image": "robot0_agentview_right_image",
        "wrist_image": "robot0_eye_in_hand_image",
    }
    observation: dict[str, np.ndarray] = {}
    for output_key, input_key in camera_mapping.items():
        if input_key not in obs:
            raise KeyError(f"RoboCasa observation is missing {input_key!r}")
        image = np.asarray(obs[input_key])
        if flip_images:
            image = np.flipud(image)
        observation[output_key] = image
    try:
        observation["proprio"] = np.concatenate(
            (
                np.asarray(obs["robot0_gripper_qpos"]),
                np.asarray(obs["robot0_eef_pos"]),
                np.asarray(obs["robot0_eef_quat"]),
            )
        )
    except KeyError as exc:
        raise KeyError(f"RoboCasa observation is missing {exc.args[0]!r}") from exc
    # proprio_ext: chassis pos (3) + chassis quat (4), opportunistic --
    # only present on mobile-base RoboCasa envs. Added 2026-09 for pi0.5,
    # whose 16-dim state needs the base pose; the Cosmos backend ignores it
    # (protocol.py treats it as an optional observation key). Silently
    # omitted (not an error) when the env has no base_pos/base_quat, e.g.
    # a fixed-base task.
    if "robot0_base_pos" in obs and "robot0_base_quat" in obs:
        observation["proprio_ext"] = np.concatenate(
            (np.asarray(obs["robot0_base_pos"]), np.asarray(obs["robot0_base_quat"]))
        ).astype(np.float64)
    # proprio_base_rel: EEF pose RELATIVE to the chassis (robosuite's own
    # robot0_base_to_eef_pos/quat sensors) -- opportunistic, same pattern as
    # proprio_ext. Added 2026-09 for GR00T-N1, whose
    # RobocasaSinglePandaGripperDataConfig state uses end_effector_position/
    # rotation_relative (base-relative), NOT the absolute proprio eef fields.
    if "robot0_base_to_eef_pos" in obs and "robot0_base_to_eef_quat" in obs:
        observation["proprio_base_rel"] = np.concatenate(
            (np.asarray(obs["robot0_base_to_eef_pos"]), np.asarray(obs["robot0_base_to_eef_quat"]))
        ).astype(np.float64)
    return observation


def create_robocasa_env(
    cfg: RemoteEvalConfig,
    *,
    seed: int | None,
    episode_idx: int,
) -> Any:
    """Create an environment with the exact official RoboCasa arguments."""
    try:
        import robocasa  # noqa: F401 - importing registers RoboCasa environments
        import robosuite
    except ImportError as exc:
        raise RuntimeError("RoboCasa evaluation requires the pinned RoboCasa and robosuite packages") from exc

    if cfg.layout_and_style_ids == "TRAIN_NONTEST":
        from robosuite.environments.base import REGISTERED_ENVS

        env_class = REGISTERED_ENVS[cfg.task_name]
        all_layout_style_ids = training_layout_style_pairs(
            getattr(env_class, "EXCLUDE_LAYOUTS", ())
        )
    else:
        all_layout_style_ids = ast.literal_eval(cfg.layout_and_style_ids) if cfg.layout_and_style_ids else None
    if all_layout_style_ids:
        scene_index = (cfg.seed + episode_idx) % len(all_layout_style_ids)
        layout_and_style_ids = (all_layout_style_ids[scene_index],)
    else:
        layout_and_style_ids = None

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
        camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
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


def _copy_observation(observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in observation.items()}


def _grasped_object_names(env: Any) -> list[str]:
    """Return privileged labels for storage only; these never enter inference."""
    objects = getattr(env, "objects", {})
    robots = getattr(env, "robots", ())
    if not isinstance(objects, Mapping) or not robots:
        return []
    gripper = getattr(robots[0], "gripper", None)
    if gripper is None:
        return []
    grasped = []
    for name, model in objects.items():
        try:
            if env._check_grasp(gripper, model):
                grasped.append(str(name))
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return sorted(grasped)


def _target_object_eef_distance(
    env: Any, observation: Mapping[str, np.ndarray]
) -> float:
    body_ids = getattr(env, "obj_body_id", {})
    if not isinstance(body_ids, Mapping) or "obj" not in body_ids:
        return float("inf")
    eef_position = np.asarray(observation["proprio"], dtype=np.float64)[2:5]
    try:
        object_position = np.asarray(
            env.sim.data.body_xpos[int(body_ids["obj"])], dtype=np.float64
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return float("inf")
    return float(np.linalg.norm(object_position - eef_position))


def _gripper_aperture(observation: Mapping[str, np.ndarray]) -> float:
    return float(np.linalg.norm(np.asarray(observation["proprio"], dtype=np.float64)[:2]))


def _save_query_sample(
    sample_root: Path,
    cfg: RemoteEvalConfig,
    episode_idx: int,
    pending: PendingQuery,
    actual_32: Mapping[str, np.ndarray],
    grasped_0: list[str],
    grasped_32: list[str],
    executed_actions: list[np.ndarray],
    *,
    layout_id: int | None,
    style_id: int | None,
) -> None:
    if (
        pending.actual_8 is None
        or pending.grasped_8 is None
        or pending.actual_16 is None
        or pending.grasped_16 is None
    ):
        return
    episode_root = sample_root / f"episode_{episode_idx:04d}"
    episode_root.mkdir(parents=True, exist_ok=True)
    stem = f"query_{pending.query_index:05d}_t{pending.timestep:04d}"
    array_path = episode_root / f"{stem}.npz"
    metadata_path = episode_root / f"{stem}.json"
    actual_32_copy = _copy_observation(actual_32)
    action_stop = min(len(executed_actions), pending.timestep + cfg.chunk_size)
    actual_actions = np.asarray(executed_actions[pending.timestep:action_stop], dtype=np.float64)
    arrays = {
        "pre_primary": pending.pre_observation["primary_image"],
        "pre_secondary": pending.pre_observation["secondary_image"],
        "pre_wrist": pending.pre_observation["wrist_image"],
        "pre_proprio": pending.pre_observation["proprio"],
        "imagined_primary_t32": pending.imagined_observation["future_image"],
        "imagined_secondary_t32": pending.imagined_observation["future_image2"],
        "imagined_wrist_t32": pending.imagined_observation["future_wrist_image"],
        "actual_primary_t8": pending.actual_8["primary_image"],
        "actual_secondary_t8": pending.actual_8["secondary_image"],
        "actual_wrist_t8": pending.actual_8["wrist_image"],
        "actual_proprio_t8": pending.actual_8["proprio"],
        "actual_primary_t16": pending.actual_16["primary_image"],
        "actual_secondary_t16": pending.actual_16["secondary_image"],
        "actual_wrist_t16": pending.actual_16["wrist_image"],
        "actual_proprio_t16": pending.actual_16["proprio"],
        "actual_primary_t32": actual_32_copy["primary_image"],
        "actual_secondary_t32": actual_32_copy["secondary_image"],
        "actual_wrist_t32": actual_32_copy["wrist_image"],
        "actual_proprio_t32": actual_32_copy["proprio"],
        "planned_actions_t0_t32": np.asarray(pending.planned_actions, dtype=np.float64),
        "executed_actions_t0_t32": actual_actions,
    }
    temporary = array_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, array_path)
    gripper = pending.planned_actions[:, -1]
    metadata = {
        "schema": "cosmos_rethink_query/v1",
        "array_file": array_path.name,
        "episode_index": episode_idx,
        "query_index": pending.query_index,
        "request_id": pending.request_id,
        "query_timestep": pending.timestep,
        "selected_seed": pending.selected_seed,
        "selected_value": pending.selected_value,
        "task_name": cfg.task_name,
        "environment_seed": (cfg.seed + episode_idx) * 256,
        "obj_instance_split": cfg.obj_instance_split,
        "layout_id": layout_id,
        "style_id": style_id,
        "imagined_horizon_actions": cfg.chunk_size,
        "first_actual_horizon_actions": cfg.num_open_loop_steps,
        "actual_actions_recorded": int(actual_actions.shape[0]),
        "planned_gripper_min": float(np.min(gripper)),
        "planned_gripper_max": float(np.max(gripper)),
        "grasped_objects_t0": grasped_0,
        "grasped_objects_t8": pending.grasped_8,
        "grasped_objects_t16": pending.grasped_16,
        "grasped_objects_t32": grasped_32,
        "any_grasp_t0": bool(grasped_0),
        "any_grasp_t8": bool(pending.grasped_8),
        "any_grasp_t16": bool(pending.grasped_16),
        "any_grasp_t32": bool(grasped_32),
        "target_grasp_t0": "obj" in grasped_0,
        "target_grasp_t8": "obj" in pending.grasped_8,
        "target_grasp_t16": "obj" in pending.grasped_16,
        "target_grasp_t32": "obj" in grasped_32,
        "min_target_object_eef_distance_t0_t8": pending.min_target_object_eef_distance_t0_t8,
        "min_gripper_aperture_t0_t8": pending.min_gripper_aperture_t0_t8,
        "max_gripper_aperture_t0_t8": pending.max_gripper_aperture_t0_t8,
        "min_target_object_eef_distance_t0_t16": pending.min_target_object_eef_distance_t0_t16,
        "min_gripper_aperture_t0_t16": pending.min_gripper_aperture_t0_t16,
        "max_gripper_aperture_t0_t16": pending.max_gripper_aperture_t0_t16,
        "min_target_object_eef_distance_t0_t32": pending.min_target_object_eef_distance,
        "min_gripper_aperture_t0_t32": pending.min_gripper_aperture,
        "max_gripper_aperture_t0_t32": pending.max_gripper_aperture,
    }
    _atomic_write_json(metadata_path, metadata)


def _request_id(run_id: str, episode_idx: int, timestep: int) -> str:
    return f"run-{run_id}/episode-{episode_idx:03d}/timestep-{timestep:04d}"


def _to_environment_action(action: np.ndarray, env_action_dim: int) -> np.ndarray:
    action = np.asarray(action)
    if action.shape == (12,):
        # Native 12-dim action (e.g. a policy whose own action space already
        # includes the mobile base, pi0.5/GR00T-N1 -- 2026-09) -- pass
        # through untouched, never reinterpreted as the 7-dim arm-only
        # convention. Only valid when the environment itself is 12-dim.
        if env_action_dim != 12:
            raise RuntimeError(
                f"remote policy returned a 12-dim action but the RoboCasa environment is "
                f"{env_action_dim}-dim"
            )
        return action
    if action.shape != (7,):
        raise RuntimeError(f"remote policy action must have shape (7,) or (12,), got {action.shape}")
    if env_action_dim == 7:
        return action
    if env_action_dim == 12:
        mobile = np.asarray([0.0, 0.0, 0.0, 0.0, -1.0], dtype=action.dtype)
        return np.concatenate((action, mobile))
    raise RuntimeError(f"unsupported RoboCasa environment action dimension: {env_action_dim}")


def _validate_single_response(
    response: Any,
    *,
    request_id: str,
    base_seed: int,
    num_queries: int,
    chunk_size: int,
    require_future_images: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict[str, np.ndarray]]:
    if num_queries != 1:
        raise RuntimeError("sequential Rethink accepts exactly one policy sample per query")
    if getattr(response, "request_id", None) != request_id:
        raise RuntimeError("RPC response request_id does not match the auditable request ID")

    seeds = np.asarray(getattr(response, "seeds", None))
    actions = np.asarray(getattr(response, "actions", None))
    values = np.asarray(getattr(response, "values", None))
    expected_seeds = np.arange(base_seed, base_seed + num_queries, dtype=np.int64)
    if seeds.shape != (num_queries,) or seeds.dtype != np.int64:
        raise RuntimeError(f"RPC returned invalid single-query seed shape/type: {seeds.shape}, {seeds.dtype}")
    if not np.array_equal(seeds, expected_seeds):
        raise RuntimeError(f"RPC seed {seeds.tolist()} != expected {expected_seeds.tolist()}")
    if actions.shape != (num_queries, chunk_size, 7) or actions.dtype != np.float64:
        raise RuntimeError(
            "RPC must return one full-precision float64 action chunk with shape "
            f"({num_queries}, {chunk_size}, 7); got {actions.shape} {actions.dtype}"
        )
    if values.shape != (num_queries,) or values.dtype != np.float64:
        raise RuntimeError(
            f"RPC must return one float64 value for the single query; got {values.shape} {values.dtype}"
        )
    if not np.isfinite(actions).all() or not np.isfinite(values).all():
        raise RuntimeError("RPC returned non-finite actions or values")

    # There is no candidate ranking here. The leading dimension is the RPC
    # batch dimension and is required to contain exactly one policy sample.
    selected_index = 0
    selected_future: dict[str, np.ndarray] = {}
    future_images = getattr(response, "future_images", None)
    if not isinstance(future_images, Mapping):
        raise RuntimeError("RPC response has no future_images mapping")
    if require_future_images:
        missing = [key for key in FUTURE_IMAGE_KEYS if key not in future_images]
        if missing:
            raise RuntimeError(f"RPC lacks requested future image outputs: {missing}")
        for key in FUTURE_IMAGE_KEYS:
            batch = np.asarray(future_images[key])
            if batch.ndim != 4 or batch.shape[0] != num_queries or batch.shape[-1] != 3:
                raise RuntimeError(f"RPC future image {key!r} has invalid shape {batch.shape}")
            if batch.dtype != np.uint8:
                raise RuntimeError(f"RPC future image {key!r} must be uint8, got {batch.dtype}")
            selected_future[key] = batch[selected_index]
    return seeds, actions, values, selected_index, selected_future


def _write_query_record(stream: TextIO, record: Mapping[str, Any]) -> None:
    stream.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def run_episode(
    cfg: RemoteEvalConfig,
    env: Any,
    task_description: str,
    client: InferenceClient,
    episode_idx: int,
    query_log: TextIO,
    log_file: TextIO,
) -> EpisodeArtifacts:
    """Run one already-reset episode using an episode-local action queue."""
    obs = None
    for _ in range(10):
        dummy_action = np.zeros(env.action_spec[0].shape)
        obs, _, _, _ = env.step(dummy_action)
    if obs is None:
        raise RuntimeError("RoboCasa stabilization produced no observation")

    action_queue: deque[np.ndarray] = deque()
    pending_queries: list[PendingQuery] = []
    executed_actions: list[np.ndarray] = []
    sample_root = cfg.output_dir / "rethink_samples" / cfg.task_name / cfg.run_id
    primary_images: list[np.ndarray] = []
    secondary_images: list[np.ndarray] = []
    wrist_images: list[np.ndarray] = []
    future_predictions: list[dict[str, np.ndarray]] = []
    success = False
    episode_length = 0
    query_count = 0

    for timestep in range(TASK_MAX_STEPS[cfg.task_name]):
        observation = prepare_observation(obs, cfg.flip_images)
        primary_images.append(observation["primary_image"])
        secondary_images.append(observation["secondary_image"])
        wrist_images.append(observation["wrist_image"])

        grasped_now = _grasped_object_names(env)
        target_object_eef_distance = _target_object_eef_distance(env, observation)
        gripper_aperture = _gripper_aperture(observation)
        for pending in list(pending_queries):
            pending.min_target_object_eef_distance = min(
                pending.min_target_object_eef_distance, target_object_eef_distance
            )
            pending.min_gripper_aperture = min(
                pending.min_gripper_aperture, gripper_aperture
            )
            pending.max_gripper_aperture = max(
                pending.max_gripper_aperture, gripper_aperture
            )
            elapsed = timestep - pending.timestep
            if elapsed <= 8:
                pending.min_target_object_eef_distance_t0_t8 = min(
                    pending.min_target_object_eef_distance_t0_t8,
                    target_object_eef_distance,
                )
                pending.min_gripper_aperture_t0_t8 = min(
                    pending.min_gripper_aperture_t0_t8, gripper_aperture
                )
                pending.max_gripper_aperture_t0_t8 = max(
                    pending.max_gripper_aperture_t0_t8, gripper_aperture
                )
            if elapsed <= cfg.num_open_loop_steps:
                pending.min_target_object_eef_distance_t0_t16 = min(
                    pending.min_target_object_eef_distance_t0_t16,
                    target_object_eef_distance,
                )
                pending.min_gripper_aperture_t0_t16 = min(
                    pending.min_gripper_aperture_t0_t16, gripper_aperture
                )
                pending.max_gripper_aperture_t0_t16 = max(
                    pending.max_gripper_aperture_t0_t16, gripper_aperture
                )
            if elapsed == 8:
                pending.actual_8 = _copy_observation(observation)
                pending.grasped_8 = list(grasped_now)
            if elapsed == cfg.num_open_loop_steps:
                pending.actual_16 = _copy_observation(observation)
                pending.grasped_16 = list(grasped_now)
            if elapsed == cfg.chunk_size:
                _save_query_sample(
                    sample_root,
                    cfg,
                    episode_idx,
                    pending,
                    observation,
                    pending.grasped_0,
                    grasped_now,
                    executed_actions,
                    layout_id=getattr(env, "layout_id", None),
                    style_id=getattr(env, "style_id", None),
                )
                pending_queries.remove(pending)

        if not action_queue:
            request_id = _request_id(cfg.run_id, episode_idx, timestep)
            query_seed = cfg.seed + query_count * cfg.query_seed_stride
            response = client.infer(
                observation,
                task_description,
                seed=query_seed,
                num_queries=cfg.num_queries_best_of_n,
                num_denoising_steps_action=cfg.num_denoising_steps_action,
                return_future_images=cfg.request_future_images,
                request_id=request_id,
            )
            seeds, actions, values, selected_index, selected_future = _validate_single_response(
                response,
                request_id=request_id,
                base_seed=query_seed,
                num_queries=cfg.num_queries_best_of_n,
                chunk_size=cfg.chunk_size,
                require_future_images=cfg.request_future_images,
            )
            selected_actions = actions[selected_index, : cfg.num_open_loop_steps]
            action_queue.extend(selected_actions[index] for index in range(len(selected_actions)))
            if selected_future:
                future_predictions.append(selected_future)
                pending_queries.append(
                    PendingQuery(
                        query_index=query_count,
                        request_id=request_id,
                        timestep=timestep,
                        selected_seed=int(seeds[selected_index]),
                        selected_value=float(values[selected_index]),
                        pre_observation=_copy_observation(observation),
                        imagined_observation=_copy_observation(selected_future),
                        planned_actions=np.array(actions[selected_index], copy=True),
                        grasped_0=list(grasped_now),
                        min_target_object_eef_distance=target_object_eef_distance,
                        min_gripper_aperture=gripper_aperture,
                        max_gripper_aperture=gripper_aperture,
                        min_target_object_eef_distance_t0_t8=target_object_eef_distance,
                        min_gripper_aperture_t0_t8=gripper_aperture,
                        max_gripper_aperture_t0_t8=gripper_aperture,
                        min_target_object_eef_distance_t0_t16=target_object_eef_distance,
                        min_gripper_aperture_t0_t16=gripper_aperture,
                        max_gripper_aperture_t0_t16=gripper_aperture,
                    )
                )
            _write_query_record(
                query_log,
                {
                    "episode_index": episode_idx,
                    "request_id": request_id,
                    "returned_seeds": [int(value) for value in seeds],
                    "selected_index": selected_index,
                    "selected_seed": int(seeds[selected_index]),
                    "timestep": timestep,
                    "values": [float(value) for value in values],
                    "value_dtype": str(values.dtype),
                    "action_dtype": str(actions.dtype),
                },
            )
            log_file.write(
                f"t={timestep}: selected seed {int(seeds[selected_index])} "
                f"with value {values[selected_index]!r}\n"
            )
            log_file.flush()
            query_count += 1

        policy_action = np.asarray(action_queue.popleft(), dtype=np.float64)
        executed_actions.append(np.array(policy_action, copy=True))
        action = _to_environment_action(policy_action, int(env.action_dim))
        obs, _, _, _ = env.step(action)
        episode_length += 1
        if env._check_success():
            success = True
            log_file.write(f"Success detected at timestep {timestep}\n")
            log_file.flush()
            break

    return EpisodeArtifacts(
        result=EpisodeResult(
            episode_index=episode_idx,
            task_description=task_description,
            success=success,
            episode_length=episode_length,
            environment_seed=(cfg.seed + episode_idx) * 256,
            query_count=query_count,
        ),
        primary_images=primary_images,
        secondary_images=secondary_images,
        wrist_images=wrist_images,
        future_predictions=future_predictions,
    )


def _save_episode_videos(
    cfg: RemoteEvalConfig,
    artifacts: EpisodeArtifacts,
    rollout_dir: Path,
    log_file: TextIO,
    video_saver: Callable[..., Any],
    future_video_saver: Callable[..., Any],
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
    if not artifacts.future_predictions:
        return
    future_video_saver(
        artifacts.primary_images,
        artifacts.secondary_images,
        artifacts.wrist_images,
        result.episode_index,
        success=result.success,
        task_description=result.task_description,
        rollout_data_dir=str(rollout_dir),
        chunk_size=cfg.chunk_size,
        num_open_loop_steps=cfg.num_open_loop_steps,
        future_primary_image_predictions=[item["future_image"] for item in artifacts.future_predictions],
        future_secondary_image_predictions=[item["future_image2"] for item in artifacts.future_predictions],
        future_wrist_image_predictions=[item["future_wrist_image"] for item in artifacts.future_predictions],
        show_diff=False,
        log_file=log_file,
        show_timestep=True,
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_evaluation(
    cfg: RemoteEvalConfig,
    *,
    client: InferenceClient,
    env_factory: Callable[..., Any] = create_robocasa_env,
    video_saver: Callable[..., Any] | None = None,
    future_video_saver: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    validate_config(cfg)
    _set_seed_everywhere(cfg.seed)
    if video_saver is None or future_video_saver is None:
        try:
            from cosmos_policy.experiments.robot.robocasa.robocasa_utils import (
                save_rollout_video,
                save_rollout_video_with_future_image_predictions,
            )
        except ImportError as exc:
            raise RuntimeError("official RoboCasa video helpers are unavailable") from exc
        video_saver = video_saver or save_rollout_video
        future_video_saver = future_video_saver or save_rollout_video_with_future_image_predictions

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir = cfg.output_dir / "rollout_data" / f"{cfg.task_name}--{cfg.run_id}"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    result_path = cfg.output_dir / "result.json"
    episodes: list[EpisodeResult] = []

    with (cfg.output_dir / "eval.log").open("a", encoding="utf-8", buffering=1) as log_file, (
        cfg.output_dir / "queries.jsonl"
    ).open("a", encoding="utf-8", buffering=1) as query_log:
        for episode_idx in range(cfg.num_trials_per_task):
            environment_seed = (cfg.seed + episode_idx) * 256
            env = env_factory(cfg, seed=environment_seed, episode_idx=episode_idx)
            try:
                # Reset exactly once. The language must be read after this reset because
                # RoboCasa can change the sampled task instance during reset.
                env.reset()
                task_description = env.get_ep_meta()["lang"]
                if not isinstance(task_description, str) or not task_description:
                    raise RuntimeError("RoboCasa returned an invalid post-reset language instruction")
                log_file.write(f"Starting episode {episode_idx}: {task_description}\n")
                artifacts = run_episode(
                    cfg,
                    env,
                    task_description,
                    client,
                    episode_idx,
                    query_log,
                    log_file,
                )
                _save_episode_videos(
                    cfg,
                    artifacts,
                    rollout_dir,
                    log_file,
                    video_saver,
                    future_video_saver,
                )
                episodes.append(artifacts.result)
            finally:
                env.close()

            successes = sum(int(item.success) for item in episodes)
            partial = _result_document(cfg, episodes, complete=False)
            _atomic_write_json(result_path, partial)
            log_file.write(f"Success: {episodes[-1].success}\n")
            log_file.write(f"# episodes completed so far: {len(episodes)}\n")
            log_file.write(f"# successes: {successes} ({successes / len(episodes) * 100:.1f}%)\n")

    result = _result_document(cfg, episodes, complete=True)
    _atomic_write_json(result_path, result)
    return result


def _result_document(
    cfg: RemoteEvalConfig,
    episodes: list[EpisodeResult],
    *,
    complete: bool,
) -> dict[str, Any]:
    successes = sum(int(item.success) for item in episodes)
    total = len(episodes)
    return {
        "complete": complete and total == cfg.num_trials_per_task,
        "episodes": [asdict(item) for item in episodes],
        "num_queries_best_of_n": cfg.num_queries_best_of_n,
        "run_id": cfg.run_id,
        "seed": cfg.seed,
        "success_rate": successes / total if total else 0.0,
        "task_name": cfg.task_name,
        "total_episodes": total,
        "total_successes": successes,
        "validated": complete and total == cfg.num_trials_per_task,
    }


def _default_run_id(task_name: str, seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    task_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", task_name).strip("-")
    digest = hashlib.sha256(f"{task_name}:{seed}:{stamp}".encode()).hexdigest()[:8]
    return f"{task_slug}-seed{seed}-{stamp}-{digest}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, dest="task_name")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--trials", required=True, type=int, dest="num_trials_per_task")
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--num-queries",
        type=int,
        default=1,
        choices=(1,),
        dest="num_queries_best_of_n",
        help="single sequential policy sample per request (must be 1)",
    )
    parser.add_argument("--query-seed-stride", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--num-open-loop-steps", type=int, default=16)
    parser.add_argument("--num-denoising-steps-action", type=int, default=5)
    parser.add_argument("--request-future-images", action="store_true")
    parser.add_argument("--obj-instance-split", default="A", choices=("A",))
    parser.add_argument("--layout-and-style-ids", default="TRAIN_NONTEST")
    parser.add_argument("--no-flip-images", action="store_false", dest="flip_images")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--controller-configs-path", type=Path, default=CONTROLLER_CONFIGS_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or _default_run_id(args.task_name, args.seed)
    cfg = RemoteEvalConfig(
        task_name=args.task_name,
        seed=args.seed,
        num_trials_per_task=args.num_trials_per_task,
        server_url=args.server_url,
        output_dir=args.output_dir,
        run_id=run_id,
        num_queries_best_of_n=args.num_queries_best_of_n,
        query_seed_stride=args.query_seed_stride,
        chunk_size=args.chunk_size,
        num_open_loop_steps=args.num_open_loop_steps,
        num_denoising_steps_action=args.num_denoising_steps_action,
        request_future_images=args.request_future_images,
        obj_instance_split=args.obj_instance_split,
        layout_and_style_ids=args.layout_and_style_ids,
        flip_images=args.flip_images,
        controller_configs_path=args.controller_configs_path,
    )
    client = CosmosPolicyClient(
        cfg.server_url,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )
    health = client.health()
    if health.get("status") != "ok":
        raise RuntimeError(f"remote Cosmos Policy server is unhealthy: {health!r}")
    result = run_evaluation(cfg, client=client)
    print(f"Success rate: {result['success_rate']:.4f}")
    print(f"Total episodes: {result['total_episodes']}")
    print(f"Total successes: {result['total_successes']}")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
