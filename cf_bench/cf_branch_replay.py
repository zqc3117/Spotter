"""V3b (RUNBOOK C-0): reconstruct env + snapshot-restore + branch-label
replay for one already-collected anchor/branch, WITHOUT touching
`rpc/run_remote_robocasa_cf_branch_collect.py` (that file is live-edited by
the Cosmos collection window; this module reproduces its label semantics
independently by calling the same module-level building blocks it uses --
`_privileged_snapshot`, `_attempt_evidence`, `evaluate_retry_recovery`,
`create_robocasa_env` -- rather than importing its `execute_steps` /
`update_tracker`, which are private closures nested inside
`run_episode_cf_branch_collect` and therefore not importable).

The replayed `local_label`/`t32_label` computation below is a byte-for-byte
port of that nested logic (verified line-by-line against
`rpc/run_remote_robocasa_cf_branch_collect.py`'s `execute_steps` and
`update_tracker`, 2026-09-03); if that file's branch-labeling logic changes,
this module needs a matching update, but it deliberately does NOT import
across that boundary so this validity line's replay behavior cannot be
silently changed out from under it by an edit landing in the other window.
"""

from __future__ import annotations

import os

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rpc.run_remote_robocasa_cf_branch_collect import (  # noqa: E402
    DEFAULT_TEST_LAYOUTS,
    PendingExpectation,
    RethinkEvalConfig,
    _array_sha256,
    _attempt_evidence,
    _privileged_snapshot,
    create_robocasa_env,
    official_layout_style_pair,
)
from rpc.run_remote_robocasa_collect import (  # noqa: E402
    TASK_MAX_STEPS,
    _copy_observation,
    _to_environment_action,
    _validate_single_response,
    prepare_observation,
)
from rpc.oracle_retry_predicates import evaluate_retry_recovery  # noqa: E402
from rpc.client import CosmosPolicyClient  # noqa: E402

from cf_bench.snapshot_io import load_snapshot, restore_simulator_snapshot  # noqa: E402


def build_env_and_cfg(
    anchor: Mapping[str, Any], *, run_id: str, output_dir: Path
) -> tuple[Any, RethinkEvalConfig]:
    """Reconstructs the exact env the anchor was collected under. Layout and
    object-instance split come from the anchor record itself
    (`obj_instance_split`). `layout_and_style_ids` strategy depends on that
    split -- TRAIN-A rounds (obj_instance_split="A") used the dynamically
    computed "TRAIN_NONTEST" pairing; dev/test rounds (obj_instance_split=
    "B") used the fixed `DEFAULT_TEST_LAYOUTS` list instead (verified
    empirically 2026-09-03: a real seed195/B anchor's recorded
    layout_id=1/style_id=1 only reproduces under DEFAULT_TEST_LAYOUTS --
    "TRAIN_NONTEST" gives the right layout but the wrong style for that
    same anchor). Caller should assert the recomputed pair equals the
    anchor's own `layout_id`/`style_id` before trusting anything downstream
    (`check_layout_style_match` does exactly that).
    """
    layout_and_style_ids = (
        "TRAIN_NONTEST" if str(anchor.get("obj_instance_split", "B")) == "A" else DEFAULT_TEST_LAYOUTS
    )
    cfg = RethinkEvalConfig(
        task_name=str(anchor["task"]),
        seed=int(anchor["env_seed"]),
        num_trials_per_task=1,
        server_url="http://127.0.0.1:1",  # unused -- no policy queries in a replay
        output_dir=output_dir,
        run_id=run_id,
        obj_instance_split=str(anchor.get("obj_instance_split", "B")),
        layout_and_style_ids=layout_and_style_ids,
        # CONTROLLER_CONFIGS_PATH (rpc/run_remote_robocasa_collect.py) is a
        # bare relative path ("cosmos_policy/experiments/robot/robocasa/...")
        # that only resolves when CWD happens to be
        # $COSMOS_POLICY_ROOT/src/cosmos-policy/ (the collector's own
        # convention). Pinned to an absolute location instead so this replay
        # module works regardless of caller CWD. COSMOS_POLICY_ROOT (see
        # env/env.sh.example) sets the prefix.
        controller_configs_path=Path(
            os.environ.get("COSMOS_POLICY_ROOT", "/workspace/cosmos_policy")
        ) / "src/cosmos-policy/cosmos_policy/experiments/robot/robocasa/robocasa_controller_configs.pkl",
    )
    episode_idx = int(anchor["episode_index"])
    env = create_robocasa_env(cfg, seed=cfg.seed, episode_idx=episode_idx)
    return env, cfg


def check_layout_style_match(anchor: Mapping[str, Any], cfg: RethinkEvalConfig) -> dict[str, Any]:
    episode_idx = int(anchor["episode_index"])
    layout_id, style_id = official_layout_style_pair(cfg, episode_idx)
    matches = layout_id == int(anchor["layout_id"]) and style_id == int(anchor["style_id"])
    return {
        "recomputed_layout_id": layout_id,
        "recomputed_style_id": style_id,
        "anchor_layout_id": int(anchor["layout_id"]),
        "anchor_style_id": int(anchor["style_id"]),
        "matches": matches,
    }


def frozen_observation_from_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The anchor's pre-branch observation, as persisted in the anchor's
    npz under the `pre_*` keys (see `run_episode_cf_branch_collect`'s
    `anchor_arrays` construction) -- renamed back to the
    primary_image/secondary_image/wrist_image/proprio keys
    `prepare_observation` produces, since that's what `_privileged_snapshot`
    and `restore_simulator_snapshot`'s equality checks expect."""
    return {
        "primary_image": np.asarray(arrays["pre_primary"]),
        "secondary_image": np.asarray(arrays["pre_secondary"]),
        "wrist_image": np.asarray(arrays["pre_wrist"]),
        "proprio": np.asarray(arrays["pre_proprio"], dtype=np.float64),
    }


def restore_anchor(
    env: Any,
    snapshot_dir: Path,
    frozen_observation: Mapping[str, np.ndarray],
    *,
    flip_images: bool,
) -> Mapping[str, np.ndarray]:
    """Loads + restores the anchor's snapshot onto `env`; raises
    RuntimeError (propagated from cf_bench.snapshot_io) if the restored
    state hash or policy-input observation does not match exactly -- this
    IS the V3b step-1 gate condition, not swallowed here."""
    snapshot = load_snapshot(snapshot_dir, env)
    _raw, policy_observation, checks = restore_simulator_snapshot(
        env, snapshot, expected_observation=frozen_observation, flip_images=flip_images
    )
    if not all(checks[k] for k in checks if k.startswith("policy_input_")):
        raise RuntimeError(f"restore_anchor: policy_input mismatch: {checks}")
    return policy_observation


def replay_branch_labels(
    env: Any,
    cfg: RethinkEvalConfig,
    task_description: str,
    start_observation: Mapping[str, np.ndarray],
    plan32: np.ndarray,
    *,
    origin_timestep: int,
) -> dict[str, Any]:
    """Executes `plan32` (32 actions) from the CURRENT env state (caller
    must have already restored it via `restore_anchor`) and returns
    {local_label, t32_label, ...}, computed with the identical
    local/t32 semantics as `run_episode_cf_branch_collect`'s branch loop:
    first 16 actions decide `local_label` (task success, or oracle
    retry-recovery if not); the second 16 (diagnostic-only, capped at the
    episode's remaining step budget) decide `t32_label` (task success OR
    already-recovered)."""
    task_name = cfg.task_name
    max_steps = TASK_MAX_STEPS[task_name]

    def update_tracker(tracker: PendingExpectation, privileged: Mapping[str, Any], relative_step: int) -> None:
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
        if isinstance(control_distance, (int, float)) and not isinstance(control_distance, bool):
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

    def execute_steps(
        actions: np.ndarray,
        current_observation: Mapping[str, np.ndarray],
        t0: Mapping[str, Any],
        *,
        step_origin: int,
        stop_on_success: bool,
    ) -> dict[str, Any]:
        distance = t0.get("target_object_eef_distance")
        tracker = PendingExpectation(
            policy_request_id="",
            query_index=0,
            origin_timestep=step_origin,
            policy_seed=0,
            selected_seed=0,
            execution_horizon=len(actions),
            mode="cf_branch_replay",
            anchor_id=None,
            retry_ordinal=None,
            task_state_t0=dict(t0),
            task_state_t8=None,
            planned_action_sha256="",
            planned_first16_sha256="",
            min_target_object_eef_distance=(
                float(distance) if isinstance(distance, (int, float)) and not isinstance(distance, bool) else None
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
        prepared = _copy_observation(current_observation)
        raw_observation: Mapping[str, Any] | None = None
        privileged = dict(t0)
        task_succeeded = False
        for index in range(len(actions)):
            policy_action = np.asarray(actions[index], dtype=np.float64)
            tracker.executed_actions.append(np.array(policy_action, copy=True))
            raw_observation, _, _, _ = env.step(_to_environment_action(policy_action, int(env.action_dim)))
            prepared = prepare_observation(raw_observation, cfg.flip_images)
            privileged = _privileged_snapshot(env, task_name, prepared)
            update_tracker(tracker, privileged, index + 1)
            if env._check_success():
                task_succeeded = True
                if stop_on_success:
                    break
        if raw_observation is None:
            raise RuntimeError("plan execution produced no observation")
        executed = np.asarray(tracker.executed_actions, dtype=np.float64)
        return {
            "prepared_observation": prepared,
            "t8": tracker.task_state_t8,
            "t_end": privileged,
            "evidence": _attempt_evidence(tracker),
            "steps": len(tracker.executed_actions),
            "task_success": task_succeeded,
            "executed_hash": _array_sha256(executed),
        }

    t0 = _privileged_snapshot(env, task_name, start_observation)
    plan32 = np.asarray(plan32, dtype=np.float64)
    first16 = plan32[: cfg.num_open_loop_steps]
    branch16 = execute_steps(first16, start_observation, t0, step_origin=origin_timestep, stop_on_success=False)

    local_label = bool(branch16["task_success"])
    branch_decision: dict[str, Any] | None = None
    if not local_label:
        if branch16["t8"] is None:
            raise RuntimeError("cf branch replay did not capture t8 evidence")
        branch_decision = evaluate_retry_recovery(
            task_name, t0, branch16["t8"], branch16["t_end"],
            attempt_evidence=branch16["evidence"], task_description=task_description,
        )
        local_label = bool(branch_decision["recovered"])

    remaining_for_t32 = min(cfg.num_open_loop_steps, max_steps - origin_timestep - branch16["steps"])
    second_half = plan32[cfg.num_open_loop_steps : cfg.num_open_loop_steps + remaining_for_t32]
    if remaining_for_t32 > 0:
        branch32 = execute_steps(
            second_half, branch16["prepared_observation"], branch16["t_end"],
            step_origin=origin_timestep + branch16["steps"], stop_on_success=True,
        )
        t32_label = bool(branch32["task_success"] or local_label)
    else:
        t32_label = local_label

    return {
        "local_label": local_label,
        "t32_label": t32_label,
        "t0": t0,
        "branch16_task_success": branch16["task_success"],
        "branch_decision": branch_decision,
    }


def run_downstream_rollout(
    env: Any,
    cfg: RethinkEvalConfig,
    client: CosmosPolicyClient,
    task_description: str,
    start_observation: Mapping[str, np.ndarray],
    first16_actions: np.ndarray,
    *,
    origin_timestep: int,
    seed_base: int,
    max_queries: int = 40,
) -> dict[str, Any]:
    """V2b: from an already-restored env at `origin_timestep`, execute the
    branch's own recorded first 16 actions (deterministic, matching what T2
    committed for this branch), then hand control to the OFFICIAL policy in
    closed loop -- repeatedly query `client`, execute
    `cfg.num_open_loop_steps` (16) actions from the returned plan, check
    success -- until the episode reaches `TASK_MAX_STEPS[cfg.task_name]` or
    succeeds. Returns `{downstream_label, steps_executed, n_queries}`.

    This does NOT reuse `replay_branch_labels`'s local/t32 semantics (task
    success at a fixed t16/t32 horizon only) -- it's a plain open-loop-
    chunked rollout to episode end, matching how the official baseline is
    normally evaluated end to end.
    """
    max_steps = TASK_MAX_STEPS[cfg.task_name]
    committed = origin_timestep
    observation = start_observation
    task_succeeded = False

    def step_chunk(actions: np.ndarray) -> tuple[Mapping[str, np.ndarray], bool]:
        nonlocal committed
        prepared = _copy_observation(observation)
        succeeded = False
        for action in actions:
            if committed >= max_steps:
                break
            policy_action = np.asarray(action, dtype=np.float64)
            raw, _, _, _ = env.step(_to_environment_action(policy_action, int(env.action_dim)))
            prepared = prepare_observation(raw, cfg.flip_images)
            committed += 1
            if env._check_success():
                succeeded = True
        return prepared, succeeded

    observation, task_succeeded = step_chunk(first16_actions)

    n_queries = 0
    while not task_succeeded and committed < max_steps and n_queries < max_queries:
        # Not part of T2's own deterministic query schedule
        # (`deterministic_policy_seed` validates episode/query-index ranges
        # sized for the ORIGINAL single-episode query budget, which a
        # downstream continuation from a mid-episode anchor exceeds) --
        # this rollout uses its own simple scheme: reproducible given
        # (seed_base, n_queries), not required to match T2's own seeds.
        seed = seed_base + n_queries
        request_id = f"downstream-{seed_base}-q{n_queries:04d}"
        response = client.infer(
            observation, task_description, seed=seed, num_queries=1,
            num_denoising_steps_action=cfg.num_denoising_steps_action,
            return_future_images=False, request_id=request_id,
        )
        _seeds, actions, _values, selected_index, _future = _validate_single_response(
            response, request_id=request_id, base_seed=seed, num_queries=1,
            chunk_size=cfg.chunk_size, require_future_images=False,
        )
        plan = np.asarray(actions[selected_index], dtype=np.float64)
        chunk = plan[: cfg.num_open_loop_steps]
        observation, task_succeeded = step_chunk(chunk)
        n_queries += 1

    return {
        "downstream_label": bool(task_succeeded),
        "steps_executed": committed - origin_timestep,
        "n_queries": n_queries,
    }
