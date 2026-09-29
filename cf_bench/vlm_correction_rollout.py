"""VLM-corrected recovery vs. plain-reseed recovery, from the same
post_failure anchor snapshot (2026-09-08).

Pipeline:
  1. Restore a post_failure anchor's simulator snapshot exactly
     (``cf_bench.cf_branch_replay.restore_anchor`` -- raises if the
     restored state hash or policy-input observation doesn't match the
     anchor byte-for-byte).
  2. Gather VLM input: up to 8 primary + 8 wrist frames covering the 16
     steps BEFORE the failure (``collect_prefailure_frames_via_replay``,
     a real deterministic replay of the ORIGINAL committed trajectory's
     policy queries -- needs a live policy server for the SAME policy/
     checkpoint the anchor was collected under; asserts the replay lands
     on the anchor's own ``state_sha256`` before trusting any frame it
     captured) plus the current post-restore frame at 512x512 from all
     three cameras (``render_camera_512``, an independent high-res
     offscreen render call, decoupled from the env's 224x224 observation
     camera config).
  3. Ask the VLM for {failure_type, object_in_gripper, correction}
     (``cf_bench.vlm_failure_diagnosis.call_vlm``).
  4. Re-restore the SAME snapshot fresh before each rollout attempt (a
     rollout mutates env state, so every arm/seed needs its own restore)
     and run to episode end or success via ``run_recovery_rollout``:
       - baseline arm: original task_description, N distinct fresh seeds.
       - corrected arm: VLM's ``correction`` appended to the original
         task_description, the SAME N seeds (paired, not resampled) so
         the two arms are compared on identical noise draws.
  5. Report recovery rate (successes / N) for each arm.

``enable_determinism()`` (cf_bench/determinism.py) must be called once in
the process hosting the policy server BEFORE any model loads -- this
module does not itself load a policy model, only calls a remote client,
so it has nothing to enable_determinism() on. Call it in the policy
server process. The VLM call always uses temperature=0 (hardcoded default
in cf_bench.vlm_failure_diagnosis.call_vlm) per the same determinism
requirement, since a temperature>0 VLM sample would make the "did the
correction help" comparison non-reproducible run to run.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cf_bench.cf_branch_replay import build_env_and_cfg, restore_anchor
from cf_bench.snapshot_io import load_snapshot
from cf_bench.vlm_failure_diagnosis import VlmDiagnosisInput, call_vlm
from rpc.run_remote_robocasa_cf_branch_collect import (  # noqa: E402
    RethinkEvalConfig,
    deterministic_policy_seed,
    _array_sha256,
)
from rpc.run_remote_robocasa_collect import (  # noqa: E402
    TASK_MAX_STEPS,
    _copy_observation,
    _to_environment_action,
    _validate_single_response,
    prepare_observation,
)

CAMERA_ROBOSUITE_NAMES = {
    "primary": "robot0_agentview_left",
    "secondary": "robot0_agentview_right",
    "wrist": "robot0_eye_in_hand",
}


def render_camera_512(env: Any, *, flip_images: bool = True, size: int = 512) -> dict[str, np.ndarray]:
    """Independent high-res offscreen render, decoupled from the env's
    configured 224x224 observation cameras -- ``env.sim.render`` is a
    direct low-level mujoco/robosuite call that accepts any width/height
    regardless of how the observation cameras were set up at env creation."""
    frames: dict[str, np.ndarray] = {}
    for camera_key, robosuite_name in CAMERA_ROBOSUITE_NAMES.items():
        raw = env.sim.render(camera_name=robosuite_name, width=size, height=size, depth=False)
        frame = np.asarray(raw, dtype=np.uint8)
        if flip_images:
            frame = np.flipud(frame)
        frames[camera_key] = np.ascontiguousarray(frame)
    return frames


def _simulator_state_sha256(env: Any) -> str:
    state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
    return _array_sha256(state)


def collect_prefailure_frames_via_replay(
    cfg: RethinkEvalConfig,
    env: Any,
    client: Any,
    task_description: str,
    *,
    episode_idx: int,
    query_timestep: int,
    expected_state_sha256: str,
    chunk_size: int = 16,
    n_frames: int = 8,
) -> dict[str, list[np.ndarray]]:
    """Re-runs the ORIGINAL committed trajectory's policy queries (query_index
    0..query_timestep/chunk_size - 1, all with retry=False -- the official
    primary-trajectory seed schedule, see ``deterministic_policy_seed``)
    from a freshly-created env at step 0, executing every returned chunk,
    and samples ``n_frames`` evenly-spaced steps of the PRIMARY and WRIST
    cameras from the FINAL chunk (the 16 steps immediately before the
    failure was detected). Requires ``env`` to be a fresh env for the same
    (task, seed, episode_idx) the anchor was collected under (episode step
    0, NOT yet restored to any snapshot) and ``client`` to be a policy
    client pointed at the SAME checkpoint/policy the anchor was collected
    with -- a different checkpoint will diverge and fail the final hash
    check below.

    Raises RuntimeError if the replay's own final simulator state hash does
    not exactly match ``expected_state_sha256`` (the anchor's own
    pre-failure snapshot hash) -- this is the mandatory "state_sha256 after replay
    must match the snapshot" gate; a caller must not trust any frame this function
    returns if it raises.
    """
    if query_timestep % chunk_size != 0:
        raise ValueError(
            f"query_timestep={query_timestep} is not a multiple of chunk_size={chunk_size}"
        )
    num_queries = query_timestep // chunk_size
    if num_queries <= 0:
        raise ValueError("anchor's query_timestep is at or before step 0 -- nothing to replay")

    obs = None
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))
    if obs is None:
        raise RuntimeError("RoboCasa stabilization produced no observation")
    observation = prepare_observation(obs, cfg.flip_images)

    primary_history: list[np.ndarray] = []
    wrist_history: list[np.ndarray] = []

    for query_index in range(num_queries):
        seed = deterministic_policy_seed(cfg, episode_idx=episode_idx, query_index=query_index, retry=False)
        response = client.infer(
            observation,
            task_description,
            seed=seed,
            num_queries=1,
            num_denoising_steps_action=cfg.num_denoising_steps_action,
            return_future_images=False,
            request_id=f"prefailure-replay-{episode_idx}-q{query_index:04d}",
        )
        _seeds, actions, _values, selected_index, _future = _validate_single_response(
            response,
            request_id=f"prefailure-replay-{episode_idx}-q{query_index:04d}",
            base_seed=seed,
            num_queries=1,
            chunk_size=cfg.chunk_size,
            require_future_images=False,
        )
        plan = np.asarray(actions[selected_index], dtype=np.float64)
        chunk = plan[:chunk_size]

        is_final_chunk = query_index == num_queries - 1
        sample_steps = (
            set(np.linspace(0, chunk_size - 1, num=n_frames, dtype=int).tolist())
            if is_final_chunk
            else set()
        )
        for step_in_chunk, action in enumerate(chunk):
            raw, _, _, _ = env.step(_to_environment_action(np.asarray(action, dtype=np.float64), int(env.action_dim)))
            observation = prepare_observation(raw, cfg.flip_images)
            if is_final_chunk and step_in_chunk in sample_steps:
                primary_history.append(np.ascontiguousarray(observation["primary_image"]))
                wrist_history.append(np.ascontiguousarray(observation["wrist_image"]))

    actual_hash = _simulator_state_sha256(env)
    if actual_hash != expected_state_sha256:
        raise RuntimeError(
            "collect_prefailure_frames_via_replay: replayed state_sha256 "
            f"({actual_hash}) does not match anchor's snapshot state_sha256 "
            f"({expected_state_sha256}) -- do not trust the captured frames "
            "(wrong checkpoint/policy, or a non-determinism regression; see "
            "cf_bench/determinism.py)"
        )
    return {"primary": primary_history, "wrist": wrist_history}


def run_recovery_rollout(
    env: Any,
    cfg: RethinkEvalConfig,
    client: Any,
    task_description: str,
    start_observation: Mapping[str, np.ndarray],
    *,
    origin_timestep: int,
    seed_base: int,
    max_queries: int = 40,
) -> dict[str, Any]:
    """Closed-loop rollout to episode end or success, querying the policy
    fresh from step 0 of the restored anchor (no pre-supplied action
    chunk, unlike ``cf_bench.cf_branch_replay.run_downstream_rollout``,
    which replays one specific already-generated branch's first chunk
    before handing off) -- this is the right shape for "give the policy a
    (possibly VLM-corrected) task_description and a fresh seed, see if it
    recovers from here," not for replaying a specific recorded branch."""
    max_steps = TASK_MAX_STEPS[cfg.task_name]
    committed = origin_timestep
    observation = _copy_observation(start_observation)
    task_succeeded = bool(env._check_success())

    n_queries = 0
    while not task_succeeded and committed < max_steps and n_queries < max_queries:
        seed = seed_base + n_queries
        request_id = f"recovery-{seed_base}-q{n_queries:04d}"
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
        for action in chunk:
            if committed >= max_steps:
                break
            raw, _, _, _ = env.step(_to_environment_action(np.asarray(action, dtype=np.float64), int(env.action_dim)))
            observation = prepare_observation(raw, cfg.flip_images)
            committed += 1
            if env._check_success():
                task_succeeded = True
                break
        n_queries += 1

    return {
        "recovered": bool(task_succeeded),
        "steps_executed": committed - origin_timestep,
        "n_queries": n_queries,
    }


def run_vlm_correction_experiment(
    anchor: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    run_id: str,
    output_dir: Path,
    policy_client: Any,
    vlm_base_url: str,
    n_seeds: int = 5,
    seed_stride: int = 10_000,
    max_queries_per_rollout: int = 40,
) -> dict[str, Any]:
    """Top-level orchestration for ONE anchor. Returns a JSON-able summary:
    {anchor_id, vlm_diagnosis, baseline: {recoveries: [...], rate}, corrected: {...}}.
    Every rollout re-restores the anchor's own snapshot fresh (a rollout
    mutates env state) -- (n_seeds baseline + n_seeds corrected) full
    restores per anchor, so this is expensive; call it once per anchor
    from a matrix/manifest driver for a real-scale run, not in a tight
    Python loop over hundreds of anchors in one process."""
    from cf_bench.cf_branch_replay import frozen_observation_from_arrays

    snapshot_dir = Path(anchor["snapshot_dir"])
    # NOTE: prefer task_text (the real language instruction, e.g. "pick the
    # sweet potato from the counter and place it in the cabinet") over
    # task (just the RoboCasa env class name, e.g. "PnPCounterToCab") --
    # both keys are usually present and non-empty, so `or` alone would
    # silently pick the wrong one. Found 2026-09-08 while inspecting a
    # real anchor record end to end.
    task_description = str(
        anchor.get("task_description") or anchor.get("task_text") or anchor.get("task")
    )
    frozen_observation = frozen_observation_from_arrays(arrays)

    env, cfg = build_env_and_cfg(anchor, run_id=run_id, output_dir=output_dir)
    try:
        policy_observation = restore_anchor(env, snapshot_dir, frozen_observation, flip_images=cfg.flip_images)
        current_frames_512 = render_camera_512(env, flip_images=cfg.flip_images)
        meta = load_snapshot(snapshot_dir, env)
        expected_state_sha256 = meta.state_sha256

        gripper_open = bool(np.linalg.norm(np.asarray(arrays["pre_proprio"], dtype=np.float64)[:2]) > 0.02)

        prefailure_synthetic = True
        primary_hist: list[np.ndarray] = [arrays["pre_primary"]]
        wrist_hist: list[np.ndarray] = [arrays["pre_wrist"]]
        for branch_id in ("b01", "b02", "b03", "b04"):
            if f"{branch_id}_actual_primary_t16" in arrays:
                primary_hist.append(arrays[f"{branch_id}_actual_primary_t16"])
                wrist_hist.append(arrays[f"{branch_id}_actual_wrist_t16"])
        primary_hist = primary_hist[:8]
        wrist_hist = wrist_hist[:8]

        vlm_input = VlmDiagnosisInput(
            task_description=task_description,
            gripper_open=gripper_open,
            prefailure_primary_frames=primary_hist,
            prefailure_wrist_frames=wrist_hist,
            current_frames=current_frames_512,
            prefailure_frames_synthetic=prefailure_synthetic,
        )
        diagnosis = call_vlm(vlm_input, base_url=vlm_base_url)

        result: dict[str, Any] = {
            "anchor_id": anchor["anchor_id"],
            "task_description": task_description,
            "vlm_diagnosis": diagnosis,
            "expected_state_sha256": expected_state_sha256,
            "baseline": {"recoveries": []},
            "corrected": {"recoveries": []},
        }
        if diagnosis.get("parse_error"):
            result["skipped"] = "vlm_parse_error"
            return result

        corrected_task_description = f"{task_description} ({diagnosis['correction']})"

        for arm, description in (
            ("baseline", task_description),
            ("corrected", corrected_task_description),
        ):
            for seed_offset in range(n_seeds):
                policy_observation = restore_anchor(
                    env, snapshot_dir, frozen_observation, flip_images=cfg.flip_images
                )
                seed_base = int(anchor["env_seed"]) * seed_stride + 1 + seed_offset
                rollout = run_recovery_rollout(
                    env, cfg, policy_client, description, policy_observation,
                    origin_timestep=int(anchor["query_timestep"]),
                    seed_base=seed_base,
                    max_queries=max_queries_per_rollout,
                )
                result[arm]["recoveries"].append(rollout)

        for arm in ("baseline", "corrected"):
            recoveries = result[arm]["recoveries"]
            result[arm]["recovery_rate"] = (
                sum(1 for r in recoveries if r["recovered"]) / len(recoveries) if recoveries else None
            )
        return result
    finally:
        try:
            env.close()
        except Exception:
            pass
