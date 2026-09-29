"""Create and reset failure states in-process, avoiding the cross-process env reconstruction bug.

## Why this module exists

The original design was "restore a previously collected failure snapshot". Testing
(2026-09-11) showed this does not work: rebuilding the environment in a new process with
``cf_bench.cf_branch_replay.build_env_and_cfg`` **reproduces layout/style exactly but not
the object instances**. One post_failure anchor per task from cf_phase0_test_both
(collected 2026-09-10), eight tasks in total:

    layout/style 8/8 correct, objects 0/8 correct
    (anchor recorded 'hot dog', rebuilt 'mug'; recorded 'cheese', rebuilt 'orange' ...)

Applying a sponge's physical state to a mug makes the failure state meaningless. This is
not a stale batch of anchors: both train (2026-09-02) and test (2026-09-10) behave the
same, so the problem is cross-process reconstruction itself.

The M1 scoring pipeline never hit this because it only reads stored npz files and never
rebuilds an environment (no simulation runs during scoring).

## How this module avoids it

No reconstruction, no reproduction problem. The env service builds its own env, drives it
into a failure state, and snapshots it **in the same process**. In-process save/restore is
reliable -- the scene objects belong to this very env, and it is the same mechanism the
collector uses for counterfactual branches.

As a side effect, three otherwise painful problems disappear: no anchor pool, no freeze
list, and no question of touching the test set (failure states are generated on the spot).

## Two ways to create a failure state

``scripted``  No policy server needed: open -> move above the object with a
              **deliberate offset** -> descend -> close (on air) -> lift (object stays
              put). The result is a real "missed grasp": gripper closed, empty, right
              next to the object.
``policy``    Needs the Cosmos policy server: run the real policy and use the oracle to
              detect a failed grasp. This is what the original spec asked for; higher
              fidelity, at the cost of needing a policy server on a GPU host.

Both call ``capture()`` at the failure point; afterwards ``restore()`` can rewind the
environment to that moment any number of times so the explorer can retry the same failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import os

import numpy as np

CONTROLLER_CONFIGS_ABS = Path(
    "/workspace/cosmos_policy/src/cosmos-policy/cosmos_policy/"
    "experiments/robot/robocasa/robocasa_controller_configs.pkl"
)

# Default layout/style. We generate failure states ourselves and do not need to match any
# collected anchor, so any fixed valid combination will do (fixed for reproducibility).
# Matches the defaults of rpc/run_remote_robocasa_policy_n1.py (--layout-and-style-ids
# defaults to TRAIN_NONTEST, --obj-instance-split defaults to B). They must match, otherwise
# what we "reproduce" is not the episode the baseline ran -- scene and objects would differ.
DEFAULT_LAYOUT_STYLE = "TRAIN_NONTEST"


@dataclass
class FailureState:
    """A failure state that can be reset to repeatedly."""

    snapshot: Any                      # cf_bench.snapshot_io.SimulatorSnapshot
    observation: Mapping[str, np.ndarray]
    task: str
    instruction: str
    object_name: str | None
    mode: str
    detail: dict[str, Any]


# Chunking parameters per policy family. Cosmos Policy is 32/16 (RethinkEvalConfig defaults);
# pi0.5 is 50/25, taken from rpc/run_remote_robocasa_policy_n1.py's
# ``--chunk-size 50 --n-action-steps 25`` (the pi05 server log reports
# ``chunk_size=50 n_action_steps=25 action_dim=12``).
# ``decision_horizon`` must equal ``num_open_loop_steps`` or validate_config fails
# (this module does not call validate_config, but keep them consistent for other callers).
POLICY_FAMILY_CHUNKING: dict[str, dict[str, int]] = {
    "cosmos": {"chunk_size": 32, "num_open_loop_steps": 16, "decision_horizon": 16},
    "pi05": {"chunk_size": 50, "num_open_loop_steps": 25, "decision_horizon": 25},
}
POLICY_FAMILY_ACTION_DIM: dict[str, int] = {"cosmos": 7, "pi05": 12}
# best-of-K width. The pi05 server has no value head (values are always 0.0), so K>1 only
# wastes compute. Both models' published baselines use a single candidate; control runs must
# match that, otherwise best-of-K makes our measured policy level incomparable with the
# published numbers. Override via environment variable; default 1.
POLICY_FAMILY_NUM_CANDIDATES: dict[str, int] = {
    "cosmos": int(os.environ.get("COSMOS_NUM_CANDIDATES", "1")),
    "pi05": 1,
}


def build_fresh_env(
    task: str,
    *,
    seed: int,
    episode_index: int,
    obj_instance_split: str = "B",
    layout_and_style_ids: str = DEFAULT_LAYOUT_STYLE,
    run_id: str = "recovery-explore",
    output_dir: Path | str = "/tmp/recovery_explore",
    policy_family: str = "cosmos",
) -> tuple[Any, Any]:
    """Build a brand-new env. Does not depend on any anchor record.

    Deliberately avoids ``cf_bench.cf_branch_replay.build_env_and_cfg``: that function is
    written to "rebuild an anchor's scene" and reads the anchor's env_seed/layout_id. We
    have no anchor and nothing to match -- we just want a clean new environment.
    """
    from rpc.run_remote_robocasa_cf_branch_collect import (
        RethinkEvalConfig,
        create_robocasa_env,
    )
    from rpc.run_remote_robocasa_collect import _set_seed_everywhere

    cfg = RethinkEvalConfig(
        task_name=task,
        seed=seed,
        num_trials_per_task=1,
        server_url="http://127.0.0.1:1",   # scripted mode never queries the policy
        output_dir=Path(output_dir),
        run_id=run_id,
        obj_instance_split=obj_instance_split,
        layout_and_style_ids=layout_and_style_ids,
        controller_configs_path=CONTROLLER_CONFIGS_ABS,
        **POLICY_FAMILY_CHUNKING[policy_family],
    )
    # Copies the construction order of rpc/run_remote_robocasa_policy_n1.py (lines 193 / 222-223):
    #     _set_seed_everywhere(cfg.seed)                     # base seed, once
    #     environment_seed = (cfg.seed + episode_idx) * 256  # derived per episode
    #     create_robocasa_env(cfg, seed=environment_seed, episode_idx=...)
    # Note that cfg.seed and the seed passed to create_robocasa_env are **different numbers**:
    # the former determines the layout (official_layout_style_pair uses cfg.seed + episode_idx),
    # the latter is the robosuite environment seed. Feeding both the same value previously
    # reproduced a different episode.
    # Verified: the n1 record PnPStoveToCounter/seed195/ep2 has environment_seed = 50432 = (195+2)*256.
    _set_seed_everywhere(int(seed))
    environment_seed = (int(seed) + int(episode_index)) * 256
    env = create_robocasa_env(cfg, seed=environment_seed, episode_idx=episode_index)
    # The official rollout (rpc/run_remote_robocasa_policy_n1.py line 219) explicitly calls
    # env.reset() once more after building the env. This is not "one extra shuffle": robocasa's
    # reset resamples object categories and placements, so without it you get the placement
    # from env construction, which is not the episode the baseline evaluated. Verified on
    # PnPCounterToCab/seed195/ep10: official has squash at [0.43,-2.27,0.94], we had avocado at
    # [0.41,-0.81,0.97] -- 1.45 m apart, different object, different counter region. This alone
    # put the harness control 24 points below the official baseline.
    env.reset()
    return env, cfg


def stabilize(env: Any, cfg: Any, steps: int = 10) -> Mapping[str, np.ndarray]:
    """Let the scene settle with zero actions -- what the collector does at the start of each episode.

    Objects in a freshly built env are still falling/bouncing; grasping before they settle
    means grasping a moving target.
    """
    from rpc.run_remote_robocasa_collect import prepare_observation

    obs = None
    for _ in range(steps):
        obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))
    if obs is None:
        raise RuntimeError("stabilize produced no observation")
    return prepare_observation(obs, cfg.flip_images)


def capture(env: Any, cfg: Any, *, task: str, instruction: str,
            object_name: str | None, mode: str, detail: dict[str, Any]) -> FailureState:
    """Snapshot the current moment. In-process only -- not written to disk, not cross-process."""
    from cf_bench.snapshot_io import capture_simulator_snapshot
    from rpc.run_remote_robocasa_collect import prepare_observation

    raw = env._get_observations(force_update=True)
    observation = prepare_observation(raw, cfg.flip_images)
    return FailureState(
        snapshot=capture_simulator_snapshot(env),
        observation={k: np.asarray(v).copy() for k, v in observation.items()},
        task=task,
        instruction=instruction,
        object_name=object_name,
        mode=mode,
        detail=detail,
    )


def restore(env: Any, cfg: Any, state: FailureState) -> Mapping[str, np.ndarray]:
    """Rewind the environment to the moment of failure.

    Uses ``cf_bench.snapshot_io.restore_simulator_snapshot`` and passes in the observation
    recorded at capture time for verification. Same process, same env, same objects, so this
    check is **meaningful** (across processes it is skipped or trivially true; see the
    module docstring).
    """
    from cf_bench.snapshot_io import restore_simulator_snapshot

    _raw, policy_observation, checks = restore_simulator_snapshot(
        env,
        state.snapshot,
        expected_observation=state.observation,
        flip_images=cfg.flip_images,
    )
    if not checks.get("simulator_state_hash", False):
        raise RuntimeError("in-process restore failed the state hash check")
    return policy_observation


def target_object_xyz(env: Any) -> np.ndarray | None:
    body_ids = getattr(env, "obj_body_id", {})
    if not isinstance(body_ids, Mapping) or "obj" not in body_ids:
        return None
    return np.asarray(env.sim.data.body_xpos[int(body_ids["obj"])], dtype=np.float64).copy()


def object_display_name(env: Any) -> str | None:
    """Human-readable object name: the cat of the ep_meta entry whose name == "obj".

    An earlier implementation picked the wrong entry (got 'yogurt' while the instruction said
    'sponge') -- that was actually cross-process reconstruction producing the wrong scene, not
    this logic. In-process it is correct.
    """
    try:
        for cfg_entry in (env.get_ep_meta().get("object_cfgs") or ()):
            if str(cfg_entry.get("name")) == "obj":
                info = cfg_entry.get("info") or {}
                cat = info.get("cat")
                if isinstance(cat, str) and cat:
                    return cat
    except Exception:
        pass
    return None


def make_scripted_failure(
    env: Any,
    cfg: Any,
    primitives: Any,
    *,
    task: str,
    miss_offset_m: float = 0.035,
    hover_m: float = 0.10,
    grasp_depth_m: float = 0.005,
    max_object_disturbance_m: float = 0.02,
    offset_ladder: tuple[float, ...] = (0.035, 0.05, 0.07, 0.09, 0.12),
) -> FailureState:
    """Create a clean "missed grasp" failure state without a policy server.

    Grasp with a deliberate offset so the gripper closes on air. The offset points along the
    horizontal object->gripper direction (i.e. out toward the gripper's current side), so the
    failure looks like "reached slightly off" rather than "grabbed at random".

    **Clean** means both of the following:
      1. the oracle says not grasped;
      2. the object was barely disturbed (displacement < ``max_object_disturbance_m``).
    The first alone is not enough -- we have seen the gripper cradle the object and lift it
    8.8 cm while ``is_grasped`` stayed False. That is an ambiguous, about-to-drop state, not
    a missed grasp, and should not be used as an exploration start.

    The offset is adaptive: snapshot once after the scene settles and rewind to it before each
    offset, so attempts do not contaminate each other. If an offset is too small and brushes
    the object, try a larger one until the result is clean.
    """
    stabilize(env, cfg)
    obj = target_object_xyz(env)
    if obj is None:
        raise RuntimeError(f"task {task} has no 'obj' body -- scripted failure needs a graspable object")

    instruction = str(env.get_ep_meta().get("lang") or "")
    name = object_display_name(env)

    # Snapshot after settling: rewind here before each offset so earlier collisions leave no trace
    settled = capture(env, cfg, task=task, instruction=instruction, object_name=name,
                      mode="settled", detail={})

    ladder = tuple(dict.fromkeys((miss_offset_m,) + tuple(offset_ladder)))
    attempts: list[dict[str, Any]] = []
    for offset in ladder:
        if attempts:                       # no rewind on the first attempt, always after that
            restore(env, cfg, settled)
            primitives.invalidate_calibration()

        primitives.open_gripper()
        eef = np.asarray(primitives.env.eef_pos, dtype=np.float64)
        horiz = eef[:2] - obj[:2]
        norm = float(np.linalg.norm(horiz))
        # If the approach direction is undefined (gripper directly above the object), fall back to +x
        direction = horiz / norm if norm > 1e-6 else np.array([1.0, 0.0])
        miss_xy = obj[:2] + direction * offset

        primitives.move_to([float(miss_xy[0]), float(miss_xy[1]), float(obj[2] + hover_m)],
                           gripper=-1.0)      # -1 = actively OPEN
        primitives.move_to([float(miss_xy[0]), float(miss_xy[1]), float(obj[2] - grasp_depth_m)],
                           gripper=-1.0)
        primitives.close_gripper()
        primitives.lift(0.08, gripper="hold")

        grasped = "obj" in set(_grasped_names(env))
        obj_after = target_object_xyz(env)
        moved = float(np.linalg.norm(obj_after - obj)) if obj_after is not None else 0.0
        attempts.append({"miss_offset_m": offset, "grasped": grasped,
                         "object_moved_m": round(moved, 4)})
        if not grasped and moved <= max_object_disturbance_m:
            break
    else:
        raise RuntimeError(
            "no offset on the ladder produced a clean missed grasp; attempts: "
            + json.dumps(attempts)
        )

    eef_now = np.asarray(primitives.env.eef_pos, dtype=np.float64)
    detail = {
        "miss_offset_m": attempts[-1]["miss_offset_m"],
        "hover_m": hover_m,
        "offset_attempts": attempts,
        # The grasp attempt can nudge the object, so record both: the position before acting
        # (which decided the offset direction) and the actual position at failure (what the
        # explorer has to find).
        "object_xyz_before_attempt": [float(v) for v in obj],
        "object_xyz_at_failure": [float(v) for v in obj_after] if obj_after is not None else None,
        "object_moved_m": attempts[-1]["object_moved_m"],
        "eef_object_distance_m": (
            float(np.linalg.norm(eef_now - obj_after)) if obj_after is not None else None
        ),
        "eef_at_failure": [float(v) for v in eef_now],
        "gripper_qpos": [float(v) for v in np.asarray(primitives.env.gripper_qpos)],
    }
    return capture(env, cfg, task=task, instruction=instruction, object_name=name,
                   mode="scripted", detail=detail)


def _grasped_names(env: Any) -> tuple[str, ...]:
    from rpc.run_remote_robocasa_collect import _grasped_object_names

    try:
        return tuple(_grasped_object_names(env))
    except Exception:
        return ()
