"""Milestone 2 for the RoboDojo env service: the TREATMENT arm.

``env_service_robodojo.py`` (milestone 1) implements the control arm and
registers every intervention method as a stub that raises
``NotImplementedError``.  This module supplies the real implementations as a
mixin, so the control path stays exactly as reviewed and the treatment path is
one reviewable file.

    class RoboDojoEnvService(RoboDojoInterventionMixin, MainThreadServeMixin, RpcFacade):

WHAT THE TREATMENT ARM ACTUALLY NEEDS
    ``judge_driver_v7.sh`` drives it with six harness verbs -- ``judge-lock`` /
    ``judge-unlock`` (860-866), ``advance`` for the look-ahead (886),
    ``open-budget`` (920), ``reset-window`` (1112) and ``close-budget`` (1150)
    -- and the model drives it through ``cli/rex.py exec-plan``, i.e. exactly
    one model-facing entry point: ``execute_plan``.  Everything else here
    exists because ``execute_plan`` dispatches to it.

THE FOUR THINGS THAT ARE GENUINELY DIFFERENT FROM ROBOCASA
  1. **Cartesian control is native, and it is cuRobo IK, not OSC.**
     RoboCasa servos an OSC controller in a closed loop.  RoboDojo's
     ``EvalEnv.take_action`` already accepts an ``ee``-typed action
     (``src/eval_client/eval_env.py:508-545``) and resolves it through
     ``robot_manager.solve_ik`` -> ``curobo_planner.solve_ik_to_joint``
     (``env/planner_manager/curobo_planner.py:337``).  We do NOT use that
     branch, because it demands an ee pose for *every* target arm and would
     therefore re-solve IK for the arm we are not moving.  We call
     ``solve_ik`` ourselves for the moving arm and emit a plain JOINT action
     that holds the other arm at its measured joints -- the same action type
     the policy speaks, and the IK status comes back to us directly, which is
     what ``servo_missed`` is made of.

  2. **Every actuation charges the episode's step limit, and we must not let
     it.**  ``take_action_batch`` increments ``take_action_cnt`` and
     ``is_episode_end`` latches failure at ``take_action_cnt >= step_lim``
     (``eval_env.py:472,958``).  If repair steps counted, the treatment arm
     would be paying for its own repairs out of the policy's budget and the
     paired comparison would be meaningless.  ``_sim_action`` therefore lifts
     ``step_lim`` for the duration of the call and restores
     ``take_action_cnt`` afterwards, counting repair steps in
     ``_intervene_steps`` instead.  Set ``INTERVENE_CHARGES_STEP_LIM=1`` to
     get the opposite (conservative) accounting.

  3. **Rewind is real here, and it was not on RoboCasa.**  RoboCasa's
     cross-process anchor restore never reproduced object instances, which is
     why that project generates failure states in-process.  Isaac Lab exposes
     ``InteractiveScene.get_state()`` / ``reset_to()``
     (``third_party/IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py:504,567``)
     covering articulation root pose/velocity, joint state, rigid- and
     deformable-object state.  That is the whole physical world.  What it does
     NOT cover is the evaluator's own bookkeeping -- ``RewardManager`` keeps
     monotonic progress counters (``env/reward_manager/reward_manager.py:12-26``)
     and ``EvalEnv`` keeps ``take_action_cnt`` / ``end_flag`` / ``success``.
     Restoring the scene without those would hand back credit for progress the
     rewind just undid, so ``_capture_snapshot`` takes both halves.

  4. **The wrist force sensor and the aperture-in-metres do not exist.**
     RoboCasa's ``closed_empty`` test reads finger separation in metres;
     RoboDojo's gripper scalar is a normalised 0..1 command and the observation
     echoes the same normalisation (``env/observation_manager/obs_manager.py:185-212``).
     ``closed_empty`` here means "commanded shut, and the measured opening came
     back at/below ``GRIP_EMPTY_NORM``" -- the fingers met.  There is no
     contact force, so the ``contact`` stop reason is absent rather than faked.

STILL UNVERIFIED (needs the simulator; see ENV_SERVICE_ROBODOJO_NOTES.md)
    Whether ``reset_to`` round-trips a grasped object faithfully, whether an
    identical re-run of a window is bit-reproducible (the paired design rests
    on it), and how often cuRobo IK fails on the X5 near the table.  Each has a
    self-test in ``selftest_intervention.py``.
"""

from __future__ import annotations

import copy
import math
import os
import time
from typing import Any, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Tunables.  Names and defaults track env_service.py wherever the concept is
# shared, so the two benchmarks' logs read alike; the metres-based ones are
# re-derived for the ARX X5's much smaller workspace.
# ---------------------------------------------------------------------------

#: Largest translation a single model action may request (metres).  RoboCasa
#: uses 0.05 on a Franka; the X5 is a smaller arm working a tabletop, and the
#: same number is about the right fraction of its reach.
MAX_STEP_M = float(os.environ.get("RD_MAX_STEP_M", "0.05"))
#: Largest rotation a single model action may request (radians, ~20 deg).
MAX_STEP_RAD = float(os.environ.get("RD_MAX_STEP_RAD", "0.35"))
#: Servo granularity: one ``take_action`` moves at most this far, so a long
#: move_to becomes several interpolated control steps instead of one jump the
#: joint interpolator would smear over ``interpolation_nums`` sim steps.
MOVE_TO_STEP_CLIP = float(os.environ.get("RD_MOVE_STEP_CLIP", "0.02"))
#: Servo iteration cap for one move_to / retreat.
MOVE_TO_MAX_STEPS = int(os.environ.get("RD_MOVE_MAX_STEPS", "40"))
#: Position tolerance that counts as "arrived" (metres).
#: RoboCasa uses 12 mm because its OSC servo cannot do better. Measured on the
#: X5 (rd_servo_diag.py, 2026-09-18): a commanded 2 cm arrives within 0.1 mm.
#: At 12 mm the servo declared "arrived" one 2 cm step into a 3 cm request and
#: reported ok, leaving the hand 1 cm short of where the judge aimed -- a third
#: of the requested move, silently. 5 mm is still 50x the measured error.
MOVE_TO_TOL = float(os.environ.get("RD_MOVE_TOL", "0.005"))
#: Overshoot guard: travelling this many times the requested distance means the
#: arm is in a configuration where Cartesian commands are not trustworthy.
OVERSHOOT_RATIO = float(os.environ.get("RD_OVERSHOOT_RATIO", "3.0"))
#: Normalised opening at/below which a commanded closure counts as empty.
GRIP_EMPTY_NORM = float(os.environ.get("RD_GRIP_EMPTY", "0.05"))
#: Control steps spent opening or closing the fingers.
GRIPPER_STEPS = int(os.environ.get("RD_GRIPPER_STEPS", "4"))
#: Rewinds allowed per episode (shared name with env_service.py).
MAX_RESETS = int(os.environ.get("MAX_RESETS", "5"))
#: Rotation servo iterations for retreat's orientation leg.
ROTATE_SUBSTEPS = int(os.environ.get("RD_ROTATE_SUBSTEPS", "6"))
#: 1 = repair steps come out of the policy's step_lim (conservative).
INTERVENE_CHARGES_STEP_LIM = os.environ.get("INTERVENE_CHARGES_STEP_LIM", "0") == "1"

#: ``SceneManager``'s per-env object dictionaries, in the order
#: ``apply_saved_poses`` places them (``env/scene_manager/scene_manager.py:210-216``).
#: These hold the things the task is ABOUT. They are NOT in the Isaac Lab
#: ``InteractiveScene`` registry -- RoboDojo builds them with its own
#: ``RigidObject``/``ArticulationObject`` classes on top of Isaac Sim prims --
#: so ``scene.get_state()`` does not see them and ``scene.reset_to()`` does not
#: put them back. Measured 2026-09-18: on ``general_pickup`` the scene state
#: contains exactly two entities, ``robot0`` and ``robot1``, and nothing else.
_OBJECT_DICTS: tuple[str, ...] = (
    "_rigid_and_dynamic_objects",
    "_articulation_objects",
    "_garment_objects",
    "_geometry_objects",
    "_fluid_objects",
)

#: Object kinds whose full state is a mesh or a particle set rather than a
#: rigid pose. A pose+velocity snapshot does NOT restore them, and pretending
#: otherwise would make a rewind look clean while the cloth stays folded.
_UNSNAPSHOTTABLE_DICTS: frozenset[str] = frozenset({"_garment_objects", "_fluid_objects"})

#: RewardManager attributes that make up the evaluator's monotonic progress
#: bookkeeping.  A rewind that leaves these alone keeps credit for progress it
#: just undid, which would silently inflate every treatment-arm score.
_REWARD_STATE_FIELDS: tuple[str, ...] = (
    "check_list", "final_check_list", "query_list",
    "trigger_check_list", "trigger_query_list",
    "score_list", "score_achieved", "score_completed_count",
    "score_meta", "score_trigger_meta",
    "final_score_list", "final_score_achieved",
    "final_score_completed_count", "final_score_meta",
    "_gated_score_lst",
)


# ---------------------------------------------------------------------------
# Quaternion helpers.  RoboDojo quaternions are WXYZ everywhere: the env feeds
# ``ee_pose[-4:]`` straight to ``transforms3d.quaternions.quat2mat``
# (``src/eval_client/eval_env.py:860``), and transforms3d is wxyz by
# definition.  Isaac Lab 3.0's own convention change is xyzw, which is exactly
# the trap this port already paid for once -- keep every quaternion in this
# file wxyz and convert at the boundary, never in the middle.
# ---------------------------------------------------------------------------

def _to_np(value: Any) -> Any:
    """torch tensor / warp array / sequence -> something numpy can take."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, both wxyz."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _rotvec_to_quat(rv: np.ndarray) -> np.ndarray:
    rv = np.asarray(rv, dtype=np.float64)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rv / theta
    s = math.sin(theta / 2.0)
    return np.array([math.cos(theta / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    if q[0] < 0:               # shortest arc
        q = -q
    w = float(np.clip(q[0], -1.0, 1.0))
    theta = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-9:
        return np.zeros(3)
    return (q[1:] / s) * theta


def _quat_geodesic_rad(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two wxyz quaternions."""
    return float(np.linalg.norm(_quat_to_rotvec(_quat_mul(_quat_normalize(a), _quat_conj(_quat_normalize(b))))))


def _frame_size() -> int:
    """``env_service_robodojo.FRAME_SIZE``, read lazily.

    Imported inside the call rather than at module scope: that module imports
    THIS one, so a module-level import here would be a cycle, and the version
    of it that "works" silently ends up with two copies of the mixin class.
    """
    try:
        from recovery_explore.env_service_robodojo import FRAME_SIZE
    except ImportError:  # the RoboDojo service is not shipped with every release; the RoboTwin service uses the same 256 px frames
        FRAME_SIZE = 256

    return int(FRAME_SIZE)


class InterventionError(RuntimeError):
    """A primitive could not be carried out.  Carries the stop reason."""

    def __init__(self, message: str, reason: str = "error") -> None:
        super().__init__(message)
        self.reason = reason


class RoboDojoInterventionMixin:
    """Milestone-2 methods.  Mixed into ``RoboDojoEnvService``.

    Relies on the host class for: ``_require_env``, ``_require_episode``,
    ``_observation``, ``_proprio_state``, ``_oracle``, ``_policy``,
    ``_episode_dir``, ``_frames_from_obs``, ``harness_advance``,
    ``_step_count``, ``_episode``.
    """

    # Model-facing ops accepted inside one plan.  ``retreat`` and ``policy``
    # are here for the same reasons as on RoboCasa: an intervention that cannot
    # undo itself degenerates into flailing, and a repair that cannot hand back
    # cannot be verified.
    _PLAN_OPS = {"move_to", "nudge", "lift", "rotate", "gripper", "retreat", "policy"}
    _PLAN_POLICY_MAX_CHUNKS = int(os.environ.get("PLAN_POLICY_MAX_CHUNKS", "2"))
    _PLAN_POLICY_MAX_TOTAL = int(os.environ.get("PLAN_POLICY_MAX_TOTAL", "4"))

    #: Refused whenever the model holds the action budget or the judge turn is
    #: locked.  Same list as env_service._PRIVILEGED_METHODS minus the methods
    #: this benchmark does not have, plus the RoboDojo-only ones.
    _PRIVILEGED_METHODS = frozenset({
        "harness_read_state_privileged", "harness_env_meta", "harness_advance",
        "harness_episode_begin", "harness_release_env", "harness_quit",
        "harness_open_action_budget", "harness_judge_lock",
        "harness_round_begin", "harness_round_status",
        "restore_snapshot", "make_failure", "reset_to_failure",
        "run_episode_to_failure", "resume_episode", "debug_reset",
        "unproject_batch", "reset_step_counter", "harness_oracle_grasp",
        "harness_checkpoint", "harness_state_digest",
    })

    # -- state ------------------------------------------------------------

    def _init_intervention_state(self) -> None:
        """Called from ``RoboDojoEnvService.__init__``."""
        self._round_budget: int | None = None
        self._round_used: int = 0
        self._round_index: int = 0
        self._round_token: str | None = None
        self._round_policy_chunks: int = 0
        self._allow_cosmos: bool = True
        self._judge_locked: bool = False
        self._judge_token: str | None = None
        self._resets_used: int = 0
        self._resets_this_window: int = 0
        self._window_snapshot: dict[str, Any] | None = None
        self._takeover_pose: dict[str, Any] | None = None
        self._window_faults: int = 0
        self._cartesian_blocked: bool = False
        self._intervene_steps: int = 0
        self._default_arm: str = os.environ.get("RD_DEFAULT_ARM", "right")
        self._ik_failures: int = 0

    def _intervention_rpc(self) -> dict[str, Any]:
        """The method table milestone 2 replaces the stubs with."""
        return {
            "harness_open_action_budget": self.harness_open_action_budget,
            "harness_close_action_budget": self.harness_close_action_budget,
            "harness_judge_lock": self.harness_judge_lock,
            "harness_judge_unlock": self.harness_judge_unlock,
            "harness_round_status": self.harness_round_status,
            "reset_window": self.reset_window,
            "execute_plan": self.execute_plan,
            "move_to": self.move_to,
            "nudge": self.nudge,
            "lift": self.lift,
            "rotate": self.rotate,
            "gripper": self.gripper,
            "retreat": self.retreat,
            "read_state": self.read_state,
            "render": self.render,
            "unproject": self.unproject,
            "restore_snapshot": self.restore_snapshot,
            "harness_checkpoint": self.harness_checkpoint,
            "harness_state_digest": self.harness_state_digest,
        }

    # -- arm / robot plumbing ---------------------------------------------

    def _robot_for(self, side: str):
        """The ``target``-type robot whose arm name starts with ``side``."""
        env = self._require_env()
        for robot in env.robot_manager.robot_list:
            if getattr(robot, "type", None) != "target":
                continue
            if str(robot.arm_name).split("_")[0] == side:
                return robot
        raise InterventionError(
            f"no target arm named {side!r}; this robot exposes "
            f"{[r.arm_name for r in env.robot_manager.robot_list if getattr(r, 'type', None) == 'target']}"
        )

    def _active_arm(self) -> str:
        """Which hand is doing the work -- measured, not assumed.

        This is NOT a constant. Observed on the pod 2026-09-18: two episodes of
        the same task, ``general_pickup``, one worked by the right arm and the
        next by the left. A fixed default sends every repair to whichever hand
        happens to be idle, which does nothing and still spends the
        intervention -- and the judge, reading a telemetry line that says the
        left hand moved 27 cm, would be writing correct plans that drive the
        right one.

        Cumulative Cartesian travel over the recent chunks decides it. Before
        anything has moved there is no answer, so the configured default
        stands; ``RD_DEFAULT_ARM`` overrides for a deliberately one-armed run.
        """
        travel = {"left": 0.0, "right": 0.0}
        for entry in self._act_log[-12:]:
            moved = entry.get("moved_cm") or []
            if len(moved) != 2:
                continue
            travel["left"] += float(moved[0] or 0.0)
            travel["right"] += float(moved[1] or 0.0)
        if max(travel.values()) <= 0.0:
            return self._default_arm
        return "left" if travel["left"] > travel["right"] else "right"

    def _side(self, arm: str | None) -> str:
        side = str(arm or self._active_arm()).lower()
        if side not in ("left", "right"):
            raise InterventionError(f"arm must be 'left' or 'right', got {arm!r}")
        return side

    def _ee_pose(self, side: str, obs: Mapping[str, Any] | None = None) -> np.ndarray:
        """Current world ee pose of one arm: [x,y,z,qw,qx,qy,qz]."""
        state = (obs if obs is not None else self._observation()).get("state", {}) or {}
        pose = state.get(f"{side}_ee_pose")
        if pose is None:
            raise InterventionError(
                f"the env does not publish {side}_ee_pose; set `world_ee_state: true` in "
                "the env config, otherwise no Cartesian primitive can work"
            )
        arr = np.asarray(pose, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 7:
            raise InterventionError(f"{side}_ee_pose has {arr.shape[0]} components, expected 7")
        return arr

    def _grip_norm(self, side: str, obs: Mapping[str, Any] | None = None) -> float:
        state = (obs if obs is not None else self._observation()).get("state", {}) or {}
        val = state.get(f"{side}_ee_joint_state")
        if val is None:
            return 0.0
        return float(np.asarray(val, dtype=np.float64).reshape(-1)[0])

    def _joint_action(
        self,
        *,
        left_arm: Sequence[float],
        left_grip: float,
        right_arm: Sequence[float],
        right_grip: float,
    ) -> dict[str, np.ndarray]:
        """One joint-typed action dict, the shape ``validate_action_dict``
        expects for a two-arm robot (``eval_env.py:697-709``)."""
        return {
            "left_arm_joint_state": np.asarray(left_arm, dtype=np.float64).reshape(-1),
            "left_ee_joint_state": np.asarray([float(np.clip(left_grip, 0.0, 1.0))], dtype=np.float64),
            "right_arm_joint_state": np.asarray(right_arm, dtype=np.float64).reshape(-1),
            "right_ee_joint_state": np.asarray([float(np.clip(right_grip, 0.0, 1.0))], dtype=np.float64),
        }

    def _current_joint_action(self, obs: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
        """A hold-everything action built from the measured joint state."""
        state = (obs if obs is not None else self._observation()).get("state", {}) or {}
        return self._joint_action(
            left_arm=np.asarray(state["left_arm_joint_state"], dtype=np.float64).reshape(-1),
            left_grip=self._grip_norm("left", obs),
            right_arm=np.asarray(state["right_arm_joint_state"], dtype=np.float64).reshape(-1),
            right_grip=self._grip_norm("right", obs),
        )

    def _solve_ik(self, side: str, target_pose: np.ndarray) -> np.ndarray | None:
        """cuRobo IK for one arm, world frame.  ``None`` = no solution.

        ``robot_manager.solve_ik`` seeds from the arm's current joints
        (``env/robot_manager/robot_manager.py:347``), so successive servo steps
        stay on the same IK branch instead of flipping elbow configuration
        between steps -- which is the whole reason the servo is incremental.
        """
        env = self._require_env()
        robot = self._robot_for(side)
        result = env.robot_manager.solve_ik(
            target_pose=[float(v) for v in np.asarray(target_pose).reshape(-1)],
            env_idx=0,
            robot=robot,
        )
        if not isinstance(result, Mapping) or result.get("status") != "Success":
            self._ik_failures += 1
            return None
        return np.asarray(result["joint_value"], dtype=np.float64).reshape(-1)

    # -- the one place the simulator is actually stepped ------------------

    def _sim_action(self, action: Mapping[str, Any]) -> None:
        """Apply one control step that does NOT spend the policy's budget.

        ``take_action_batch`` increments ``take_action_cnt`` and then
        ``is_episode_end`` latches failure once it reaches ``step_lim``
        (``eval_env.py:472,958-962``).  Repair steps must not be charged there
        or the treatment arm pays for its own repairs and the paired
        comparison loses its meaning.  So: lift the limit for the duration of
        the call, restore the counter afterwards, and keep our own tally.

        The OTHER ``is_episode_end`` branch -- ``not self.success`` -- is left
        alone deliberately.  That is the task's own failure predicate, and a
        repair that irrecoverably breaks the task SHOULD latch.
        """
        env = self._require_env()
        before = int(env.take_action_cnt[0])
        saved_lim = env.step_lim
        if not INTERVENE_CHARGES_STEP_LIM:
            env.step_lim = int(saved_lim) + 1_000_000
        try:
            env.take_action(dict(action))
        finally:
            env.step_lim = saved_lim
            if not INTERVENE_CHARGES_STEP_LIM:
                env.take_action_cnt[0] = before
        self._intervene_steps += 1

    def _spend_action(self, name: str) -> None:
        if self._round_budget is None:
            return
        if self._round_used >= self._round_budget:
            raise InterventionError(
                f"round action budget exhausted ({self._round_used}/{self._round_budget}): "
                "this round is over. Stop calling actions and write your summary.",
                reason="budget",
            )
        self._round_used += 1

    def _require_cartesian(self) -> None:
        if self._cartesian_blocked:
            raise InterventionError(
                "this intervention has already had a Cartesian command fail from this arm "
                "configuration; move_to, nudge, lift, rotate and retreat are refused for the "
                "rest of it. What remains: open the fingers, or hand back to the policy.",
                reason="servo_missed",
            )

    # -- servo ------------------------------------------------------------

    def _servo_to(
        self,
        side: str,
        target_xyz: np.ndarray,
        target_quat: np.ndarray | None,
        grip: float,
        *,
        max_steps: int = MOVE_TO_MAX_STEPS,
        tol: float = MOVE_TO_TOL,
    ) -> dict[str, Any]:
        """Incremental IK servo toward a world pose.  The workhorse.

        One ``take_action`` per iteration, each asking for at most
        ``MOVE_TO_STEP_CLIP`` of travel, re-reading the measured pose in
        between.  Stops on arrival, on the iteration cap, when IK fails, or
        when the measured pose stops converging -- the last being the only
        honest detector for "the arm is fighting a contact", since there is no
        force sensor on this robot.
        """
        obs = self._observation()
        start = self._ee_pose(side, obs)
        requested = float(np.linalg.norm(np.asarray(target_xyz, dtype=np.float64) - start[:3]))
        goal_quat = _quat_normalize(target_quat) if target_quat is not None else _quat_normalize(start[3:])

        travelled = 0.0
        last_dist = requested
        stalled = 0
        reason = "arrived"
        steps = 0
        for _ in range(int(max_steps)):
            obs = self._observation()
            cur = self._ee_pose(side, obs)
            delta = np.asarray(target_xyz, dtype=np.float64) - cur[:3]
            dist = float(np.linalg.norm(delta))
            rot_err = _quat_geodesic_rad(goal_quat, cur[3:])
            # ``steps > 0``: always take at least one control step, even when
            # the hand is already within tolerance -- the gripper command
            # rides on the same action, and a nudge smaller than the tolerance
            # that also asks for a close must still close.
            if dist <= tol and rot_err <= 0.08 and steps > 0:
                reason = "arrived"
                break
            step = delta if dist <= MOVE_TO_STEP_CLIP else delta * (MOVE_TO_STEP_CLIP / dist)
            waypoint = np.concatenate([cur[:3] + step, goal_quat])
            joints = self._solve_ik(side, waypoint)
            if joints is None:
                reason = "ik_failed"
                break
            other = "left" if side == "right" else "right"
            state = obs.get("state", {}) or {}
            arms = {
                side: joints,
                other: np.asarray(state[f"{other}_arm_joint_state"], dtype=np.float64).reshape(-1),
            }
            grips = {side: grip, other: self._grip_norm(other, obs)}
            self._sim_action(self._joint_action(
                left_arm=arms["left"], left_grip=grips["left"],
                right_arm=arms["right"], right_grip=grips["right"],
            ))
            steps += 1
            after = self._ee_pose(side)
            travelled += float(np.linalg.norm(after[:3] - cur[:3]))
            new_dist = float(np.linalg.norm(np.asarray(target_xyz, dtype=np.float64) - after[:3]))
            if new_dist > last_dist - 1e-4:
                stalled += 1
                if stalled >= 3:
                    reason = "stalled"
                    break
            else:
                stalled = 0
            last_dist = new_dist
            if bool(self._require_env().end_flag[0]):
                reason = "episode_end"
                break
        else:
            reason = "step_cap"

        final = self._ee_pose(side)
        final_dist = float(np.linalg.norm(np.asarray(target_xyz, dtype=np.float64) - final[:3]))
        ok = final_dist <= tol and reason in ("arrived",)
        overshot = requested > 1e-4 and travelled > OVERSHOOT_RATIO * requested
        out = {
            "ok": bool(ok),
            "arm": side,
            "requested_dist_m": round(requested, 4),
            "final_dist_m": round(final_dist, 4),
            "travelled_m": round(travelled, 4),
            "control_steps": int(steps),
            "servo_stop": reason,
            "overshoot": bool(overshot),
            "eef": [round(float(v), 4) for v in final[:3]],
            "quat": [round(float(v), 4) for v in final[3:]],
            "rot_err_deg": round(float(np.degrees(_quat_geodesic_rad(goal_quat, final[3:]))), 1),
        }
        if not ok:
            # One failed Cartesian command means the arm is somewhere the IK
            # cannot work from; the remaining Cartesian ops would fail the
            # same way, and letting the model retry them just burns budget.
            self._cartesian_blocked = True
            self._window_faults += 1
        return out

    # -- snapshots --------------------------------------------------------

    def _clone_scene_state(self, state: Any) -> Any:
        """Deep-clone ``InteractiveScene.get_state()``.

        The tensors it returns are VIEWS into the simulation buffers -- keeping
        them without cloning gives a "snapshot" that silently tracks the live
        world and restores nothing.
        """
        if isinstance(state, Mapping):
            return {k: self._clone_scene_state(v) for k, v in state.items()}
        if hasattr(state, "clone"):
            return state.clone()
        return copy.deepcopy(state)

    def _scene(self) -> Any:
        """The Isaac Lab ``InteractiveScene``.

        NOT ``env.scene``: ``EvalEnv`` wraps ``BaseEnv``, which keeps the
        DirectRLEnv in ``self.sim`` and reads the scene off it
        (``env/environment/base_env.py:236`` -> ``self.sim.scene.env_origins``).
        The first draft assumed ``env.scene`` and every window silently came up
        with ``reset_available: false``, because the capture is deliberately
        wrapped in a try/except -- rewind is a bonus, not a precondition. The
        fallback chain is ordered so a future refactor that promotes ``scene``
        onto EvalEnv keeps working.
        """
        env = self._require_env()
        for holder, attr in ((env, "scene"), (getattr(env, "sim", None), "scene"),
                             (getattr(env, "robot_manager", None), "scene")):
            scene = getattr(holder, attr, None) if holder is not None else None
            if scene is not None and hasattr(scene, "get_state") and hasattr(scene, "reset_to"):
                return scene
        raise InterventionError(
            "cannot find an InteractiveScene with get_state/reset_to on this env "
            "(tried env.scene, env.sim.scene, env.robot_manager.scene); "
            "without it there is no rewind"
        )

    def _scene_objects(self):
        """(key, object, dict_name) for every scene object in env 0."""
        manager = getattr(self._require_env(), "scene_manager", None)
        if manager is None:
            return
        for attr in _OBJECT_DICTS:
            per_env = getattr(manager, attr, None)
            if not per_env:
                continue
            try:
                entries = per_env[0]
            except (IndexError, KeyError, TypeError):
                continue
            if not isinstance(entries, Mapping):
                continue
            for name, obj in entries.items():
                if obj is None:
                    continue
                yield f"{attr}/{name}", obj, attr

    def _capture_objects(self) -> dict[str, Any]:
        """Pose + velocity (+ joints) of every task object.

        ``apply_saved_pose`` (``objects/rigid.py:218``, ``objects/articulation.py:261``)
        is the proven read/write surface for these: local pose, scale and the
        two velocities, plus joint positions for the articulated ones. We take
        the same fields, at their CURRENT values rather than the layout's.
        """
        out: dict[str, Any] = {}
        for key, obj, attr in self._scene_objects():
            if attr in _UNSNAPSHOTTABLE_DICTS:
                continue
            entry: dict[str, Any] = {}
            try:
                pos, ori = obj.get_local_pose()
                entry["pos"] = np.asarray(_to_np(pos), dtype=np.float64).copy()
                entry["ori"] = np.asarray(_to_np(ori), dtype=np.float64).copy()
            except Exception:
                continue          # no pose surface: nothing to restore either
            for field, getter in (("lin", "get_linear_velocity"),
                                  ("ang", "get_angular_velocity"),
                                  ("joints", "get_current_joint_positions")):
                fn = getattr(obj, getter, None)
                if fn is None:
                    continue
                try:
                    value = fn()
                except Exception:
                    continue
                if value is not None:
                    entry[field] = np.asarray(_to_np(value), dtype=np.float64).copy()
            out[key] = entry
        return out

    def _restore_objects(self, objects: Mapping[str, Any]) -> int:
        """Put the task objects back. Returns how many were restored.

        No settling steps: the velocities are restored exactly, so stepping
        here would advance the world past the state we are restoring rather
        than reach it. (``apply_saved_poses`` steps 20 times because it is
        PLACING objects from a layout file, which is a different problem.)
        """
        restored = 0
        live = {key: obj for key, obj, _ in self._scene_objects()}
        for key, entry in objects.items():
            obj = live.get(key)
            if obj is None:
                continue
            try:
                obj.set_local_pose(translation=entry["pos"], orientation=entry["ori"])
            except Exception:
                continue
            for field, setter in (("lin", "set_linear_velocity"),
                                  ("ang", "set_angular_velocity"),
                                  ("joints", "set_current_joint_positions")):
                value = entry.get(field)
                fn = getattr(obj, setter, None)
                if value is None or fn is None:
                    continue
                try:
                    fn(value)
                except Exception:
                    pass
            restored += 1
        return restored

    def _unsnapshottable_objects(self) -> list[str]:
        """Objects a pose snapshot cannot capture (cloth, fluid). Reported, not
        hidden: on those tasks a rewind is partial and the caller must know."""
        return [key for key, _, attr in self._scene_objects() if attr in _UNSNAPSHOTTABLE_DICTS]

    def _capture_snapshot(self, tag: str) -> dict[str, Any]:
        env = self._require_env()
        episode = self._episode or {}
        reward = env.reward_manager
        return {
            "tag": str(tag),
            "ts": time.time(),
            "scene": self._clone_scene_state(self._scene().get_state()),
            # The robots come from the InteractiveScene; everything the task is
            # actually about comes from here. See _OBJECT_DICTS.
            "objects": self._capture_objects(),
            "unsnapshottable": self._unsnapshottable_objects(),
            "reward": {
                field: copy.deepcopy(getattr(reward, field))
                for field in _REWARD_STATE_FIELDS
                if hasattr(reward, field)
            },
            "take_action_cnt": [int(v) for v in env.take_action_cnt],
            "end_flag": [bool(v) for v in env.end_flag],
            "success": [bool(v) for v in getattr(env, "success", [])],
            "step_lim": int(env.step_lim),
            "step_count": int(self._step_count),
            "committed_timestep": int(episode.get("committed_timestep") or 0),
            "next_query_index": int(episode.get("next_query_index") or 0),
            "intervene_steps": int(self._intervene_steps),
        }

    def _apply_snapshot(self, snap: Mapping[str, Any]) -> None:
        env = self._require_env()
        self._scene().reset_to(snap["scene"], env_ids=None, is_relative=False)
        self._restore_objects(snap.get("objects") or {})
        reward = env.reward_manager
        for field, value in (snap.get("reward") or {}).items():
            setattr(reward, field, copy.deepcopy(value))
        env.take_action_cnt[:] = list(snap["take_action_cnt"])
        env.end_flag[:] = list(snap["end_flag"])
        if hasattr(env, "success") and snap.get("success"):
            env.success[:] = list(snap["success"])
        env.step_lim = int(snap["step_lim"])
        self._step_count = int(snap["step_count"])
        if self._episode is not None:
            self._episode["committed_timestep"] = int(snap["committed_timestep"])
            self._episode["next_query_index"] = int(snap["next_query_index"])
        self._intervene_steps = int(snap.get("intervene_steps", self._intervene_steps))
        # The policy server keeps its own observation window; after a rewind
        # that window describes a future that no longer happened.
        try:
            self._policy().reset()
        except Exception:
            pass

    # -- harness verbs ----------------------------------------------------

    def harness_open_action_budget(
        self,
        budget: int,
        allow_cosmos: bool = True,
        round_index: int | None = None,
        control_token: str | None = None,
    ) -> dict[str, Any]:
        """HARNESS ONLY: open one intervention window.  Does not rewind."""
        self._round_budget = int(budget)
        self._round_used = 0
        self._round_policy_chunks = 0
        self._round_token = str(control_token) if control_token else None
        self._allow_cosmos = bool(allow_cosmos)
        self._round_index = int(round_index) if round_index is not None else self._round_index + 1
        self._resets_this_window = 0
        self._window_faults = 0
        self._cartesian_blocked = False
        try:
            self._window_snapshot = self._capture_snapshot("window_open")
        except Exception as exc:       # rewind is a bonus, not a precondition
            self._window_snapshot = None
            self._log({"ts": time.time(), "method": "snapshot_failed", "error": repr(exc)})
        self._takeover_pose = self._capture_takeover_pose()
        return {
            "ok": True,
            "round_index": self._round_index,
            "round_budget": self._round_budget,
            "round_used": self._round_used,
            "allow_cosmos": self._allow_cosmos,
            "step_count": int(self._step_count),
            "token_required": self._round_token is not None,
            "reset_available": self._window_snapshot is not None,
            "resets_left": int(MAX_RESETS - self._resets_used),
            "objects_snapshotted": len((self._window_snapshot or {}).get("objects") or {}),
            # Non-empty = this task has cloth or fluid, whose real state a
            # pose snapshot cannot hold. A rewind here is PARTIAL.
            "objects_not_snapshottable": list((self._window_snapshot or {}).get("unsnapshottable") or []),
        }

    def harness_close_action_budget(self, control_token: str | None = None) -> dict[str, Any]:
        if self._round_token and str(control_token or "") != self._round_token:
            raise PermissionError(
                "closing the action budget requires the harness control token issued "
                "when the window was opened; this is a harness-only operation."
            )
        used = self._round_used
        resets = self._resets_this_window
        self._round_token = None
        self._round_budget = 0
        self._allow_cosmos = True
        self._resets_this_window = 0
        self._window_snapshot = None
        self._cartesian_blocked = False
        return {
            "ok": True,
            "actions_used": int(used),
            "resets": int(resets),
            "resets_used_total": int(self._resets_used),
            "intervene_control_steps": int(self._intervene_steps),
            "ik_failures": int(self._ik_failures),
        }

    def harness_judge_lock(self, control_token: str | None = None) -> dict[str, Any]:
        self._judge_locked = True
        self._judge_token = str(control_token) if control_token else None
        return {"ok": True, "judge_locked": True, "token_required": self._judge_token is not None}

    def harness_judge_unlock(self, control_token: str | None = None) -> dict[str, Any]:
        if self._judge_token and str(control_token or "") != self._judge_token:
            raise PermissionError(
                "unlocking the judge turn requires the harness control token issued "
                "when it was locked; this is a harness-only operation."
            )
        self._judge_locked = False
        self._judge_token = None
        return {"ok": True, "judge_locked": False}

    def harness_round_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "round_index": self._round_index,
            "round_budget": self._round_budget,
            "round_used": self._round_used,
            "allow_cosmos": self._allow_cosmos,
            "step_count": int(self._step_count),
            "intervene_control_steps": int(self._intervene_steps),
        }
        if self._env is not None:
            status["oracle"] = self._oracle()
        return status

    def harness_checkpoint(self, tag: str = "manual") -> dict[str, Any]:
        """HARNESS ONLY: take a snapshot without opening a window.

        Exists for the determinism and snapshot-fidelity self-tests, which need
        to capture, run, restore and compare outside any intervention.
        """
        self._window_snapshot = self._capture_snapshot(str(tag))
        return {"ok": True, "tag": str(tag), "step_count": int(self._step_count)}

    def harness_state_digest(self) -> dict[str, Any]:
        """HARNESS ONLY: a comparable fingerprint of the whole simulated world.

        Exists to answer the two questions the paired design rests on, and that
        no amount of "the RPC returned ok" can answer:

          * **Snapshot fidelity** -- capture, disturb, rewind, compare. If the
            digest does not come back identical, ``reset_window`` is putting
            the policy back into a DIFFERENT world than the control arm saw,
            and every paired comparison is measuring the rewind instead of the
            intervention.
          * **Re-run determinism** -- rewind, replay the same chunks, compare.
            If a replay diverges, a matched control/treatment pair is not
            matched and the design needs unpaired statistics.

        Returns per-entity tensor checksums rather than the tensors: the scene
        state is megabytes, and what a test needs is "same or not, and if not,
        by how much".
        """
        scene = self._scene()
        env = self._require_env()
        state = scene.get_state()
        digest: dict[str, Any] = {}
        for kind, entities in (state or {}).items():
            if not isinstance(entities, Mapping):
                continue
            for name, fields in entities.items():
                if not isinstance(fields, Mapping):
                    continue
                for field, tensor in fields.items():
                    try:
                        arr = np.asarray(
                            tensor.detach().cpu() if hasattr(tensor, "detach") else tensor,
                            dtype=np.float64,
                        )
                    except Exception:
                        continue
                    digest[f"{kind}/{name}/{field}"] = {
                        "sum": round(float(np.nansum(arr)), 6),
                        "absmax": round(float(np.nanmax(np.abs(arr))) if arr.size else 0.0, 6),
                        "shape": list(arr.shape),
                    }
        for key, entry in self._capture_objects().items():
            for field, arr in entry.items():
                arr = np.asarray(arr, dtype=np.float64)
                digest[f"object/{key}/{field}"] = {
                    "sum": round(float(np.nansum(arr)), 6),
                    "absmax": round(float(np.nanmax(np.abs(arr))) if arr.size else 0.0, 6),
                    "shape": list(arr.shape),
                }
        obs = self._observation()
        return {
            "scene": digest,
            "unsnapshottable": self._unsnapshottable_objects(),
            "left_ee_pose": [round(float(v), 6) for v in self._ee_pose("left", obs)],
            "right_ee_pose": [round(float(v), 6) for v in self._ee_pose("right", obs)],
            "opening": {side: round(self._grip_norm(side, obs), 6) for side in ("left", "right")},
            "take_action_cnt": int(env.take_action_cnt[0]),
            "end_flag": bool(env.end_flag[0]),
            "success": bool(env.success[0]) if hasattr(env, "success") else None,
            "step_count": int(self._step_count),
            "committed_timestep": int((self._episode or {}).get("committed_timestep") or 0),
            "oracle": self._oracle(),
        }

    def restore_snapshot(self, which: str = "window") -> dict[str, Any]:
        """HARNESS ONLY: put the world back to the stored snapshot."""
        if which != "window" or not self._window_snapshot:
            return {"ok": False, "error": "no snapshot stored"}
        self._apply_snapshot(self._window_snapshot)
        return {"ok": True, "step_count": int(self._step_count), "tag": self._window_snapshot.get("tag")}

    def reset_window(self) -> dict[str, Any]:
        """Rewind to the start of THIS intervention and refill the budget.

        Undoes only the model's own actions: the policy's state comes back
        exactly as it was when the window opened, so the control arm's
        trajectory is unaffected and no extra policy steps are granted.
        """
        self._require_env()
        if self._round_budget is None or self._round_budget <= 0:
            return {"ok": False, "error": "not inside an intervention window; nothing to rewind"}
        if self._resets_used >= MAX_RESETS:
            return {
                "ok": False,
                "error": f"no resets left this episode (used {MAX_RESETS})",
                "resets_used": int(self._resets_used),
            }
        if not self._window_snapshot:
            return {"ok": False, "error": "the snapshot for this window was not captured; cannot rewind"}
        self._apply_snapshot(self._window_snapshot)
        self._resets_used += 1
        self._resets_this_window += 1
        self._round_used = 0
        self._round_policy_chunks = 0
        self._window_faults = 0
        self._cartesian_blocked = False
        return {
            "ok": True,
            "note": "rewound to the start of this intervention; action budget refilled",
            "actions_budget": int(self._round_budget),
            "actions_used": 0,
            "resets_used": int(self._resets_used),
            "resets_left": int(MAX_RESETS - self._resets_used),
        }

    # -- primitives -------------------------------------------------------

    def _capture_takeover_pose(self) -> dict[str, Any] | None:
        try:
            obs = self._observation()
            return {
                side: {
                    "pose": [float(v) for v in self._ee_pose(side, obs)],
                    "grip": float(self._grip_norm(side, obs)),
                }
                for side in ("left", "right")
            } | {"step_count": int(self._step_count)}
        except Exception:
            return None

    def _grip_for(self, side: str, gripper: Any, obs: Mapping[str, Any] | None = None) -> float:
        """Resolve a plan's ``gripper`` field to a normalised command.

        ``"hold"`` means "whatever the fingers are doing now", which on this
        robot is the measured normalised opening -- not a separate -1/+1
        channel as on RoboCasa's OSC action.
        """
        if gripper is None or gripper == "hold":
            return self._grip_norm(side, obs)
        if isinstance(gripper, str):
            key = gripper.lower()
            if key in ("open", "release"):
                return 1.0
            if key in ("close", "grasp", "shut"):
                return 0.0
            raise InterventionError(f"gripper must be hold/open/close, got {gripper!r}")
        # A NUMBER is always RoboCasa's OSC gripper convention: -1 opens,
        # +1 closes, 0 holds.  That is what ``cli/rex.py`` sends
        # (rex.py:230,255,259,263) and what the judge prompts teach, so plans
        # written for RoboCasa carry it too.  RoboDojo's own channel is a
        # normalised opening where 1.0 is OPEN -- the opposite at both ends --
        # so a number is interpreted, never clamped: clamping +1 would leave
        # the fingers open when the model asked for a grasp.
        value = float(gripper)
        if value < 0.0:
            return 1.0
        if value > 0.0:
            return 0.0
        return self._grip_norm(side, obs)

    def move_to(self, xyz: Any, gripper: Any = "hold", arm: str | None = None,
                quat: Any = None) -> dict[str, Any]:
        """Servo the end-effector to a world position, wrist held or as given."""
        self._require_cartesian()
        side = self._side(arm)
        self._spend_action("move_to")
        target = np.asarray(xyz, dtype=np.float64).reshape(-1)
        if target.shape[0] != 3:
            raise InterventionError("xyz must be 3 numbers (world metres)")
        cur = self._ee_pose(side)
        dist = float(np.linalg.norm(target - cur[:3]))
        clipped = False
        if dist > MAX_STEP_M and not self._is_visited(side, target):
            target = cur[:3] + (target - cur[:3]) * (MAX_STEP_M / dist)
            clipped = True
        out = self._servo_to(
            side, target,
            _quat_normalize(np.asarray(quat, dtype=np.float64)) if quat is not None else None,
            self._grip_for(side, gripper),
        )
        out["clipped"] = clipped
        return out

    def _is_visited(self, side: str, target: np.ndarray) -> bool:
        """Was this point reached under the policy's own power this episode?

        A move back to somewhere the arm has already been is not a leap of
        faith, so it is exempt from the single-step distance cap -- the same
        rule RoboCasa's re-stage point uses.  ``_act_log`` records the ee pose
        at the end of every policy chunk (``env_service_robodojo.harness_advance``).
        """
        key = f"{side}_ee_pose"
        for entry in self._act_log:
            pose = entry.get(key)
            if not pose:
                continue
            if float(np.linalg.norm(np.asarray(pose[:3], dtype=np.float64) - target)) <= 0.08:
                return True
        take = self._takeover_pose or {}
        held = (take.get(side) or {}).get("pose")
        if held and float(np.linalg.norm(np.asarray(held[:3], dtype=np.float64) - target)) <= 0.08:
            return True
        return False

    def nudge(self, dxyz: Any, gripper: Any = "hold", arm: str | None = None,
              max_norm: float | None = MAX_STEP_M) -> dict[str, Any]:
        self._require_cartesian()
        side = self._side(arm)
        delta = np.asarray(dxyz, dtype=np.float64).reshape(-1)
        if delta.shape[0] != 3:
            raise InterventionError("dxyz must be 3 numbers (metres)")
        norm = float(np.linalg.norm(delta))
        clipped = False
        if max_norm and norm > float(max_norm):
            delta = delta * (float(max_norm) / norm)
            clipped = True
        self._spend_action("nudge")
        cur = self._ee_pose(side)
        out = self._servo_to(side, cur[:3] + delta, None, self._grip_for(side, gripper))
        out["clipped"] = clipped
        out["dxyz"] = [round(float(v), 4) for v in delta]
        return out

    def lift(self, dz: float, gripper: Any = "hold", arm: str | None = None) -> dict[str, Any]:
        return self.nudge([0.0, 0.0, float(dz)], gripper=gripper, arm=arm)

    def rotate(self, drot: Any, gripper: Any = "hold", arm: str | None = None) -> dict[str, Any]:
        """Rotate the wrist by an axis-angle delta, position held."""
        self._require_cartesian()
        side = self._side(arm)
        rv = np.asarray(drot, dtype=np.float64).reshape(-1)
        if rv.shape[0] != 3:
            raise InterventionError("drot must be a 3-vector (axis-angle, radians)")
        angle = float(np.linalg.norm(rv))
        clipped = False
        if angle > MAX_STEP_RAD:
            rv = rv * (MAX_STEP_RAD / angle)
            clipped = True
        self._spend_action("rotate")
        cur = self._ee_pose(side)
        goal = _quat_mul(_rotvec_to_quat(rv), cur[3:])
        out = self._servo_to(side, cur[:3], goal, self._grip_for(side, gripper))
        out["clipped"] = clipped
        out["drot"] = [round(float(v), 4) for v in rv]
        return out

    def gripper(self, action: str = "close", arm: str | None = None) -> dict[str, Any]:
        """Open or close the fingers, arm held where it is.

        ``closed_empty`` is inferred from the measured opening after the
        command settles: this robot has no force sensor and no finger-gap in
        metres, so "commanded shut and the fingers met" is the only available
        empty-closure test.
        """
        side = self._side(arm)
        target = self._grip_for(side, action)
        self._spend_action("gripper")
        obs = self._observation()
        state = obs.get("state", {}) or {}
        left = np.asarray(state["left_arm_joint_state"], dtype=np.float64).reshape(-1)
        right = np.asarray(state["right_arm_joint_state"], dtype=np.float64).reshape(-1)
        grips = {"left": self._grip_norm("left", obs), "right": self._grip_norm("right", obs)}
        grips[side] = target
        for _ in range(GRIPPER_STEPS):
            self._sim_action(self._joint_action(
                left_arm=left, left_grip=grips["left"],
                right_arm=right, right_grip=grips["right"],
            ))
            if bool(self._require_env().end_flag[0]):
                break
        width = self._grip_norm(side)
        closing = target <= 0.5
        return {
            "ok": True,
            "arm": side,
            "action": str(action),
            "commanded": round(float(target), 3),
            "opening_norm": round(float(width), 3),
            "closed_empty": bool(closing and width <= GRIP_EMPTY_NORM),
        }

    def retreat(self, to: str = "takeover", arm: str | None = None) -> dict[str, Any]:
        """Put the hand back where it was when this intervention took over.

        The takeover point was reached by the policy under its own power, so
        it is a point the arm demonstrably can occupy -- which is what makes
        this the one recovery move worth having when a repair leaves the hand
        somewhere unworkable.  Scene objects are not touched: this is not a
        rewind.
        """
        self._require_cartesian()
        side = self._side(arm)
        if to not in ("takeover", "pre_grasp"):
            raise InterventionError(f"retreat 'to' must be takeover or pre_grasp, got {to!r}")
        # RoboCasa's ``pre_grasp`` is "the approach point just before the last
        # empty closure", reconstructed from its finger-width history. This
        # robot reports a normalised opening, not a width, and the state
        # machine that records that point does not exist here yet -- so the
        # request is honoured with the nearest thing we do have, and the reply
        # says so rather than pretending.
        take = (self._takeover_pose or {}).get(side)
        if not take:
            raise InterventionError(
                "no takeover pose recorded for this intervention; retreat is unavailable"
            )
        self._spend_action("retreat")
        pose = np.asarray(take["pose"], dtype=np.float64)
        was_open = float(take["grip"]) > GRIP_EMPTY_NORM
        if was_open:
            # Retreating means giving back whatever the repair picked up.
            obs = self._observation()
            state = obs.get("state", {}) or {}
            grips = {"left": self._grip_norm("left", obs), "right": self._grip_norm("right", obs)}
            grips[side] = 1.0
            for _ in range(GRIPPER_STEPS):
                self._sim_action(self._joint_action(
                    left_arm=np.asarray(state["left_arm_joint_state"], dtype=np.float64).reshape(-1),
                    left_grip=grips["left"],
                    right_arm=np.asarray(state["right_arm_joint_state"], dtype=np.float64).reshape(-1),
                    right_grip=grips["right"],
                ))
        rot_before = round(float(np.degrees(_quat_geodesic_rad(pose[3:], self._ee_pose(side)[3:]))), 1)
        out = self._servo_to(
            side, pose[:3], pose[3:], float(take["grip"]),
            max_steps=MOVE_TO_MAX_STEPS * 2,
        )
        out.update({
            "to": "takeover",
            "requested_to": str(to),
            "note": ("pre_grasp is not recorded on RoboDojo; went to the takeover pose instead"
                     if to == "pre_grasp" else None),
            "fingers": "open" if was_open else "close",
            "rot_before_deg": rot_before,
            "rot_after_deg": out.get("rot_err_deg"),
        })
        return out

    # -- observation surfaces the model may use ---------------------------

    def read_state(self) -> dict[str, Any]:
        """Model-facing proprioception.  NO oracle, NO success predicate."""
        state = self._proprio_state()
        state.pop("recent_chunks", None)
        return {
            "state": state,
            "arm_default": self._default_arm,
            "actions_budget": (int(self._round_budget) if self._round_budget is not None else None),
            "actions_used": int(self._round_used),
            "resets_left": int(MAX_RESETS - self._resets_used),
            "cartesian_blocked": bool(self._cartesian_blocked),
        }

    def render(self) -> dict[str, Any]:
        """All three camera views, in the shape ``cli/rex.py render`` expects.

        The CLI writes the PNGs itself (``rex.py:232-240`` iterates ``rgb`` and
        calls ``write_png``), so this returns pixels, not paths -- returning
        paths would leave the model with an empty frames directory and no error.
        No size parameter, for env_service.py's reason: a camera has the
        resolution it has, and re-rendering one instant at several resolutions
        is a free extra look no robot gets.
        """
        obs = self._observation()
        shot = self._frames_from_obs(obs)
        return {
            "size": _frame_size(),
            "rgb": {alias: image.tolist() for alias, image in shot.items()},
            "cameras": {alias: self._camera_meta(alias) for alias in shot},
            "main_camera": "primary",
            "depth": None,
        }

    def _camera_meta(self, alias: str) -> dict[str, Any]:
        """Intrinsics and world pose of one camera, or why they are missing."""
        try:
            from recovery_explore.env_service_robodojo import CAMERA_ALIASES
        except ImportError:  # RoboDojo's camera names; RoboTwin overrides CAMERA_ALIASES on the mixin anyway
            CAMERA_ALIASES = {"cam_head": "primary", "cam_left_wrist": "secondary", "cam_right_wrist": "wrist"}

        cam_key = {v: k for k, v in CAMERA_ALIASES.items()}.get(alias, alias)
        try:
            cam = self._camera_object(cam_key)
            intr = np.asarray(cam.data.intrinsic_matrices[0].detach().cpu(), dtype=np.float64)
            return {
                "intrinsics": [[round(float(v), 4) for v in row] for row in intr],
                "pos_w": [round(float(v), 4) for v in np.asarray(cam.data.pos_w[0].detach().cpu())],
                "quat_w_ros": [round(float(v), 5) for v in np.asarray(cam.data.quat_w_ros[0].detach().cpu())],
                "size": _frame_size(),
            }
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def unproject(self, row: int, col: int, camera: str = "primary") -> dict[str, Any]:
        """Pixel -> world point, using the camera's depth buffer.

        Needs depth in the observation: ObsManager only publishes it when the
        vision config asks for it (``env/observation_manager/obs_manager.py:50-51,141-148``),
        and it publishes ``approximate_depth`` (uint16 millimetres) by default
        with float metres only under ``depth: true``.  Refuse loudly rather
        than guess -- a silently wrong 3D target sends the arm into the table.
        """
        env = self._require_env()
        try:
            from recovery_explore.env_service_robodojo import CAMERA_ALIASES
        except ImportError:  # RoboDojo's camera names; RoboTwin overrides CAMERA_ALIASES on the mixin anyway
            CAMERA_ALIASES = {"cam_head": "primary", "cam_left_wrist": "secondary", "cam_right_wrist": "wrist"}

        alias_to_key = {v: k for k, v in CAMERA_ALIASES.items()}
        cam_key = alias_to_key.get(str(camera), str(camera))
        obs = self._observation()
        data = (obs.get("vision") or {}).get(cam_key)
        if not isinstance(data, Mapping):
            raise InterventionError(f"camera {camera!r} is not in this observation")
        if "depth" in data:
            depth = np.asarray(data["depth"], dtype=np.float64)
        elif "approximate_depth" in data:
            depth = np.asarray(data["approximate_depth"], dtype=np.float64) / 1000.0
        else:
            raise InterventionError(
                "this env publishes no depth; set `depth: true` (or leave "
                "`approximate_depth` on) in the vision config before using unproject"
            )
        v, u = int(row), int(col)   # rex speaks (row, col); the pinhole maths speaks (u=col, v=row)
        h, w = depth.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            raise InterventionError(f"pixel ({u},{v}) is outside the {w}x{h} image")
        z = float(depth[v, u])
        if not np.isfinite(z) or z <= 0:
            raise InterventionError(f"no depth at pixel ({u},{v})")

        cam = self._camera_object(cam_key)
        intr = np.asarray(cam.data.intrinsic_matrices[0].detach().cpu(), dtype=np.float64)
        fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
        point_cam = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])
        pos_w = np.asarray(cam.data.pos_w[0].detach().cpu(), dtype=np.float64)
        quat_ros = np.asarray(cam.data.quat_w_ros[0].detach().cpu(), dtype=np.float64)  # wxyz
        rot = _quat_to_matrix(quat_ros)
        world = pos_w + rot @ point_cam
        return {
            "camera": str(camera),
            "size": _frame_size(),
            "row": int(row),
            "col": int(col),
            "depth": round(z, 4),
            "world_xyz": [round(float(x), 4) for x in world],
        }

    def _camera_object(self, cam_key: str) -> Any:
        env = self._require_env()
        manager = env.camera_manager
        names = list(getattr(manager, "camera_names", []) or [])
        cameras = manager.cameras[0]
        if names and cam_key in names:
            return cameras[names.index(cam_key)]
        raise InterventionError(
            f"cannot map camera {cam_key!r} onto camera_manager.cameras[0]; "
            f"known names: {names}"
        )

    # -- the model's single entry point -----------------------------------

    def _resolve_plan_target(self, target: Any) -> tuple[list[float], dict[str, Any]]:
        """A plan's ``target`` -> world xyz. Accepts what the prompt documents.

        PLAN_SPEC offers the judge two ways to aim, and the pixel form is the
        one it actually reaches for -- it is looking at a picture, not at a
        coordinate frame:

            {"xyz": [x, y, z]}
            {"pixel": [row, col], "camera": "primary"|"secondary"|"wrist", "dz": 0.05}

        The first draft here accepted only a bare 3-list, so the very first
        real Qwen intervention (2026-09-18, general_pickup ep0) died with
        ``TypeError: float() argument must be ... not 'dict'`` on a plan that
        was correct by the prompt's own rules, twice in a row, and the judge
        gave the episode up.

        A bare ``[x, y, z]`` is also accepted: harmless, and some plans write it.
        """
        if isinstance(target, (list, tuple)):
            xyz = [float(v) for v in target]
            if len(xyz) != 3:
                raise InterventionError("a bare target must be 3 numbers [x, y, z]")
            return xyz, {"from": "xyz"}
        if not isinstance(target, Mapping):
            raise InterventionError("target must be an object with 'xyz' or 'pixel'")
        if "xyz" in target:
            xyz = [float(v) for v in target["xyz"]]
            if len(xyz) != 3:
                raise InterventionError("target.xyz must have 3 numbers")
            return xyz, {"from": "xyz"}
        if "pixel" not in target:
            raise InterventionError("target needs either 'xyz' or 'pixel'")
        try:
            row, col = (int(v) for v in target["pixel"])
        except Exception:
            raise InterventionError("target.pixel must be [row, col]")
        camera = str(target.get("camera") or "primary")
        hit = self.unproject(row=row, col=col, camera=camera)
        # unproject returns ``world_xyz``, not ``xyz``. Reading the wrong key
        # raises inside move_to and aborts the whole plan -- and pixel aiming
        # is the judge's most-used form.
        xyz = [float(v) for v in hit["world_xyz"]]
        dz = float(target.get("dz") or 0.0)
        xyz[2] += dz
        return xyz, {"from": "pixel", "pixel": [row, col], "camera": camera, "dz": dz}

    # -- the plan gate ----------------------------------------------------

    def _plan_gate(self, plan: Sequence[Any]) -> tuple[int, str] | None:
        """Reject a whole plan BEFORE any step runs. ``(index, why)`` or None.

        The judge's prompt tells the model these rules are enforced by the
        simulator, and they have to actually be enforced or the prompt is
        lying. Each one is a lesson the RoboCasa rounds paid for; the wording
        is kept close to ``env_service._repair_gate_check`` /
        ``_policy_step_gate`` so a judge that learned them there obeys them
        here.

        R1  A plan may not START with a policy step, and two policy steps may
            not sit next to each other. Round one shipped ``policy`` and a
            large share of plans were nothing but that: control handed straight
            back, nothing repaired, one intervention and one plan spent. The
            treatment arm went DOWN.
        R2  No manual close until a hand-back in this intervention has already
            failed. The repair for a missed grasp is to put the open fingers on
            a better line and let the policy close -- the policy is better at
            the last two centimetres than a string of hand-written nudges.
        R3  A manual close must be followed by a lift in the same plan, so the
            grasp is verified rather than assumed.
        R4  Once a step in this intervention has missed, overshot or been
            clipped, only ``gripper`` and ``policy`` remain. Cartesian commands
            from that arm configuration fail the same way.

        Rejecting the whole plan rather than stopping midway is deliberate: a
        half-applied plan hands the model a world it has to reverse-engineer.
        """
        steps = [st if isinstance(st, dict) else {} for st in plan]
        ops = [str(st.get("op")) for st in steps]

        if self._cartesian_blocked or self._window_faults > 0:
            for i, op in enumerate(ops):
                if op not in ("gripper", "policy"):
                    return i, (
                        "a step in this intervention already missed, overshot or was clipped; "
                        "from that arm configuration further Cartesian moves (move_to, nudge, "
                        "lift, rotate, retreat) fail the same way and are refused for the rest "
                        "of this intervention. Open the fingers if they are shut on nothing, "
                        "hand back one chunk, or end the intervention as fixed."
                    )

        for i, op in enumerate(ops):
            if op != "policy":
                continue
            if i == 0:
                return i, (
                    "a plan may not start with a policy step: hand control back only after you "
                    "have repaired something yourself. Put the move_to / nudge / gripper / "
                    "rotate that fixes the fault first, then the policy step."
                )
            if ops[i - 1] == "policy":
                return i, (
                    "two policy steps in a row are not allowed: each hand-back has to be earned "
                    "by a repair of your own. Merge them into one policy step with "
                    '"chunks": 2, or put a repair between them.'
                )

        for i, st in enumerate(steps):
            if ops[i] != "gripper":
                continue
            state = str(st.get("state") or st.get("action") or "")
            if state not in ("close", "grasp", "shut"):
                continue
            if self._round_policy_chunks <= 0 and "policy" not in ops[:i]:
                return i, (
                    "the first repair is to place the open fingers on a good line and hand back "
                    "to the policy for the close; a manual close is allowed only after a "
                    "hand-back in this intervention has failed. Replace the close with a policy "
                    "step, or re-stage and hand back."
                )
            if "lift" not in ops[i + 1:]:
                return i, (
                    "a manual close must be followed by a lift in the same plan so the grasp is "
                    'verified before anything else happens; add {"op": "lift", "dz": 0.04, '
                    '"gripper": "hold"} after it.'
                )
        return None

    def execute_plan(self, plan: Any, max_steps: int = 8) -> dict[str, Any]:
        """Run a plan step by step, stopping at the first abort condition.

        Stop reasons, same vocabulary the driver's prompts already speak
        (``judge_driver_v7.sh:950-1000``): ``plan_complete``, ``budget``,
        ``closed_empty``, ``servo_missed``, ``move_clipped``, ``overshoot``,
        ``policy_cap``, ``episode_end``, ``error``.  ``contact`` is absent --
        this robot has no force sensor, and a fabricated one would make the
        model trust a signal that is not there.
        """
        self._require_env()
        if not isinstance(plan, (list, tuple)) or not plan:
            raise ValueError("plan must be a non-empty list of steps")
        if len(plan) > int(max_steps):
            raise ValueError(f"plan has {len(plan)} steps; at most {int(max_steps)} are allowed")

        log: list[dict[str, Any]] = []
        stop_reason = "plan_complete"
        stopped_at: int | None = None

        # Whole-plan gate first: nothing runs if the plan breaks a rule the
        # prompt promised the simulator enforces.
        bad = self._plan_gate(plan)
        if bad is not None:
            index, why = bad
            state = self._proprio_state()
            state.pop("recent_chunks", None)
            return {
                "ok": False,
                "executed": [{"step": index, "op": str(plan[index].get("op"))
                              if isinstance(plan[index], dict) else "?", "error": why}],
                "stop_reason": "error",
                "stopped_at": -1,
                "remaining": list(plan),
                "remaining_count": len(plan),
                "actions_used": int(self._round_used),
                "actions_budget": (int(self._round_budget) if self._round_budget is not None else None),
                "resets_left": int(MAX_RESETS - self._resets_used),
                "cartesian_blocked": bool(self._cartesian_blocked),
                "state": state,
                "frames": {},
            }

        for i, raw in enumerate(plan):
            if not isinstance(raw, dict) or "op" not in raw:
                stop_reason, stopped_at = "error", i
                log.append({"step": i, "error": "each step needs an 'op'"})
                break
            op = str(raw["op"])
            if op not in self._PLAN_OPS:
                stop_reason, stopped_at = "error", i
                log.append({"step": i, "op": op,
                            "error": f"unknown op; allowed: {sorted(self._PLAN_OPS)}"})
                break
            arm = raw.get("arm")
            entry: dict[str, Any] = {"step": i, "op": op, "arm": self._side(arm)}
            try:
                grip = raw.get("gripper", "hold")
                if op == "move_to":
                    xyz, how = self._resolve_plan_target(raw.get("target"))
                    entry["target"] = [round(v, 4) for v in xyz]
                    entry["target_from"] = how
                    res = self.move_to(xyz=xyz, gripper=grip, arm=arm, quat=raw.get("quat"))
                elif op == "nudge":
                    res = self.nudge(dxyz=raw["dxyz"], gripper=grip, arm=arm)
                elif op == "lift":
                    res = self.lift(dz=float(raw["dz"]), gripper=grip, arm=arm)
                    entry["dz"] = round(float(raw["dz"]), 4)
                elif op == "rotate":
                    res = self.rotate(drot=raw["drot"], gripper=grip, arm=arm)
                elif op == "retreat":
                    res = self.retreat(to=str(raw.get("to") or "takeover"), arm=arm)
                elif op == "gripper":
                    res = self.gripper(action=str(raw.get("state") or raw.get("action") or "close"), arm=arm)
                else:  # policy hand-back
                    res, done = self._plan_policy_step(raw, entry)
                    if done is not None:
                        stop_reason, stopped_at = done, i
                        log.append(entry)
                        break
            except InterventionError as exc:
                entry["error"] = str(exc)
                log.append(entry)
                stopped_at, stop_reason = i, exc.reason
                break
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                log.append(entry)
                stopped_at, stop_reason = i, "error"
                break

            entry.update({k: v for k, v in res.items() if k not in ("ok",)})
            entry["ok"] = bool(res.get("ok", True))
            log.append(entry)

            if res.get("closed_empty"):
                stop_reason, stopped_at = "closed_empty", i
                break
            if res.get("overshoot"):
                stop_reason, stopped_at = "overshoot", i
                break
            if op in ("move_to", "nudge", "lift", "rotate", "retreat") and not res.get("ok", True):
                stop_reason, stopped_at = ("move_clipped" if res.get("clipped") else "servo_missed"), i
                break
            if bool(self._require_env().end_flag[0]):
                stop_reason, stopped_at = "episode_end", i
                break
            if self._round_budget is not None and self._round_used >= self._round_budget:
                stop_reason, stopped_at = "budget", i
                break

        remaining = [] if stopped_at is None else list(plan[stopped_at + 1:])
        state = self._proprio_state()
        state.pop("recent_chunks", None)
        return {
            "ok": stop_reason == "plan_complete",
            "executed": log,
            "stop_reason": stop_reason,
            "stopped_at": stopped_at,
            "remaining": remaining,
            "remaining_count": len(remaining),
            "actions_used": int(self._round_used),
            "actions_budget": (int(self._round_budget) if self._round_budget is not None else None),
            "resets_left": int(MAX_RESETS - self._resets_used),
            "cartesian_blocked": bool(self._cartesian_blocked),
            "state": state,
            # Pixels, not paths: rex.py stitches the three views into one
            # montage client-side (rex.py:273-289) and only then writes a file.
            "frames": self.render()["rgb"],
        }

    def _plan_policy_step(self, raw: Mapping[str, Any], entry: dict[str, Any]):
        """The ``policy`` op: hand back to the frozen policy for a few chunks.

        Capped twice -- per step and per intervention -- for the reason the
        RoboCasa runs found the hard way: a judge that can hand back without
        limit stops repairing and just re-runs the policy until something
        changes, which is not an intervention.
        """
        n = max(1, min(int(raw.get("chunks", 1)), self._PLAN_POLICY_MAX_CHUNKS))
        used = int(self._round_policy_chunks)
        if used + n > self._PLAN_POLICY_MAX_TOTAL:
            n = max(0, self._PLAN_POLICY_MAX_TOTAL - used)
        if n <= 0:
            entry["error"] = (
                f"policy hand-back allowance for this intervention "
                f"({self._PLAN_POLICY_MAX_TOTAL} chunks) is used up; finish the repair "
                "with your own steps or answer fixed"
            )
            return {}, "policy_cap"
        if self._round_budget is not None and self._round_used + n > self._round_budget:
            n = max(0, int(self._round_budget) - int(self._round_used))
        if n <= 0:
            entry["error"] = "no budget left for a policy step"
            return {}, "budget"
        adv = self.harness_advance(num_chunks=n)
        self._round_used += n
        self._round_policy_chunks = used + n
        entry.update({
            "chunks": n,
            "policy_chunks_left": max(0, self._PLAN_POLICY_MAX_TOTAL - self._round_policy_chunks),
            "committed_timestep": adv.get("committed_timestep"),
            "policy_done": bool(adv.get("done")),
        })
        return {"ok": True}, None


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion -> 3x3 rotation matrix."""
    w, x, y, z = _quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
