"""The TREATMENT arm for the RoboTwin 2.0 env service (SAPIEN 3, aloha-agilex).

A port of ``robodojo_intervention.py`` (RoboDojo / Isaac) to RoboTwin.  Same
RPC names, same argument shapes, same return fields and the same stop-reason
vocabulary, so ``cli/harness.py``, ``cli/rex.py`` and ``judge_driver_v7.sh``
drive it unchanged.  Mixed in first:

    class RoboTwinEnvService(RobotwinInterventionMixin, ..., RpcFacade):

HOST CONTRACT -- everything this mixin reads from or calls on its host
=====================================================================
Methods (all required unless marked optional):
    _require_env() -> TASK_ENV     the live ``envs._base_task.Base_Task``
                                   subclass instance; raises if none
    _observation() -> dict         ``TASK_ENV.get_obs()`` (renders cameras);
                                   only called for model-facing output
                                   (frames / final state), never inside the
                                   servo loop
    _proprio_state(obs=None)->dict model-facing proprioception; may contain
                                   ``recent_chunks`` (this mixin pops it)
    _oracle() -> dict              privileged success predicate (harness only)
    _frames_from_obs(obs)->dict    {alias: HxWx3 uint8} -- what ``render`` and
                                   ``execute_plan`` return as ``frames``
    harness_advance(num_chunks)    run the frozen policy for N chunks; returns
                                   a dict with ``committed_timestep`` / ``done``
    _log(record: dict)             append one JSON-able record to the call log
    _policy() (optional)           object with ``reset()``; called after a
                                   rewind unless the host defines
    _reset_policy_after_rewind() (optional) -- preferred over _policy().reset()
                                   when present (the lingbot-va websocket
                                   policy resets via ``infer({"reset": True,
                                   "prompt": ...})``, not a ``reset()`` method)
Attributes:
    _env              the TASK_ENV or None (``harness_round_status`` checks it)
    _episode          dict or None; this mixin reads/writes
                      ``committed_timestep`` and ``next_query_index``
    _step_count       int, policy control steps executed this episode
    _act_log          list of per-chunk dicts written by ``harness_advance``;
                      this mixin reads ``moved_cm`` ([left_cm, right_cm]) to
                      find the working arm and ``{side}_ee_pose`` (7-list,
                      xyz + wxyz quat, WORLD frame) for the visited-point
                      exemption.  If the host logs the robot's native
                      ``endpose`` rather than the TCP, set class attribute
                      ``ACT_LOG_POSE_POINT = "ee"`` so the 12 cm offset is
                      converted (default assumes "ee" -- see below).
Optional class attributes:
    CAMERA_ALIASES    {robotwin_camera_name: alias}; default maps head_camera
                      -> primary, front_camera -> secondary, left_camera ->
                      left_wrist, right_camera -> right_wrist
    FRAME_SIZE        reported as ``size`` by ``render``; default = the head
                      camera's height
Host responsibilities:
    * call ``self._init_intervention_state()`` in ``__init__``;
    * register ``self._intervention_rpc()`` in the RPC table;
    * refuse ``self._PRIVILEGED_METHODS`` while ``_round_budget`` is truthy
      or ``_judge_locked`` is set;
    * clear ``_act_log`` and call ``_init_intervention_state()`` (or at least
      reset ``_resets_used``) at episode begin.

TASK_ENV surface used (read from RoboTwin source, sapien 3.0.0b1):
    take_action(a14, action_type="qpos"), take_action_cnt (int), step_lim,
    eval_success, check_success(), get_arm_pose(side), scene (sapien.Scene,
    ``physx_system`` is a PhysxCpuSystem with pack()/unpack()), robot
    (envs/robot/robot.py: left/right_plan_path, get_*_arm_jointState,
    get_*_tcp_pose, *_gripper_val, *_gripper, *_entity, *_arm_joints),
    cameras (envs/camera/camera.py).

THE THINGS THAT ARE GENUINELY DIFFERENT FROM ROBODOJO
  1. **Arm IK is the robot's own cuRobo planner.**  ``robot.left_plan_path``
     / ``right_plan_path`` (``envs/robot/robot.py:425,459``) plan a
     collision-aware (table cuboid) trajectory to an ``endpose``-convention
     world pose; we take its last waypoint as the joint target and execute it
     as a 14-dim ``qpos`` action through ``take_action`` -- one move per
     servo iteration, the non-moving arm held at its current DRIVE TARGETS.
     A zero-length TOPP path makes ``take_action`` leave that arm uncommanded
     (``_base_task.py:1560-1575``), so the hold is exact.  Measured on dep-2:
     a 3 cm move lands within 0.9 mm after one step.
  2. **``take_action`` counts and latches.**  It increments
     ``take_action_cnt``, silently no-ops once ``take_action_cnt ==
     step_lim`` and latches ``eval_success`` the moment ``check_success()``
     holds mid-motion (``_base_task.py:1482-1491,1659-1664``).
     ``_sim_action`` lifts ``step_lim`` and restores the counter
     (``INTERVENE_CHARGES_STEP_LIM=1`` inverts), exactly as on RoboDojo.  A
     success reached DURING a repair latches like any other; see "known
     limitations" in the report.
  3. **Rewind is one PhysX blob plus the Python-side flags.**
     ``scene.physx_system.pack()`` holds every rigid-dynamic pose/velocity and
     every articulation's qpos/qvel AND joint drive targets / drive velocity
     targets (probed: a perturbed drive target comes back).  It does NOT hold
     the RoboTwin-side state: ``take_action_cnt``, ``eval_success``, the
     commanded gripper values ``robot.{left,right}_gripper_val`` (which
     ``check_success`` of e.g. ``handover_block`` reads via
     ``is_right_gripper_open``) and any task flags -- those are captured as the
     scalar attributes of the TASK_ENV and the robot.  An explicit per-entity
     capture rides along both as a belt-and-braces restore and as the digest.
  4. **Gripper convention.**  The qpos action's gripper channel is a
     normalised COMMAND, 1 = open, 0 = shut -- the same as RoboDojo, so a
     NUMBER in a plan is still RoboCasa's OSC convention (-1 open, +1 close)
     and is inverted, never clamped.  ``hold`` = the current COMMANDED value
     (``robot.*_gripper_val``), not the measured opening: holding at the
     measured opening would loosen a grasp whose fingers are blocked by the
     object.  The fingers bottom out at the joint lower limit 0.0 m, which the
     ``gripper_scale`` normalisation calls 0.18 -- so ``closed_empty`` uses the
     finger joint's own limits: measured opening / limit span <= GRIP_EMPTY_NORM.
  5. **Control point.**  The robot's ``endpose`` (what the policy observes and
     the planner consumes) sits 12 cm behind the fingertip centre (TCP) along
     the gripper's x axis.  The judge aims at OBJECTS in world metres, so the
     primitives servo the TCP by default (``RT_CONTROL_POINT=tcp``); every
     reply says which point it reports.  Set ``RT_CONTROL_POINT=ee`` to servo
     the endpose instead.

Quaternions are WXYZ everywhere (sapien ``Pose.q`` and transforms3d), and all
positions are WORLD frame metres (robot base is at y=-0.65 facing +y).
"""

from __future__ import annotations

import copy
import math
import os
import time
from typing import Any, Mapping, Sequence

import numpy as np

# Pure-math helpers and the shared error type: reused from the RoboDojo port
# (numpy-only module, no simulator import) so the two benchmarks raise the
# same exception class the harness already knows.
from recovery_explore.robodojo_intervention import (  # noqa: F401  (re-exported)
    InterventionError,
    _quat_conj,
    _quat_geodesic_rad,
    _quat_mul,
    _quat_normalize,
    _quat_to_matrix,
    _quat_to_rotvec,
    _rotvec_to_quat,
)

# ---------------------------------------------------------------------------
# Tunables.  Same names/defaults as the RoboDojo port where the concept is
# shared (env-var prefix RT_ instead of RD_).
# ---------------------------------------------------------------------------

#: Longest single servo SEGMENT (metres).  A longer move_to / nudge / lift /
#: move_both is executed as consecutive segments of at most this length.
MAX_STEP_M = float(os.environ.get("RT_MAX_STEP_M", "0.05"))
#: Largest total translation one model action may request (metres); only a
#: request beyond this is clipped (a point the arm already visited is exempt
#: for the single-arm moves).
# path mode (RT_MOTION=path, the default since 2026-09-22) plans the whole move at once, so the
# travel clip is lifted to the table's reach; the old servo keeps its 25 cm clip.
MAX_TOTAL_M = float(os.environ.get("RT_MAX_TOTAL_M",
                                   "1.5" if os.environ.get("RT_MOTION", "path") == "path" else "0.25"))
#: Largest rotation a single model action may request (radians, ~20 deg).
MAX_STEP_RAD = float(os.environ.get("RT_MAX_STEP_RAD", "0.35"))
#: Servo granularity: one ``take_action`` asks for at most this much travel.
MOVE_TO_STEP_CLIP = float(os.environ.get("RT_MOVE_STEP_CLIP", "0.02"))
#: Servo iteration cap for one move_to / nudge.
MOVE_TO_MAX_STEPS = int(os.environ.get("RT_MOVE_MAX_STEPS", "40"))
#: Position tolerance that counts as "arrived" (metres).  Same 5 mm as
#: RoboDojo; measured RoboTwin landing error is ~1 mm (see report).
MOVE_TO_TOL = float(os.environ.get("RT_MOVE_TOL", "0.005"))
#: Once inside MOVE_TO_TOL the servo keeps correcting (at most
#: FINE_MAX_STEPS more control steps) until inside this.  Measured on dep-2:
#: cuRobo's aloha-agilex patch (``planner.py``: -0.02 rad about z for the left
#: arm, -0.01 for the right) leaves a fixed ~3 mm / ~1 mm landing bias that
#: the integral aim below removes.
FINE_TOL = float(os.environ.get("RT_FINE_TOL", "0.002"))
FINE_MAX_STEPS = int(os.environ.get("RT_FINE_MAX_STEPS", "2"))
#: Anti-windup: the integral aim point never leads the measured hand by more
#: than MOVE_TO_STEP_CLIP plus this (metres).
AIM_WINDUP_M = float(os.environ.get("RT_AIM_WINDUP_M", "0.01"))
#: Orientation tolerance that counts as "arrived" (radians, ~4.6 deg).
ROT_TOL = float(os.environ.get("RT_ROT_TOL", "0.08"))
#: A stall that has already covered NEAR_MIN_FRAC of the way and ends within NEAR_TOL is reported as
#: "arrived_near" (ok, residual reported) instead of a servo miss. rtj12c handover_mic: a right-hand
#: nudge went 44 of 50 mm, 10 mm short with the wrist 4.9 deg off (ROT_TOL 4.6), and was called a
#: miss -- R4 then locked every Cartesian move for the rest of the intervention. Hands working close
#: together (handover, dual lift) hit this constantly.
NEAR_TOL = float(os.environ.get("RT_NEAR_TOL", "0.015"))
NEAR_MIN_FRAC = float(os.environ.get("RT_NEAR_MIN_FRAC", "0.6"))
#: Travelling more than this multiple of the requested distance = overshoot.
OVERSHOOT_RATIO = float(os.environ.get("RT_OVERSHOOT_RATIO", "3.0"))
#: Measured finger opening (fraction of the finger joint's limit span) at or
#: below which a commanded closure counts as having closed on nothing.
GRIP_EMPTY_NORM = float(os.environ.get("RT_GRIP_EMPTY", "0.05"))
#: ``take_action`` calls spent opening or closing the fingers (each is one
#: 50-substep hold when the arm does not move; 4 = 0.8 s sim time; probed:
#: an empty close reaches the joint limit on the 3rd).
GRIPPER_STEPS = int(os.environ.get("RT_GRIPPER_STEPS", "4"))
#: Motion mode (2026-09-22, user; RPent's move_to): "path" = ONE cuRobo plan from the current
#: pose to the target, the whole joint path executed (subsampled to PATH_SUBSTEPS waypoints),
#: then the final residual reported -- no 2 cm re-solving, no strict arrival gate, no lockout of
#: later Cartesian moves after a miss. "servo" = the old incremental servo.
MOTION_MODE = os.environ.get("RT_MOTION", "path")
PATH_SUBSTEPS = int(os.environ.get("RT_PATH_SUBSTEPS", "25"))
PATH_SETTLE = int(os.environ.get("RT_PATH_SETTLE", "6"))
#: A path move whose residual is within these counts as ok; beyond them the plan stops so the
#: judge can look again (the residual is always reported), but nothing is locked.
PATH_OK_M = float(os.environ.get("RT_PATH_OK_M", "0.02"))
PATH_OK_RAD = float(os.environ.get("RT_PATH_OK_RAD", "0.26"))
#: Rewinds allowed per episode.
MAX_RESETS = int(os.environ.get("MAX_RESETS", "5"))
#: 1 = repair steps come out of the policy's step_lim (conservative).
# 2026-09-22 (user): repair steps spend the episode step limit like the policy's own steps.
INTERVENE_CHARGES_STEP_LIM = os.environ.get("INTERVENE_CHARGES_STEP_LIM", "1") == "1"
#: Which point of the hand the primitives servo and report: "tcp" (between
#: the fingertips; default) or "ee" (RoboTwin's endpose, 12 cm behind it).
CONTROL_POINT = os.environ.get("RT_CONTROL_POINT", "tcp").strip().lower()
#: Offset from endpose to TCP along the gripper x axis (metres).  Derived from
#: ``Robot._trans_endpose``: tcp = link + R*[bias], ee = link + R*[bias-0.12].
TCP_OFFSET_M = 0.12

_SIDES = ("left", "right")

#: Scalar attributes of TASK_ENV / Robot that must NOT be rewound: they are
#: configuration or I/O plumbing, not simulated state.
_SCALAR_DENYLIST = frozenset({
    "render_freq", "save_data", "save_dir", "save_freq", "eval_video_path",
    "eval_mode", "task_name", "ep_num", "FRAME_IDX", "need_plan", "need_topp",
    "data_type", "instruction", "step_lim",   # step_lim handled explicitly
})

_DEFAULT_CAMERA_ALIASES = {
    "head_camera": "primary",
    "front_camera": "secondary",
    "left_camera": "left_wrist",
    "right_camera": "right_wrist",
}


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str, np.bool_, np.integer, np.floating))


def _tcp_from_ee(pose: Sequence[float]) -> np.ndarray:
    """endpose [xyz, wxyz] -> TCP pose (same orientation)."""
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    rot = _quat_to_matrix(pose[3:7])
    return np.concatenate([pose[:3] + rot[:, 0] * TCP_OFFSET_M, pose[3:7]])


def _ee_from_tcp(pose: Sequence[float]) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    rot = _quat_to_matrix(pose[3:7])
    return np.concatenate([pose[:3] - rot[:, 0] * TCP_OFFSET_M, pose[3:7]])


class RobotwinInterventionMixin:
    """Treatment-arm methods.  Mixed into ``RoboTwinEnvService`` (first base).

    See the module docstring for the complete host contract.
    """

    _PLAN_OPS = {"move_to", "nudge", "lift", "rotate", "gripper", "retreat", "policy",
                 "move_both", "lift_both", "locate"}
    #: Ops that servo an arm (a failure stops the plan as servo_missed /
    #: move_clipped and blocks further Cartesian steps).
    _CARTESIAN_OPS = ("move_to", "nudge", "lift", "rotate", "retreat", "move_both", "lift_both")
    #: Ops that drive BOTH arms together in the same control steps.
    _BIMANUAL_OPS = ("move_both", "lift_both")
    _PLAN_POLICY_MAX_CHUNKS = int(os.environ.get("PLAN_POLICY_MAX_CHUNKS", "2"))
    _PLAN_POLICY_MAX_TOTAL = int(os.environ.get("PLAN_POLICY_MAX_TOTAL", "4"))

    #: Refused whenever the model holds the action budget or the judge turn is
    #: locked (the host enforces it).  Same list as the RoboDojo port.
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

    #: What ``_act_log[*]["{side}_ee_pose"]`` holds: "ee" (RoboTwin endpose,
    #: what obs["endpose"] carries) or "tcp".  Host may override.
    ACT_LOG_POSE_POINT = "ee"

    # -- state ------------------------------------------------------------

    def _init_intervention_state(self) -> None:
        """Called from the host's ``__init__`` (and at episode begin)."""
        self._round_budget: int | None = None
        self._round_used: int = 0
        self._round_index: int = 0
        self._round_token: str | None = None
        self._round_policy_chunks: int = 0
        # Hand-backs already tried in THIS intervention window, across rewinds. reset_window zeroes
        # _round_policy_chunks (the chunk cap), which also erased the evidence R2 needs: after a
        # rewind a manual close was refused again even though 3 hand-backs had failed (rtj12b
        # pick_dual_bottles: 6 identical "reopen + hand back" attempts, close never allowed).
        self._window_handbacks: int = 0
        # A non-policy step has run since the last hand-back (R1 is about "earn each hand-back").
        self._repaired_since_handback: bool = False
        # No-repeat gate: plans that ran in the current attempt, and those of attempts already
        # rewound (failed) in this window. A rewound attempt must not be replayed verbatim.
        self._attempt_plan_sigs: list[tuple] = []
        self._failed_plan_sigs: list[tuple[int, tuple]] = []
        self._attempt_no: int = 1
        self._allow_cosmos: bool = True
        self._judge_locked: bool = False
        self._judge_token: str | None = None
        self._resets_used: int = 0
        self._resets_this_window: int = 0
        self._window_snapshot: dict[str, Any] | None = None
        self._checkpoint: dict[str, Any] | None = None
        self._takeover_pose: dict[str, Any] | None = None
        self._window_faults: int = 0
        self._cartesian_blocked: bool = False
        self._intervene_steps: int = 0
        self._default_arm: str = os.environ.get("RT_DEFAULT_ARM", "right")
        self._ik_failures: int = 0

    def _intervention_rpc(self) -> dict[str, Any]:
        return {
            "harness_open_action_budget": self.harness_open_action_budget,
            "harness_close_action_budget": self.harness_close_action_budget,
            "harness_judge_lock": self.harness_judge_lock,
            "harness_judge_unlock": self.harness_judge_unlock,
            "harness_round_begin": self.harness_round_begin,
            "harness_round_status": self.harness_round_status,
            "reset_window": self.reset_window,
            "execute_plan": self.execute_plan,
            "move_to": self.move_to,
            "nudge": self.nudge,
            "lift": self.lift,
            "rotate": self.rotate,
            "gripper": self.gripper,
            "retreat": self.retreat,
            "move_both": self.move_both,
            "lift_both": self.lift_both,
            "read_state": self.read_state,
            "render": self.render,
            "unproject": self.unproject,
            "restore_snapshot": self.restore_snapshot,
            "harness_checkpoint": self.harness_checkpoint,
            "harness_state_digest": self.harness_state_digest,
        }

    # -- arm / robot plumbing ---------------------------------------------

    def _robot(self):
        return self._require_env().robot

    def _active_arm(self) -> str:
        """Which hand is doing the work -- measured from the policy's own
        recent per-arm Cartesian travel (``_act_log[*].moved_cm``), not
        assumed.  RoboTwin tasks like adjust_bottle pick the arm per seed."""
        travel = {"left": 0.0, "right": 0.0}
        for entry in (getattr(self, "_act_log", None) or [])[-12:]:
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
        if side not in _SIDES:
            raise InterventionError(f"arm must be 'left' or 'right', got {arm!r}")
        return side

    def _endpose(self, side: str) -> np.ndarray:
        """RoboTwin's native endpose (world, xyz + wxyz) -- planner convention."""
        return np.asarray(self._require_env().get_arm_pose(side), dtype=np.float64).reshape(-1)

    def _ee_pose(self, side: str, obs: Mapping[str, Any] | None = None) -> np.ndarray:
        """Current world pose of the CONTROL POINT of one arm: [x,y,z,qw,qx,qy,qz].

        Read from the articulation directly (``obs`` is accepted for
        signature compatibility and ignored): cheap, and never stale -- the
        observation dict is only refreshed by a camera render.
        """
        ee = self._endpose(side)
        return _tcp_from_ee(ee) if CONTROL_POINT == "tcp" else ee

    def _to_planner_pose(self, pose: np.ndarray) -> np.ndarray:
        """Control-point pose -> the endpose the planner consumes."""
        return _ee_from_tcp(pose) if CONTROL_POINT == "tcp" else np.asarray(pose, dtype=np.float64)

    def _logged_pose(self, pose: Sequence[float]) -> np.ndarray:
        """An ``_act_log`` pose -> control-point pose."""
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        src = str(getattr(self, "ACT_LOG_POSE_POINT", "ee")).lower()
        if pose.shape[0] >= 7 and src != CONTROL_POINT:
            return _tcp_from_ee(pose) if CONTROL_POINT == "tcp" else _ee_from_tcp(pose)
        return pose

    def _finger_joint(self, side: str):
        robot = self._robot()
        return (robot.left_gripper if side == "left" else robot.right_gripper)[0][0]

    def _grip_norm(self, side: str, obs: Mapping[str, Any] | None = None) -> float:
        """MEASURED finger opening, 0 = fingers met, 1 = fully open.

        Normalised by the finger joint's own limits (aloha: [0, 0.04765] m),
        not ``gripper_scale`` ([-0.01, 0.045]): the command range extends past
        the physical stop, so a fully shut empty hand would read 0.18 there.
        """
        robot = self._robot()
        entity = robot.left_entity if side == "left" else robot.right_entity
        joint = self._finger_joint(side)
        joints = entity.get_active_joints()
        idx = [j.get_name() for j in joints].index(joint.get_name())
        q = float(entity.get_qpos()[idx])
        lo, hi = (float(v) for v in np.asarray(joint.get_limits(), dtype=np.float64).reshape(-1)[:2])
        if hi - lo < 1e-9:
            return 0.0
        return float(np.clip((q - lo) / (hi - lo), 0.0, 1.0))

    def _grip_cmd(self, side: str) -> float:
        """The COMMANDED gripper value (what ``hold`` keeps)."""
        robot = self._robot()
        return float(robot.left_gripper_val if side == "left" else robot.right_gripper_val)

    def _arm_targets(self, side: str) -> np.ndarray:
        """The arm's current joint DRIVE TARGETS (6), what take_action starts from."""
        robot = self._robot()
        js = robot.get_left_arm_jointState() if side == "left" else robot.get_right_arm_jointState()
        return np.asarray(js[:-1], dtype=np.float64)

    def _joint_action(self, *, left_arm: Sequence[float], left_grip: float,
                      right_arm: Sequence[float], right_grip: float) -> np.ndarray:
        """One 14-dim qpos action: left 6 + left grip + right 6 + right grip."""
        return np.concatenate([
            np.asarray(left_arm, dtype=np.float64).reshape(-1),
            [float(np.clip(left_grip, 0.0, 1.0))],
            np.asarray(right_arm, dtype=np.float64).reshape(-1),
            [float(np.clip(right_grip, 0.0, 1.0))],
        ])

    def _hold_action(self, overrides: Mapping[str, Any] | None = None) -> np.ndarray:
        """Hold both arms at their drive targets and grippers at their commands."""
        arms = {s: self._arm_targets(s) for s in _SIDES}
        grips = {s: self._grip_cmd(s) for s in _SIDES}
        for key, value in (overrides or {}).items():
            side, what = key.split("_", 1)
            (arms if what == "arm" else grips)[side] = value
        return self._joint_action(left_arm=arms["left"], left_grip=grips["left"],
                                  right_arm=arms["right"], right_grip=grips["right"])

    def _solve_ik(self, side: str, target_pose: np.ndarray) -> np.ndarray | None:
        """cuRobo (the robot's own planner) for one arm, world frame.

        ``target_pose`` is a CONTROL-POINT pose.  Plans from the arm's current
        qpos, so successive servo steps stay on one IK branch.  ``None`` = no
        solution.
        """
        robot = self._robot()
        planner_pose = self._to_planner_pose(np.asarray(target_pose, dtype=np.float64))
        plan = robot.left_plan_path if side == "left" else robot.right_plan_path
        try:
            result = plan([float(v) for v in planner_pose])
        except Exception as exc:  # cuRobo raises on malformed / unreachable input
            self._ik_failures += 1
            self._last_ik_error = f"{type(exc).__name__}: {exc}"
            return None
        if not isinstance(result, Mapping) or result.get("status") != "Success":
            self._ik_failures += 1
            return None
        pos = np.asarray(result["position"], dtype=np.float64)
        if pos.ndim != 2 or pos.shape[0] == 0:
            self._ik_failures += 1
            return None
        return pos[-1].reshape(-1)

    def _episode_over(self) -> bool:
        env = self._require_env()
        if bool(getattr(env, "eval_success", False)):
            return True
        lim = getattr(env, "step_lim", None)
        return lim is not None and int(env.take_action_cnt) >= int(lim)

    # -- the one place the simulator is actually stepped ------------------

    def _sim_action(self, action: Sequence[float]) -> None:
        """Apply one 14-dim qpos control step that does NOT spend the policy's
        budget: ``take_action`` increments ``take_action_cnt`` and no-ops at
        ``take_action_cnt == step_lim``; lift the limit for the call, restore
        the counter afterwards, keep our own tally.  The task's own success
        latch (``eval_success``) is left alone."""
        env = self._require_env()
        before = int(env.take_action_cnt)
        saved_lim = env.step_lim
        if not INTERVENE_CHARGES_STEP_LIM and saved_lim is not None:
            env.step_lim = int(saved_lim) + 1_000_000
        try:
            env.take_action(np.asarray(action, dtype=np.float64).reshape(-1), action_type="qpos")
        finally:
            env.step_lim = saved_lim
            if not INTERVENE_CHARGES_STEP_LIM:
                env.take_action_cnt = before
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
        if MOTION_MODE == "path":      # no lockout after a miss in path mode (the residual is reported)
            return
        if self._cartesian_blocked:
            raise InterventionError(
                "this intervention has already had a Cartesian command fail from this arm "
                "configuration; move_to, nudge, lift, rotate and retreat are refused for the "
                "rest of it. What remains: open the fingers, or hand back to the policy.",
                reason="servo_missed",
            )

    # -- servo ------------------------------------------------------------

    def _servo_to(self, side: str, target_xyz: np.ndarray, target_quat: np.ndarray | None,
                  grip: float, *, max_steps: int = MOVE_TO_MAX_STEPS,
                  tol: float = MOVE_TO_TOL, fine: bool = True) -> dict[str, Any]:
        """Incremental cuRobo servo of one arm's control point to a world pose.

        ``fine=False`` (an intermediate waypoint of a segmented move) accepts
        arrival inside ``tol`` without the FINE_TOL correction steps.

        One ``take_action`` per iteration, each asking for at most
        ``MOVE_TO_STEP_CLIP`` of travel, re-measuring in between.  Stops on
        arrival, iteration cap, planner failure, non-convergence (the only
        contact detector available -- there is no wrist F/T), or episode end.
        """
        target_xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        start = self._ee_pose(side)
        requested = float(np.linalg.norm(target_xyz - start[:3]))
        goal_quat = _quat_normalize(target_quat) if target_quat is not None else _quat_normalize(start[3:])
        other = "left" if side == "right" else "right"

        travelled = 0.0
        last_dist = requested
        stalled = 0
        reason = "arrived"
        steps = 0
        fine_steps = 0
        aim = None           # integral aim point: cancels the planner's fixed landing bias
        for _ in range(int(max_steps)):
            cur = self._ee_pose(side)
            delta = target_xyz - cur[:3]
            dist = float(np.linalg.norm(delta))
            rot_err = _quat_geodesic_rad(goal_quat, cur[3:])
            # At least one control step: the gripper command rides on it.
            if steps > 0 and rot_err <= ROT_TOL and (
                    dist <= min(tol, FINE_TOL)
                    or (dist <= tol and (fine_steps >= FINE_MAX_STEPS or not fine))):
                reason = "arrived"
                break
            if steps > 0 and dist <= tol:
                fine_steps += 1
            step = delta if dist <= MOVE_TO_STEP_CLIP else delta * (MOVE_TO_STEP_CLIP / dist)
            aim = (cur[:3] if aim is None else aim) + step
            lead = aim - cur[:3]
            lead_n = float(np.linalg.norm(lead))
            cap = MOVE_TO_STEP_CLIP + AIM_WINDUP_M
            if lead_n > cap:
                aim = cur[:3] + lead * (cap / lead_n)
            joints = self._solve_ik(side, np.concatenate([aim, goal_quat]))
            if joints is None:
                reason = "ik_failed"
                break
            self._sim_action(self._hold_action({f"{side}_arm": joints, f"{side}_grip": grip,
                                                f"{other}_grip": self._grip_cmd(other)}))
            steps += 1
            after = self._ee_pose(side)
            travelled += float(np.linalg.norm(after[:3] - cur[:3]))
            new_dist = float(np.linalg.norm(target_xyz - after[:3]))
            new_rot = _quat_geodesic_rad(goal_quat, after[3:])
            if new_dist > last_dist - 1e-4 and not (new_dist <= tol and new_rot <= ROT_TOL):
                stalled += 1
                if stalled >= 3:
                    reason = "stalled"
                    break
            else:
                stalled = 0
            last_dist = new_dist
            if self._episode_over():
                reason = "episode_end"
                break
        else:
            reason = "step_cap"

        final = self._ee_pose(side)
        final_dist = float(np.linalg.norm(target_xyz - final[:3]))
        if reason == "step_cap" and final_dist <= tol:
            reason = "arrived"
        if (reason in ("stalled", "step_cap") and final_dist <= NEAR_TOL
                and requested > 1e-4 and travelled >= NEAR_MIN_FRAC * requested):
            reason = "arrived_near"
        ok = (final_dist <= tol and reason == "arrived") or reason == "arrived_near"
        overshot = requested > 1e-4 and travelled > OVERSHOOT_RATIO * requested
        out = {
            "ok": bool(ok),
            "arm": side,
            "control_point": CONTROL_POINT,
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
        if reason == "episode_end" and bool(getattr(self._require_env(), "eval_success", False)):
            out["task_success_latched"] = True
        if not ok:
            self._cartesian_blocked = True
            self._window_faults += 1
        return out

    @staticmethod
    def _n_segments(dist: float) -> int:
        return max(1, int(math.ceil(float(dist) / MAX_STEP_M - 1e-9)))

    def _segmented_servo(self, side: str, target_xyz: np.ndarray, target_quat: np.ndarray | None,
                         grip: float) -> dict[str, Any]:
        """``_servo_to`` along a straight line in segments of at most
        ``MAX_STEP_M``; the last one lands on ``target_xyz`` with the fine
        correction.  Stops at the first segment that does not arrive."""
        target_xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        start = self._ee_pose(side)[:3]
        vec = target_xyz - start
        requested = float(np.linalg.norm(vec))
        n = self._n_segments(requested)
        travelled, steps, seg_overshoot, done = 0.0, 0, False, 0
        out: dict[str, Any] = {}
        for k in range(1, n + 1):
            last = k == n
            waypoint = target_xyz if last else start + vec * (k / n)
            out = self._servo_to(side, waypoint, target_quat, grip, fine=last)
            travelled += float(out["travelled_m"])
            steps += int(out["control_steps"])
            seg_overshoot = seg_overshoot or bool(out["overshoot"])
            done = k
            if not out["ok"]:
                break
        final_dist = float(np.linalg.norm(target_xyz - self._ee_pose(side)[:3]))
        out.update({
            "requested_dist_m": round(requested, 4),
            "final_dist_m": round(final_dist, 4),
            "travelled_m": round(travelled, 4),
            "control_steps": int(steps),
            "overshoot": bool(seg_overshoot or (requested > 1e-4 and travelled > OVERSHOOT_RATIO * requested)),
            "segments": int(n),
            "segments_done": int(done),
        })
        return out

    def _plan_full_path(self, side: str, target_pose: np.ndarray) -> np.ndarray | None:
        """One cuRobo plan for one arm (world control-point pose) -> the whole joint path."""
        robot = self._robot()
        planner_pose = self._to_planner_pose(np.asarray(target_pose, dtype=np.float64))
        plan = robot.left_plan_path if side == "left" else robot.right_plan_path
        try:
            result = plan([float(v) for v in planner_pose])
        except Exception as exc:
            self._ik_failures += 1
            self._last_ik_error = f"{type(exc).__name__}: {exc}"
            return None
        if not isinstance(result, Mapping) or result.get("status") != "Success":
            self._ik_failures += 1
            return None
        pos = np.asarray(result["position"], dtype=np.float64)
        if pos.ndim != 2 or pos.shape[0] == 0:
            self._ik_failures += 1
            return None
        return pos

    @staticmethod
    def _subsample(path: np.ndarray, k: int) -> np.ndarray:
        if k >= 1 and len(path) > k:
            return path[np.linspace(0, len(path) - 1, k).astype(int)]
        return path

    def _path_move(self, side: str, target_xyz: np.ndarray, target_quat: np.ndarray | None,
                   grip: float) -> dict[str, Any]:
        """RPent-style move: plan once, run the whole path, report where the hand ended up."""
        target_xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        start = self._ee_pose(side)
        requested = float(np.linalg.norm(target_xyz - start[:3]))
        goal_quat = _quat_normalize(target_quat) if target_quat is not None else _quat_normalize(start[3:])
        other = "left" if side == "right" else "right"
        path = self._plan_full_path(side, np.concatenate([target_xyz, goal_quat]))
        reason, steps = "arrived", 0
        if path is None:
            reason = "plan_failed"
        else:
            for q in self._subsample(path, PATH_SUBSTEPS):
                self._sim_action(self._hold_action({f"{side}_arm": q.reshape(-1), f"{side}_grip": grip,
                                                    f"{other}_grip": self._grip_cmd(other)}))
                steps += 1
                if self._episode_over():
                    reason = "episode_end"
                    break
        if reason == "arrived" and path is not None:
            # the PD loop lags the last waypoint: hold it until the hand stops moving (<= PATH_SETTLE)
            last = path[-1].reshape(-1)
            prev = self._ee_pose(side)[:3]
            for _ in range(PATH_SETTLE):
                self._sim_action(self._hold_action({f"{side}_arm": last, f"{side}_grip": grip,
                                                    f"{other}_grip": self._grip_cmd(other)}))
                steps += 1
                if self._episode_over():
                    reason = "episode_end"
                    break
                now = self._ee_pose(side)[:3]
                if float(np.linalg.norm(now - prev)) < 0.002:
                    break
                prev = now
        final = self._ee_pose(side)
        final_dist = float(np.linalg.norm(target_xyz - final[:3]))
        rot = _quat_geodesic_rad(goal_quat, final[3:])
        travelled = float(np.linalg.norm(final[:3] - start[:3]))
        if reason == "arrived" and not (final_dist <= PATH_OK_M and rot <= PATH_OK_RAD):
            reason = "residual"
        out = {
            "ok": reason in ("arrived", "episode_end") and final_dist <= PATH_OK_M,
            "arm": side, "control_point": CONTROL_POINT, "motion": "path",
            "requested_dist_m": round(requested, 4), "final_dist_m": round(final_dist, 4),
            "travelled_m": round(travelled, 4), "control_steps": int(steps),
            "servo_stop": reason, "overshoot": False,
            "eef": [round(float(v), 4) for v in final[:3]],
            "quat": [round(float(v), 4) for v in final[3:]],
            "rot_err_deg": round(float(np.degrees(rot)), 1),
        }
        if reason == "episode_end" and bool(getattr(self._require_env(), "eval_success", False)):
            out["task_success_latched"] = True
        return out

    def _path_move_both(self, targets: Mapping[str, np.ndarray], grips: Mapping[str, float]) -> dict[str, Any]:
        """Both arms: one plan each, executed in lockstep (both paths resampled to one length)."""
        start = {s: self._ee_pose(s) for s in _SIDES}
        paths, failed = {}, []
        for s, xyz in targets.items():
            p = self._plan_full_path(s, np.concatenate([np.asarray(xyz, dtype=np.float64), _quat_normalize(start[s][3:])]))
            if p is None:
                failed.append(s)
            else:
                paths[s] = p
        steps, reason = 0, "arrived" if not failed else "plan_failed"
        if not failed:
            k = min(PATH_SUBSTEPS, max(len(p) for p in paths.values()))
            res = {s: p[np.linspace(0, len(p) - 1, k).astype(int)] for s, p in paths.items()}
            for i in range(k):
                parts = {f"{s}_arm": res[s][i].reshape(-1) for s in res}
                parts.update({f"{s}_grip": float(grips[s]) for s in _SIDES})
                self._sim_action(self._hold_action(parts))
                steps += 1
                if self._episode_over():
                    reason = "episode_end"
                    break
        final = {s: self._ee_pose(s) for s in _SIDES}
        per = {}
        for s, xyz in targets.items():
            fd = float(np.linalg.norm(np.asarray(xyz) - final[s][:3]))
            per[s] = {"final_dist_m": round(fd, 4), "travelled_m": round(float(np.linalg.norm(final[s][:3] - start[s][:3])), 4),
                      "eef": [round(float(v), 4) for v in final[s][:3]]}
        worst = max(v["final_dist_m"] for v in per.values())
        if reason == "arrived" and worst > PATH_OK_M:
            reason = "residual"
        return {"ok": reason in ("arrived", "episode_end") and worst <= PATH_OK_M, "motion": "path",
                "servo_stop": reason, "control_steps": int(steps), "per_arm": per, "overshoot": False,
                "plan_failed_arms": failed}

    def _servo_both(self, targets: Mapping[str, np.ndarray], grips: Mapping[str, float], *,
                    max_steps: int = MOVE_TO_MAX_STEPS, tol: float = MOVE_TO_TOL,
                    fine: bool = True) -> dict[str, Any]:
        """Servo BOTH control points in the same control steps.

        Each iteration asks every moving arm for the same FRACTION of its
        remaining way (the longer way advances ``MOVE_TO_STEP_CLIP``), so two
        hands given the same displacement move in lockstep and keep their
        relative offset -- a jointly held object is carried, not torn.  Each
        arm keeps its own integral aim (the planner's per-arm landing bias) and
        its own wrist orientation.  Arms not in ``targets`` hold their drive
        targets.  Same stop vocabulary as ``_servo_to``.
        """
        sides = [s for s in _SIDES if s in targets]
        tgt = {s: np.asarray(targets[s], dtype=np.float64).reshape(3) for s in sides}
        start = {s: self._ee_pose(s) for s in _SIDES}
        quat = {s: _quat_normalize(start[s][3:]) for s in sides}
        requested = {s: float(np.linalg.norm(tgt[s] - start[s][:3])) for s in sides}
        travelled = {s: 0.0 for s in sides}
        aim: dict[str, Any] = {s: None for s in sides}
        last_dist = max(requested.values()) if sides else 0.0
        stalled, steps, fine_steps = 0, 0, 0
        reason, failed_arm = "arrived", None
        for _ in range(int(max_steps)):
            cur = {s: self._ee_pose(s) for s in sides}
            delta = {s: tgt[s] - cur[s][:3] for s in sides}
            dist = {s: float(np.linalg.norm(delta[s])) for s in sides}
            dmax = max(dist.values())
            rot_ok = all(_quat_geodesic_rad(quat[s], cur[s][3:]) <= ROT_TOL for s in sides)
            if steps > 0 and rot_ok and (
                    dmax <= min(tol, FINE_TOL)
                    or (dmax <= tol and (fine_steps >= FINE_MAX_STEPS or not fine))):
                reason = "arrived"
                break
            if steps > 0 and dmax <= tol:
                fine_steps += 1
            scale = 1.0 if dmax <= MOVE_TO_STEP_CLIP else MOVE_TO_STEP_CLIP / dmax
            joints: dict[str, np.ndarray] = {}
            for s in sides:
                a = (cur[s][:3] if aim[s] is None else aim[s]) + delta[s] * scale
                lead = a - cur[s][:3]
                lead_n = float(np.linalg.norm(lead))
                cap = MOVE_TO_STEP_CLIP + AIM_WINDUP_M
                if lead_n > cap:
                    a = cur[s][:3] + lead * (cap / lead_n)
                aim[s] = a
                j = self._solve_ik(s, np.concatenate([a, quat[s]]))
                if j is None:
                    failed_arm = s
                    break
                joints[s] = j
            if failed_arm is not None:
                reason = "ik_failed"
                break
            overrides: dict[str, Any] = {f"{s}_arm": joints[s] for s in sides}
            overrides.update({f"{s}_grip": float(grips.get(s, self._grip_cmd(s))) for s in _SIDES})
            self._sim_action(self._hold_action(overrides))
            steps += 1
            after = {s: self._ee_pose(s) for s in sides}
            for s in sides:
                travelled[s] += float(np.linalg.norm(after[s][:3] - cur[s][:3]))
            new_dist = max(float(np.linalg.norm(tgt[s] - after[s][:3])) for s in sides)
            new_rot_ok = all(_quat_geodesic_rad(quat[s], after[s][3:]) <= ROT_TOL for s in sides)
            if new_dist > last_dist - 1e-4 and not (new_dist <= tol and new_rot_ok):
                stalled += 1
                if stalled >= 3:
                    reason = "stalled"
                    break
            else:
                stalled = 0
            last_dist = new_dist
            if self._episode_over():
                reason = "episode_end"
                break
        else:
            reason = "step_cap"

        final = {s: self._ee_pose(s) for s in _SIDES}
        final_dist = {s: float(np.linalg.norm(tgt[s] - final[s][:3])) for s in sides}
        fmax = max(final_dist.values()) if sides else 0.0
        if reason == "step_cap" and fmax <= tol:
            reason = "arrived"
        if (reason in ("stalled", "step_cap") and sides and fmax <= NEAR_TOL
                and all(requested[s] <= 1e-4 or travelled[s] >= NEAR_MIN_FRAC * requested[s] for s in sides)):
            reason = "arrived_near"        # same rule as _servo_to
        ok = (fmax <= tol and reason == "arrived") or reason == "arrived_near"
        overshot = any(requested[s] > 1e-4 and travelled[s] > OVERSHOOT_RATIO * requested[s] for s in sides)
        off0 = start["right"][:3] - start["left"][:3]
        off1 = final["right"][:3] - final["left"][:3]
        out: dict[str, Any] = {
            "ok": bool(ok),
            "arm": "both",
            "moving_arms": list(sides),
            "control_point": CONTROL_POINT,
            "requested_dist_m": round(max(requested.values()) if sides else 0.0, 4),
            "final_dist_m": round(fmax, 4),
            "travelled_m": round(max(travelled.values()) if sides else 0.0, 4),
            "control_steps": int(steps),
            "servo_stop": reason,
            "overshoot": bool(overshot),
            "eef": {s: [round(float(v), 4) for v in final[s][:3]] for s in _SIDES},
            "per_arm": {s: {"requested_dist_m": round(requested[s], 4),
                            "final_dist_m": round(final_dist[s], 4),
                            "travelled_m": round(travelled[s], 4),
                            "rot_err_deg": round(float(np.degrees(
                                _quat_geodesic_rad(quat[s], final[s][3:]))), 1)}
                        for s in sides},
            # Change of the right-minus-left hand offset (metres): ~0 when the
            # hands moved together, as they must for a jointly held object.
            "hand_offset_drift_m": round(float(np.linalg.norm(off1 - off0)), 4),
        }
        if failed_arm is not None:
            out["failed_arm"] = failed_arm
        if reason == "episode_end" and bool(getattr(self._require_env(), "eval_success", False)):
            out["task_success_latched"] = True
        if not ok:
            self._cartesian_blocked = True
            self._window_faults += 1
        return out

    # -- snapshots --------------------------------------------------------

    def _articulations(self):
        return list(self._require_env().scene.get_all_articulations())

    def _dynamic_actors(self):
        """(key, entity, rigid-dynamic component) for every movable actor."""
        import sapien  # local: keeps the module importable without SAPIEN

        out = []
        for i, entity in enumerate(self._require_env().scene.get_all_actors()):
            comp = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            if comp is None:
                continue
            out.append((f"{i}:{entity.get_name()}", entity, comp))
        return out

    def _capture_entities(self) -> dict[str, Any]:
        """Explicit per-entity physical state.  Used for the belt-and-braces
        restore after ``unpack`` and as the fidelity digest."""
        state: dict[str, Any] = {"actors": {}, "articulations": {}}
        for key, entity, comp in self._dynamic_actors():
            pose = entity.get_pose()
            state["actors"][key] = {
                "pose": np.concatenate([np.asarray(pose.p, dtype=np.float64),
                                        np.asarray(pose.q, dtype=np.float64)]),
                "lin": np.asarray(comp.get_linear_velocity(), dtype=np.float64).copy(),
                "ang": np.asarray(comp.get_angular_velocity(), dtype=np.float64).copy(),
                "kinematic": bool(comp.get_kinematic()),
            }
        for i, art in enumerate(self._articulations()):
            root = art.get_root_pose()
            joints = art.get_active_joints()
            state["articulations"][f"{i}:{art.get_name()}"] = {
                "root_pose": np.concatenate([np.asarray(root.p, dtype=np.float64),
                                             np.asarray(root.q, dtype=np.float64)]),
                "qpos": np.asarray(art.get_qpos(), dtype=np.float64).copy(),
                "qvel": np.asarray(art.get_qvel(), dtype=np.float64).copy(),
                "drive_target": np.asarray([float(np.asarray(j.get_drive_target()).reshape(-1)[0])
                                            if j.get_dof() else 0.0 for j in joints], dtype=np.float64),
                "drive_vel_target": np.asarray([float(np.asarray(j.get_drive_velocity_target()).reshape(-1)[0])
                                                if j.get_dof() else 0.0 for j in joints], dtype=np.float64),
            }
        return state

    def _restore_entities(self, state: Mapping[str, Any]) -> int:
        import sapien

        restored = 0
        live = {key: (entity, comp) for key, entity, comp in self._dynamic_actors()}
        for key, entry in (state.get("actors") or {}).items():
            if key not in live:
                continue
            entity, comp = live[key]
            pose = np.asarray(entry["pose"], dtype=np.float64)
            entity.set_pose(sapien.Pose(pose[:3], pose[3:]))
            if not entry.get("kinematic"):
                comp.set_linear_velocity(np.asarray(entry["lin"], dtype=np.float32))
                comp.set_angular_velocity(np.asarray(entry["ang"], dtype=np.float32))
            restored += 1
        arts = {f"{i}:{a.get_name()}": a for i, a in enumerate(self._articulations())}
        for key, entry in (state.get("articulations") or {}).items():
            art = arts.get(key)
            if art is None:
                continue
            root = np.asarray(entry["root_pose"], dtype=np.float64)
            art.set_root_pose(sapien.Pose(root[:3], root[3:]))
            art.set_qpos(np.asarray(entry["qpos"], dtype=np.float32))
            art.set_qvel(np.asarray(entry["qvel"], dtype=np.float32))
            for j, dt, dv in zip(art.get_active_joints(), entry["drive_target"], entry["drive_vel_target"]):
                j.set_drive_target(float(dt))
                j.set_drive_velocity_target(float(dv))
            restored += 1
        return restored

    def _capture_scalars(self, obj: Any) -> dict[str, Any]:
        return {
            k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in vars(obj).items()
            if _is_scalar(v) and k not in _SCALAR_DENYLIST
        }

    def _capture_snapshot(self, tag: str) -> dict[str, Any]:
        env = self._require_env()
        episode = getattr(self, "_episode", None) or {}
        return {
            "tag": str(tag),
            "ts": time.time(),
            # bytes: immutable, so this is a true copy, not a view.
            "physx": env.scene.physx_system.pack(),
            "entities": copy.deepcopy(self._capture_entities()),
            # RoboTwin-side bookkeeping: take_action_cnt, eval_success,
            # stage_success_tag, plan_success, task flags ...
            "env_scalars": self._capture_scalars(env),
            # the commanded gripper values check_success reads
            "robot_scalars": self._capture_scalars(env.robot),
            "step_lim": env.step_lim,
            "step_count": int(getattr(self, "_step_count", 0)),
            "committed_timestep": int(episode.get("committed_timestep") or 0),
            "next_query_index": int(episode.get("next_query_index") or 0),
            "intervene_steps": int(self._intervene_steps),
            "act_log_len": len(getattr(self, "_act_log", None) or []),
        }

    def _apply_snapshot(self, snap: Mapping[str, Any]) -> None:
        env = self._require_env()
        env.scene.physx_system.unpack(snap["physx"])
        self._restore_entities(snap.get("entities") or {})
        for k, v in (snap.get("env_scalars") or {}).items():
            setattr(env, k, v)
        for k, v in (snap.get("robot_scalars") or {}).items():
            setattr(env.robot, k, v)
        env.step_lim = snap["step_lim"]
        self._step_count = int(snap["step_count"])
        episode = getattr(self, "_episode", None)
        if episode is not None:
            episode["committed_timestep"] = int(snap["committed_timestep"])
            episode["next_query_index"] = int(snap["next_query_index"])
        self._intervene_steps = int(snap.get("intervene_steps", self._intervene_steps))
        # Render-side poses (wrist cameras ride on links) follow physics.
        try:
            env._update_render()
        except Exception:
            try:
                env.scene.update_render()
            except Exception:
                pass
        # The policy keeps an observation window; after a rewind it describes
        # a future that did not happen.
        try:
            hook = getattr(self, "_reset_policy_after_rewind", None)
            if callable(hook):
                hook()
            else:
                self._policy().reset()
        except Exception:
            pass

    # -- harness verbs ----------------------------------------------------

    def harness_open_action_budget(self, budget: int, allow_cosmos: bool = True,
                                   round_index: int | None = None,
                                   control_token: str | None = None) -> dict[str, Any]:
        """HARNESS ONLY: open one intervention window.  Does not rewind."""
        self._round_budget = int(budget)
        self._round_used = 0
        self._round_policy_chunks = 0
        self._window_handbacks = 0
        self._repaired_since_handback = False
        self._attempt_plan_sigs = []
        self._failed_plan_sigs = []
        self._attempt_no = 1
        self._round_token = str(control_token) if control_token else None
        self._allow_cosmos = bool(allow_cosmos)
        self._round_index = int(round_index) if round_index is not None else self._round_index + 1
        self._resets_this_window = 0
        self._window_faults = 0
        self._cartesian_blocked = False
        try:
            self._window_snapshot = self._capture_snapshot("window_open")
        except Exception as exc:
            self._window_snapshot = None
            self._log({"ts": time.time(), "method": "snapshot_failed", "error": repr(exc)})
        self._takeover_pose = self._capture_takeover_pose()
        snap = self._window_snapshot or {}
        return {
            "ok": True,
            "round_index": self._round_index,
            "round_budget": self._round_budget,
            "round_used": self._round_used,
            "allow_cosmos": self._allow_cosmos,
            "step_count": int(getattr(self, "_step_count", 0)),
            "token_required": self._round_token is not None,
            "reset_available": self._window_snapshot is not None,
            "resets_left": int(MAX_RESETS - self._resets_used),
            "objects_snapshotted": len((snap.get("entities") or {}).get("actors") or {})
            + len((snap.get("entities") or {}).get("articulations") or {}),
            # RoboTwin has no cloth/fluid: every task is rigid + articulated.
            "objects_not_snapshottable": [],
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

    def harness_round_begin(self, budget: int, round_index: int | None = None) -> dict[str, Any]:
        """HARNESS ONLY: rewind to the last ``harness_checkpoint`` (if any) and
        open a fresh action budget -- RoboCasa's round semantics, with the
        checkpoint standing in for RoboCasa's failure snapshot."""
        rewound = False
        if self._checkpoint:
            self._apply_snapshot(self._checkpoint)
            rewound = True
        out = self.harness_open_action_budget(budget=budget, round_index=round_index)
        out["rewound_to_checkpoint"] = rewound
        out.update(self.harness_round_status())
        return out

    def harness_round_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "round_index": self._round_index,
            "round_budget": self._round_budget,
            "round_used": self._round_used,
            "allow_cosmos": self._allow_cosmos,
            "step_count": int(getattr(self, "_step_count", 0)),
            "intervene_control_steps": int(self._intervene_steps),
        }
        if getattr(self, "_env", None) is not None:
            status["oracle"] = self._oracle()
        return status

    def harness_checkpoint(self, tag: str = "manual") -> dict[str, Any]:
        """HARNESS ONLY: take a snapshot without opening a window.

        Stored both as the window snapshot (so ``restore-window`` restores it,
        as on RoboDojo) and as the round checkpoint ``harness_round_begin``
        rewinds to."""
        snap = self._capture_snapshot(str(tag))
        self._window_snapshot = snap
        self._checkpoint = snap
        return {"ok": True, "tag": str(tag), "step_count": int(getattr(self, "_step_count", 0))}

    def harness_state_digest(self) -> dict[str, Any]:
        """HARNESS ONLY: a comparable fingerprint of the whole simulated world.

        Same shape as the RoboDojo digest (``scene`` = {key: {sum, absmax,
        shape}}), plus ``values`` per key so a fidelity test can report the
        worst per-component delta instead of just "differs"."""
        env = self._require_env()
        digest: dict[str, Any] = {}
        ents = self._capture_entities()
        for kind in ("actors", "articulations"):
            for name, fields in ents[kind].items():
                for field, value in fields.items():
                    if isinstance(value, bool):
                        continue
                    arr = np.asarray(value, dtype=np.float64)
                    digest[f"{kind}/{name}/{field}"] = {
                        "sum": round(float(np.nansum(arr)), 6),
                        "absmax": round(float(np.nanmax(np.abs(arr))) if arr.size else 0.0, 6),
                        "shape": list(arr.shape),
                        "values": [round(float(v), 7) for v in arr.reshape(-1)],
                    }
        try:
            success_now = bool(env.check_success())
        except Exception as exc:
            success_now = f"error: {type(exc).__name__}: {exc}"
        return {
            "scene": digest,
            "unsnapshottable": [],
            "left_ee_pose": [round(float(v), 6) for v in self._ee_pose("left")],
            "right_ee_pose": [round(float(v), 6) for v in self._ee_pose("right")],
            "control_point": CONTROL_POINT,
            "opening": {s: round(self._grip_norm(s), 6) for s in _SIDES},
            "gripper_cmd": {s: round(self._grip_cmd(s), 6) for s in _SIDES},
            "take_action_cnt": int(env.take_action_cnt),
            "end_flag": bool(self._episode_over()),
            "success": bool(getattr(env, "eval_success", False)),
            "check_success": success_now,
            "step_count": int(getattr(self, "_step_count", 0)),
            "committed_timestep": int((getattr(self, "_episode", None) or {}).get("committed_timestep") or 0),
            "oracle": self._oracle(),
        }

    def restore_snapshot(self, which: str = "window", **_ignored: Any) -> dict[str, Any]:
        """HARNESS ONLY: put the world back to the stored snapshot.

        ``cli/harness.py restore-snapshot`` also sends ``anchor_id`` /
        ``anchors_jsonl`` (RoboCasa's cross-process anchors); those are not
        supported here and are refused explicitly rather than misread."""
        if _ignored.get("anchor_id") is not None:
            return {"ok": False, "error": "cross-process anchors are not supported on RoboTwin"}
        if which != "window" or not self._window_snapshot:
            return {"ok": False, "error": "no snapshot stored"}
        self._apply_snapshot(self._window_snapshot)
        return {"ok": True, "step_count": int(getattr(self, "_step_count", 0)),
                "tag": self._window_snapshot.get("tag")}

    def reset_window(self) -> dict[str, Any]:
        """Rewind to the start of THIS intervention and refill the budget."""
        self._require_env()
        if self._round_budget is None or self._round_budget <= 0:
            return {"ok": False, "error": "not inside an intervention window; nothing to rewind"}
        if self._resets_used >= MAX_RESETS:
            return {"ok": False, "error": f"no resets left this episode (used {MAX_RESETS})",
                    "resets_used": int(self._resets_used)}
        if not self._window_snapshot:
            return {"ok": False, "error": "the snapshot for this window was not captured; cannot rewind"}
        self._apply_snapshot(self._window_snapshot)
        # The window's chunks never happened: drop their _act_log entries too,
        # or the active-arm vote would still count them.
        n = self._window_snapshot.get("act_log_len")
        log = getattr(self, "_act_log", None)
        if isinstance(n, int) and isinstance(log, list) and len(log) > n:
            del log[n:]
        self._resets_used += 1
        self._resets_this_window += 1
        self._round_used = 0
        self._round_policy_chunks = 0
        self._repaired_since_handback = False   # the scene is back; _window_handbacks is kept on purpose
        self._failed_plan_sigs.extend((self._attempt_no, sig) for sig in self._attempt_plan_sigs)
        self._attempt_plan_sigs = []
        self._attempt_no += 1
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
            out: dict[str, Any] = {
                side: {"pose": [float(v) for v in self._ee_pose(side)],
                       "grip": float(self._grip_cmd(side))}
                for side in _SIDES
            }
            out["step_count"] = int(getattr(self, "_step_count", 0))
            return out
        except Exception:
            return None

    def _grip_for(self, side: str, gripper: Any, obs: Mapping[str, Any] | None = None) -> float:
        """A plan's ``gripper`` field -> normalised COMMAND (1 open, 0 shut).

        ``hold`` keeps the current COMMAND.  A NUMBER is RoboCasa's OSC
        convention (-1 open, +1 close, 0 hold) and is INVERTED into RoboTwin's
        channel, never clamped (clamping +1 would leave the fingers open when
        the model asked for a grasp).
        """
        if gripper is None or gripper == "hold":
            return self._grip_cmd(side)
        if isinstance(gripper, str):
            key = gripper.lower()
            if key in ("open", "release"):
                return 1.0
            if key in ("close", "grasp", "shut"):
                return 0.0
            raise InterventionError(f"gripper must be hold/open/close, got {gripper!r}")
        value = float(gripper)
        if value < 0.0:
            return 1.0
        if value > 0.0:
            return 0.0
        return self._grip_cmd(side)

    def move_to(self, xyz: Any, gripper: Any = "hold", arm: str | None = None,
                quat: Any = None) -> dict[str, Any]:
        """Servo the control point to a WORLD position (metres), wrist held or as given (wxyz).

        Up to ``MAX_TOTAL_M`` of travel runs as consecutive ``MAX_STEP_M``
        segments; only a longer request (to a point the arm has not visited)
        is clipped to ``MAX_TOTAL_M``."""
        self._require_cartesian()
        side = self._side(arm)
        target = np.asarray(xyz, dtype=np.float64).reshape(-1)
        if target.shape[0] != 3:
            raise InterventionError("xyz must be 3 numbers (world metres)")
        self._spend_action("move_to")
        cur = self._ee_pose(side)
        dist = float(np.linalg.norm(target - cur[:3]))
        clipped = False
        if dist > MAX_TOTAL_M and not self._is_visited(side, target):
            target = cur[:3] + (target - cur[:3]) * (MAX_TOTAL_M / dist)
            clipped = True
        _q = _quat_normalize(np.asarray(quat, dtype=np.float64)) if quat is not None else None
        out = (self._path_move(side, target, _q, self._grip_for(side, gripper)) if MOTION_MODE == "path"
               else self._segmented_servo(side, target, _q, self._grip_for(side, gripper)))
        out["clipped"] = clipped
        if clipped:
            out["asked_dist_m"] = round(dist, 4)
            out["clipped_to_m"] = round(MAX_TOTAL_M, 4)
        return out

    def _is_visited(self, side: str, target: np.ndarray) -> bool:
        """Was this point reached under the policy's own power this episode?
        Such a move is exempt from the single-step distance cap."""
        key = f"{side}_ee_pose"
        for entry in getattr(self, "_act_log", None) or []:
            pose = entry.get(key)
            if not pose:
                continue
            p = self._logged_pose(pose)
            if float(np.linalg.norm(p[:3] - target)) <= 0.08:
                return True
        held = ((self._takeover_pose or {}).get(side) or {}).get("pose")
        if held and float(np.linalg.norm(np.asarray(held[:3], dtype=np.float64) - target)) <= 0.08:
            return True
        return False

    def nudge(self, dxyz: Any, gripper: Any = "hold", arm: str | None = None,
              max_norm: float | None = MAX_TOTAL_M) -> dict[str, Any]:
        """Relative move of the control point (world metres), segmented like
        ``move_to``; clipped only beyond ``max_norm`` (default MAX_TOTAL_M)."""
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
        out = (self._path_move(side, cur[:3] + delta, None, self._grip_for(side, gripper)) if MOTION_MODE == "path"
               else self._segmented_servo(side, cur[:3] + delta, None, self._grip_for(side, gripper)))
        out["clipped"] = clipped
        if clipped:
            out["asked_dist_m"] = round(norm, 4)
            out["clipped_to_m"] = round(float(max_norm), 4)
        out["dxyz"] = [round(float(v), 4) for v in delta]
        return out

    def lift(self, dz: float, gripper: Any = "hold", arm: str | None = None) -> dict[str, Any]:
        return self.nudge([0.0, 0.0, float(dz)], gripper=gripper, arm=arm)

    def _both_spec(self, side: str, spec: Any, cur: np.ndarray) -> np.ndarray:
        """One arm's part of a move_both: ``{"xyz": [..]}`` (world) or
        ``{"dxyz": [..]}`` (relative) -> world target of the control point."""
        if not isinstance(spec, Mapping) or (("xyz" in spec) == ("dxyz" in spec)):
            raise InterventionError(
                f'move_both "{side}" must be an object with exactly one of "xyz" (world metres) '
                f'or "dxyz" (metres, relative), got {spec!r}')
        key = "xyz" if "xyz" in spec else "dxyz"
        try:
            vec = np.asarray(spec[key], dtype=np.float64).reshape(-1)
        except Exception:
            vec = np.zeros(0)
        if vec.shape[0] != 3 or not np.all(np.isfinite(vec)):
            raise InterventionError(f'move_both "{side}".{key} must be 3 numbers (metres)')
        return vec.copy() if key == "xyz" else cur[:3] + vec

    def move_both(self, left: Any = None, right: Any = None, gripper_left: Any = "hold",
                  gripper_right: Any = "hold") -> dict[str, Any]:
        """Servo BOTH hands together, in the same control steps.

        ``left`` / ``right``: ``{"xyz": [x, y, z]}`` or ``{"dxyz": [dx, dy, dz]}``
        (an omitted side holds still).  The same dxyz for both hands keeps
        their relative offset, so an object held in both is carried, not torn.
        Longer than ``MAX_STEP_M`` runs in segments; the longer hand's travel
        beyond ``MAX_TOTAL_M`` clips both hands by the same factor.  One
        model action; repair control steps do not spend the policy's step
        limit."""
        self._require_cartesian()
        if left is None and right is None:
            raise InterventionError('move_both needs "left" and/or "right"')
        cur = {s: self._ee_pose(s) for s in _SIDES}
        targets = {s: self._both_spec(s, spec, cur[s])
                   for s, spec in (("left", left), ("right", right)) if spec is not None}
        grips = {"left": self._grip_for("left", gripper_left),
                 "right": self._grip_for("right", gripper_right)}
        self._spend_action("move_both")
        deltas = {s: targets[s] - cur[s][:3] for s in targets}
        dmax = max(float(np.linalg.norm(d)) for d in deltas.values())
        clipped = dmax > MAX_TOTAL_M
        if clipped:
            k = MAX_TOTAL_M / dmax
            deltas = {s: d * k for s, d in deltas.items()}
        if MOTION_MODE == "path":
            out = self._path_move_both({s: cur[s][:3] + d for s, d in deltas.items()}, grips)
            final = {s: self._ee_pose(s) for s in _SIDES}
            off0 = cur["right"][:3] - cur["left"][:3]
            off1 = final["right"][:3] - final["left"][:3]
            reqs = {s: float(np.linalg.norm(d)) for s, d in deltas.items()}
            for s in out["per_arm"]:
                out["per_arm"][s]["requested_dist_m"] = round(reqs[s], 4)
            out.update({
                "requested_dist_m": round(max(reqs.values()), 4),
                "final_dist_m": round(max(v["final_dist_m"] for v in out["per_arm"].values()), 4),
                "travelled_m": round(max(v["travelled_m"] for v in out["per_arm"].values()), 4),
                "hand_offset_drift_m": round(float(np.linalg.norm(off1 - off0)), 4),
                "clipped": bool(clipped),
                "dxyz": {s: [round(float(v), 4) for v in d] for s, d in deltas.items()},
            })
            if clipped:
                out["asked_dist_m"] = round(dmax, 4)
                out["clipped_to_m"] = round(MAX_TOTAL_M, 4)
            return out
        n = self._n_segments(min(dmax, MAX_TOTAL_M))
        steps, done = 0, 0
        travelled = {s: 0.0 for s in deltas}
        out: dict[str, Any] = {}
        for i in range(1, n + 1):
            last = i == n
            wps = {s: cur[s][:3] + d * (i / n) for s, d in deltas.items()}
            out = self._servo_both(wps, grips, fine=last)
            steps += int(out["control_steps"])
            for s in travelled:
                travelled[s] += float(out["per_arm"][s]["travelled_m"])
            done = i
            if not out["ok"]:
                break
        final = {s: self._ee_pose(s) for s in _SIDES}
        for s, d in deltas.items():
            req = float(np.linalg.norm(d))
            out["per_arm"][s].update({
                "requested_dist_m": round(req, 4),
                "final_dist_m": round(float(np.linalg.norm(cur[s][:3] + d - final[s][:3])), 4),
                "travelled_m": round(travelled[s], 4),
            })
        reqs = [float(np.linalg.norm(d)) for d in deltas.values()]
        off0 = cur["right"][:3] - cur["left"][:3]
        off1 = final["right"][:3] - final["left"][:3]
        out.update({
            "requested_dist_m": round(max(reqs), 4),
            "final_dist_m": round(max(v["final_dist_m"] for v in out["per_arm"].values()), 4),
            "travelled_m": round(max(travelled.values()), 4),
            "control_steps": int(steps),
            "overshoot": bool(out.get("overshoot") or any(
                r > 1e-4 and travelled[s] > OVERSHOOT_RATIO * r for s, r in zip(deltas, reqs))),
            "hand_offset_drift_m": round(float(np.linalg.norm(off1 - off0)), 4),
            "segments": int(n),
            "segments_done": int(done),
            "clipped": bool(clipped),
            "dxyz": {s: [round(float(v), 4) for v in d] for s, d in deltas.items()},
        })
        if clipped:
            out["asked_dist_m"] = round(dmax, 4)
            out["clipped_to_m"] = round(MAX_TOTAL_M, 4)
        return out

    def lift_both(self, dz: float, gripper: Any = "hold") -> dict[str, Any]:
        """Both hands straight up (or down) by ``dz`` metres together -- the
        pot lifted by its two handles."""
        d = {"dxyz": [0.0, 0.0, float(dz)]}
        return self.move_both(left=d, right=dict(d), gripper_left=gripper, gripper_right=gripper)

    def rotate(self, drot: Any, gripper: Any = "hold", arm: str | None = None) -> dict[str, Any]:
        """Rotate the wrist by a WORLD-frame axis-angle delta, control point held."""
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
        out = (self._path_move(side, cur[:3], goal, self._grip_for(side, gripper)) if MOTION_MODE == "path"
               else self._servo_to(side, cur[:3], goal, self._grip_for(side, gripper)))
        out["clipped"] = clipped
        out["drot"] = [round(float(v), 4) for v in rv]
        return out

    def _drive_gripper(self, side: str, target: float) -> None:
        for _ in range(GRIPPER_STEPS):
            self._sim_action(self._hold_action({f"{side}_grip": target}))
            if self._episode_over():
                break

    def gripper(self, action: str = "close", arm: str | None = None) -> dict[str, Any]:
        """Open or close the fingers, arm held.  ``closed_empty`` = commanded
        shut and the MEASURED finger opening came back at the joint stop."""
        side = self._side(arm)
        target = self._grip_for(side, action)
        self._spend_action("gripper")
        self._drive_gripper(side, target)
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
        Scene objects are not touched: this is not a rewind."""
        self._require_cartesian()
        side = self._side(arm)
        if to not in ("takeover", "pre_grasp"):
            raise InterventionError(f"retreat 'to' must be takeover or pre_grasp, got {to!r}")
        take = (self._takeover_pose or {}).get(side)
        if not take:
            raise InterventionError("no takeover pose recorded for this intervention; retreat is unavailable")
        self._spend_action("retreat")
        pose = np.asarray(take["pose"], dtype=np.float64)
        was_open = float(take["grip"]) > 0.5
        if was_open:
            # Retreating means giving back whatever the repair picked up.
            self._drive_gripper(side, 1.0)
        rot_before = round(float(np.degrees(_quat_geodesic_rad(pose[3:], self._ee_pose(side)[3:]))), 1)
        out = (self._path_move(side, pose[:3], pose[3:], float(take["grip"])) if MOTION_MODE == "path"
               else self._servo_to(side, pose[:3], pose[3:], float(take["grip"]), max_steps=MOVE_TO_MAX_STEPS * 2))
        out.update({
            "to": "takeover",
            "requested_to": str(to),
            "note": ("pre_grasp is not recorded on RoboTwin; went to the takeover pose instead"
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
            "control_point": CONTROL_POINT,
        }

    def _camera_aliases(self) -> dict[str, str]:
        return dict(getattr(self, "CAMERA_ALIASES", None) or _DEFAULT_CAMERA_ALIASES)

    def _camera_object(self, cam_key: str):
        """A RoboTwin camera name (or alias) -> the sapien camera component."""
        aliases = self._camera_aliases()
        key = {v: k for k, v in aliases.items()}.get(cam_key, cam_key)
        cams = self._require_env().cameras
        if key == "left_camera" and getattr(cams, "left_camera", None) is not None:
            return key, cams.left_camera
        if key == "right_camera" and getattr(cams, "right_camera", None) is not None:
            return key, cams.right_camera
        names = list(getattr(cams, "static_camera_name", []) or [])
        if key in names:
            return key, cams.static_camera_list[names.index(key)]
        raise InterventionError(
            f"unknown camera {cam_key!r}; known: {sorted(set(names) | {'left_camera', 'right_camera'})} "
            f"or aliases {sorted(aliases.values())}"
        )

    def _frame_size(self, shot: Mapping[str, Any] | None = None) -> Any:
        size = getattr(self, "FRAME_SIZE", None)
        if size is not None:
            return int(size)
        for image in (shot or {}).values():
            return int(np.asarray(image).shape[0])
        return None

    def render(self) -> dict[str, Any]:
        """Camera views in the shape ``cli/rex.py render`` expects (pixels, not paths)."""
        obs = self._observation()
        shot = self._frames_from_obs(obs)
        return {
            "size": self._frame_size(shot),
            "rgb": {alias: np.asarray(image).tolist() for alias, image in shot.items()},
            "cameras": {alias: self._camera_meta(alias) for alias in shot},
            "main_camera": "primary",
            "depth": None,
        }

    def _camera_meta(self, alias: str) -> dict[str, Any]:
        try:
            _, cam = self._camera_object(alias)
            model = np.asarray(cam.get_model_matrix(), dtype=np.float64)   # cam->world, OpenGL axes
            return {
                "intrinsics": [[round(float(v), 4) for v in row]
                               for row in np.asarray(cam.get_intrinsic_matrix(), dtype=np.float64)],
                "pos_w": [round(float(v), 4) for v in model[:3, 3]],
                "cam2world_gl": [[round(float(v), 5) for v in row] for row in model],
                "size": [int(cam.get_height()), int(cam.get_width())],
            }
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def unproject(self, row: int, col: int, camera: str = "primary") -> dict[str, Any]:
        """Pixel -> world point, from the camera's rendered ``Position`` buffer.

        SAPIEN renders a per-pixel camera-space position (OpenGL axes) with
        every picture, so this needs no depth in the observation (demo_clean
        has ``depth: false``).  The picture is the one the last
        ``get_obs`` took, so the call refreshes the observation first.
        """
        self._observation()
        key, cam = self._camera_object(str(camera))
        pos = np.asarray(cam.get_picture("Position"), dtype=np.float64)   # H x W x 4
        v, u = int(row), int(col)          # rex speaks (row, col)
        h, w = pos.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            raise InterventionError(f"pixel (row={v}, col={u}) is outside the {h}x{w} image")
        p_cam = pos[v, u, :3]
        depth = float(-p_cam[2])
        if not np.isfinite(depth) or depth <= 0 or (pos.shape[2] > 3 and pos[v, u, 3] >= 1.0):
            raise InterventionError(f"no surface at pixel (row={v}, col={u}) (background)")
        model = np.asarray(cam.get_model_matrix(), dtype=np.float64)
        world = model[:3, :3] @ p_cam + model[:3, 3]
        return {
            "camera": str(camera),
            "size": [h, w],
            "row": int(row),
            "col": int(col),
            "depth": round(depth, 4),
            "world_xyz": [round(float(x), 4) for x in world],
        }

    def locate_region(self, bbox: Sequence[int], camera: str = "primary") -> dict[str, Any]:
        """Depth query over a picture REGION (2026-09-22, user; RPent's query_world_map): the box
        [row0, col0, row1, col1] on the FRAME_SIZE copy the judge sees, rescaled to the camera's
        native grid, every surface pixel back-projected from the rendered depth; returns the median
        world point, the box's 3-D extent and the highest surface point. Same perception a real
        RGB-D camera gives; no object identity or pose from the simulator."""
        try:
            r0, c0, r1, c1 = (int(v) for v in bbox)
        except Exception:
            raise InterventionError("bbox must be [row0, col0, row1, col1]")
        if r1 < r0: r0, r1 = r1, r0
        if c1 < c0: c0, c1 = c1, c0
        self._observation()
        key, cam = self._camera_object(str(camera))
        pos = np.asarray(cam.get_picture("Position"), dtype=np.float64)
        h, w = pos.shape[:2]
        frame = self._frame_size() or h
        sr, sc = h / float(frame), w / float(frame)
        R0, R1 = max(0, int(r0 * sr)), min(h, int(math.ceil((r1 + 1) * sr)))
        C0, C1 = max(0, int(c0 * sc)), min(w, int(math.ceil((c1 + 1) * sc)))
        patch = pos[R0:R1, C0:C1]
        if patch.size == 0:
            raise InterventionError(f"bbox {bbox} is outside the image")
        pts = patch[..., :3].reshape(-1, 3)
        valid = np.isfinite(pts).all(axis=1) & (-pts[:, 2] > 0)
        if patch.shape[-1] > 3:
            valid &= patch[..., 3].reshape(-1) < 1.0
        if not valid.any():
            raise InterventionError(f"no surface inside bbox {bbox} (background)")
        model = np.asarray(cam.get_model_matrix(), dtype=np.float64)
        world = pts[valid] @ model[:3, :3].T + model[:3, 3]
        med = np.median(world, axis=0)
        lo, hi = np.percentile(world, 10, axis=0), np.percentile(world, 90, axis=0)
        return {"camera": str(camera), "bbox": [r0, c0, r1, c1], "pixels": int(valid.sum()),
                "median_xyz": [round(float(v), 4) for v in med],
                "extent_m": [round(float(v), 4) for v in (hi - lo)],
                "top_z": round(float(np.percentile(world[:, 2], 95)), 4)}

    # -- the model's single entry point -----------------------------------

    def _resolve_plan_target(self, target: Any) -> tuple[list[float], dict[str, Any]]:
        """A plan's ``target`` -> world xyz.  Accepts ``{"xyz": [...]}``,
        a bare ``[x, y, z]`` and ``{"pixel": [row, col], "camera": ..., "dz": ...}``."""
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
        if "bbox" in target:
            q = self.locate_region(target["bbox"], camera=str(target.get("camera") or "primary"))
            xyz = list(q["median_xyz"])
            dz = float(target.get("dz") or 0.0)
            xyz[2] += dz
            return xyz, {"from": "bbox", **q, "dz": dz}
        if "pixel" not in target:
            raise InterventionError("target needs 'xyz', 'pixel' or 'bbox'")
        try:
            row, col = (int(v) for v in target["pixel"])
        except Exception:
            raise InterventionError("target.pixel must be [row, col]")
        camera = str(target.get("camera") or "primary")
        # The judge reads pixels off the FRAME_SIZE x FRAME_SIZE copies it is shown (256 x 256),
        # unproject needs the camera's native grid (D435: 320 x 240): rescale each axis.
        frame = self._frame_size()
        nrow, ncol = row, col
        if frame:
            try:
                _, cam = self._camera_object(camera)
                nh, nw = int(cam.get_height()), int(cam.get_width())
                nrow = int(round((row + 0.5) * nh / float(frame) - 0.5))
                ncol = int(round((col + 0.5) * nw / float(frame) - 0.5))
            except InterventionError:
                raise
            except Exception:
                pass
        hit = self.unproject(row=nrow, col=ncol, camera=camera)
        xyz = [float(v) for v in hit["world_xyz"]]
        dz = float(target.get("dz") or 0.0)
        xyz[2] += dz
        return xyz, {"from": "pixel", "pixel": [row, col], "native_pixel": [nrow, ncol],
                     "camera": camera, "surface_xyz": hit["world_xyz"], "dz": dz}

    def _plan_fix(self, plan: Sequence[Any]) -> tuple[list[dict[str, Any]], list[str]]:
        """Repair a plan locally instead of rejecting it (2026-09-21, user).

        rtF1 spent 357 of 1670 plan segments on rejections, nearly all of them rule violations the
        judge could not seem to avoid; every rejection burned a round and taught it nothing. What
        the rules protect is still enforced, but here, by rewriting the plan and telling the judge
        what was changed in the execution report:
          * after a Cartesian fault, Cartesian steps are dropped rather than the plan refused;
          * a plan identical to one that already failed is nudged 1 cm higher on its first
            positional step, so the retry at least differs;
          * a manual close with no lift after it gets the verifying lift appended;
          * steps beyond the remaining action budget are truncated instead of erroring out.
        Returns the new plan and the notes to show the judge.
        """
        steps = [dict(st) if isinstance(st, Mapping) else {} for st in plan]
        notes: list[str] = []

        if self._cartesian_blocked or self._window_faults > 0:
            keep = [st for st in steps if str(st.get("op")) in ("gripper", "policy")]
            if len(keep) != len(steps):
                notes.append("a step of this intervention already missed, overshot or was clipped, so the "
                             "Cartesian steps were dropped from your plan and only the gripper / policy "
                             "steps ran. Rewind if you need to move the arm again.")
                steps = keep or [{"op": "policy", "chunks": 1}]

        sig = self._plan_sig(steps)
        if any(str(st.get("op")) != "policy" for st in steps):
            for att, old in self._failed_plan_sigs:
                if old == sig:
                    for st in steps:
                        for key in ("dxyz", "xyz"):
                            v = st.get(key)
                            if isinstance(v, (list, tuple)) and len(v) == 3:
                                st[key] = [float(v[0]), float(v[1]), float(v[2]) + 0.01]
                                break
                        else:
                            if isinstance(st.get("target"), Mapping) and "dz" in st["target"]:
                                st["target"] = dict(st["target"], dz=float(st["target"]["dz"]) + 0.01)
                            elif st.get("dz") is not None:
                                st["dz"] = float(st["dz"]) + 0.01
                            else:
                                continue
                        break
                    notes.append(f"this plan repeated attempt {att}, which failed; it was raised by 1 cm so "
                                 "the retry is not identical. A centimetre is rarely the difference -- change "
                                 "the approach, the arm or who makes the close yourself.")
                    break

        # Grasp paradigm (user, rtG1 feedback): fingers WIDE open before the approach, then close.
        # A close with no open earlier in the plan while the hand is not already open gets an open
        # step put at the very start, so the approach is made with open fingers.
        opened = set()
        fixed: list[dict[str, Any]] = []
        for st in steps:
            op = str(st.get("op"))
            side = str(st.get("arm") or self._active_arm()).lower() if hasattr(self, "_active_arm") else "right"
            if op == "gripper" and str(st.get("state") or st.get("action") or "") in ("open", "release"):
                opened.add(side)
            if op == "gripper" and str(st.get("state") or st.get("action") or "") in ("close", "grasp", "shut") \
                    and side not in opened:
                try:
                    already_open = float(self._grip_norm(side)) >= 0.8
                except Exception:
                    already_open = True
                if not already_open:
                    fixed.insert(0, {"op": "gripper", "state": "open", "arm": side})
                    opened.add(side)
                    notes.append(f"the {side} fingers were not open, so an open step was put at the start of "
                                 "your plan: approach with the fingers wide open, then close.")
            fixed.append(st)
        steps = fixed

        out: list[dict[str, Any]] = []
        for i, st in enumerate(steps):
            out.append(st)
            op = str(st.get("op"))
            closes = (op == "gripper" and str(st.get("state") or st.get("action") or "") in ("close", "grasp", "shut")) or (
                op in self._BIMANUAL_OPS and any(self._is_close_value(st.get(k))
                                                 for k in ("gripper", "gripper_left", "gripper_right")))
            if closes and op != "lift_both" and not ({"lift", "lift_both"} & {str(x.get("op")) for x in steps[i + 1:]}):
                out.append({"op": "lift", "dz": 0.04, "gripper": "hold", **({"arm": st["arm"]} if st.get("arm") else {})})
                notes.append("a verifying lift was added after your close, so the hold is checked before "
                             "anything else happens.")
        steps = out

        if self._round_budget:
            left = max(0, int(self._round_budget) - int(self._round_used))
            cost = [st for st in steps if str(st.get("op")) not in ("policy", "locate")]
            if left and len(cost) > left:
                trimmed, spent = [], 0
                for st in steps:
                    if str(st.get("op")) != "policy":
                        if spent >= left:
                            continue
                        spent += 1
                    trimmed.append(st)
                notes.append(f"only {left} action(s) were left in this round, so the plan was cut to fit.")
                steps = trimmed or [{"op": "policy", "chunks": 1}]
        return steps, notes

    def _plan_gate(self, plan: Sequence[Any]) -> tuple[int, str] | None:
        """Reject a whole plan BEFORE any step runs (rules R1-R4, verbatim
        from the RoboDojo port so a judge that learned them there obeys them
        here).  ``(index, why)`` or None."""
        steps = [st if isinstance(st, dict) else {} for st in plan]
        ops = [str(st.get("op")) for st in steps]

        # R5 (no verbatim replay): after a rewind, a plan identical to one that already ran in a
        # failed attempt of this window reproduces the same failure (rtj12b: "reopen + hand back"
        # six times in a row). Positions compare at 5 mm, angles at 0.05 rad.
        sig = self._plan_sig(steps)
        if any(op != "policy" for op in ops):
            for att, old in self._failed_plan_sigs:
                if old == sig:
                    return 0, (
                        f"this plan is the same as one you already ran in attempt {att} of this "
                        "intervention, and that attempt failed and was rewound. Running it again "
                        "gives the same result. Change at least one real thing: the target position "
                        "(by 5 mm or more), the approach direction or height, the arm, the gripper "
                        "state, or grasp yourself instead of handing back."
                    )

        # R4
        if self._cartesian_blocked or self._window_faults > 0:
            for i, op in enumerate(ops):
                if op not in ("gripper", "policy"):
                    return i, (
                        "a step in this intervention already missed, overshot or was clipped; "
                        "from that arm configuration further Cartesian moves (move_to, nudge, "
                        "lift, rotate, retreat, move_both, lift_both) fail the same way and are "
                        "refused for the rest "
                        "of this intervention. Open the fingers if they are shut on nothing, "
                        "hand back one chunk, or end the intervention as fixed."
                    )
        # R1 and the adjacency rule were dropped on 2026-09-21 (user): a plan may start with a
        # policy step and may hand back twice in a row -- if the judge reads the scene that way,
        # that is its call. They cost 178 rejected plans in rtF1 and taught nothing.
        # R2 / R3.  A two-hand move that closes a hand on the way
        # (gripper / gripper_left / gripper_right) is a manual close as well.
        for i, st in enumerate(steps):
            if ops[i] == "gripper":
                state = str(st.get("state") or st.get("action") or "")
                if state not in ("close", "grasp", "shut"):
                    continue
            elif ops[i] in self._BIMANUAL_OPS:
                if not any(self._is_close_value(st.get(k))
                           for k in ("gripper", "gripper_left", "gripper_right")):
                    continue
            else:
                continue
            # Episode-level evidence also counts: once the policy has shut a hand on nothing twice in
            # this episode, a manual grasp is allowed from the first plan of any intervention
            # (rtj8a open_laptop: 3 empty closures across windows, every new intervention started
            # from zero, the judge never got to grasp itself).
            # 2026-09-21 (user): a manual close in the first repair is allowed too. The only thing
            # still enforced is that a close is verified by a lift, and _plan_fix adds that itself.
            if ops[i] != "lift_both" and not ({"lift", "lift_both"} & set(ops[i + 1:])):
                return i, (
                    "a manual close must be followed by a lift in the same plan so the grasp is "
                    'verified before anything else happens; add {"op": "lift", "dz": 0.04, '
                    '"gripper": "hold"} after it.'
                )
        return None

    @staticmethod
    def _plan_sig(steps: Sequence[Mapping[str, Any]]) -> tuple:
        """Comparable form of a plan: op, arm, and every numeric argument rounded
        (metres to 5 mm, radians to 0.05, pixels to 4 px)."""
        def rnd(v: Any, q: float) -> Any:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return round(round(float(v) / q) * q, 4)
            if isinstance(v, (list, tuple)):
                return tuple(rnd(x, q) for x in v)
            if isinstance(v, Mapping):
                return tuple(sorted((k, rnd(x, 4.0 if k == "pixel" else q)) for k, x in v.items()))
            return v
        out = []
        for st in steps:
            items = []
            for k, v in sorted(st.items()):
                q = 0.05 if k == "drot" else 0.005
                items.append((k, rnd(v, q) if k not in ("op", "arm", "state", "gripper", "action",
                                                         "gripper_left", "gripper_right", "to")
                              else str(v).lower()))
            out.append(tuple(items))
        return tuple(out)

    @staticmethod
    def _is_close_value(value: Any) -> bool:
        if isinstance(value, str):
            return value.lower() in ("close", "grasp", "shut")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value) > 0.0          # OSC convention: +1 = close
        return False

    def _plan_reply(self, *, ok: bool, log: list, stop_reason: str, stopped_at: Any,
                    remaining: list, frames: bool, rewritten: Sequence[str] = ()) -> dict[str, Any]:
        state = self._proprio_state()
        state.pop("recent_chunks", None)
        return {
            "ok": bool(ok),
            # what _plan_fix changed before running the plan, so the judge sees it in the report
            **({"rewritten": list(rewritten)} if rewritten else {}),
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
            "frames": self.render()["rgb"] if frames else {},
        }

    def execute_plan(self, plan: Any, max_steps: int = 8) -> dict[str, Any]:
        """Run a plan step by step, stopping at the first abort condition.

        Stop reasons: ``plan_complete``, ``budget``, ``closed_empty``,
        ``servo_missed``, ``move_clipped``, ``overshoot``, ``policy_cap``,
        ``episode_end``, ``error``.  No ``contact`` (no wrist F/T sensor).

        Ops: move_to, nudge, lift, rotate, gripper, retreat, policy (one arm,
        ``"arm"``), and the two-hand ops
        ``{"op": "move_both", "left": {"xyz"|"dxyz": [..]}, "right": {...},
        "gripper_left": ..., "gripper_right": ...}`` and
        ``{"op": "lift_both", "dz": 0.05, "gripper": "hold"}``.
        """
        self._require_env()
        if not isinstance(plan, (list, tuple)) or not plan:
            raise ValueError("plan must be a non-empty list of steps")
        if len(plan) > int(max_steps):
            raise ValueError(f"plan has {len(plan)} steps; at most {int(max_steps)} are allowed")

        plan, fix_notes = self._plan_fix(plan)
        bad = self._plan_gate(plan)
        if bad is not None:
            index, why = bad
            op = str(plan[index].get("op")) if isinstance(plan[index], dict) else "?"
            return self._plan_reply(ok=False, log=[{"step": index, "op": op, "error": why}],
                                    stop_reason="error", stopped_at=-1, remaining=list(plan),
                                    frames=False)
        if any(isinstance(st, dict) and str(st.get("op")) != "policy" for st in plan):
            self._attempt_plan_sigs.append(self._plan_sig([st if isinstance(st, dict) else {} for st in plan]))

        log: list[dict[str, Any]] = []
        stop_reason = "plan_complete"
        stopped_at: int | None = None
        for i, raw in enumerate(plan):
            if not isinstance(raw, dict) or "op" not in raw:
                stop_reason, stopped_at = "error", i
                log.append({"step": i, "error": "each step needs an 'op'"})
                break
            op = str(raw["op"])
            if op not in self._PLAN_OPS:
                stop_reason, stopped_at = "error", i
                log.append({"step": i, "op": op, "error": f"unknown op; allowed: {sorted(self._PLAN_OPS)}"})
                break
            if self._episode_over():
                # Success latched or the policy's step limit is spent: nothing
                # may run after the episode has ended.
                stop_reason, stopped_at = "episode_end", (i - 1 if i else -1)
                if i == 0:
                    log.append({"step": 0, "op": op, "error": "the episode has already ended"})
                break
            arm = raw.get("arm")
            entry: dict[str, Any] = {"step": i, "op": op}
            try:
                entry["arm"] = "both" if op in self._BIMANUAL_OPS else self._side(arm)
                grip = raw.get("gripper", "hold")
                if op == "move_both":
                    res = self.move_both(left=raw.get("left"), right=raw.get("right"),
                                         gripper_left=raw.get("gripper_left", grip),
                                         gripper_right=raw.get("gripper_right", grip))
                elif op == "lift_both":
                    if "dz" not in raw:
                        raise InterventionError('lift_both needs "dz" (metres, + is up)', reason="error")
                    entry["dz"] = round(float(raw["dz"]), 4)
                    res = self.lift_both(dz=float(raw["dz"]), gripper=grip)
                elif op == "move_to":
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
                elif op == "locate":
                    # perception only: no motion, no action budget, no sim step
                    if "bbox" in raw:
                        res = self.locate_region(raw["bbox"], camera=str(raw.get("camera") or "primary"))
                    else:
                        row, col = (int(v) for v in raw["pixel"])
                        xyz, how = self._resolve_plan_target({"pixel": [row, col], "camera": raw.get("camera") or "primary"})
                        res = {"median_xyz": [round(v, 4) for v in xyz], **how}
                    res["ok"] = True
                else:
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

            entry.update({k: v for k, v in res.items() if k != "ok"})
            entry["ok"] = bool(res.get("ok", True))
            log.append(entry)
            if op != "policy":
                self._repaired_since_handback = True

            if res.get("closed_empty"):
                stop_reason, stopped_at = "closed_empty", i
                break
            if res.get("servo_stop") == "episode_end":
                stop_reason, stopped_at = "episode_end", i
                break
            if res.get("overshoot"):
                stop_reason, stopped_at = "overshoot", i
                break
            if op in self._CARTESIAN_OPS and not res.get("ok", True):
                stop_reason, stopped_at = ("move_clipped" if res.get("clipped") else
                                           "move_residual" if res.get("motion") == "path" else "servo_missed"), i
                break
            if self._episode_over():
                stop_reason, stopped_at = "episode_end", i
                break
            if self._round_budget is not None and self._round_used >= self._round_budget:
                stop_reason, stopped_at = "budget", i
                break

        remaining = [] if stopped_at is None else list(plan[max(stopped_at, -1) + 1:])
        return self._plan_reply(ok=stop_reason == "plan_complete", log=log, stop_reason=stop_reason,
                                stopped_at=stopped_at, remaining=remaining, frames=True, rewritten=fix_notes)

    def _plan_policy_step(self, raw: Mapping[str, Any], entry: dict[str, Any]):
        """The ``policy`` op: hand back to the frozen policy, capped per step
        and per intervention."""
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
        self._window_handbacks += 1
        self._repaired_since_handback = False
        entry.update({
            "chunks": n,
            "policy_chunks_left": max(0, self._PLAN_POLICY_MAX_TOTAL - self._round_policy_chunks),
            "committed_timestep": adv.get("committed_timestep"),
            "policy_done": bool(adv.get("done")),
        })
        return {"ok": True}, None
