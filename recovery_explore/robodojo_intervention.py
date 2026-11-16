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
import json
import math
import os
from pathlib import Path
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
# 2026-10-20 (user rule): a move executes its whole route; no 5 cm clipping. RD_MAX_STEP_M=0.05 restores it.
MAX_STEP_M = float(os.environ.get("RD_MAX_STEP_M", "0.6"))
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
# measured finger opening; the ARX X5 fingers bottom out at 0.185 (probe 2026-10-20)
GRIP_EMPTY_NORM = float(os.environ.get("RD_GRIP_EMPTY", "0.20"))
#: Control steps spent opening or closing the fingers.
GRIPPER_STEPS = int(os.environ.get("RD_GRIPPER_STEPS", "4"))
GRIPPER_STEPS_MAX = int(os.environ.get("RD_GRIPPER_STEPS_MAX", "30"))
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



def _quat_slerp(q0, q1, t: float):
    """Spherical interpolation between wxyz quaternions (shortest arc)."""
    q0 = _quat_normalize(np.asarray(q0, dtype=np.float64)); q1 = _quat_normalize(np.asarray(q1, dtype=np.float64))
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        return _quat_normalize(q0 + t * (q1 - q0))
    th = np.arccos(min(1.0, d))
    return _quat_normalize((np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th))

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
    _PLAN_OPS = {"move_to", "nudge", "lift", "rotate", "gripper", "retreat", "policy", "grasp",
                 "place", "pick_place", "press", "push", "pi0_pick", "reach_check"}
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
        "harness_checkpoint", "harness_state_digest", "debug_depth_probe", "debug_gripper_geom", "debug_objects", "debug_obs_drift",
        "debug_estimate_truth", "debug_camera_check",
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
            "object_marks": self.object_marks,
            "camera_params": self.camera_params,
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

    # ---- depth (RGB-D cameras; needs RD_DEPTH=1 so the annotators exist) -------------------
    def _depth_view(self, cam: str):
        """One depth read: (depth HxW metres along the optical axis, K 3x3, T_cam2world 4x4).
        Switches ObsManager's depth collection on for this single get_obs only."""
        env = self._require_env()
        om = env.obs_manager; cm = env.camera_manager
        prev = om.collect_depth
        om.collect_depth = True
        try:
            obs = env.get_obs()
        finally:
            om.collect_depth = prev
        obs0 = obs[0] if isinstance(obs, (list, tuple)) else obs
        if isinstance(obs0, dict) and 0 in obs0: obs0 = obs0[0]
        vis = obs0["vision"][cam]
        if "depth" not in vis:
            raise InterventionError(f"no depth for {cam}: start the service with RD_DEPTH=1")
        depth = np.asarray(vis["depth"], dtype=np.float64)
        idx = list(cm.camera_names[0]).index(cam)
        K = np.asarray(cm.get_camera_intrinsics(idx, 0), dtype=np.float64)
        T = np.asarray(cm.get_camera_extrinsics(idx, 0), dtype=np.float64)
        return depth, K, T

    def debug_objects(self, cam: str = "cam_head") -> dict:
        """HARNESS ONLY (test-target picking): connected blobs above the dominant surface in one camera."""
        from recovery_explore import rd_grasp
        depth, K, T = self._depth_view(cam)
        pts, _ = rd_grasp.backproject(depth, K, T, stride=2)
        c = np.median(pts[:, :2], axis=0)
        tz = rd_grasp.support_plane(pts, c, radius=1.0)
        hands = self._hands_world()
        m = (pts[:, 2] > tz + rd_grasp.ABOVE) & (pts[:, 2] < tz + 0.4)
        for h in hands:
            m &= np.linalg.norm(pts - h, axis=1) > rd_grasp.HAND_R
        for a, b in self._arm_bones():
            m &= rd_grasp._seg_dist(pts, a, b) > rd_grasp.BONE_R
        above = pts[m]
        cells = np.floor(above[:, :2] / rd_grasp.CELL).astype(int)
        occ = {}
        for i, cc in enumerate(map(tuple, cells)):
            occ.setdefault(cc, []).append(i)
        seen, blobs = set(), []
        for start in occ:
            if start in seen:
                continue
            comp, stack = [], [start]; seen.add(start)
            while stack:
                cx, cy = stack.pop(); comp.append((cx, cy))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        n = (cx + dx, cy + dy)
                        if n in occ and n not in seen:
                            seen.add(n); stack.append(n)
            idx = [i for cc in comp for i in occ[cc]]
            if len(idx) >= 30:
                b = above[idx]
                pl = rd_grasp.plan_top_down(b, tz)
                blobs.append({"center": pl["center_xy"].round(3).tolist(), "top_z": round(pl["top_z"], 3),
                              "width": round(pl["width"], 3), "length": round(pl["length"], 3), "n": len(idx)})
        blobs.sort(key=lambda d: -d["n"])
        return {"support_z": round(tz, 3), "blobs": blobs[:12]}

    def object_marks(self, cam: str = "cam_head", max_marks: int = 12) -> dict:
        """Objects standing on the support surface, from this camera's depth: pixel of each top centre in the primary
        picture (0..PIXEL_FRAME, [row, col]) and its footprint / height. Hands and arm links are excluded."""
        from recovery_explore import rd_grasp
        depth, K, T = self._depth_view(cam)
        h, w = depth.shape[:2]
        frame = float(os.environ.get("PIXEL_FRAME", "1000"))
        pts, _ = rd_grasp.backproject(depth, K, T, stride=2)
        tz = rd_grasp.support_plane(pts, np.median(pts[:, :2], axis=0), radius=1.0)
        m = (pts[:, 2] > tz + rd_grasp.ABOVE) & (pts[:, 2] < tz + 0.4)
        hands = self._hands_world()
        for hp in hands:
            m &= np.linalg.norm(pts - hp, axis=1) > rd_grasp.HAND_R
        bones_all = self._arm_bones()
        for a, b in bones_all:
            m &= rd_grasp._seg_dist(pts, a, b) > rd_grasp.BONE_R
        above = pts[m]
        cells = np.floor(above[:, :2] / rd_grasp.CELL).astype(int)
        occ = {}
        for i, cc in enumerate(map(tuple, cells)):
            occ.setdefault(cc, []).append(i)
        seen, marks = set(), []
        for start in occ:
            if start in seen:
                continue
            comp, stack = [], [start]; seen.add(start)
            while stack:
                cx, cy = stack.pop(); comp.append((cx, cy))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        n = (cx + dx, cy + dy)
                        if n in occ and n not in seen:
                            seen.add(n); stack.append(n)
            idx = [i for cc in comp for i in occ[cc]]
            if len(idx) < 30:
                continue
            pl = rd_grasp.plan_top_down(above[idx], tz)
            c = np.array([float(pl["center_xy"][0]), float(pl["center_xy"][1]), float(pl["top_z"])])
            if c[1] < -0.43:                       # the arms' own base plates at the front edge
                continue
            if any(np.linalg.norm(c - np.asarray(hp)) < 0.12 for hp in hands):   # a gripper body the hand mask missed
                continue
            if c[2] - tz > 0.25:                   # rdv5: arm links / elbow housing the 6 cm bone mask missed
                continue
            bp = above[idx]
            if bones_all and np.mean(np.min([rd_grasp._seg_dist(bp, a, b) for a, b in bones_all], axis=0) < 0.10) >= 0.25:
                continue
            pc = (c - T[:3, 3]) @ T[:3, :3]
            x, y, z = pc[0], -pc[1], -pc[2]
            if z <= 0.05:
                continue
            col = x / z * K[0, 0] + K[0, 2]; row = y / z * K[1, 1] + K[1, 2]
            if not (0 <= row < h and 0 <= col < w):
                continue
            wd, ln = sorted([float(pl["width"]), float(pl["length"])])
            marks.append({"pixel": [int(round(row * frame / h)), int(round(col * frame / w))],
                          "xyz": c.round(3).tolist(), "narrow_cm": round(100 * wd, 1), "long_cm": round(100 * ln, 1),
                          "height_cm": round(100 * (c[2] - tz), 1), "n": len(idx), "depth_m": round(float(z), 3)})
        marks.sort(key=lambda d: -d["n"])
        marks = marks[:int(max_marks)]
        marks.sort(key=lambda d: (d["pixel"][0] // 150, d["pixel"][1]))   # reading order: back to front, left to right
        for i, mk in enumerate(marks, 1):
            mk["id"] = i
        return {"camera": "primary", "support_z": round(float(tz), 3), "marks": marks}

    _JUDGE_CAMS = (("primary", "cam_head"), ("secondary", "cam_left_wrist"), ("wrist", "cam_right_wrist"))

    def _judge_camera(self, cam: str, obs: Mapping[str, Any] | None = None) -> dict:
        """K in the judge's 0..PIXEL_FRAME [row, col] frame and camera->world R, t with OpenCV camera axes."""
        env = self._require_env(); cm = env.camera_manager
        frame = float(os.environ.get("PIXEL_FRAME", "1000"))
        idx = list(cm.camera_names[0]).index(cam)
        K = np.asarray(cm.get_camera_intrinsics(idx, 0), dtype=np.float64)
        T = np.asarray(cm.get_camera_extrinsics(idx, 0), dtype=np.float64)
        obs = obs if obs is not None else self._observation()
        h, w = np.asarray(obs["vision"][cam]["color"]).shape[:2]
        sx, sy = frame / w, frame / h
        Kj = np.array([[K[0, 0] * sx, 0.0, K[0, 2] * sx], [0.0, K[1, 1] * sy, K[1, 2] * sy], [0.0, 0.0, 1.0]])
        R = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])     # USD camera axes (y up, z back) -> OpenCV (y down, z forward)
        return {"K": Kj, "R": R, "t": T[:3, 3].copy(), "native_wh": [int(w), int(h)]}

    def camera_params(self) -> dict:
        """RD_CAM_PARAMS: what the judge needs to turn a [row, col] + depth into a world point, per camera."""
        obs = self._observation()
        out = {"pixel_frame": float(os.environ.get("PIXEL_FRAME", "1000")), "cameras": {}}
        for alias, cam in self._JUDGE_CAMS:
            try:
                c = self._judge_camera(cam, obs)
            except Exception as exc:
                out["cameras"][alias] = {"error": f"{type(exc).__name__}: {exc}"}; continue
            out["cameras"][alias] = {"K": c["K"].round(2).tolist(), "R": c["R"].round(4).tolist(),
                                     "t": c["t"].round(4).tolist(), "native_wh": c["native_wh"]}
        out["hands"] = {sd: self._ee_pose(sd, obs)[:3].round(3).tolist() for sd in ("left", "right")}
        return out

    def _object_truth(self) -> list:
        return [{"name": k, "pos": [round(float(v), 4) for v in np.asarray(e["pos"]).reshape(-1)[:3]]}
                for k, e in self._capture_objects().items() if "pos" in e]

    def debug_estimate_truth(self, estimates: list) -> dict:
        """ANALYSIS ONLY (never sent to the judge): ground truth for the judge's own position estimates."""
        from recovery_explore import rd_grasp
        frame = float(os.environ.get("PIXEL_FRAME", "1000"))
        objs = self._object_truth()
        views, rows = {}, []
        for est in estimates or []:
            row = {"est": est}
            try:
                alias = str(est.get("camera") or "primary")
                cam = dict(self._JUDGE_CAMS).get(alias, "cam_head")
                if cam not in views:
                    views[cam] = self._depth_view(cam)
                depth, K, T = views[cam]
                h, w = depth.shape[:2]
                r, cc = float(est["pixel"][0]) * h / frame, float(est["pixel"][1]) * w / frame
                p = rd_grasp.pixel_to_world(depth, K, T, r, cc)
                rr, ci = int(np.clip(round(r), 0, h - 1)), int(np.clip(round(cc), 0, w - 1))
                row["depth_gt"] = round(float(depth[rr, ci]), 4)
                row["pixel_world_gt"] = None if p is None else [round(float(v), 4) for v in np.asarray(p).reshape(-1)[:3]]
                ref = np.asarray(est.get("xyz") if est.get("xyz") is not None else (p if p is not None else [0, 0, 0]),
                                 dtype=np.float64)
                near = min(objs, key=lambda o: float(np.linalg.norm(np.asarray(o["pos"]) - ref))) if objs else None
                row["nearest_object_gt"] = near
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        return {"rows": rows, "objects_gt": objs, "camera_params": self.camera_params()}

    def debug_camera_check(self, out_dir: str) -> dict:
        """ANALYSIS ONLY: project every task object's simulator position into each judge camera and draw it."""
        from PIL import Image, ImageDraw
        obs = self._observation(); frame = float(os.environ.get("PIXEL_FRAME", "1000"))
        objs = self._object_truth(); res = {}
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for alias, cam in self._JUDGE_CAMS:
            c = self._judge_camera(cam, obs)
            img = Image.fromarray(np.asarray(obs["vision"][cam]["color"], dtype=np.uint8)[..., :3]).convert("RGB")
            w, h = img.size; dr = ImageDraw.Draw(img); pts = []
            for o in objs + [{"name": f"hand_{sd}", "pos": self._ee_pose(sd, obs)[:3].tolist()} for sd in ("left", "right")]:
                pc = c["R"].T @ (np.asarray(o["pos"], dtype=np.float64) - c["t"])
                if pc[2] <= 0.02:
                    continue
                uvw = c["K"] @ pc
                rowp, colp = uvw[1] / uvw[2], uvw[0] / uvw[2]          # judge frame, 0..PIXEL_FRAME
                y, x = rowp * h / frame, colp * w / frame
                pts.append({"name": o["name"], "pixel": [round(rowp), round(colp)], "depth": round(float(pc[2]), 3)})
                dr.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(255, 0, 0), width=2)
                dr.text((x + 7, y - 7), o["name"].split("/")[-1][:18], fill=(255, 0, 0))
            img.save(Path(out_dir) / f"camcheck_{alias}.png"); res[alias] = pts
        return {"projected": res, "out_dir": out_dir}

    def debug_ik_probe(self, side: str, points: list, approaches: list) -> dict:
        """HARNESS ONLY: IK success for each (point, approach) -- approach = [ax, ay, az] the EE +x direction; the
        fingers close along the horizontal perpendicular. No motion."""
        def mat2quat(Rm):
            tr = Rm[0, 0] + Rm[1, 1] + Rm[2, 2]
            if tr > 0:
                s_ = np.sqrt(tr + 1.0) * 2; q = [0.25 * s_, (Rm[2, 1] - Rm[1, 2]) / s_, (Rm[0, 2] - Rm[2, 0]) / s_, (Rm[1, 0] - Rm[0, 1]) / s_]
            elif Rm[0, 0] > Rm[1, 1] and Rm[0, 0] > Rm[2, 2]:
                s_ = np.sqrt(1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2; q = [(Rm[2, 1] - Rm[1, 2]) / s_, 0.25 * s_, (Rm[0, 1] + Rm[1, 0]) / s_, (Rm[0, 2] + Rm[2, 0]) / s_]
            elif Rm[1, 1] > Rm[2, 2]:
                s_ = np.sqrt(1.0 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2; q = [(Rm[0, 2] - Rm[2, 0]) / s_, (Rm[0, 1] + Rm[1, 0]) / s_, 0.25 * s_, (Rm[1, 2] + Rm[2, 1]) / s_]
            else:
                s_ = np.sqrt(1.0 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2; q = [(Rm[1, 0] - Rm[0, 1]) / s_, (Rm[0, 2] + Rm[2, 0]) / s_, (Rm[1, 2] + Rm[2, 1]) / s_, 0.25 * s_]
            q = np.asarray(q); return q / np.linalg.norm(q)
        out = []
        for pt in points:
            row = []
            for a in approaches:
                x = np.asarray(a, dtype=np.float64); x /= np.linalg.norm(x)
                y = np.cross([0.0, 0.0, 1.0], x)
                if np.linalg.norm(y) < 1e-6:
                    y = np.array([1.0, 0.0, 0.0])
                y /= np.linalg.norm(y); z = np.cross(x, y)
                q = mat2quat(np.stack([x, y, z], axis=1))
                ok = False
                for yaw_flip in (False, True):
                    qq = q if not yaw_flip else mat2quat(np.stack([x, -y, -z], axis=1))
                    if self._solve_ik(side, np.concatenate([np.asarray(pt, dtype=np.float64), qq])) is not None:
                        ok = True; break
                row.append(ok)
            out.append(row)
        return {"side": side, "ok": out}

    def debug_obs_drift(self, n: int = 50) -> dict:
        """HARNESS ONLY: physics time / scene state before and after n observation reads (no actions)."""
        env = self._require_env()
        simctx = env.sim.sim

        def snap():
            t = None
            for attr in ("current_time", "get_physics_dt"):
                pass
            try:
                t = float(simctx.current_time)
            except Exception:
                try:
                    t = float(simctx.get_simulation_time())
                except Exception:
                    t = None
            objs = {}
            try:
                for name, obj, _a in list(self._scene_objects())[:6]:
                    pos = obj.data.root_pos_w[0].detach().cpu().numpy()
                    objs[name] = [round(float(v), 5) for v in pos]
            except Exception as exc:
                objs = {"error": repr(exc)}
            ee = {s_: [round(float(v), 5) for v in self._ee_pose(s_)[:3]] for s_ in ("left", "right")}
            return {"sim_time": t, "objects": objs, "ee": ee, "take_action_cnt": int(env.take_action_cnt[0])}
        before = snap()
        for _ in range(int(n)):
            env.get_obs()
        after = snap()
        return {"n": int(n), "before": before, "after": after}

    def debug_gripper_geom(self) -> dict:
        """HARNESS ONLY (offline calibration of robot geometry, i.e. what the URDF says): every body of each
        arm expressed in that arm's end-effector frame."""
        env = self._require_env(); rm = env.robot_manager
        out = {}
        for robot, key in zip(rm.robot_list, rm.robot_key):
            side = rm.process_name(robot.arm_name).split("_")[0]
            names = list(key.body_names)
            pose = key.data.body_link_pose_w[0].detach().cpu().numpy()
            ee = self._ee_pose(side)
            Rw = _quat_to_matrix(ee[3:])
            rows = {}
            for i, n in enumerate(names):
                p_ee = Rw.T @ (pose[i, :3] - ee[:3])
                rows[n] = [round(float(v), 4) for v in p_ee]
            out[side] = {"ee": [round(float(v), 4) for v in ee], "bodies_in_ee_frame": rows}
        return out

    def debug_depth_probe(self, cam: str = "cam_head") -> dict:
        """HARNESS ONLY (validation): depth stats and world points of a pixel grid under two camera conventions."""
        depth, K, T = self._depth_view(cam)
        h, w = depth.shape[:2]
        pts = {}
        for conv in ("opencv", "usd"):
            out = []
            for r in (h // 4, h // 2, 3 * h // 4):
                for c in (w // 4, w // 2, 3 * w // 4):
                    z = float(depth[r, c])
                    x = (c - K[0, 2]) / K[0, 0] * z; y = (r - K[1, 2]) / K[1, 1] * z
                    pc = np.array([x, y, z]) if conv == "opencv" else np.array([x, -y, -z])
                    pw = T[:3, :3] @ pc + T[:3, 3]
                    out.append([r, c, round(z, 3)] + [round(float(v), 3) for v in pw])
            pts[conv] = out
        finite = depth[np.isfinite(depth)]
        return {"shape": [h, w], "depth_min": float(finite.min()), "depth_med": float(np.median(finite)),
                "depth_max": float(finite.max()), "K": K.round(2).tolist(), "T": T.round(3).tolist(),
                "points_r_c_z_xyz": pts, "left_ee": self._ee_pose("left")[:3].round(3).tolist(),
                "right_ee": self._ee_pose("right")[:3].round(3).tolist()}

    def _grip_open_measured(self, side: str) -> float:
        m = self._grip_measured().get(side)
        return float(m) if m is not None else self._grip_norm(side)

    def _grip_norm(self, side: str, obs: Mapping[str, Any] | None = None) -> float:
        # The command to keep: our own last gripper command on this side if we issued one in
        # this episode (the echo is rate-limited and would freeze a close half way), else the echo.
        cmd = getattr(self, "_grip_cmd_issued", {}).get(side)
        if cmd is not None:
            return float(cmd)
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
        if os.environ.get("RD_FEED_POLICY_HISTORY", "0") == "1":
            # The official loop pushes one observation per executed action; doing the same for judge steps keeps
            # VPP2's stride-25 history window a continuous record of the scene when control is handed back.
            try:
                self._policy().observe(self._observation())
            except Exception:
                pass

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
        # 2026-10-20 (user rule): a missed move stops and reports but never locks later moves.
        if self._cartesian_blocked and os.environ.get("ROBODOJO_PLAN_GATES", "0") == "1":
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
        last_rot = _quat_geodesic_rad(goal_quat, start[3:])   # rdv5: wrist turning is progress too
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
            max_rot = np.radians(float(os.environ.get("RD_ROT_STEP_DEG", "20")))
            frac = 1.0 if rot_err <= max_rot else max_rot / rot_err
            joints = None
            for _shrink in (1.0, 0.5, 0.25):
                wq = _quat_slerp(cur[3:], goal_quat, min(1.0, frac * _shrink))
                waypoint = np.concatenate([cur[:3] + step * _shrink, wq])
                joints = self._solve_ik(side, waypoint)
                if joints is not None:
                    break
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
            new_rot = _quat_geodesic_rad(goal_quat, after[3:])
            if new_dist > last_dist - 1e-4 and new_rot > last_rot - np.radians(1.0):
                stalled += 1
                if stalled >= 3:
                    reason = "stalled"
                    break
            else:
                stalled = 0
            last_dist = new_dist
            last_rot = new_rot
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
        if not hasattr(self, "_grip_cmd_issued"):
            self._grip_cmd_issued = {}
        self._grip_cmd_issued[side] = float(target)
        # Step until the measured fingers stop moving (or GRIPPER_STEPS_MAX): the controller only
        # moves the command 20% of the stroke per step relative to the real finger position.
        prev = None
        _gv = os.environ.get("RD_GRASP_VERIFY", "0") == "1"; _stable = 0
        for k in range(GRIPPER_STEPS_MAX):
            self._sim_action(self._joint_action(
                left_arm=left, left_grip=grips["left"],
                right_arm=right, right_grip=grips["right"],
            ))
            if bool(self._require_env().end_flag[0]):
                break
            cur = self._grip_open_measured(side)
            if k + 1 >= GRIPPER_STEPS and prev is not None and abs(cur - prev) < (0.003 if _gv else 0.005):
                _stable += 1
                if not _gv or _stable >= 2:   # rdv6: two stable readings, so a mid-closure value is never read
                    break
            else:
                _stable = 0
            prev = cur
        width = self._grip_open_measured(side)
        closing = target <= 0.5
        return {
            "ok": True,
            "arm": side,
            "action": str(action),
            "commanded": round(float(target), 3),
            "opening_norm": round(float(width), 3),
            "closed_empty": bool(closing and width <= GRIP_EMPTY_NORM),
        }

    def _gripper_verified(self, action: str = "close", arm: str | None = None) -> dict[str, Any]:
        """rdv6 (Spotter issue #13): a manual close used to report only the settled opening. With RD_GRASP_VERIFY=1 a
        close that did not shut empty is followed by a small lift and a second reading. Reporting only: the plan
        goes on whatever the outcome; a partial close is logged."""
        res = self.gripper(action=action, arm=arm)
        if os.environ.get("RD_GRASP_VERIFY", "0") != "1" or float(res.get("commanded", 1.0)) > 0.5:
            return res
        side = res.get("arm") or self._side(arm)
        w0 = float(res.get("opening_norm") or 0.0)
        ver = {"opening_settled": round(w0, 3)}
        if res.get("closed_empty"):
            ver["outcome"] = "closed_empty"
        else:
            dz = float(os.environ.get("RD_GRASP_VERIFY_LIFT", "0.03"))
            try:
                pose = self._ee_pose(side)
                r = self._servo_to(side, pose[:3] + np.array([0.0, 0.0, dz]), pose[3:], float(res.get("commanded", 0.0)))
                self._cartesian_blocked = False
                ver["lift_m"] = dz; ver["lift_stop"] = r.get("servo_stop")
            except InterventionError:
                raise
            except Exception as exc:
                ver["lift_error"] = f"{type(exc).__name__}: {exc}"
            w1 = float(self._grip_open_measured(side))
            ver["opening_after_lift"] = round(w1, 3)
            if w1 <= GRIP_EMPTY_NORM:
                ver["outcome"] = "slipped"
            elif w1 < GRIP_EMPTY_NORM + 0.06:
                ver["outcome"] = "partial"
            else:
                ver["outcome"] = "held_by_width"
            ver["note"] = ("the opening alone does not prove the object is held: check in the next picture that it "
                           "left its support")
            if ver["outcome"] == "partial":
                try:
                    with open(os.path.join(str(self._episode_dir()), "partial_closes.jsonl"), "a") as fh:
                        fh.write(json.dumps({"arm": side, **ver}) + "\n")
                except Exception:
                    pass
        res["verify"] = ver
        return res

    def retreat(self, to: str = "takeover", arm: str | None = None, release: bool = False) -> dict[str, Any]:
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
        kept = False
        if was_open and os.environ.get("RD_HONEST", "0") == "1" and not release:
            # rdv7: the old retreat opened a hand that had been open at takeover -- dropping whatever the repair had
            # just picked up (pi0_pick -> retreat lost the object). Keep a hand that is holding something closed.
            w_now = float(self._grip_open_measured(side))
            cmd = (getattr(self, "_grip_cmd_issued", {}) or {}).get(side)
            if GRIP_EMPTY_NORM + 0.02 < w_now < 0.85 and (cmd is None or float(cmd) <= 0.5):
                was_open = False; kept = True
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
            side, pose[:3], pose[3:], (0.0 if kept else float(take["grip"])),
            max_steps=MOVE_TO_MAX_STEPS * 2,
        )
        if kept:
            out["kept_holding"] = True
        out.update({
            "to": "takeover",
            "requested_to": str(to),
            "note": ("pre_grasp is not recorded on RoboDojo; went to the takeover pose instead"
                     if to == "pre_grasp" else None),
            "fingers": "kept closed on the held object" if kept else ("open" if was_open else "close"),
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

    def _cam_key(self, camera: str) -> str:
        try:
            from recovery_explore.env_service_robodojo import CAMERA_ALIASES
        except ImportError:
            CAMERA_ALIASES = {"cam_head": "primary", "cam_left_wrist": "secondary", "cam_right_wrist": "wrist"}
        return {v: k for k, v in CAMERA_ALIASES.items()}.get(str(camera), str(camera))

    def unproject(self, row: float, col: float, camera: str = "primary") -> dict[str, Any]:
        """Pixel -> world point from the camera's depth (RGB-D). row/col are 0..PIXEL_FRAME within that one
        camera image (Qwen points in 0-1000), converted to the native resolution here."""
        from recovery_explore import rd_grasp
        cam_key = self._cam_key(camera)
        depth, K, T = self._depth_view(cam_key)
        h, w = depth.shape[:2]
        frame = float(os.environ.get("PIXEL_FRAME", "1000"))
        r = min(h - 1, max(0.0, float(row) * h / frame)); c = min(w - 1, max(0.0, float(col) * w / frame))
        pw = rd_grasp.pixel_to_world(depth, K, T, r, c)
        if pw is None:
            raise InterventionError(f"no depth near pixel ({row},{col}) of {camera}")
        return {"camera": str(camera), "row": float(row), "col": float(col), "frame": frame,
                "world_xyz": [round(float(x), 4) for x in pw]}

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

    def _undo_if_unreachable(self, side: str, start, res: dict) -> dict:
        """No-harm rule for the tools: an op that could not reach its goal puts the arm back to its starting pose
        (position + wrist) so the policy resumes from the state it was in."""
        if res.get("outcome") == "unreachable" and start is not None:
            r = self._servo_to(side, np.asarray(start[:3]), np.asarray(start[3:]), self._grip_norm(side))
            self._cartesian_blocked = False
            res["restored_start_pose"] = r.get("servo_stop")
        return res

    # -- grasp: depth segmentation + top-down grasp (ARX X5) ----------------

    _TIP_X = float(os.environ.get("RD_TIP_X", "0.158"))      # link6 -> fingertip along the approach axis (probe: contact at table+0.158)
    _PAD_X = float(os.environ.get("RD_PAD_X", "0.140"))      # link6 -> middle of the finger pads
    _MAX_OPEN = float(os.environ.get("RD_MAX_OPEN", "0.080"))  # widest object the fingers close across (pads open 0.088)

    def _arm_bones(self) -> list:
        """Consecutive link positions of both arms as segments (robot self-geometry from its joint state)."""
        env = self._require_env(); rm = env.robot_manager
        bones = []
        for robot, key in zip(rm.robot_list, rm.robot_key):
            names = list(key.body_names)
            pose = key.data.body_link_pose_w[0].detach().cpu().numpy()
            chain = [n for n in ("base_link", "link1", "link2", "link3", "link4", "link5", "link6") if n in names]
            pts = [pose[names.index(n), :3].astype(np.float64) for n in chain]
            for a, b in zip(pts[:-1], pts[1:]):
                bones.append((a, b))
            for f in ("link7", "link8"):
                if f in names and "link6" in names:
                    bones.append((pose[names.index("link6"), :3].astype(np.float64), pose[names.index(f), :3].astype(np.float64)))
        for side in ("left", "right"):          # rdv5: the gripper body + fingers along the approach axis
            try:
                ee = self._ee_pose(side); ax = _quat_to_matrix(ee[3:])[:, 0]
                bones.append((ee[:3] - 0.05 * ax, ee[:3] + 0.12 * ax))
            except Exception:
                pass
        return bones

    _ARM_BASE = {"left": np.array([-0.3, -0.45]), "right": np.array([0.3, -0.45])}   # dual_x5.yml default_root_pos
    _TILTS_DEG = tuple(float(t) for t in os.environ.get("RD_TILTS", "0,30,45,60").split(","))   # outward leans
    _REACH = float(os.environ.get("RD_REACH", "0.58"))

    def _tilted(self, q_down, side: str, target_xy, tilt_deg: float):
        """Rotate a fingers-down wrist so the approach axis leans back toward the arm's base by tilt_deg
        (rotation about the horizontal axis perpendicular to base->target)."""
        if tilt_deg == 0:
            return np.asarray(q_down, dtype=np.float64)
        d = np.asarray(target_xy, dtype=np.float64)[:2] - self._ARM_BASE[side]
        n = np.linalg.norm(d)
        d = d / n if n > 1e-6 else np.array([0.0, 1.0])
        axis = np.array([-d[1], d[0], 0.0])           # horizontal, perpendicular to base->target
        # approach starts as -z; leaning it back toward the base means rotating it toward -d
        a = np.radians(tilt_deg) / 2.0
        q_axis = np.array([np.cos(a), *(np.sin(a) * axis)])
        q1 = _quat_normalize(_quat_mul(q_axis, np.asarray(q_down, dtype=np.float64)))
        app = _quat_to_matrix(q1)[:, 0]
        if float(app[:2] @ d) < 0:                    # must lean OUTWARD (away from the base): flip otherwise
            q_axis = np.array([np.cos(-a), *(np.sin(-a) * axis)])
            q1 = _quat_normalize(_quat_mul(q_axis, np.asarray(q_down, dtype=np.float64)))
        return q1

    def _ee_for_point(self, point, quat, offset: float):
        """Wrist position that puts the point `offset` metres along the approach axis (EE +x) from link6."""
        return np.asarray(point, dtype=np.float64) - _quat_to_matrix(np.asarray(quat))[:, 0] * float(offset)

    def _restage_for_handback(self) -> dict:
        """Empty hands back to their episode-start pose, fingers open, before the policy resumes."""
        home = getattr(self, "_home_pose", None) or {}
        out = {}
        for side in ("left", "right"):
            if side not in home:
                continue
            if self._grip_open_measured(side) > GRIP_EMPTY_NORM and self._grip_open_measured(side) < 0.9:
                out[side] = "holding: left in place"
                continue
            pose = np.asarray(home[side], dtype=np.float64)
            r = self._servo_to(side, pose[:3], pose[3:], 1.0)
            self._cartesian_blocked = False
            out[side] = r.get("servo_stop")
        return out

    def _hands_world(self) -> list:
        out = []
        for side in ("left", "right"):
            ee = self._ee_pose(side); Rw = _quat_to_matrix(ee[3:])
            for x in (0.0, 0.06, 0.12):
                out.append(ee[:3] + Rw @ np.array([x, 0.0, 0.0]))
        return out

    # -- rdv7: evidence for honest outcomes (depth cameras + proprioception only, no sim state) -------
    def _scene_cloud(self, side: str):
        from recovery_explore import rd_grasp
        pts = []
        for cam in ("cam_head", f"cam_{side}_wrist"):
            try:
                d, K, T = self._depth_view(cam)
                pts.append(rd_grasp.backproject(d, K, T, stride=2)[0])
            except Exception:
                pass
        return np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))

    def _band_count(self, side: str, xy, z_lo: float, z_hi: float, radius: float = 0.03) -> int:
        """Depth points in a thin horizontal band around xy, off the arm links (-1 when no depth). The band sits
        just above a support, below where a lifted hand's fingers are, so it reads 'is something still standing
        here' without counting the hand."""
        P = self._scene_cloud(side)
        if not len(P):
            return -1
        m = (np.linalg.norm(P[:, :2] - np.asarray(xy[:2], dtype=np.float64), axis=1) < radius) & (P[:, 2] > z_lo) & (P[:, 2] < z_hi)
        P = P[m]
        try:
            for a, b in self._arm_bones():
                if not len(P):
                    break
                ab = b - a; L2 = float(ab @ ab) or 1e-9
                t = np.clip(((P - a) @ ab) / L2, 0.0, 1.0)
                P = P[np.linalg.norm(P - (a + t[:, None] * ab), axis=1) > 0.035]
        except Exception:
            pass
        return int(len(P))

    def _cam_KT(self, cam_key: str):
        env = self._require_env(); cm = env.camera_manager
        idx = list(cm.camera_names[0]).index(cam_key)
        return (np.asarray(cm.get_camera_intrinsics(idx, 0), dtype=np.float64),
                np.asarray(cm.get_camera_extrinsics(idx, 0), dtype=np.float64))

    def _triangulate_target(self, target) -> tuple[list[float], dict[str, Any]]:
        """rdv7 (RD_TRIANGULATE): the same physical point marked in two or three calibrated cameras -> 3-D point
        (least squares over the rays). Rejected when the rays are nearly parallel (< 1 deg) or a reprojection is
        off by more than 3 native pixels -- then the marks are probably not the same point."""
        pix = target.get("pixels") or []; cams = target.get("cameras") or []
        if len(pix) < 2 or len(pix) != len(cams):
            raise InterventionError("triangulation needs 'pixels': [[row, col], ...] and 'cameras': [...] of the same length (2 or 3)")
        frame = float(os.environ.get("PIXEL_FRAME", "1000")); size = float(_frame_size())
        origins, dirs, natives = [], [], []
        for (row, col), cam in zip(pix, cams):
            K, T = self._cam_KT(self._cam_key(str(cam)))
            r = float(row) * size / frame; c = float(col) * size / frame
            v = np.array([(c - K[0, 2]) / K[0, 0], -(r - K[1, 2]) / K[1, 1], -1.0])
            d = T[:3, :3] @ (v / np.linalg.norm(v))
            origins.append(T[:3, 3]); dirs.append(d / np.linalg.norm(d)); natives.append((r, c, K, T))
        ang = max(float(np.degrees(np.arccos(np.clip(abs(dirs[i] @ dirs[j]), -1, 1))))
                  for i in range(len(dirs)) for j in range(i + 1, len(dirs)))
        if ang < 1.0:
            raise InterventionError(f"triangulation rejected: the camera rays are nearly parallel ({ang:.2f} deg)")
        A = np.zeros((3, 3)); b = np.zeros(3)
        for o, d in zip(origins, dirs):
            M = np.eye(3) - np.outer(d, d); A += M; b += M @ o
        p = np.linalg.solve(A, b)
        errs = []
        for r, c, K, T in natives:
            pc = T[:3, :3].T @ (p - T[:3, 3]); z = -pc[2]
            if z <= 0:
                raise InterventionError("triangulation rejected: the point lies behind a camera")
            cc = K[0, 0] * pc[0] / z + K[0, 2]; rr = -K[1, 1] * pc[1] / z + K[1, 2]
            errs.append(float(np.hypot(rr - r, cc - c)))
        if max(errs) > 3.0:
            raise InterventionError("triangulation rejected: reprojection error %.1f px (> 3): the marks are probably "
                                    "not the same physical point" % max(errs))
        xyz = [float(v) for v in p]; dz = float(target.get("dz") or 0.0); xyz[2] += dz
        return xyz, {"from": "triangulated", "cameras": [str(c) for c in cams], "ray_angle_deg": round(ang, 1),
                     "reproj_px": [round(e, 1) for e in errs], "dz": dz}

    def grasp(self, target: Any = None, arm: str | None = None) -> dict[str, Any]:
        side = self._side(arm) if arm else (getattr(self, '_held', None) or {}).get('side') or self._side(None)
        start = self._ee_pose(side)
        res = self._grasp_impl(target=target, arm=arm)
        if res.get("outcome") in ("closed_empty", "slipped") and os.environ.get("RD_GRASP_RETRY", "1") == "1":
            first = res.get("outcome")
            used = res.get("arm_used") or side
            try:
                self.gripper("open", arm=used)
                self._extra_depth = 0.015
                seed0 = getattr(self, '_last_grasp_seed', None)
                self._skip_hand_check = seed0 is not None
                res2 = self._grasp_impl(target=list(map(float, seed0)) if seed0 is not None else target, arm=used)
            except Exception as exc:  # budget exhausted etc.: keep the first result
                res2 = None
            finally:
                self._extra_depth = 0.0
                self._skip_hand_check = False
            if res2 is not None and res2.get("outcome") not in ("unreachable", "no_object", "too_wide", "target_on_robot"):
                res = {**res2, "first_try": first}
        return self._undo_if_unreachable(res.get('arm_used') or side, start, res)

    def _grasp_impl(self, target: Any = None, arm: str | None = None) -> dict[str, Any]:
        """Find the object at `target` (pixel in one camera, or xyz; default: under the working hand) in the
        RGB-D point cloud, grasp it top-down across its short side, lift 6 cm, report whether it is held."""
        from recovery_explore import rd_grasp
        self._require_cartesian()
        side = self._side(arm)
        self._spend_action("grasp")
        saved_budget = self._round_budget
        self._round_budget = None            # the sub-steps below are one action
        try:
            hands = self._hands_world()
            self._on_robot = None
            if target is None:
                ee = self._ee_pose(side); seed = (ee[:3] + _quat_to_matrix(ee[3:]) @ np.array([self._TIP_X, 0, 0]))
                how = {"from": "hand"}
            else:
                xyz, how = self._resolve_plan_target(target)
                seed = np.asarray(xyz, dtype=np.float64)
                for hs in (() if getattr(self, '_skip_hand_check', False) else ("left", "right")):     # rdv5: the pixel landed on an empty hand, not on an object
                    ee = self._ee_pose(hs); ax = _quat_to_matrix(ee[3:])[:, 0]
                    d = float(rd_grasp._seg_dist(seed[None, :], ee[:3] - 0.05 * ax, ee[:3] + 0.17 * ax)[0])
                    if d < 0.06 and self._grip_open_measured(hs) > 0.6:
                        if os.environ.get("RD_ROBOT_SNAP", "0") == "1":   # rdv6: snap off the robot below
                            self._on_robot = hs
                            break
                        return {"ok": False, "outcome": "target_on_robot", "target_from": how, "seed": seed.round(3).tolist(),
                                "detail": f"the target point is on the robot's own open {hs} hand in the current view "
                                          "(the object may be hidden under it): aim at a visible part of the object, "
                                          "or move that hand away first"}
            self._last_grasp_seed = np.asarray(seed, dtype=np.float64).copy()
            pts = []
            for cam in ("cam_head", f"cam_{side}_wrist"):
                try:
                    depth, K, T = self._depth_view(cam)
                    pts.append(rd_grasp.backproject(depth, K, T, stride=2)[0])
                except Exception:
                    pass
            if not pts:
                return {"ok": False, "outcome": "no_depth", "target_from": how}
            cloud = np.concatenate(pts, axis=0)
            tz = rd_grasp.support_plane(cloud, seed[:2])
            if tz is None:
                return {"ok": False, "outcome": "no_object", "detail": "no support surface near the target", "target_from": how}
            blob, snapped = rd_grasp.segment_snap(cloud, seed[:2], tz, hands, self._arm_bones())
            if getattr(self, "_on_robot", None):
                hs = self._on_robot
                if blob is None or snapped > float(os.environ.get("RD_ROBOT_SNAP_MAX", "0.08")):
                    return {"ok": False, "outcome": "target_on_robot", "target_from": how, "seed": seed.round(3).tolist(),
                            "detail": f"the target point is on the robot's own open {hs} hand and no object lies within "
                                      f"{float(os.environ.get('RD_ROBOT_SNAP_MAX', '0.08')):.2f} m of it off the robot "
                                      "(nearest found: %s m): aim at a visible part of the object, or move that hand away first"
                                      % (round(snapped, 3) if blob is not None else "none")}
                self._robot_snap = {"from_hand": hs, "snapped_from_robot_m": round(snapped, 3)}
                try:
                    with open(os.path.join(str(self._episode_dir()), "robot_snaps.jsonl"), "a") as fh:
                        fh.write(json.dumps(self._robot_snap) + "\n")
                except Exception:
                    pass
            if blob is None:
                return {"ok": False, "outcome": "no_object", "support_z": round(tz, 3), "seed": seed.round(3).tolist(), "target_from": how}
            pl = rd_grasp.plan_top_down(blob, tz)
            info = {"target_from": how, "seed": seed.round(3).tolist(), "support_z": round(tz, 3),
                    "snapped_to_nearest_object_m": round(snapped, 3),
                    "object_center": pl["center_xy"].round(3).tolist(), "object_top_z": round(pl["top_z"], 3),
                    "object_width_m": round(pl["width"], 3), "object_length_m": round(pl["length"], 3)}
            if getattr(self, "_on_robot", None):
                info.update(getattr(self, "_robot_snap", {}) or {})
            if pl["width"] > self._MAX_OPEN:
                part = blob[np.linalg.norm(blob[:, :2] - seed[:2], axis=1) < 0.035] if snapped == 0 else blob[:0]
                pp = rd_grasp.plan_top_down(part, tz) if len(part) >= 30 else None
                if pp is None or pp["width"] > self._MAX_OPEN:
                    return {"ok": False, "outcome": "too_wide", **info}
                pl = pp; info["grasped_part_of_wider_object"] = True
            gz = max(pl["top_z"] - float(np.clip(0.4 * max(pl["height"], 0.0), 0.012, 0.045)) - getattr(self, "_extra_depth", 0.0), tz + 0.012)
            ee_z = max(gz + self._PAD_X, tz + self._TIP_X + 0.002)
            cx, cy = (float(v) for v in pl["center_xy"])
            axes = [pl["minor"]]
            if pl["length"] <= self._MAX_OPEN:
                axes.append(np.array([-pl["minor"][1], pl["minor"][0]]))   # the long axis also fits the fingers
            other = "left" if side == "right" else "right"
            # the named arm first; the other arm only if it is free (open hand, nothing held)
            arms = [side] + ([other] if self._grip_open_measured(other) > 0.6 else [])
            only = getattr(self, "_grasp_only", None)
            if only:
                arms = [a for a in arms if a in only]
            reach = lambda a: float(np.hypot(cx - self._ARM_BASE[a][0], cy - self._ARM_BASE[a][1]))
            arms = sorted([a for a in arms if reach(a) <= self._REACH], key=reach)
            if not arms:
                return {"ok": False, "outcome": "unreachable", "detail": "object is farther than %.2f m from the base of every free arm" % self._REACH, **info}
            steps = []
            quat = None
            for arm_try in arms:
                cur_y = _quat_to_matrix(self._ee_pose(arm_try)[3:])[:, 1]
                cands = []
                for tilt in self._TILTS_DEG:
                    for ax in axes:
                        for sgn in (1, -1):
                            cands.append((tilt, self._tilted(rd_grasp.top_down_quat(ax, sgn * cur_y), arm_try, (cx, cy), tilt)))
                g_open = self.gripper("open", arm=arm_try); steps.append(("open", arm_try, g_open.get("opening_norm")))
                grasp_pt = np.array([cx, cy, gz])
                good = []                       # rdv5: IK-check each approach before spending steps on it
                for tilt, q in cands:
                    try:
                        ok_ik = (self._solve_ik(arm_try, np.concatenate([self._ee_for_point(grasp_pt, q, self._PAD_X + 0.08), q])) is not None and
                                 self._solve_ik(arm_try, np.concatenate([self._ee_for_point(grasp_pt, q, self._PAD_X), q])) is not None)
                    except Exception:
                        ok_ik = True
                    if ok_ik:
                        good.append((tilt, q))
                if len(good) < len(cands):
                    steps.append(("ik_precheck", arm_try, len(good), len(cands)))
                ended = False
                for tilt, q in (good or cands):
                    pre = self._ee_for_point(grasp_pt, q, self._PAD_X + 0.08)
                    r = self._servo_to(arm_try, pre, q, 1.0)
                    steps.append(("above", arm_try, int(tilt), r.get("servo_stop"), r.get("rot_err_deg")))
                    self._cartesian_blocked = False
                    if r.get("ok") or r.get("servo_stop") == "arrived":
                        quat = q; break
                    if r.get("servo_stop") == "episode_end":
                        break
                    if r.get("servo_stop") == "episode_end":
                        ended = True; break
                if quat is not None:
                    side = arm_try
                    break
                if bool(self._require_env().end_flag[0]):
                    break
                if ended:
                    break
            info["arm_used"] = side
            if quat is None:
                return {"ok": False, "outcome": "unreachable", "steps": steps, **info}
            honest = os.environ.get("RD_HONEST", "0") == "1"
            if honest:                      # rdv7: is something standing in the bottom band before the close?
                band = (tz + 0.008, min(tz + 0.03, float(pl["top_z"])))
                n_before = self._band_count(side, (cx, cy), band[0], band[1]) if band[1] > band[0] + 0.004 else -1
            ee_goal = self._ee_for_point(np.array([cx, cy, gz]), quat, self._PAD_X)
            ee_goal[2] = max(ee_goal[2], tz + 0.004 + self._TIP_X * float(-_quat_to_matrix(quat)[2, 0]))
            r = self._servo_to(side, ee_goal, quat, 1.0)
            steps.append(("down", r.get("servo_stop"), r.get("final_dist_m")))
            self._cartesian_blocked = False
            if honest and r.get("servo_stop") != "arrived" and float(r.get("final_dist_m") or 1.0) > float(os.environ.get("RD_DESCENT_TOL", "0.015")):
                # rdv7: the old code closed here anyway, at an unaligned pose -- most of those closes came back empty.
                up = self._servo_to(side, self._ee_for_point(np.array([cx, cy, gz]), quat, self._PAD_X + 0.08), quat, 1.0)
                steps.append(("back_up", up.get("servo_stop"))); self._cartesian_blocked = False
                return {"ok": False, "outcome": "descent_blocked", "descent_stop": r.get("servo_stop"),
                        "descent_short_m": round(float(r.get("final_dist_m") or 0.0), 3),
                        "detail": "the fingers stopped %.1f cm above the grasp height (%s: something in the way, or a "
                                  "joint limit); the gripper was NOT closed and the hand went back up. Choose another "
                                  "point on the object, another approach (the other arm, pi0_pick) or move what is in "
                                  "the way." % (100 * float(r.get("final_dist_m") or 0.0), r.get("servo_stop")),
                        "steps": steps, **info}
            g = self.gripper("close", arm=side); steps.append(("close", g.get("opening_norm")))
            r = self._servo_to(side, self._ee_pose(side)[:3] + np.array([0.0, 0.0, 0.06]), quat, 0.0)
            steps.append(("lift", r.get("servo_stop")))
            self._cartesian_blocked = False
            after = self._grip_open_measured(side)
            closed_at = g.get("opening_norm") or 0.0
            if closed_at <= GRIP_EMPTY_NORM:
                outcome = "closed_empty"
            elif after <= GRIP_EMPTY_NORM:
                outcome = "slipped"
            else:
                outcome = "held"
            if honest and outcome == "held":
                # rdv7: the opening alone does not prove a hold (a neighbour or the holder can stop the fingers).
                n_after = self._band_count(side, (cx, cy), band[0], band[1]) if n_before >= 0 else -1
                info["support_points_before"] = n_before; info["support_points_after"] = n_after
                if n_before >= 15 and n_after >= 0.6 * n_before:
                    outcome = "held_by_width_only"
                    info["detail"] = ("the fingers stopped on something, but the depth cameras still see the object "
                                      "standing where it was after the 6 cm lift: probably not held")
                elif n_before >= 15 and n_after <= 0.25 * n_before:
                    info["verify"] = "object left its support"
                else:
                    info["verify"] = "unclear from depth -- check the picture"
            if outcome == "held":
                self._held = {"side": side, "grasp_z": float(gz), "support_z": float(tz), "quat": np.asarray(quat),
                              "height": float(pl["height"]), "width": float(pl["width"])}
            return {"ok": outcome == "held", "outcome": outcome, "opening_after_lift": round(after, 3),
                    "grasp_z": round(gz, 3), "steps": steps, **info}
        finally:
            self._round_budget = saved_budget

    # -- sub-goal ops: place / pick_place / press / push --------------------

    def _surface_z_near(self, xyz, side: str, radius: float = 0.04):
        """Height of the surface around a target point from the head + that arm's wrist depth (median of the top
        decile within radius, so a target on a box rim or a plate reads the rim/plate, not the table)."""
        from recovery_explore import rd_grasp
        pts = []
        for cam in ("cam_head", f"cam_{side}_wrist"):
            try:
                d, K, T = self._depth_view(cam)
                pts.append(rd_grasp.backproject(d, K, T, stride=2)[0])
            except Exception:
                pass
        if not pts:
            return float(xyz[2])
        P = np.concatenate(pts, axis=0)
        P = P[np.linalg.norm(P[:, :2] - np.asarray(xyz[:2]), axis=1) < radius]
        for h in self._hands_world():
            P = P[np.linalg.norm(P - h, axis=1) > 0.07] if len(P) else P
        try:                                   # rdv5: drop points on any arm segment (links, fingers)
            for a, b in self._arm_bones():
                if not len(P):
                    break
                ab = b - a; L2 = float(ab @ ab) or 1e-9
                t = np.clip(((P - a) @ ab) / L2, 0.0, 1.0)
                P = P[np.linalg.norm(P - (a + t[:, None] * ab), axis=1) > 0.05]
        except Exception:
            pass
        if len(P) < 10:
            return float(xyz[2])
        z = float(np.percentile(P[:, 2], 90))
        if z > float(xyz[2]) + 0.15 and len(P) < 30:
            return float(xyz[2])
        return z

    def _carry_to(self, side: str, xy, ee_z: float, quat, grip: float, steps: list) -> bool:
        """Rise, travel above (x, y), descend to ee_z keeping the wrist; True when the descent arrived or stalled
        on contact."""
        # The X5 is short: fingers-down it cannot rise far. Rise 5 cm (cap 1.02 m); if even that fails, travel at a
        # small clearance above the higher of start and goal instead of giving up.
        if float(np.hypot(xy[0] - self._ARM_BASE[side][0], xy[1] - self._ARM_BASE[side][1])) > self._REACH:
            steps.append(("out_of_reach", side)); return False
        cur = self._ee_pose(side)
        safe = min(max(float(cur[2]), ee_z) + 0.05, 1.02)
        r = self._servo_to(side, np.array([cur[0], cur[1], safe]), quat, grip)
        steps.append(("rise", r.get("servo_stop"), r.get("final_dist_m"))); self._cartesian_blocked = False
        if r.get("servo_stop") != "arrived" and (r.get("final_dist_m") or 1) > 0.02:
            safe = max(float(self._ee_pose(side)[2]), ee_z + 0.02)
        # travel with the given wrist; if the IK cannot reach the goal that way, try the wrist turned about the
        # vertical by 90/180/270 degrees (a parallel gripper holds the same object either way)
        q0 = np.asarray(quat, dtype=np.float64) if quat is not None else self._ee_pose(side)[3:]
        # the goal is the point the finger pads should reach: ee_z was computed for a vertical approach
        pad_pt = np.array([xy[0], xy[1], ee_z - self._PAD_X])
        arrived = False
        cands = []                              # rdv5: IK-check every candidate before spending steps on it
        for tilt in self._TILTS_DEG:
            for k in range(4):
                a = 0.5 * k * np.pi / 2
                qz = np.array([np.cos(a), 0.0, 0.0, np.sin(a)])
                qk = self._tilted(_quat_normalize(_quat_mul(qz, q0)), side, xy, tilt)
                over = self._ee_for_point(pad_pt, qk, self._PAD_X); over[2] = max(over[2], safe)
                try:
                    ok_ik = (self._solve_ik(side, np.concatenate([over, qk])) is not None and
                             self._solve_ik(side, np.concatenate([self._ee_for_point(pad_pt, qk, self._PAD_X), qk])) is not None)
                except Exception:
                    ok_ik = True
                cands.append((tilt, k, qk, over, ok_ik))
        good = [c for c in cands if c[4]]
        if len(good) < len(cands):
            steps.append(("ik_precheck", len(good), len(cands)))
        for tilt, k, qk, over, _ in (good or cands):
            if True:
                r = self._servo_to(side, over, qk, grip)
                steps.append(("over", int(tilt), k * 90, r.get("servo_stop"), r.get("final_dist_m"))); self._cartesian_blocked = False
                if r.get("servo_stop") == "arrived" or (r.get("final_dist_m") or 1) <= 0.03:
                    quat, arrived = qk, True
                    break
                if r.get("servo_stop") == "episode_end":
                    break
        if not arrived:
            return False
        r = self._servo_to(side, self._ee_for_point(pad_pt, quat, self._PAD_X), quat, grip)
        steps.append(("down", r.get("servo_stop"), r.get("final_dist_m"))); self._cartesian_blocked = False
        return True

    def place(self, target: Any, arm: str | None = None, release_dz: float = 0.01) -> dict[str, Any]:
        side = self._side(arm) if arm else (getattr(self, '_held', None) or {}).get('side') or self._side(None)
        start = self._ee_pose(side)
        res = self._place_impl(target, arm=arm, release_dz=release_dz)
        return self._undo_if_unreachable(res.get('arm_used') or side, start, res)

    def _place_impl(self, target: Any, arm: str | None = None, release_dz: float = 0.01) -> dict[str, Any]:
        """Put the object in this hand down at `target` (a surface point: pixel or xyz) and let go, then lift 8 cm.
        The release height comes from the grasp (how far below the fingers the object reaches) and the depth at
        the target, so it works on the table, a plate, a shelf or into a box."""
        self._require_cartesian()
        held = getattr(self, "_held", None)
        side = self._side(arm) if arm else (held["side"] if held else self._side(None))
        self._spend_action("place")
        saved_budget = self._round_budget; self._round_budget = None
        try:
            xyz, how = self._resolve_plan_target(target)
            surf = self._surface_z_near(xyz, side)
            hang = (held["grasp_z"] - held["support_z"]) if held and held["side"] == side else 0.03
            quat = held["quat"] if held and held["side"] == side else self._ee_pose(side)[3:]
            ee_z = surf + hang + float(release_dz) + self._PAD_X
            steps = []
            n_dest0 = (self._band_count(side, xyz, surf + 0.008, surf + 0.06, radius=0.04)
                       if os.environ.get("RD_HONEST", "0") == "1" else -1)
            ok = self._carry_to(side, xyz, ee_z, quat, self._grip_norm(side), steps)
            if not ok:
                return {"ok": False, "outcome": "unreachable", "target_from": how, "surface_z": round(surf, 3), "steps": steps}
            g = self.gripper("open", arm=side); steps.append(("open", g.get("opening_norm")))
            r = self._servo_to(side, self._ee_pose(side)[:3] + np.array([0, 0, 0.08]), None, 1.0)
            steps.append(("lift", r.get("servo_stop"))); self._cartesian_blocked = False
            self._held = None
            if os.environ.get("RD_HONEST", "0") == "1":
                # rdv7: look at the destination after letting go instead of assuming the object is there.
                n_at = self._band_count(side, xyz, surf + 0.008, surf + 0.06, radius=0.04)
                dn = [s for s in steps if s and s[0] == "down"]
                stalled = bool(dn) and dn[-1][1] != "arrived" and float(dn[-1][2] or 1.0) > 0.03
                out = "released_at_target" if (n_at >= max(15, (n_dest0 if n_dest0 > 0 else 0) + 15)) else "released_unverified"
                return {"ok": True, "outcome": out, "target_from": how, "surface_z": round(surf, 3),
                        "release_ee_z": round(ee_z, 3), "dest_points_before": n_dest0, "dest_points_after": n_at,
                        "released_above_target_m": (round(float(dn[-1][2] or 0.0), 3) if stalled else None),
                        "detail": (None if out == "released_at_target" else
                                   "the depth cameras do not see a new object resting at the destination: look at the "
                                   "picture before handing back"), "steps": steps}
            return {"ok": True, "outcome": "released", "target_from": how, "surface_z": round(surf, 3),
                    "release_ee_z": round(ee_z, 3), "steps": steps}
        finally:
            self._round_budget = saved_budget

    def pick_place(self, target: Any, destination: Any, arm: str | None = None) -> dict[str, Any]:
        """grasp(target) then, if held, place(destination). One action."""
        try:                                     # rdv5: never grasp with an arm that cannot reach the destination
            dxyz, dhow = self._resolve_plan_target(destination)
            fits = [a for a in ("left", "right")
                    if float(np.hypot(dxyz[0] - self._ARM_BASE[a][0], dxyz[1] - self._ARM_BASE[a][1])) <= self._REACH]
        except InterventionError:
            raise
        except Exception:
            fits = ["left", "right"]
        if not fits:
            return {"ok": False, "outcome": "unreachable", "stage": "destination",
                    "detail": "destination is farther than %.2f m from the base of both arms" % self._REACH}
        want = self._side(arm) if arm else None
        if want is not None and want not in fits:
            arm = fits[0]
        self._grasp_only = fits if want is None else [arm]
        try:
            g = self.grasp(target=target, arm=arm)
        finally:
            self._grasp_only = None
        if g.get("outcome") != "held":
            return {**g, "ok": False, "stage": "grasp"}
        p = self.place(destination, arm=g.get("arm_used") or arm)
        if os.environ.get("RD_HONEST", "0") == "1" and p.get("ok"):
            pout = {"released_at_target": "placed_at_target", "released_unverified": "placed_unverified"}.get(p.get("outcome"), "placed")
            return {"ok": True, "outcome": pout, "detail": p.get("detail"),
                    "grasp": {k: g.get(k) for k in ("object_center", "object_width_m", "arm_used", "verify")}, "place": p}
        return {"ok": p.get("ok", False), "outcome": "placed" if p.get("ok") else "grasped_not_placed",
                "grasp": {k: g.get(k) for k in ("object_center", "object_width_m", "arm_used")}, "place": p}

    def press(self, target: Any, arm: str | None = None, depth: float = 0.012) -> dict[str, Any]:
        side = self._side(arm) if arm else (getattr(self, '_held', None) or {}).get('side') or self._side(None)
        start = self._ee_pose(side)
        res = self._press_impl(target, arm=arm, depth=depth)
        return self._undo_if_unreachable(res.get('arm_used') or side, start, res)

    def _press_impl(self, target: Any, arm: str | None = None, depth: float = 0.012) -> dict[str, Any]:
        """Closed fingers straight down onto `target` (button, key, lever top) until `depth` past the surface, then up."""
        from recovery_explore import rd_grasp
        self._require_cartesian()
        side = self._side(arm)
        self._spend_action("press")
        saved_budget = self._round_budget; self._round_budget = None
        try:
            xyz, how = self._resolve_plan_target(target)
            surf = self._surface_z_near(xyz, side, radius=0.015)
            quat = rd_grasp.top_down_quat(_quat_to_matrix(self._ee_pose(side)[3:])[:2, 1])
            steps = []
            g = self.gripper("close", arm=side); steps.append(("close", g.get("opening_norm")))
            ok = self._carry_to(side, xyz, surf + self._TIP_X + 0.02, quat, 0.0, steps)
            if not ok:
                return {"ok": False, "outcome": "unreachable", "target_from": how, "steps": steps}
            r = self._servo_to(side, np.array([xyz[0], xyz[1], surf + self._TIP_X - float(depth)]), quat, 0.0)
            steps.append(("press", r.get("servo_stop"), r.get("final_dist_m"))); self._cartesian_blocked = False
            r2 = self._servo_to(side, np.array([xyz[0], xyz[1], surf + self._TIP_X + 0.06]), quat, 0.0)
            steps.append(("up", r2.get("servo_stop"))); self._cartesian_blocked = False
            if os.environ.get("RD_HONEST", "0") == "1":
                # rdv7: "pressed" only if the fingertips actually got down to the surface (a press that ran out of IK
                # or of steps above the button used to be reported as pressed too).
                low = float((r.get("eef") or [0, 0, 9])[2]); above = low - (surf + self._TIP_X)
                if r.get("servo_stop") == "episode_end":
                    out = "episode_end"
                elif above <= 0.004:
                    out = "pressed"
                else:
                    out = "press_short"
                return {"ok": out == "pressed", "outcome": out, "target_from": how, "surface_z": round(surf, 3),
                        "press_stop": r.get("servo_stop"), "stopped_above_surface_m": round(max(0.0, above), 3),
                        "steps": steps}
            return {"ok": True, "outcome": "pressed", "target_from": how, "surface_z": round(surf, 3), "steps": steps}
        finally:
            self._round_budget = saved_budget

    def push(self, target: Any, to: Any = None, direction: Any = None, distance: float = 0.05,
             arm: str | None = None) -> dict[str, Any]:
        side = self._side(arm)
        start = self._ee_pose(side)
        res = self._push_impl(target, to=to, direction=direction, distance=distance, arm=arm)
        return self._undo_if_unreachable(side, start, res)

    def _push_impl(self, target: Any, to: Any = None, direction: Any = None, distance: float = 0.05,
             arm: str | None = None) -> dict[str, Any]:
        """Slide the object at `target` along the table: toward `to` (a point) or along `direction` [dx, dy] for
        `distance` metres. Closed fingers come down behind the object and move through it."""
        from recovery_explore import rd_grasp
        self._require_cartesian()
        side = self._side(arm)
        self._spend_action("push")
        saved_budget = self._round_budget; self._round_budget = None
        try:
            xyz, how = self._resolve_plan_target(target)
            pts = []
            for cam in ("cam_head", f"cam_{side}_wrist"):
                try:
                    d, K, T = self._depth_view(cam); pts.append(rd_grasp.backproject(d, K, T, stride=2)[0])
                except Exception:
                    pass
            cloud = np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))
            tz = rd_grasp.support_plane(cloud, np.asarray(xyz[:2])) if len(cloud) else None
            blob, _ = rd_grasp.segment_snap(cloud, np.asarray(xyz[:2]), tz, self._hands_world(), self._arm_bones()) if tz is not None else (None, 0)
            if blob is None:
                return {"ok": False, "outcome": "no_object", "target_from": how}
            c = np.median(blob[:, :2], axis=0); top = float(np.percentile(blob[:, 2], 95))
            if to is not None:
                dest, _h = self._resolve_plan_target(to)
                v = np.asarray(dest[:2]) - c; dist = float(np.linalg.norm(v))
            else:
                v = np.asarray(direction, dtype=np.float64)[:2]; dist = float(distance)
            if np.linalg.norm(v) < 1e-6:
                return {"ok": False, "outcome": "no_direction", "target_from": how}
            u = v / np.linalg.norm(v)
            ext = float(np.max((blob[:, :2] - c) @ u) - np.min((blob[:, :2] - c) @ u))
            start = c - u * (ext / 2 + 0.04)
            z_tip = tz + min(0.02, max(0.008, 0.5 * (top - tz)))
            quat = rd_grasp.top_down_quat(np.array([-u[1], u[0]]))   # fingers across the push direction
            steps = []
            g = self.gripper("close", arm=side); steps.append(("close", g.get("opening_norm")))
            ok = self._carry_to(side, start, z_tip + self._TIP_X, quat, 0.0, steps)
            if not ok:
                return {"ok": False, "outcome": "unreachable", "target_from": how, "steps": steps}
            end = c + u * (dist - ext / 2 + 0.02) if to is not None else c + u * (dist)
            r = self._servo_to(side, np.array([end[0], end[1], z_tip + self._TIP_X]), quat, 0.0)
            steps.append(("push", r.get("servo_stop"), r.get("final_dist_m"))); self._cartesian_blocked = False
            r2 = self._servo_to(side, self._ee_pose(side)[:3] + np.array([0, 0, 0.08]), quat, 0.0)
            steps.append(("up", r2.get("servo_stop"))); self._cartesian_blocked = False
            if os.environ.get("RD_HONEST", "0") == "1":
                # rdv7: measure where the object is now instead of reporting "pushed" unconditionally (the old code
                # said pushed even when the push motion hit an IK failure, a stall or the step cap).
                moved = along = None
                try:
                    cloud2 = self._scene_cloud(side)
                    blob2, _sn = rd_grasp.segment_snap(cloud2, c, tz, self._hands_world(), self._arm_bones()) if len(cloud2) else (None, 0)
                    if blob2 is not None:
                        c2 = np.median(blob2[:, :2], axis=0)
                        moved = float(np.linalg.norm(c2 - c)); along = float((c2 - c) @ u)
                except Exception:
                    pass
                blocked = r.get("servo_stop") != "arrived" and float(r.get("final_dist_m") or 1.0) > 0.02
                if moved is not None and moved < 0.01:
                    out = "no_motion"
                elif blocked:
                    out = "push_blocked"
                elif moved is None:
                    out = "pushed_unverified"
                else:
                    out = "pushed"
                return {"ok": out == "pushed", "outcome": out, "target_from": how, "object_center": c.round(3).tolist(),
                        "push_dir": u.round(3).tolist(), "distance_m": round(dist, 3), "push_stop": r.get("servo_stop"),
                        "push_short_m": round(float(r.get("final_dist_m") or 0.0), 3),
                        "object_moved_m": (round(moved, 3) if moved is not None else None),
                        "moved_along_push_m": (round(along, 3) if along is not None else None), "steps": steps}
            return {"ok": True, "outcome": "pushed", "target_from": how, "object_center": c.round(3).tolist(),
                    "push_dir": u.round(3).tolist(), "distance_m": round(dist, 3), "steps": steps}
        finally:
            self._round_budget = saved_budget

    # -- the model's single entry point -----------------------------------

    # -- rdv10: parameter spellings, reachability report, arm "auto" ----------------------------

    _REACH_OPS = {"grasp", "place", "pick_place", "press", "push", "move_to"}

    def _norm_plan_params(self, op: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        """Accept the spellings the judge sends for nudge / lift / rotate (rdv9: {"dx": .., "dy": .., "dz": ..} for a
        nudge crashed with KeyError 'dxyz')."""
        r = dict(raw)

        def num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        def vec3(v):
            return isinstance(v, (list, tuple)) and len(v) == 3

        if op == "nudge" and "dxyz" not in r:
            for k in ("delta", "offset", "xyz", "d", "dxyz_m"):
                if vec3(r.get(k)):
                    r["dxyz"] = list(r[k])
                    break
            else:
                if any(k in r for k in ("dx", "dy", "dz")):
                    r["dxyz"] = [num(r.get("dx", 0)), num(r.get("dy", 0)), num(r.get("dz", 0))]
        if op == "lift" and "dz" not in r:
            for k in ("z", "distance", "height", "dist"):
                if k in r:
                    r["dz"] = num(r[k])
                    break
            else:
                if vec3(r.get("dxyz")):
                    r["dz"] = num(r["dxyz"][2])
        if op == "rotate" and "drot" not in r:
            if any(k in r for k in ("droll", "dpitch", "dyaw")):
                r["drot"] = [num(r.get("droll", 0)), num(r.get("dpitch", 0)), num(r.get("dyaw", 0))]
            else:
                for k in ("rpy", "angles", "delta", "axis_angle"):
                    if vec3(r.get(k)):
                        r["drot"] = list(r[k])
                        break
        return r

    def _reach_report(self, xyz) -> dict[str, Any]:
        """Which arm can reach a world point: horizontal distance to each arm base vs RD_REACH, and whether the scripted
        IK finds a fingers-down pose (pads at the point and 8 cm above it) for any of the usual wrist tilts. Uses only the
        target point and the arms' proprioception."""
        from recovery_explore import rd_grasp
        p = np.asarray(xyz, dtype=np.float64).reshape(-1)[:3]
        arms: dict[str, Any] = {}
        by = []
        for side in ("left", "right"):
            dist = float(np.hypot(p[0] - self._ARM_BASE[side][0], p[1] - self._ARM_BASE[side][1]))
            past = dist - self._REACH
            ik = None
            if past <= 0.05:
                ik = False
                try:
                    cur_y = _quat_to_matrix(self._ee_pose(side)[3:])[:, 1]
                    for ax in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
                        for tilt in self._TILTS_DEG:
                            q = self._tilted(rd_grasp.top_down_quat(ax, cur_y), side, p[:2], tilt)
                            if all(self._solve_ik(side, np.concatenate([self._ee_for_point(p, q, off), q])) is not None
                                   for off in (self._PAD_X, self._PAD_X + 0.08)):
                                ik = True
                                break
                        if ik:
                            break
                except Exception:
                    ik = None
            arms[side] = {"dist_m": round(dist, 3), "past_reach_m": round(max(past, 0.0), 3), "ik": ik}
            if past <= 0 and ik is not False:
                by.append(side)
        return {"reachable_by": by or ["none"], "arms": arms}

    def _reach_for_step(self, op: str, raw: Mapping[str, Any]) -> dict[str, Any] | None:
        """Reach report for a step's target (and destination). Sets unreachable_both when a point the op has to reach
        (the object; for place / pick_place also the destination) is beyond RD_REACH from BOTH arm bases -- the same
        distance at which the op itself refuses -- so no step is spent on a motion that cannot happen."""
        if op == "pick_place":
            pts = (("target", raw.get("target"), True), ("destination", raw.get("destination") or raw.get("to"), True))
        elif op == "push":
            pts = (("target", raw.get("target"), True), ("to", raw.get("to"), False))
        else:
            pts = (("target", raw.get("target"), True),)
        out: dict[str, Any] = {}
        must = []
        for name, tgt, required in pts:
            if tgt is None:
                continue
            try:
                xyz, _how = self._resolve_plan_target(tgt)
            except Exception as exc:
                out[name] = {"error": f"reach check skipped: {type(exc).__name__}"}
                continue
            rep = self._reach_report(xyz)
            out[name] = {"xyz": [round(float(v), 3) for v in xyz], **rep}
            if required and all(a["past_reach_m"] > 0 for a in rep["arms"].values()):
                must.append((name, min(a["past_reach_m"] for a in rep["arms"].values())))
        if not out:
            return None
        if must:
            name, margin = must[0]
            out["unreachable_both"] = True
            out["detail"] = (f"the {name} point is {margin:.2f} m beyond the reach of BOTH arms' scripted motion; "
                             "hand this step back to the policy (it reaches poses the scripted IK cannot) or first "
                             "bring the object closer with a reachable push or a hand-to-hand handover")
        return out

    def _auto_arm(self, reach: Mapping[str, Any] | None) -> str:
        """arm "auto": the nearest arm that reaches the step's target (else its destination), else the active arm."""
        for key in ("target", "destination"):
            t = (reach or {}).get(key) or {}
            cand = [s for s in t.get("reachable_by", []) if s in ("left", "right")]
            if cand:
                return min(cand, key=lambda s: t["arms"][s]["dist_m"])
        return self._active_arm()

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
        if "pixels" in target and os.environ.get("RD_TRIANGULATE", "0") == "1":
            return self._triangulate_target(target)
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
        # 2026-10-20 (user rule, judge-rules-autonomy): the server never rejects a plan for how it
        # repairs -- R1-R4 below are disabled. Safety stops (empty close, missed servo, blow-up) still
        # stop and report inside execute_plan but never lock later actions. ROBODOJO_PLAN_GATES=1 restores them.
        if os.environ.get("ROBODOJO_PLAN_GATES", "0") != "1":
            return None
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
            # format auto-correction (2026-10-22, GPT-6.1 dev: 7 plans died on these names)
            if op in ("open", "close", "grip", "release"):
                raw = dict(raw, op="gripper", state=("open" if op in ("open", "release") else "close")); op = "gripper"
            elif op in ("rotate_wrist", "wrist_rotate", "turn"):
                raw = dict(raw, op="rotate"); op = "rotate"
            elif op in ("pick", "pick_up", "regrasp"):
                raw = dict(raw, op="grasp"); op = "grasp"
            elif op in ("check_reach", "reachability", "reach"):
                raw = dict(raw, op="reach_check"); op = "reach_check"
            raw = self._norm_plan_params(op, raw)   # rdv10: dx/dy/dz etc. (KeyError 'dxyz' in rdv9 dev)
            if op not in self._PLAN_OPS:
                stop_reason, stopped_at = "error", i
                log.append({"step": i, "op": op,
                            "error": f"unknown op; allowed: {sorted(self._PLAN_OPS)}"})
                break
            arm = raw.get("arm")
            reach_pre = None
            if os.environ.get("RD_REACH_REPORT", "0") == "1" and op in self._REACH_OPS:
                try:
                    reach_pre = self._reach_for_step(op, raw)
                except Exception as exc:
                    reach_pre = {"error": f"reach check failed: {type(exc).__name__}"}
            if str(arm).lower() == "auto":
                arm = self._auto_arm(reach_pre)
                raw = dict(raw, arm=arm)
            entry: dict[str, Any] = {"step": i, "op": op, "arm": self._side(arm)}
            if reach_pre is not None:
                entry["reach"] = reach_pre
                if reach_pre.get("unreachable_both"):
                    entry.update({"outcome": "unreachable_both", "ok": False, "detail": reach_pre.get("detail")})
                    log.append(entry)
                    continue
            try:
                grip = raw.get("gripper", "hold")
                if op == "move_to":
                    xyz, how = self._resolve_plan_target(raw.get("target"))
                    entry["target"] = [round(v, 4) for v in xyz]
                    entry["target_from"] = how
                    res = self.move_to(xyz=xyz, gripper=grip, arm=arm, quat=raw.get("quat"))
                elif op == "nudge":
                    if "dxyz" not in raw:
                        raise InterventionError('nudge needs "dxyz": [dx, dy, dz] in metres')
                    res = self.nudge(dxyz=raw["dxyz"], gripper=grip, arm=arm)
                elif op == "lift":
                    if "dz" not in raw:
                        raise InterventionError('lift needs "dz" in metres')
                    res = self.lift(dz=float(raw["dz"]), gripper=grip, arm=arm)
                    entry["dz"] = round(float(raw["dz"]), 4)
                elif op == "rotate":
                    if "drot" not in raw:
                        raise InterventionError('rotate needs "drot": [rx, ry, rz] (axis-angle, radians)')
                    res = self.rotate(drot=raw["drot"], gripper=grip, arm=arm)
                elif op == "reach_check":
                    xyz, how = self._resolve_plan_target(raw.get("target"))
                    res = {"ok": True, "outcome": "reach_report", "target": [round(float(v), 4) for v in xyz],
                           "target_from": how, **self._reach_report(xyz)}
                elif op == "retreat":
                    res = self.retreat(to=str(raw.get("to") or "takeover"), arm=arm, release=bool(raw.get("release", False)))
                elif op == "gripper":
                    res = self._gripper_verified(action=str(raw.get("state") or raw.get("action") or "close"), arm=arm)
                elif op == "grasp":
                    res = self.grasp(target=raw.get("target"), arm=arm)
                elif op == "place":
                    res = self.place(target=raw.get("target"), arm=arm)
                elif op == "pick_place":
                    res = self.pick_place(target=raw.get("target"), destination=raw.get("destination") or raw.get("to"), arm=arm)
                elif op == "press":
                    res = self.press(target=raw.get("target"), arm=arm)
                elif op == "push":
                    res = self.push(target=raw.get("target"), to=raw.get("to"), direction=raw.get("direction"),
                                    distance=float(raw.get("distance") or 0.05), arm=arm)
                elif op == "pi0_pick":
                    n = max(1, min(int(raw.get("chunks", 8)), 8))
                    if self._round_budget is not None:
                        n = max(1, min(n, int(self._round_budget) - int(self._round_used)))
                    self._spend_action("pi0_pick")
                    adv = self.harness_advance(num_chunks=n, stop_on_grasp=True)
                    self._grip_cmd_issued = {}
                    hit = adv.get("stopped") == "grasp_detected"
                    entry.update({"outcome": "grasped" if hit else "no_grasp", "chunks_run": adv.get("chunks_done"),
                                  "policy_done": bool(adv.get("done")), "task_success": bool(adv.get("task_success"))})
                    if os.environ.get("RD_HONEST", "0") == "1":
                        entry["opening_after"] = {s: round(float(self._grip_open_measured(s)), 3) for s in ("left", "right")}
                    res = {"ok": hit}
                else:  # policy hand-back
                    if os.environ.get("RD_RESTAGE", "0") == "1":
                        entry["restage"] = self._restage_for_handback()
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
            if os.environ.get("RD_HONEST", "0") == "1" and op in ("move_to", "nudge", "lift", "rotate", "retreat") \
                    and not entry.get("outcome"):
                entry["outcome"] = "arrived" if res.get("ok") else f"stopped_short_{res.get('servo_stop')}"
            if reach_pre and str(entry.get("outcome", "")).startswith("unreachable"):
                t = reach_pre.get("target") or {}
                other = [s for s in t.get("reachable_by", []) if s in ("left", "right") and s != entry.get("arm")]
                if other:
                    entry["hint"] = f"the {other[0]} arm reaches this target: send that arm or \"arm\": \"auto\""
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
