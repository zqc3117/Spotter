"""Robot-agnostic action primitives for the recovery_explore env service.

COPY-AND-ADAPT of ``vendor/rpent/robocasa/primitives.py`` (Copyright 2026 The
RPent Authors, Apache-2.0 -- see ``vendor/rpent/LICENSE``, upstream commit in
``vendor/rpent/UPSTREAM_COMMIT.txt``). Every behaviour below that carries an
explanatory comment carries it because it was hard-won upstream; the comments
are copied with the code.

WHAT SURVIVED THE PORT TO PandaMobile, and why it is legitimate to port:

* ``move_to``'s world->action mapping is an ONLINE-CALIBRATED 3x3 position
  jacobian (probe the 3 unit arm-xyz actions, measure the world dpos each
  produces, invert). Nothing in it knows which robot it is driving -- it
  measures the robot it is actually attached to. That is the single reason
  this primitive is reusable across RPent's PandaOmron and our PandaMobile.
* ``_resolve_grip``'s hold-servo: the gripper command is a close-VELOCITY, so a
  sustained +1 squeezes a small object out. Servoing back to the width the
  fingers had when the motion began is the carry-safe hold.
* per-step clamping of the world delta (``step_clip``) and of the action itself
  to [-1, 1].

WHAT WAS STRIPPED (all Omron/mobile-base/RLDX specific, none of it ours):
``navigate_to`` / ``move_base`` / ``_calibrate_forward`` / ``_yaw`` (Omron base
driving), the ``mobilebase0_navview`` camera, ``RLDXSkill`` and every VLA hook,
the world-map/heavy-npy dumping, the frame recorder, ``reset`` and its
``RLDX_ALLOW_RESET`` gate (this service NEVER resets the episode -- the whole
point is a restored failure anchor), and ``dump_success_criteria`` /
``task_progress`` (privileged text/counters, which the oracle-masking contract
does not allow us to hand the explorer).

WHAT WAS ADDED: ``lift(dz)``, a thin wrapper over ``move_delta`` -- straight-up
displacement is the single most common recovery motion and RPent had no name
for it.

The env handle passed in is duck-typed (``EnvHandle`` below) so these primitives
stay testable with a fake and carry no robosuite import of their own; the real
implementation lives in ``recovery_explore/env_service.py``.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np

# Composite-controller action scaling: action 1.0 -> this much target delta.
# Identical in the PandaOmron (RPent) and PandaMobile (ours) composite configs
# -- both use robosuite's OSC_POSE defaults -- but it only ever biases the
# INITIAL jacobian probe, which move_to then measures for real.
OSC_POS_SCALE = 0.05  # action 1.0 -> 0.05 m target delta
OSC_ROT_SCALE = 0.5  # action 1.0 -> 0.5 rad

# PandaMobile composite action layout (12): [eef_pos(3), eef_rot(3), gripper(1),
# base_motion(4), control_mode(1)] -- same layout RPent's env_server.step
# documents for PandaOmron, and the same one
# rpc/run_remote_robocasa_collect.py::_to_environment_action pads a 7-dim policy
# action into (arm 7 + [0,0,0,0,-1]).
ARM_ACTION_DIM = 12
BASE_MODE_INDEX = 11
GRIPPER_INDEX = 6


class EnvHandle(Protocol):
    """The minimal env surface these primitives need.

    Implemented in-process by ``env_service._DirectEnvHandle``; upstream this was
    an RPC client (``vendor/rpent/robocasa/env_client.py``).
    """

    def step(self, flat_action: np.ndarray) -> Any: ...

    @property
    def eef_pos(self) -> np.ndarray: ...

    @property
    def gripper_qpos(self) -> np.ndarray: ...


class RecoveryPrimitives:
    """Closed-loop eef primitives over a single, already-restored env."""

    def __init__(self, env: EnvHandle, *, action_dim: int = ARM_ACTION_DIM) -> None:
        self.env = env
        self.action_dim = int(action_dim)
        self._pos_jac: np.ndarray | None = None  # 3x3 action(arm xyz) -> world dpos

    # ---- action helpers ----
    def _zero(self, base_mode: float = -1.0) -> np.ndarray:
        a = np.zeros(self.action_dim)
        if self.action_dim > BASE_MODE_INDEX:
            # base_mode < 0 => the composite controller drives the ARM, not the
            # base. Every primitive here is arm-only.
            a[BASE_MODE_INDEX] = base_mode
        return a

    @staticmethod
    def _hold_gripper_val(g: float) -> float:
        # +1 close/hold, -1 open
        return float(np.clip(g, -1, 1))

    def _resolve_grip(self, gripper: float | str | None, target_q: float) -> float:
        """Return the a[6] gripper command for a motion step.

        gripper="hold"/None (DEFAULT for moves) -> SERVO the fingers back to
        `target_q`, the width they had when the motion began. This is the
        carry-safe hold: the gripper action is a CLOSE-VELOCITY command, so a
        sustained +1 keeps driving the fingers shut and SQUEEZES a small object
        OUT (verified upstream: bread qpos 0.0376 -> 0.0005 during a +1 carry).
        Servoing to the grasped width holds the object without crushing it and
        without letting it drift open. A numeric gripper (+1 close / -1 open) is
        an EXPLICIT override and passes through unchanged.
        """
        if isinstance(gripper, str) or gripper is None:
            cur = float(self.env.gripper_qpos[0])
            return float(
                np.clip(60.0 * (cur - target_q), -1.0, 1.0)
            )  # +a[6] closes (qpos down)
        return self._hold_gripper_val(gripper)

    def _step_arm(
        self,
        dpos: Sequence[float] = (0, 0, 0),
        drot: Sequence[float] = (0, 0, 0),
        gripper: float = -1.0,
        n: int = 1,
    ) -> np.ndarray:
        a = self._zero(base_mode=-1.0)
        a[0:3] = np.clip(np.asarray(dpos, dtype=np.float64) / OSC_POS_SCALE, -1, 1)
        a[3:6] = np.clip(np.asarray(drot, dtype=np.float64) / OSC_ROT_SCALE, -1, 1)
        a[GRIPPER_INDEX] = self._hold_gripper_val(gripper)
        for _ in range(n):
            self.env.step(a)
        return self.env.eef_pos

    # ---- mobile base ----
    def move_base(self, cmd: Sequence[float], steps: int = 6) -> dict[str, Any]:
        """Drive the mobile base. ``cmd`` = [forward/back, left/right, turn, lift], each in [-1, 1], for ``steps`` consecutive steps.

        The last 5 action dims are [base(4), mode]; with mode > 0 the composite controller drives the base instead of the arm.
        pi0.5 is a native 12-dim policy that moves the base; the VLM's action set must be as wide as the policy's, hence this primitive.
        """
        if self.action_dim <= BASE_MODE_INDEX:
            return {"ok": False, "error": "这个环境的动作是 7 维，没有底盘"}
        c = np.clip(np.asarray(cmd, dtype=np.float64).reshape(4), -1.0, 1.0)
        a = self._zero(base_mode=1.0)
        a[GRIPPER_INDEX] = self._hold_gripper_val("hold")
        a[GRIPPER_INDEX + 1: BASE_MODE_INDEX] = c
        p0 = np.asarray(self.env.eef_pos, dtype=np.float64).copy()
        for _ in range(int(steps)):
            self.env.step(a)
        p1 = np.asarray(self.env.eef_pos, dtype=np.float64)
        self.invalidate_calibration()
        return {"ok": True, "cmd": c.tolist(), "steps": int(steps),
                "eef_moved_m": [round(float(v), 4) for v in (p1 - p0)],
                "eef": p1.tolist()}

    # ---- online jacobian + move_to ----
    def _calibrate_pos_jacobian(self, gripper: float = -1.0) -> np.ndarray:
        """Probe 3 unit arm-xyz actions, measure world dpos -> 3x3 jacobian J s.t.
        world_dpos ~= J @ action_xyz. move_to inverts J to map a desired world
        delta into an action. This is what makes the primitive robot-agnostic."""
        cols = []
        for axis in range(3):
            p0 = np.asarray(self.env.eef_pos, dtype=np.float64).copy()
            a = self._zero()
            a[axis] = 0.4
            a[GRIPPER_INDEX] = gripper
            for _ in range(3):
                self.env.step(a)
            d = (np.asarray(self.env.eef_pos, dtype=np.float64) - p0) / (0.4 * 3)
            cols.append(d)
            # settling back is not needed (the loop is closed and re-reads)
        self._pos_jac = np.stack(cols, axis=1)  # 3x3: world_dpos = J @ a_xyz
        return self._pos_jac

    def invalidate_calibration(self) -> None:
        """Forget the jacobian (call after a snapshot restore or a policy rollout
        moved the arm far from where the probe was taken)."""
        self._pos_jac = None

    def move_to(
        self,
        xyz: Sequence[float],
        gripper: float | str | None = "hold",
        step_clip: float = 0.02,
        max_steps: int = 200,
        tol: float = 0.012,
    ) -> dict[str, Any]:
        """Closed-loop OSC servo of the eef to a WORLD xyz target.

        gripper="hold" (DEFAULT) maintains the CURRENT finger width -- carry a
        grasped object WITHOUT crushing it (a sustained +1 squeezes small objects
        out) or letting it drop. Pass +1 to actively CLOSE, -1 to actively OPEN.
        """
        target = np.asarray(xyz, dtype=np.float64)
        target_q = float(self.env.gripper_qpos[0])  # finger width to hold
        if self._pos_jac is None:
            self._calibrate_pos_jacobian(gripper=self._resolve_grip(gripper, target_q))
        assert self._pos_jac is not None
        Jinv = np.linalg.pinv(self._pos_jac)
        for i in range(int(max_steps)):
            cur = np.asarray(self.env.eef_pos, dtype=np.float64)
            err = target - cur
            dist = float(np.linalg.norm(err))
            if dist < tol:
                return {
                    "ok": True,
                    "steps": i,
                    "final_dist": dist,
                    "eef": cur.tolist(),
                    "gripper_qpos": round(float(self.env.gripper_qpos[0]), 4),
                }
            step_world = err if dist <= step_clip else err / dist * step_clip
            a_xyz = np.clip(Jinv @ step_world, -1, 1)
            a = self._zero()
            a[0:3] = a_xyz
            a[GRIPPER_INDEX] = self._resolve_grip(gripper, target_q)
            self.env.step(a)
        cur = np.asarray(self.env.eef_pos, dtype=np.float64)
        return {
            "ok": False,
            "steps": int(max_steps),
            "final_dist": float(np.linalg.norm(target - cur)),
            "eef": cur.tolist(),
            "gripper_qpos": round(float(self.env.gripper_qpos[0]), 4),
        }

    def move_delta(
        self,
        dxyz: Sequence[float],
        gripper: float | str | None = "hold",
        step_clip: float = 0.02,
        max_steps: int = 80,
    ) -> dict[str, Any]:
        return self.move_to(
            np.asarray(self.env.eef_pos, dtype=np.float64)
            + np.asarray(dxyz, dtype=np.float64),
            gripper,
            step_clip,
            max_steps,
        )

    def lift(
        self,
        dz: float,
        gripper: float | str | None = "hold",
        step_clip: float = 0.02,
        max_steps: int = 80,
    ) -> dict[str, Any]:
        """ADDED (not upstream): raise (dz>0) / lower (dz<0) the eef by ``dz``
        metres in WORLD z, holding the finger width. A thin wrapper over
        ``move_delta`` -- "lift what you just grasped" is the single most common
        recovery motion and deserves a name the explorer can reach for."""
        return self.move_delta(
            (0.0, 0.0, float(dz)),
            gripper=gripper,
            step_clip=step_clip,
            max_steps=max_steps,
        )

    def rotate_pitch(
        self, target_pitch: float = 0.6, gripper: float = 1.0, n: int = 12
    ) -> dict[str, Any]:
        """Tilt the wrist forward (axis-angle about control-x). Reuses arm drot."""
        per = float(np.clip(target_pitch, -1.5, 1.5)) / n
        for _ in range(n):
            self._step_arm(drot=(per, 0, 0), gripper=gripper, n=1)
        return {"ok": True, "eef": np.asarray(self.env.eef_pos).tolist()}

    def rotate(
        self,
        drot: Sequence[float] = (0.0, 0.0, 0.0),
        gripper: float | str | None = "hold",
        max_rad: float | None = None,
        steps: int = 12,
    ) -> dict[str, Any]:
        """Rotate the wrist by an axis-angle delta, in the SAME 3 dof the policy has.

        The action vector is [dpos(3), drot(3), gripper, base(4), mode]; the policy
        writes all six arm dof every step while these primitives only ever wrote
        dpos, which left the explorer strictly weaker than the thing it is
        supposed to be correcting. ``rotate`` closes that gap.

        ``max_rad`` clamps the axis-angle magnitude SERVER-SIDE, for the same
        reason the translation primitives clamp displacement: a correction is
        meant to be small, and a budget written into the interface cannot be
        talked out of the way a prompt can.
        """
        d = np.asarray(drot, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(d))
        clipped = False
        if max_rad is not None and n > float(max_rad) > 0:
            d = d / n * float(max_rad)
            clipped = True
        # hold must behave like the translation primitives: servo back to the starting aperture (_resolve_grip). The old code fed qpos
        # (about 0.04 when open) straight in as the gripper command; after clipping that is positive = close. In 67 measured rotates,
        # 24 closed an open hand to 4 mm, so the judge's later "open-hand" descent was actually a fist.
        # open / close / numeric values are still passed through explicitly.
        if isinstance(gripper, str) and gripper in ("open", "close"):
            gripper = {"open": -1.0, "close": 1.0}[gripper]
        q0 = float(self.env.gripper_qpos[0])
        steps = max(1, int(steps))
        per = d / steps
        for _ in range(steps):
            g = self._resolve_grip(gripper, q0)
            self._step_arm(drot=tuple(float(v) for v in per), gripper=g, n=1)
        return {
            "ok": True,
            "requested_rad": round(n, 5),
            "applied_rad": round(float(np.linalg.norm(d)), 5),
            "clipped": clipped,
            "eef": np.asarray(self.env.eef_pos).tolist(),
            "gripper_qpos": round(float(self.env.gripper_qpos[0]), 4),
        }

    def set_gripper(self, gripper: float = 1.0, steps: int = 10) -> dict[str, Any]:
        g = self._hold_gripper_val(gripper)
        a = self._zero()
        a[GRIPPER_INDEX] = g
        for _ in range(int(steps)):
            self.env.step(a)
        return {"ok": True, "gripper_qpos": np.asarray(self.env.gripper_qpos).tolist()}

    def close_gripper(self, steps: int = 10) -> dict[str, Any]:
        return self.set_gripper(+1.0, steps=steps)

    def open_gripper(self, steps: int = 10) -> dict[str, Any]:
        return self.set_gripper(-1.0, steps=steps)

    # upstream name, kept so recipes ported from RPent still read
    def release(self, steps: int = 10) -> dict[str, Any]:
        return self.set_gripper(-1.0, steps=steps)

    def scripted_grasp(
        self,
        xyz: Sequence[float],
        approach_z: float = 0.10,
        grasp_z_offset: float = 0.0,
        step_clip: float = 0.02,
    ) -> dict[str, Any]:
        """Open -> hover above target -> descend -> close -> lift. Coarse; the
        Cosmos policy (``call_cosmos``) is the closed-loop grasp for hard objects."""
        t = np.asarray(xyz, dtype=np.float64)
        self.set_gripper(-1.0, steps=4)
        r = self.move_to(t + [0, 0, approach_z], gripper=-1.0, step_clip=step_clip)
        if not r["ok"]:
            return {**r, "stage": "approach"}
        r = self.move_to(
            t + [0, 0, grasp_z_offset], gripper=-1.0, step_clip=0.012, tol=0.01
        )
        if not r["ok"]:
            return {**r, "stage": "descent"}
        self.set_gripper(+1.0, steps=14)
        r = self.lift(approach_z + 0.05, gripper="hold", step_clip=0.015)
        if not r["ok"]:
            return {**r, "stage": "lift"}
        return {
            "ok": True,
            "gripper_qpos": np.asarray(self.env.gripper_qpos).tolist(),
            "eef": np.asarray(self.env.eef_pos).tolist(),
        }


__all__ = ["EnvHandle", "RecoveryPrimitives", "OSC_POS_SCALE", "OSC_ROT_SCALE"]
