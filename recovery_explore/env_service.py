"""Persistent RoboCasa environment service: ONE process, ONE env, HTTP-RPC.

WHY a service at all: building a RoboCasa env (scene generation + MuJoCo compile)
costs minutes, and a restored failure anchor is only meaningful while the very
process that restored it keeps holding it. An interactive explorer that shells out
per step would pay that cost every call and, worse, could not keep the restored
state alive between calls. So: build once, restore into it, keep it, and expose
small verbs over HTTP.

Transport: ``recovery_explore/rpc_transport.py`` (copy-and-adapt of
``vendor/rpent/utils/rpc/*``). ``MainThreadServeMixin`` is MANDATORY here, not a
style choice -- MuJoCo's EGL context is thread-bound, and the HTTP server is a
threading server, so every dispatch must be funnelled back onto the thread that
built the env. See that module's docstring.

The env is built the way THIS repo builds it -- PandaMobile, the official pickled
controller config, our three cameras, the anchor's own layout/style and object
split -- via ``cf_bench.cf_branch_replay.build_env_and_cfg``
(``recovery_explore/snapshot_bridge.py``). It is NOT built the way RPent's
``vendor/rpent/robocasa/env_server.py`` builds one (PandaOmron +
``load_composite_controller_config`` + RoboCasa365 splits): that path targets a
different robot and a robosuite fork we deliberately do not install.

THE MODEL-FACING CONTRACT IS FROZEN (2026-09-11). The guiding rule is: the
model may only ask for what a real robot carrying a deployed VLM would have.
Everything else is harness orchestration or privileged scoring, lives behind a
``harness_*``/orchestration method, and is reachable only from
``cli/harness.py`` -- never from ``cli/rex.py``.

Model-facing endpoints (exposed by ``cli/rex.py``, seven tools):
    read_state()                 -- eef pose, gripper width, step count. NEVER oracle.
    render()                     -- three cameras at a FIXED 512, depth + camera params
    unproject(row, col, camera)  -- pixel -> world xyz
    move_to(xyz, gripper)        -- servo; the servo tuning is fixed here
    lift(dz, gripper)
    gripper(action)
    call_cosmos(num_chunks)      -- prompt/seed/K are the SERVICE's, not the model's
    env_meta()                   -- instruction + bookkeeping, NO object_name

Harness-only endpoints (exposed by ``cli/harness.py``):
    make_failure, restore_snapshot, reset_to_failure, reset_step_counter,
    unproject_batch, harness_env_meta, harness_read_state_privileged,
    harness_oracle_grasp, run_episode_to_failure, resume_episode

Oracle masking: ``read_state`` takes NO
arguments and ALWAYS masks -- the privileged quantities are not computed at all,
so a masked run cannot leak the answer through a log line that "had it but did
not show it". There is no longer a model-side switch to turn masking off: the
unmasked read moved to ``harness_read_state_privileged``, which ``rex.py`` does
not expose. Likewise the grasp predicate (``env._check_grasp`` via
``_grasped_object_names``) moved to ``harness_oracle_grasp``: the model judges a
grasp from gripper width plus before/after images, the way a robot must.

Every business call appends ONE JSON line to the call log -- harness calls
INCLUDED, so a privileged read is always visible in the audit trail.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # package import (normal: run as `python -m recovery_explore.env_service`)
    from recovery_explore import cosmos_call, snapshot_bridge
    from recovery_explore.primitives import RecoveryPrimitives
    from recovery_explore.rpc_transport import (
        MainThreadServeMixin,
        RpcFacade,
        configure_egl_device,
    )
except ImportError:  # pragma: no cover - direct `python env_service.py`
    import cosmos_call  # type: ignore[no-redef]
    import snapshot_bridge  # type: ignore[no-redef]
    from primitives import RecoveryPrimitives  # type: ignore[no-redef]
    from rpc_transport import (  # type: ignore[no-redef]
        MainThreadServeMixin,
        RpcFacade,
        configure_egl_device,
    )

# Our three cameras. Same mapping as
# cf_bench/vlm_correction_rollout.py::CAMERA_ROBOSUITE_NAMES (imported at call
# time rather than duplicated; this constant is only the fallback default so the
# module imports cleanly off the simulation host).
DEFAULT_CAMERA_KEYS = ("primary", "secondary", "wrist")
MAIN_CAMERA = "primary"  # robot0_agentview_left -- the depth/unprojection camera
DEFAULT_PORT = 8410
DEFAULT_RENDER_SIZE = 512

# ---------------------------------------------------------------------------
#  Frozen model-facing constants.
#
#  These were CLI flags. They are constants now because a flag is an invitation
#  to search over it, and none of them is information a robot has:
#    * render size -- a camera has the resolution it has.
#    * servo tuning (step clip / max steps / tolerance) -- controller internals.
#    * K for best-of-K -- a deployment-side budget, not a model decision.
#  Freezing them also keeps every explored episode comparable: a recipe mined at
#  one step_clip is not a recipe at another.
# ---------------------------------------------------------------------------
COSMOS_NUM_CANDIDATES = 4       # best-of-K width, fixed server-side (cosmos family)
MOVE_TO_STEP_CLIP = 0.02        # metres per servo step
MOVE_TO_MAX_STEPS = 200
MOVE_TO_TOL = 0.012             # metres
LIFT_STEP_CLIP = 0.02
LIFT_MAX_STEPS = 80
GRIPPER_STEPS = 10

# Per-action translation/rotation limits, clipped server-side.
# Basis: perturbation experiments measured 3 cm toward the object as best, 5 cm next, and 8 cm+
# falling back to the no-intervention level; first-round human review also kept noting that
# "once the VLM takes over, it moves more than the policy itself". The budget lives in the
# interface rather than the prompt -- a prompt can be argued with, an interface cannot.
# Gripper aperture (proprioception, not oracle): aperture = L2 norm of the two finger qpos. Measured:
# resting 0.0288 m, commanded open up to 0.056 m, holding a yogurt 0.032 m, closed on nothing 0.0007 m.
# Note the aperture while holding something can exceed rest (the object props the fingers apart), so
# **aperture cannot tell "holding" from "open"**; the only certain state is "fully closed = nothing between the fingers".
# GRIP_OPEN_W was first set to 0.045, above rest, and the state machine never fired once.
GRIP_EMPTY_W = 0.012      # below this: fully closed, nothing between the fingers
# Mug tasks (CoffeeSetupMug / CoffeeServeMug) grasp the handle: holding the handle gives 6-12 mm aperture,
# colliding with the 12 mm empty-close threshold. On seed 500, 12 of the 17 episodes the judge broke were
# handle grasps misread as empty, then repeatedly retreated away. A truly empty close goes below 1 mm, so mug tasks use 4 mm.
GRIP_EMPTY_W_MUG = 0.004

def _grip_empty_w_for(task) -> float:
    return GRIP_EMPTY_W_MUG if "Mug" in str(task or "") else GRIP_EMPTY_W
GRIP_NOTEMPTY_W = 0.018   # above this: no longer fully closed (hysteresis band to avoid re-triggering on jitter)
ANCHOR_TOL_M = 0.08       # target within this radius of a "visited point" lifts the per-action translation limit
RECALL_MAX_M = 0.60       # per-action limit once lifted
# Max resets per episode (back to the intervention-start state, with a fresh budget). 0 = no rollback at all --
# used to see "can the model get it right in one go". launch_svc.sh passes it in as an environment variable.
MAX_RESETS = int(os.environ.get("MAX_RESETS", "5"))
BASE_MAX_STEPS = 8        # max consecutive base steps per action
BASE_MAX_CMD = 0.5        # per-action base command magnitude limit

MAX_STEP_M = 0.05          # per-action translation limit (m)
# Wrist contact-force abort threshold (N). If execute_plan sees a value above this mid-plan it stops
# instead of pushing on with the original plan -- every step after a collision was written on a wrong assumption.
CONTACT_FORCE_ABORT_N = float(os.environ.get("CONTACT_FORCE_ABORT_N", "40"))
MAX_STEP_RAD = 0.35        # per-action rotation limit (rad, about 20 degrees)
ROTATE_SUBSTEPS = 12

# Base of the service's OWN Cosmos seed counter. Deliberately far above the
# evaluation schedules in rpc/run_remote_robocasa_*.py (whose alternate seeds are
# cfg.seed + 1_000_000 + episode*10_000 + query, i.e. low millions) so an
# exploration seed can never collide with an evaluation seed.
COSMOS_SEED_ORIGIN = 7_000_000


def _camera_names() -> dict[str, str]:
    from cf_bench.vlm_correction_rollout import CAMERA_ROBOSUITE_NAMES

    return dict(CAMERA_ROBOSUITE_NAMES)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _summarize(value: Any, *, depth: int = 0) -> Any:
    """Log-friendly rendering of an argument: arrays become shape/dtype tags."""
    if isinstance(value, np.ndarray):
        return f"<ndarray {list(value.shape)} {value.dtype}>"
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, str):
        return value if len(value) <= 120 else value[:117] + "..."
    if depth >= 2:
        return f"<{type(value).__name__}>"
    if isinstance(value, Mapping):
        return {str(k): _summarize(v, depth=depth + 1) for k, v in list(value.items())[:12]}
    if isinstance(value, (list, tuple)):
        return [_summarize(v, depth=depth + 1) for v in value[:12]]
    return f"<{type(value).__name__}>"



# ---- Hard constraints on repair plans (per task category, no scene coordinates) -------------------------------
# Evidence (seed 195, four learning batches + n90abc + t42a): 13 of 15 attributable rescues were
# "open hand + short repositioning + hand back to the policy"; the judge closing the gripper itself missed 30-55 times per round and rescued 2;
# after servo_missed/overshoot/move_clipped within an intervention, further Cartesian actions (incl. retreat)
# almost always failed again. These three rules are checked server-side: the whole plan is vetted before execution and returned unchanged on violation.
_BLOWUP_TASK_PREFIXES = ("TurnOnMicrowave", "TurnOffMicrowave", "CoffeePressButton",
                         "CoffeeServeMug", "CoffeeSetupMug", "TurnOnStove", "TurnOffStove")
_BLOWUP_MOVE_TO_MAX_M = 0.06     # in these tasks move_to is short-range only (long moves are the main source of runaway)
_FAULT_STOPS = ("servo_missed", "overshoot", "move_clipped")


class EpisodeStepBudgetExhausted(RuntimeError):
    """The episode's step clock is at its budget while a judge window is charging it.

    Raised by ``_raw_step`` BEFORE the step that would take the episode past
    ``max_steps``, so the judge's own moves never run the episode longer than the
    policy alone may run it. The message contains "budget": ``execute_plan`` maps
    such an exception to stop_reason "budget".
    """


def _repair_gate_check(task: str, plan, policy_chunks_so_far: int, faults_this_window: int,
                       eef_pos=None, resolve_xyz=None):
    """Return (step_index, message) to reject the whole plan; None to accept it.

    R1 PnP: until this intervention has handed back to the policy, the plan may not contain a gripper close -- position, hand back,
       and close only if the policy fails to grasp; every close must be followed by a lift for verification.
    R2 Runaway-prone tasks (microwave buttons, coffee machine, stove knobs): move_to limited to <= 6 cm.
    R3 A fault already occurred in this intervention: later plans may only use gripper / policy (no Cartesian actions, incl. retreat).
    """
    ops = [str(st.get("op")) for st in plan if isinstance(st, dict)]
    if faults_this_window > 0:
        for i, op in enumerate(ops):
            if op not in ("gripper", "policy"):
                return i, ("a step in this intervention already missed, overshot or was clipped; "
                           "from that arm configuration further Cartesian moves (move_to, nudge, lift, "
                           "rotate, retreat) fail the same way and are refused for the rest of this "
                           "intervention. Open the fingers if they are shut on nothing, hand back one "
                           "chunk, or end the intervention as fixed.")
    if str(task).startswith("PnP"):
        for i, st in enumerate(plan):
            if not isinstance(st, dict) or str(st.get("op")) != "gripper":
                continue
            state = str(st.get("state") or st.get("action") or "")
            if state != "close":
                continue
            if policy_chunks_so_far <= 0 and "policy" not in ops[:i]:
                return i, ("on a pick-and-place task the first repair is to place the open fingers on a "
                           "good line and hand back to the policy for the close; a manual close is "
                           "allowed only after a hand-back in this intervention has failed. Replace the "
                           "close with a policy step, or restage and hand back.")
            if "lift" not in ops[i + 1:]:
                return i, ("a manual close must be followed by a lift in the same plan so the grasp is "
                           "verified before anything else happens; add {\"op\": \"lift\", "
                           "\"dz\": 0.04, \"gripper\": \"hold\"} after it.")
    if str(task).startswith(_BLOWUP_TASK_PREFIXES) and eef_pos is not None and resolve_xyz is not None:
        cur = np.asarray(eef_pos, dtype=np.float64)
        for i, st in enumerate(plan):
            if not isinstance(st, dict) or str(st.get("op")) != "move_to":
                continue
            try:
                xyz, _ = resolve_xyz(st.get("target"))
            except Exception:
                continue
            d = float(np.linalg.norm(np.asarray(xyz, dtype=np.float64) - cur))
            if d > _BLOWUP_MOVE_TO_MAX_M:
                return i, (f"on this fixture the Cartesian controller is unstable far from the current "
                           f"pose: a move_to of {d*100:.0f} cm is refused (limit {_BLOWUP_MOVE_TO_MAX_M*100:.0f} cm). "
                           "Use a short lift or nudge and hand back to the policy instead.")
            cur = np.asarray(xyz, dtype=np.float64)
    return None

class _DirectEnvHandle:
    """In-process adapter giving ``primitives.RecoveryPrimitives`` its env surface.

    Upstream (``vendor/rpent/robocasa/primitives.py``) this was an RPC client to a
    separate env process; here the env lives in THIS process, so the handle is a
    thin shim that also keeps the service's observation and step counter current.
    """

    def __init__(self, service: "RecoveryEnvService") -> None:
        self._service = service

    def step(self, flat_action: np.ndarray) -> Any:
        return self._service._raw_step(np.asarray(flat_action, dtype=np.float64))

    @property
    def eef_pos(self) -> np.ndarray:
        return np.asarray(self._service._obs_value("robot0_eef_pos"), dtype=np.float64)

    @property
    def eef_quat(self) -> np.ndarray:
        return np.asarray(self._service._obs_value("robot0_eef_quat"), dtype=np.float64)

    @property
    def gripper_qpos(self) -> np.ndarray:
        return np.asarray(
            self._service._obs_value("robot0_gripper_qpos"), dtype=np.float64
        )


class RecoveryEnvService(MainThreadServeMixin, RpcFacade):
    """Holds exactly one RoboCasa env, restored to one failure anchor."""

    def __init__(
        self,
        *,
        anchors_jsonl: Path | None,
        output_dir: Path,
        run_id: str,
        log_path: Path,
        policy_endpoint: str | None = None,
        render_size: int = DEFAULT_RENDER_SIZE,
        debug_resets: int = 0,
        policy_family: str = "cosmos",
        step_budget_scale: float = 1.0,
    ) -> None:
        self._anchors_jsonl = Path(anchors_jsonl) if anchors_jsonl else None
        self._output_dir = Path(output_dir)
        self._run_id = run_id
        self._log_path = Path(log_path)
        self._policy_endpoint = policy_endpoint
        self._render_size = int(render_size)
        # The policy family sets chunk length, action dimension and best-of-K width. Both families share the protocol (/v1/infer
        # returning InferenceResponse); only these three numbers differ, so one service runs both.
        if policy_family not in ("cosmos", "pi05"):
            raise ValueError(f"unknown policy_family {policy_family!r}")
        self._policy_family = str(policy_family)
        # Step-budget multiplier: TASK_MAX_STEPS is the official benchmark constant; once scaled, results are only comparable within the batch,
        # not against the baseline or earlier experiments. Control and treatment arms share one service, so it applies to both.
        if not 0.1 <= float(step_budget_scale) <= 5.0:
            raise ValueError("step_budget_scale must be in [0.1, 5.0]")
        self._step_budget_scale = float(step_budget_scale)

        self._env: Any = None
        self._cfg: Any = None
        self._env_key: tuple[Any, ...] | None = None
        self._anchor: dict[str, Any] | None = None
        self._failure: Any = None
        self._call_seq: int = 0
        self._instance_nonce: str = f"{os.getpid():05d}{int(time.time()) % 100000:05d}"
        # The judge must see "what the policy itself did". The second-round prompt had only one overview image:
        # the model knew neither which commands the policy issued nor where/at which step the last empty grasp happened,
        # so it guessed from four 256px images and restarted the object search far away every time.
        # Both buffers are cleared in harness_episode_begin and live for one episode.
        self._act_log: list[dict[str, Any]] = []
        self._grasp_events: list[dict[str, Any]] = []
        # Points the end effector has visited. move_to's long-range allowance only honors these: the arm really was there,
        # so going back retraces the path rather than teleporting.
        self._waypoints: list[dict[str, Any]] = []
        # Snapshot taken when the intervention window opens. reset = restore this state and re-issue an action budget,
        # at most MAX_RESETS per episode. It only undoes the VLM's own actions; the policy side is identical to no intervention.
        self._window_snapshot: dict[str, Any] | None = None
        self._last_action_poses: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
        self._resets_used: int = 0
        # Collision records stay on the harness side for post-hoc stats and are never returned to the model (see _finish_action).
        self._collateral_log: list[dict[str, Any]] = []
        self._resets_this_window: int = 0
        # Debug mode: lets the model roll back itself, at most debug_resets times. 0 = off (default).
        # Outside debug the model cannot roll back -- a failed grasp on a real robot cannot be rewound, only rescued from a worse state.
        # Enabling it lets exploration run controlled comparisons on the same failure state (change one variable, try again);
        # the resulting lessons must be labeled as obtained with rollback available.
        self._debug_resets_budget: int = int(debug_resets)
        self._debug_resets_used: int = 0
        # Round budget (rounds experiment): the harness uses harness_round_begin to roll back and set this round's action cap;
        # each model-side primitive (nudge/move_to/lift/gripper/call_cosmos) costs one. Once spent, calls are rejected
        # and the harness decides to roll back into the next round -- the model cannot roll back and does not see who triggers round boundaries.
        self._round_budget: int | None = None
        self._round_used: int = 0
        self._round_index: int = 0
        # End-effector pose and gripper aperture at the moment this intervention took over. The retreat primitive returns here -- a
        # "soft undo" a real robot can also do: put the hand back where the policy itself had been, without touching the scene.
        self._takeover_pose: dict[str, Any] | None = None
        # Number of servo faults in this intervention (servo_missed / overshoot / move_clipped), used by _repair_gate.
        self._window_faults: int = 0
        # End-effector pose/gripper at the start of each policy chunk (this episode), plus the approach pose before the latest empty grasp.
        # retreat(to="pre_grasp") returns to the latter: a point the policy reached during a normal approach, so handing back
        # from there lets the policy redo its own grasp; the takeover point is where it withdrew after failing.
        self._pose_hist: list[dict[str, Any]] = []
        self._pre_grasp_pose: dict[str, Any] | None = None
        # Token for closing the intervention window. The harness hands out a random string on open and must present it on close.
        # Without it, "close budget" would be a backdoor the model could call itself: one call zeroes the budget,
        # the privilege gate in _dispatch goes away, and the oracle becomes readable. See _dispatch.
        self._round_token: str | None = None
        # Judging-round lock. The budget is closed during judging, so the _round_budget gate alone leaves it unguarded --
        # the judge still has Bash and the endpoint address then. The driver locks/unlocks around each judging call,
        # and unlocking needs the token. While locked, as while the budget is open, privileged methods are rejected.
        self._judge_locked: bool = False
        self._judge_token: str | None = None
        # judge experiment: call_cosmos is disabled during interventions. Allowing it would give the treatment arm free extra
        # policy steps (call_cosmos does not advance the episode's step clock itself; each step in a window is counted by
        # _raw_step, but it is still a policy call off the seed schedule), making the comparison with the control arm unfair.
        self._allow_cosmos: bool = True
        # While a judge window is open on a harness episode, every simulator step is charged to
        # that episode's step clock, the judge's own moves included (see _raw_step).
        # harness_open_action_budget turns it on and close turns it off. It stays on through the
        # policy op of execute_plan: harness_advance advances the same clock step by step and
        # writes back the same total, so nothing is counted twice. The control arm never opens a
        # window, so its behaviour is unchanged.
        self._charge_clock: bool = False
        self._restored: snapshot_bridge.RestoredAnchor | None = None
        self._raw_obs: Mapping[str, Any] | None = None
        self._policy_obs: Mapping[str, np.ndarray] | None = None
        self._step_count = 0
        self._primitives: RecoveryPrimitives | None = None
        # The service owns the Cosmos RNG draw; see _next_cosmos_seed_base.
        self._cosmos_seed_cursor = 0
        # Where a generated failure sits in its episode's step budget. 0 for a
        # scripted failure (its setup steps are not episode steps); the real
        # committed timestep for one produced by run_episode_to_failure.
        self._failure_origin_timestep = 0
        # Bookkeeping for the full-episode harness loop (task/seed/episode index
        # /instruction/query cursor), so resume_episode can continue the SAME
        # episode rather than starting a new deterministic schedule.
        self._episode: dict[str, Any] | None = None

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        super().__init__()

    # ---- RPC plumbing -----------------------------------------------------

    def _register_rpc(self) -> None:
        self._rpc.update(
            {
                # -- model-facing (cli/rex.py) ------------------------------
                "read_state": self.read_state,
                "render": self.render,
                "unproject": self.unproject,
                "move_to": self.move_to,
                "lift": self.lift,
                "gripper": self.gripper,
                "rotate": self.rotate,
                "reset_window": self.reset_window,
                "execute_plan": self.execute_plan,
                "call_cosmos": self.call_cosmos,
                "env_meta": self.env_meta,
                # -- harness-only (cli/harness.py) --------------------------
                # Orchestration: building/rewinding the state under study.
                "restore_snapshot": self.restore_snapshot,
                "make_failure": self.make_failure,
                "reset_to_failure": self.reset_to_failure,
                "nudge": self.nudge,
                "debug_reset": self.debug_reset,
                "harness_round_begin": self.harness_round_begin,
                "harness_episode_begin": self.harness_episode_begin,
                "harness_release_env": self.harness_release_env,
                "harness_quit": self.harness_quit,
                "harness_open_action_budget": self.harness_open_action_budget,
                "harness_close_action_budget": self.harness_close_action_budget,
                "harness_judge_lock": self.harness_judge_lock,
                "harness_judge_unlock": self.harness_judge_unlock,
                "harness_advance": self.harness_advance,
                "harness_round_status": self.harness_round_status,
                "reset_step_counter": self.reset_step_counter,
                "run_episode_to_failure": self.run_episode_to_failure,
                "resume_episode": self.resume_episode,
                # Privileged scoring: the oracle the model must never see.
                "harness_env_meta": self.harness_env_meta,
                "harness_read_state_privileged": self.harness_read_state_privileged,
                "harness_oracle_grasp": self.harness_oracle_grasp,
                # Batch unprojection is a harness convenience (a model asks
                # about a handful of pixels, one at a time, like a VLM would).
                "unproject_batch": self.unproject_batch,
            }
        )

    def _log(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, default=str, sort_keys=True)
        with self._log_path.open("a") as stream:
            stream.write(line + "\n")

    # Methods always rejected while the model holds control. The judge has Bash and can bypass cli/rex.py to
    # curl this endpoint directly, so "the prompt never mentioned it" is no protection -- it must be blocked server-side,
    # otherwise the claim "proprioception only, throughout" cannot be verified.
    #
    # ``harness_close_action_budget`` is deliberately NOT in this set: the gate condition is "budget open",
    # so blocking the budget close would mean an opened window can never be closed (the driver's close-budget would hit
    # PermissionError, and the next chunk's harness_advance would stall too). It is protected by
    # a token instead -- see self._round_token.
    _PRIVILEGED_METHODS = frozenset({
        "harness_read_state_privileged", "harness_oracle_grasp", "harness_env_meta",
        "harness_advance", "harness_episode_begin", "harness_release_env",
        "harness_open_action_budget", "harness_judge_lock",
        "harness_round_begin", "harness_round_status", "restore_snapshot",
        "make_failure", "reset_to_failure", "run_episode_to_failure",
        "resume_episode", "debug_reset", "unproject_batch", "reset_step_counter",
        "harness_quit",
    })

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        """Time every business call and append one JSON log line.

        ``healthz`` is excluded on purpose: ``wait_for_ready`` polls it twice a
        second while the env builds, and a log full of health probes buries the
        actual call history.
        """
        if method == "healthz":
            return super()._dispatch(method, args, kwargs)
        # Action budget open = the model is driving; judging lock held = the model is looking. In either case
        # every privileged method is rejected.
        if (self._round_budget or self._judge_locked) and method in self._PRIVILEGED_METHODS:
            raise PermissionError(
                f"{method} is a harness-only method and is refused while the model "
                "holds the action budget; the model may use only proprioception, "
                "images and the movement primitives."
            )
        started = time.monotonic()
        record: dict[str, Any] = {
            "ts": _utc_now(),
            "method": method,
            "args": _summarize(list(args)),
            "kwargs": _summarize(dict(kwargs)),
            "anchor_id": self._restored.anchor_id if self._restored else None,
            "step_count": self._step_count,
        }
        try:
            result = super()._dispatch(method, args, kwargs)
        except Exception as exc:
            record.update(
                {
                    "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            self._log(record)
            raise
        record.update(
            {
                "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
                "ok": True,
                "error": None,
            }
        )
        self._log(record)
        return result

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    # ---- internal env helpers --------------------------------------------

    def _require_env(self) -> Any:
        """The precondition is simply that an env exists.

        Deliberately NOT gated on ``self._restored``: that field is set only by
        ``restore_snapshot``, so gating on it locked out the whole ``make_failure``
        path, which supplies an env of its own (and, mid-construction, has not yet
        produced a failure state to point at either).
        """
        if self._env is None:
            raise RuntimeError(
                "no env loaded: the harness must call run_episode_to_failure "
                "(preferred), make_failure, or restore_snapshot first"
            )
        return self._env

    def _current_task(self) -> str:
        """Task name, whichever entry point produced the current state."""
        if self._restored is not None:
            return str(self._restored.task)
        if self._failure is not None:
            return str(self._failure.task)
        raise RuntimeError("no state loaded: call make_failure or restore_snapshot")

    def _origin_timestep(self) -> int:
        """Where this state sits in the episode's step budget.

        A restored anchor carries the timestep it was captured at. A SCRIPTED
        failure starts its own budget at 0 -- the steps spent scripting the miss
        are setup, not part of the recovery the explorer is given. A failure
        produced by ``run_episode_to_failure`` carries the real committed
        timestep it was reached at, because there the preceding steps ARE the
        episode and must count against ``TASK_MAX_STEPS``.
        """
        if self._restored is not None:
            return int(self._restored.query_timestep)
        return int(self._failure_origin_timestep)

    def _current_instruction(self) -> str:
        """The CURRENT episode's own task instruction, verbatim.

        This is what ``call_cosmos`` sends as the prompt. The model does not get
        to choose it: the Cosmos policy only accepts instructions present in its
        precomputed T5 cache, so an arbitrary prompt 500s (see
        memory: "Cosmos policy prompts must be T5-cached"). A deployed VLM does
        receive the task instruction, so handing it over in ``env_meta`` is fair;
        letting the model rewrite it is not -- it would be tuning the policy's
        conditioning, which no robot can do at run time.
        """
        if self._restored is not None:
            return str(self._restored.task_description)
        if self._failure is not None:
            return str(self._failure.instruction)
        raise RuntimeError("no state loaded: call make_failure or run_episode_to_failure")

    def _next_cosmos_seed_base(self, count: int) -> int:
        """Reserve ``count`` consecutive Cosmos seeds from the service's counter.

        WHY the model cannot pass a seed: a seed is not information a robot has.
        Exposing it invites "reroll until lucky" -- retry the same state with a
        new draw until the policy happens to succeed -- which is not a
        transferable skill and confounds attribution (we could no longer tell a
        recovery recipe that works from a lucky sample). The counter is
        monotonic, so every call gets a fresh, non-overlapping block and the run
        stays reproducible as a SEQUENCE even though no single call names a seed.
        """
        base = COSMOS_SEED_ORIGIN + self._cosmos_seed_cursor
        self._cosmos_seed_cursor += max(1, int(count))
        return base

    def _next_call_seq(self) -> int:
        self._call_seq += 1
        return self._call_seq

    def _state_id(self) -> str:
        """Stable id for the state being explored, for request ids and logs.

        A restored anchor has an anchor_id; a generated failure has none, so it
        is identified by task+mode instead.
        """
        if self._restored is not None:
            return str(self._restored.anchor_id)
        if self._failure is not None:
            return f"{self._failure.task}-{self._failure.mode}"
        return "unloaded"

    def _obs_value(self, key: str) -> Any:
        if self._raw_obs is None:
            raise RuntimeError("no observation yet: call restore_snapshot first")
        if key not in self._raw_obs:
            raise KeyError(f"RoboCasa observation is missing {key!r}")
        return self._raw_obs[key]

    def _refresh_observation(self) -> None:
        """Re-read the env's observation into the service's caches."""
        from rpc.run_remote_robocasa_collect import prepare_observation

        env = self._env
        getter = getattr(env, "_get_observations", None)
        if not callable(getter):
            raise RuntimeError("RoboCasa environment lacks _get_observations")
        try:
            raw = getter(force_update=True)
        except TypeError:
            raw = getter()
        self._raw_obs = raw
        self._policy_obs = prepare_observation(raw, self._cfg.flip_images)

    def _episode_steps_left(self) -> int | None:
        """Steps left on the episode's clock (the budget harness_advance enforces); None without one."""
        ep = self._episode
        if not ep:
            return None
        budget = int(ep.get("max_steps") or self._task_max_steps(str(ep["task"])))
        return budget - int(ep["failure_committed_timestep"])

    def _raw_step(self, flat_action: np.ndarray) -> Any:
        """Step the env with an ALREADY env-dimensioned action, keeping caches.

        While a judge window charges the clock (``_charge_clock``), the step is also
        one step of the episode, exactly like a policy step in harness_advance, and
        it is refused once the episode's step budget is used up.
        """
        from rpc.run_remote_robocasa_collect import prepare_observation

        env = self._require_env()
        charge = bool(self._charge_clock) and self._episode is not None
        if charge:
            left = self._episode_steps_left()
            if left is not None and left <= 0:
                ep = self._episode
                raise EpisodeStepBudgetExhausted(
                    f"the episode's step budget is used up ({int(ep['failure_committed_timestep'])} of "
                    f"{int(ep['failure_committed_timestep']) + left} steps, your own moves included); "
                    "nothing more can run in this episode")
        raw, _reward, _done, _info = env.step(np.asarray(flat_action, dtype=np.float64))
        self._raw_obs = raw
        self._policy_obs = prepare_observation(raw, self._cfg.flip_images)
        self._step_count += 1
        if charge:
            ep = self._episode
            ep["failure_committed_timestep"] = int(ep["failure_committed_timestep"]) + 1
            # harness_advance keeps these two equal at the end of every call; so does this.
            self._failure_origin_timestep = int(ep["failure_committed_timestep"])
        return raw

    def _primitive_driver(self) -> RecoveryPrimitives:
        if self._primitives is None:
            env = self._require_env()
            self._primitives = RecoveryPrimitives(
                _DirectEnvHandle(self), action_dim=int(env.action_dim)
            )
        return self._primitives

    # ---- endpoints --------------------------------------------------------

    def harness_env_meta(self) -> dict[str, Any]:
        """FULL description of what this process holds -- ``object_name`` included.

        Harness-only. ``env_meta`` below is the model-facing subset.
        """
        return {
            "run_id": self._run_id,
            "anchors_jsonl": str(self._anchors_jsonl) if self._anchors_jsonl else None,
            "anchor_id": self._restored.anchor_id if self._restored else None,
            "state_id": self._state_id(),
            "debug_resets_left": max(0, self._debug_resets_budget - self._debug_resets_used),
            "actions_left_this_round": (
                None if self._round_budget is None
                else max(0, self._round_budget - self._round_used)
            ),
            "mode": (
                "anchor" if self._restored is not None
                else (self._failure.mode if self._failure is not None else None)
            ),
            "task": (
                self._restored.task if self._restored
                else (self._failure.task if self._failure else None)
            ),
            "instruction": (
                self._restored.task_description if self._restored
                else (self._failure.instruction if self._failure else None)
            ),
            "object_name": (
                self._restored.object_name if self._restored
                else (self._failure.object_name if self._failure else None)
            ),
            "state_sha256": self._restored.state_sha256 if self._restored else None,
            "origin_timestep": self._origin_timestep() if (
                self._restored is not None or self._failure is not None
            ) else None,
            "episode": dict(self._episode) if self._episode else None,
            "step_count": self._step_count,
            "render_size": self._render_size,
            "policy_endpoint": self._policy_endpoint,
            "policy_family": self._policy_family,
            "step_budget_scale": self._step_budget_scale,
            "action_dim": int(self._env.action_dim) if self._env is not None else None,
            "cosmos_seed_cursor": int(self._cosmos_seed_cursor),
        }

    def env_meta(self) -> dict[str, Any]:
        """What the MODEL is allowed to know about this process.

        ``object_name`` is deliberately absent. A deployed VLM is given the task
        instruction -- "pick the sponge from the counter and place it in the
        cabinet" -- and must find the referent in the image itself; it is never
        handed the simulator's category label for the target body. Leaving the
        label in turned every perception failure into a lookup, which is exactly
        the skill we are trying to measure. The full record stays available to
        the harness via ``harness_env_meta``.
        """
        full = self.harness_env_meta()
        return {
            key: value
            for key, value in full.items()
            if key not in ("object_name", "state_sha256", "episode", "anchors_jsonl")
        }

    def restore_snapshot(
        self,
        anchor_id: str,
        anchors_jsonl: str | None = None,
    ) -> dict[str, Any]:
        """Restore a failure anchor's snapshot, hash-verified, into this env.

        The state_sha256 check is NOT re-implemented here: it lives inside
        ``cf_bench.snapshot_io.restore_simulator_snapshot`` (which raises
        "restored flattened simulator state hash differs from anchor") and is
        reached through ``snapshot_bridge.restore``. A mismatch propagates to the
        caller as an RPC error -- never swallowed, never downgraded to a warning.

        ONE process holds ONE env: a second anchor from a different
        (task, seed, episode_index) is refused, because honouring it would mean
        tearing down and rebuilding the env, which is exactly the cost this
        service exists to avoid. Launch another process on the next port instead.
        Re-restoring the SAME anchor is expected and cheap -- every rollout
        mutates the env, so each attempt starts with a fresh restore.
        """
        path = Path(anchors_jsonl) if anchors_jsonl else self._anchors_jsonl
        if path is None:
            raise ValueError(
                "no anchors.jsonl: pass anchors_jsonl or launch with --anchors-jsonl"
            )
        anchor = snapshot_bridge.find_anchor(path, anchor_id)
        key = (
            str(anchor["task"]),
            int(anchor["env_seed"]),
            int(anchor["episode_index"]),
            str(anchor.get("obj_instance_split", "B")),
        )
        if self._env is None:
            self._env, self._cfg = snapshot_bridge.build_env(
                anchor, run_id=self._run_id, output_dir=self._output_dir
            )
            self._env_key = key
        elif key != self._env_key:
            raise RuntimeError(
                f"this process holds an env for {self._env_key}; anchor {anchor_id!r} "
                f"needs {key}. Start a second env_service on the next port."
            )

        restored = snapshot_bridge.restore(self._env, self._cfg, anchor)
        self._anchor = dict(anchor)
        self._restored = restored
        self._policy_obs = restored.policy_observation
        self._refresh_observation()
        self._step_count = 0
        # A restore teleports the arm; the jacobian was calibrated somewhere else.
        if self._primitives is not None:
            self._primitives.invalidate_calibration()
        return {
            "anchor_id": restored.anchor_id,
            "task": restored.task,
            "instruction": restored.task_description,
            "object_name": restored.object_name,
            "state_sha256": restored.state_sha256,
            "origin_timestep": restored.query_timestep,
            "step_count": self._step_count,
        }

    def make_failure(
        self,
        task: str,
        seed: int = 901,
        episode_index: int = 0,
        mode: str = "scripted",
        miss_offset_m: float = 0.035,
        obj_instance_split: str = "B",
    ) -> dict[str, Any]:
        """Build a fresh env and drive it into a REAL failure state, in-process.

        This is the alternative to ``restore_snapshot`` and the one to prefer.
        Cross-process anchor restore cannot reproduce the object instance (see
        ``failure_factory``'s module docstring: layout/style reproduce 8/8, the
        object 0/8, on BOTH the 2026-09-02 train pool and the 2026-09-10 test
        pool), which makes a restored anchor physically meaningless. Generating
        the failure here sidesteps that entirely -- and removes the anchor pool,
        the frozen list, and the whole test-set-contamination question with it.

        The captured state is kept in memory; ``reset_to_failure`` rewinds to it
        as many times as the explorer wants.
        """
        from . import failure_factory as ff
        from .primitives import RecoveryPrimitives

        if self._env is not None:
            raise RuntimeError(
                "this process already holds an env; make_failure builds a fresh one. "
                "Start another env_service on the next port."
            )
        self._env, self._cfg = ff.build_fresh_env(
            task,
            seed=int(seed),
            episode_index=int(episode_index),
            obj_instance_split=str(obj_instance_split),
            run_id=self._run_id,
            output_dir=self._output_dir,
            policy_family=self._policy_family,
        )
        self._env_key = (str(task), int(seed), int(episode_index), str(obj_instance_split))
        self._primitives = RecoveryPrimitives(
            _DirectEnvHandle(self), action_dim=int(self._env.action_dim)
        )
        self._refresh_observation()

        if mode != "scripted":
            raise ValueError(
                f"mode={mode!r} not implemented yet; 'policy' needs a Cosmos "
                "policy server and is the next step"
            )
        state = ff.make_scripted_failure(
            self._env, self._cfg, self._primitives,
            task=task, miss_offset_m=float(miss_offset_m),
        )
        self._failure = state
        # A scripted miss is not part of any episode: its setup steps are not
        # episode steps, so the recovery budget starts at 0 and there is no
        # episode to resume. run_episode_to_failure sets both of these properly.
        self._failure_origin_timestep = 0
        self._episode = None
        self._refresh_observation()
        self._step_count = 0
        self._primitives.invalidate_calibration()
        return {
            "task": state.task,
            "instruction": state.instruction,
            "object_name": state.object_name,
            "mode": state.mode,
            "detail": state.detail,
            "step_count": self._step_count,
        }

    def debug_reset(self) -> dict[str, Any]:
        """Model-side rollback -- only available when the service is started with --debug-resets N, with a count cap.

        Off by default. When off, real-robot semantics hold: a failed grasp cannot be rewound, only rescued from the current (worse)
        state. When on, it serves exploration: change one variable at a time on the same failure state as a controlled comparison.

        Rejects once the budget is spent rather than silently degrading -- otherwise the model would believe it rolled back when it did not,
        and every later observation would be misattributed.
        """
        if self._debug_resets_budget <= 0:
            raise RuntimeError(
                "reset is not available: this service was not started with --debug-resets. "
                "A real robot cannot rewind; continue from the current state."
            )
        if self._debug_resets_used >= self._debug_resets_budget:
            raise RuntimeError(
                f"debug reset budget exhausted ({self._debug_resets_used}/"
                f"{self._debug_resets_budget}); continue from the current state."
            )
        self._debug_resets_used += 1
        out = self.reset_to_failure()
        out["debug_resets_used"] = self._debug_resets_used
        out["debug_resets_left"] = self._debug_resets_budget - self._debug_resets_used
        return out

    # ---- Per-action snapshots and contact reporting during an intervention ---------------------------------------
    # Human review twice recorded "the intervention knocked over the container next to it". The model needs to know it hit
    # something, but not by reading all rigid-body poses (a real robot has no such sensor) -- what it gets back is the wrist
    # contact force, a single proprioceptive quantity; pose differences stay on the harness side for post-hoc stats.
    # The fallback is reset_window: return to before this intervention started; corrections from earlier windows are unaffected.

    def _wrist_contact_force(self) -> float:
        """Magnitude of external contact force on the gripper body (on the order of newtons).

        This is **proprioception**, not privileged information: a real wrist F/T sensor provides exactly this quantity.
        It replaces the old "all rigid-body pose difference" -- that read sim.data.body_xpos,
        which no real sensor can provide, so reporting it to the model leaked internal simulator state.
        Touching, knocking over and squeezing all leave a spike here, enough to serve as an "I just hit something" signal.
        """
        env = self._require_env()
        try:
            m, d = env.sim.model, env.sim.data
            total = 0.0
            for name in ("gripper0_right_right_finger", "gripper0_right_left_finger",
                         "gripper0_right_eef", "robot0_right_hand"):
                try:
                    bid = m.body_name2id(name)
                except Exception:
                    continue
                total = max(total, float(np.linalg.norm(np.asarray(d.cfrc_ext[bid])[3:6])))
            return round(total, 2)
        except Exception:
            return float("nan")

    def _free_body_poses(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Poses of all rigid bodies with a free joint.

        **For post-hoc harness analysis only; never return this to the model.** No real sensor
        provides the poses of all objects in the scene; sending them to the judge as an action return value would let it decide from
        internal simulator state and invalidate the experimental conclusions. Rounds three and four before 2026-09-14
        did exactly this (the collateral field); it has been removed.
        """
        env = self._require_env()
        m, d = env.sim.model, env.sim.data
        out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for b in range(int(m.nbody)):
            n = int(m.body_jntnum[b])
            if n == 1 and int(m.jnt_type[int(m.body_jntadr[b])]) == 0:  # mjJNT_FREE
                out[b] = (np.asarray(d.body_xpos[b]).copy(), np.asarray(d.body_xquat[b]).copy())
        return out

    @staticmethod
    def _tilt_deg(q0: np.ndarray, q1: np.ndarray) -> float:
        """Angle (degrees) between the body z axes of two orientations. Above 45 degrees it has basically fallen over."""
        def zax(q: np.ndarray) -> np.ndarray:
            w, x, y, z = (float(v) for v in q)
            return np.asarray([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        c = float(np.clip(float(np.dot(zax(q0), zax(q1))), -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    def _grasped_body_ids(self) -> set[int]:
        """Bodies currently held. Something held moving with the hand is intended, not a collision."""
        try:
            from rpc.run_remote_robocasa_collect import _grasped_object_names as _gon
            env = self._require_env()
            ids = getattr(env, "obj_body_id", {}) or {}
            return {int(ids[n]) for n in set(_gon(env)) if n in ids}
        except Exception:
            return set()

    def _push_undo(self, name: str) -> None:
        """Record rigid-body poses in the scene before an action; compared afterwards to write what was hit into the return value."""
        try:
            self._last_action_poses = self._free_body_poses()
        except Exception:
            self._last_action_poses = None

    def _finish_action(self, result: dict[str, Any]) -> dict[str, Any]:
        """Compare free rigid bodies after an action and write what was hit into the return value."""
        if self._round_budget is None or self._last_action_poses is None:
            return result
        # The model only gets proprioception back: wrist contact force. Real robots have this sensor.
        out = {**result,
               "wrist_contact_force": self._wrist_contact_force(),
               "actions_used": int(self._round_used),
               "actions_budget": int(self._round_budget)}
        # Actual object displacement is recorded only on the harness side for post-hoc collision rates; never returned.
        try:
            before = self._last_action_poses or {}
            after = self._free_body_poses()
            held = self._grasped_body_ids()
            worst_move = worst_tilt = 0.0
            for b, (p1, q1) in after.items():
                if b in held:
                    continue
                p0, q0 = before.get(b, (p1, q1))
                worst_move = max(worst_move, float(np.linalg.norm(np.asarray(p1) - np.asarray(p0))) * 100.0)
                worst_tilt = max(worst_tilt, self._tilt_deg(np.asarray(q0), np.asarray(q1)))
            self._collateral_log.append({"t": int(self._step_count),
                                         "moved_cm": round(worst_move, 1),
                                         "tilt_deg": round(worst_tilt)})
        except Exception:
            pass
        return out

    def reset_window(self) -> dict[str, Any]:
        """Return to the state at the start of this intervention and re-issue a full action budget. At most MAX_RESETS per episode.

        This is the only "start over" mechanism in the judge experiment. It undoes only the VLM's own actions -- the policy state
        is identical to "never intervened" and the policy gets no extra steps, so the control arm is unaffected.
        The model can call it itself (on noticing it knocked something over), and the harness also calls it when the round ends with
        "not fixed, start over". Both paths use the same function and the same counter.
        """
        from . import failure_factory as ff

        env = self._require_env()
        if self._round_budget is None or self._round_budget <= 0:
            return {"ok": False, "error": "not inside an intervention window; nothing to rewind"}
        if self._resets_used >= MAX_RESETS:
            return {"ok": False, "error": f"no resets left this episode (used {MAX_RESETS})",
                    "resets_used": int(self._resets_used)}
        entry = self._window_snapshot
        if not entry or entry.get("snapshot") is None:
            return {"ok": False, "error": "the snapshot for this window was not captured; cannot rewind"}
        obs = ff.restore(env, self._cfg, entry["snapshot"])
        self._policy_obs = dict(obs)
        self._refresh_observation()
        self._step_count = int(entry.get("step_count", self._step_count))
        self._resets_used += 1
        self._resets_this_window += 1
        self._round_used = 0
        self._round_policy_chunks = 0
        self._window_faults = 0
        self._last_action_poses = None
        if self._primitives is not None:
            self._primitives.invalidate_calibration()
        return {"ok": True,
                "note": "rewound to the start of this intervention; action budget refilled",
                "actions_budget": int(self._round_budget), "actions_used": 0,
                "resets_used": int(self._resets_used),
                "resets_left": int(MAX_RESETS - self._resets_used),
                # the episode clock is not rewound: the driver prints these in the replan prompt
                **self._policy_clock()}

    def _spend_action(self, name: str) -> None:
        """Charge one model-side action against the current round budget (if any)."""
        if self._round_budget is None:
            return
        if self._round_used >= self._round_budget:
            raise RuntimeError(
                f"round action budget exhausted ({self._round_used}/{self._round_budget}): "
                "this round is over. Stop calling actions and write your summary."
            )
        self._round_used += 1
        self._push_undo(name)

    def harness_open_action_budget(
        self, budget: int, allow_cosmos: bool = True, round_index: int | None = None,
        control_token: str | None = None,
    ) -> dict[str, Any]:
        """HARNESS ONLY: open the action budget only, **without rolling back** (the judge experiment's intervention window).

        The only difference from ``harness_round_begin`` is that it does not call ``reset_to_failure``: in the judge
        experiment the intervention happens mid-trajectory, so rolling back would also erase what needs fixing.
        """
        self._round_budget = int(budget)
        self._round_used = 0
        self._round_policy_chunks = 0
        self._round_token = str(control_token) if control_token else None
        self._allow_cosmos = bool(allow_cosmos)
        self._round_index = int(round_index) if round_index is not None else self._round_index + 1
        self._resets_this_window = 0
        self._last_action_poses = None
        self._window_snapshot = None
        # From here until close, every step the judge takes is a step of the episode.
        self._charge_clock = self._episode is not None
        from . import failure_factory as ff
        ep = self._episode or {}
        try:
            self._window_snapshot = {
                "action": "window_open",
                "snapshot": ff.capture(
                    self._require_env(), self._cfg, task=str(ep.get("task", "")),
                    instruction=str(ep.get("instruction", "")),
                    object_name=None, mode="reset", detail={"action": "window_open"},
                ),
                "poses": self._free_body_poses(),
                "step_count": int(self._step_count),
            }
        except Exception:
            self._window_snapshot = None
        self._takeover_pose = self._capture_takeover_pose()
        self._window_faults = 0
        # Note: do not return harness_round_status() here -- it carries the privileged success predicate
        # (harness_oracle_grasp). The return value of opening a window is only one driver variable away from the model,
        # so return only this window's bookkeeping.
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
        }

    def _capture_takeover_pose(self) -> dict[str, Any] | None:
        """Record the end-effector position, orientation and gripper aperture at takeover, for retreat."""
        try:
            from rpc.run_remote_robocasa_collect import _gripper_aperture
            return {
                "eef_pos": [float(v) for v in np.asarray(self._obs_value("robot0_eef_pos"))],
                "eef_quat": [float(v) for v in np.asarray(self._obs_value("robot0_eef_quat"))],
                "aperture_m": float(_gripper_aperture(self._policy_obs)),
                "step_count": int(self._step_count),
            }
        except Exception:
            return None

    def retreat(self, to: str = "takeover") -> dict[str, Any]:
        """Return the hand to its pose at the takeover of this intervention and restore the gripper to its takeover aperture, without touching the scene.

        This is the "hand back as found" of non-rollback runs: after a failed repair, stop repairing from the runaway spot and instead retreat
        to a point the policy itself reached, then hand back. Difference from reset_window: only the arm moves, not objects,
        and the step clock is not rewound. Position uses a closed-loop servo (the takeover point was visited this episode, so the per-step limit does not apply);
        orientation rotates back step by step from the axis-angle difference; if it cannot (residual stops shrinking) it stops and reports the residual.
        """
        self._require_env()
        if to == "pre_grasp":
            pg = self._pre_grasp_pose
            if not pg:
                raise RuntimeError("no pre-grasp pose recorded this episode (no empty closure yet); "
                                   "use retreat without 'to', or a move_to")
            pose = {"eef_pos": pg["eef"], "eef_quat": pg["quat"], "aperture_m": 0.05}
        else:
            to = "takeover"
            pose = self._takeover_pose
            if not pose:
                raise RuntimeError("no takeover pose recorded for this intervention; retreat is unavailable")
        self._spend_action("retreat")
        drv = self._primitive_driver()
        was_open = float(pose["aperture_m"]) >= GRIP_EMPTY_W
        grip = -1.0 if was_open else 1.0
        # Release first: undoing means dropping anything picked up during the repair (if the hand was open at takeover).
        if was_open:
            drv.open_gripper(steps=GRIPPER_STEPS)
        cur = np.asarray(self._obs_value("robot0_eef_pos"), dtype=np.float64)
        target = np.asarray(pose["eef_pos"], dtype=np.float64)
        req = float(np.linalg.norm(target - cur))
        mv = drv.move_to(target, gripper=grip, step_clip=MOVE_TO_STEP_CLIP,
                         max_steps=MOVE_TO_MAX_STEPS * 3, tol=MOVE_TO_TOL)
        self._refresh_observation()
        rot_before, rot_after = self._retreat_rotation(np.asarray(pose["eef_quat"], dtype=np.float64), grip)
        self._refresh_observation()
        out = {
            "ok": bool(mv.get("ok", False)),
            "final_dist": mv.get("final_dist"),
            "final_dist_m": (round(float(mv["final_dist"]), 4) if mv.get("final_dist") is not None else None),
            "requested_dist_m": round(req, 4),
            "rot_before_deg": rot_before,
            "rot_after_deg": rot_after,
            "fingers": "open" if was_open else "close",
            "to": to,
            "eef": np.asarray(self._obs_value("robot0_eef_pos")).tolist(),
            "gripper_qpos": mv.get("gripper_qpos"),
            "step_count": int(self._step_count),
        }
        return self._finish_action(out)

    def _retreat_rotation(self, target_quat: np.ndarray, grip: float) -> tuple[float | None, float | None]:
        """Rotate the wrist back to target_quat (xyzw) in a few steps, stopping if the residual stops shrinking. Returns (before, after) residual in degrees."""
        try:
            from scipy.spatial.transform import Rotation as R
        except Exception:
            return None, None
        drv = self._primitive_driver()

        def residual() -> np.ndarray:
            q = np.asarray(self._obs_value("robot0_eef_quat"), dtype=np.float64)
            err = R.from_quat(target_quat) * R.from_quat(q).inv()
            return err.as_rotvec()

        try:
            rv = residual()
        except Exception:
            return None, None
        before = float(np.degrees(np.linalg.norm(rv)))
        last = float(np.linalg.norm(rv))
        for _ in range(4):
            if last < 0.05:
                break
            drv.rotate(rv, gripper=grip, max_rad=MAX_STEP_RAD, steps=ROTATE_SUBSTEPS)
            self._refresh_observation()
            rv = residual()
            now = float(np.linalg.norm(rv))
            if now > last * 0.9:   # not converging: wrong frame or blocked by structure, stop rotating
                break
            last = now
        return round(before, 1), round(float(np.degrees(last)), 1)

    def harness_close_action_budget(self, control_token: str | None = None) -> dict[str, Any]:
        """HARNESS ONLY: close the intervention window; the action interface returns to its normal (unavailable) state.

        This method cannot be in ``_PRIVILEGED_METHODS`` (the gate condition is "budget open", so blocking it would mean the window
        could never be closed), so it uses a token instead: the token handed out on open must be presented again on close.
        The model never gets the token, so it cannot close the privilege gate itself.
        """
        if self._round_token and str(control_token or "") != self._round_token:
            raise PermissionError(
                "closing the action budget requires the harness control token issued "
                "when the window was opened; this is a harness-only operation."
            )
        used = self._round_used
        self._round_token = None
        self._round_budget = 0
        self._charge_clock = False
        self._allow_cosmos = True
        resets = self._resets_this_window
        self._resets_this_window = 0
        self._window_snapshot = None
        self._last_action_poses = None
        return {"ok": True, "actions_used": int(used), "resets": int(resets),
                "resets_used_total": int(self._resets_used)}

    def harness_judge_lock(self, control_token: str | None = None) -> dict[str, Any]:
        """HARNESS ONLY: enter a judging round. While locked, privileged methods are all rejected, as while the budget is open.

        A judging round has no action budget (the model only looks), so the ``_round_budget`` gate does not apply;
        without this lock the judge could curl ``harness_read_state_privileged`` during judging and get
        the success predicate. The lock itself is in ``_PRIVILEGED_METHODS`` (cannot lock again while locked, which is fine),
        and unlocking needs the token.
        """
        self._judge_locked = True
        self._judge_token = str(control_token) if control_token else None
        return {"ok": True, "judge_locked": True, "token_required": self._judge_token is not None}

    def harness_judge_unlock(self, control_token: str | None = None) -> dict[str, Any]:
        """HARNESS ONLY: end the judging round. Raises PermissionError on a token mismatch."""
        if self._judge_token and str(control_token or "") != self._judge_token:
            raise PermissionError(
                "unlocking the judge turn requires the harness control token issued "
                "when it was locked; this is a harness-only operation."
            )
        self._judge_locked = False
        self._judge_token = None
        return {"ok": True, "judge_locked": False}

    def harness_round_begin(self, budget: int, round_index: int | None = None) -> dict[str, Any]:
        """HARNESS ONLY: rewind to the failure snapshot and open a fresh action budget."""
        out = self.reset_to_failure()
        self._round_budget = int(budget)
        self._round_used = 0
        self._round_policy_chunks = 0
        self._round_index = int(round_index) if round_index is not None else self._round_index + 1
        self._allow_cosmos = True
        out.update(self.harness_round_status())
        return out

    def harness_round_status(self) -> dict[str, Any]:
        """HARNESS ONLY: round bookkeeping plus the privileged success predicate."""
        status: dict[str, Any] = {
            "round_index": self._round_index,
            "round_budget": self._round_budget,
            "round_used": self._round_used,
            "allow_cosmos": self._allow_cosmos,
            "step_count": int(self._step_count),
        }
        if self._env is not None:
            status.update(self.harness_oracle_grasp())
        return status

    def nudge(self, dxyz: Any, gripper: Any = "hold", max_norm: float | None = MAX_STEP_M) -> dict[str, Any]:
        """Relative displacement correction -- the model's only means of moving at deployment.

        The difference from absolute ``move_to`` is fundamental: a relative displacement keeps the pose the policy chose and only offsets it,
        so a small correction most likely stays near the policy's support; absolute relocation sends the arm to poses the policy has never
        seen, which in practice left it unable to take over again (in 25 trials it broke half of the trajectories that would have self-recovered).

        ``max_norm`` clips the displacement norm **server-side**: the budget lives in the interface rather than the prompt, so the model structurally
        cannot escape it. None means no clipping (only for harness calibration experiments).
        """
        self._require_env()
        self._spend_action("nudge")
        d = np.asarray(dxyz, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(d))
        clipped = False
        if max_norm is not None and n > float(max_norm) > 0:
            d = d / n * float(max_norm)
            clipped = True
        out = self._primitives.move_delta(
            tuple(float(v) for v in d), gripper=gripper,
            step_clip=MOVE_TO_STEP_CLIP, max_steps=LIFT_MAX_STEPS,
        )
        self._refresh_observation()
        out["requested_norm_m"] = n
        out["applied_norm_m"] = float(np.linalg.norm(d))
        out["clipped"] = clipped
        return self._finish_action(out)

    def reset_to_failure(self) -> dict[str, Any]:
        """Rewind the env to the captured failure state. Repeatable, verified.

        Unlike ``restore_snapshot``'s cross-process path, the observation check
        inside ``restore_simulator_snapshot`` is MEANINGFUL here: same process,
        same env, same objects, so a mismatch is a real error rather than a
        check that silently cannot apply.
        """
        from . import failure_factory as ff

        if self._failure is None:
            raise RuntimeError("no failure state: call make_failure first")
        self._require_env()
        ff.restore(self._env, self._cfg, self._failure)
        self._refresh_observation()
        self._step_count = 0
        if self._primitives is not None:
            self._primitives.invalidate_calibration()
        return {"ok": True, "task": self._failure.task, "step_count": self._step_count}

    def _proprio_state(self) -> dict[str, Any]:
        """Proprioception, and nothing else. Shared by both read paths."""
        from rpc.run_remote_robocasa_collect import _gripper_aperture

        self._require_env()
        assert self._policy_obs is not None
        return {
            "eef_pos": [float(v) for v in np.asarray(self._obs_value("robot0_eef_pos"))],
            "eef_quat": [float(v) for v in np.asarray(self._obs_value("robot0_eef_quat"))],
            # Same scalar the collectors call "gripper aperture": the L2 norm of
            # the two finger qpos (rpc/run_remote_robocasa_collect.py).
            "gripper_width": float(_gripper_aperture(self._policy_obs)),
            "step_count": int(self._step_count),
            # The policy's own action history + the landing point of each grasp attempt. All proprioception:
            # the commands are the policy's, the width is the two finger qpos; no oracle.
            # During an intervention the model uses these two to locate "where the last empty grasp happened" instead of re-finding the object in the image.
            "grasp_attempts": list(self._grasp_events),
            "pre_grasp_pose": (dict(self._pre_grasp_pose) if self._pre_grasp_pose else None),
            "recent_chunks": list(self._act_log[-6:]),
            # Points the end effector has visited. When a move_to target is within 8 cm of any of them, the per-action translation limit
            # is lifted from 5 cm to 60 cm (retracing the path, not teleporting).
            "visited_points": list(self._waypoints[-40:]),
            "resets_left": int(MAX_RESETS - self._resets_used),
            # This episode's policy step clock. Chunks the model hands back to the policy are charged here; its own moves are not.
            # Without this number, it treats handing back as a free probe.
            **self._policy_clock(),
            "far_move_rule": (f"move-to may travel up to {RECALL_MAX_M} m in one call when its target is "
                              f"within {ANCHOR_TOL_M} m of any visited_points / grasp_attempts position; "
                              f"otherwise it is clipped to {MAX_STEP_M} m"),
        }

    def read_state(self) -> dict[str, Any]:
        """Proprio state ONLY. Takes no arguments; ALWAYS masks.

        The ``mask_oracle`` parameter is gone, on purpose. A default-on switch is
        still a switch: it survives in the RPC surface, in the CLI ``--help``, and
        in anyone's muscle memory, and one forgotten flag silently converts an
        exploration run into a privileged run whose findings do not transfer to a
        VLM (which has no object_xyz, no contact query, no success flag). There is
        now no argument that can turn masking off -- the privileged read is a
        DIFFERENT method, ``harness_read_state_privileged``, and ``cli/rex.py``
        does not expose it at any flag.
        """
        return {"state": self._proprio_state()}

    def harness_read_state_privileged(self) -> dict[str, Any]:
        """HARNESS ONLY: proprio state PLUS the oracle block.

        Never routed through ``cli/rex.py``. This is the scoring read the paired
        trial needs (did the intervention move the object? was it grasped?), and
        it is logged like every other call so a privileged read always leaves a
        trace in the audit trail.
        """
        from rpc.run_remote_robocasa_collect import _grasped_object_names

        env = self._require_env()
        state = self._proprio_state()
        grasped = _grasped_object_names(env)
        return {
            "state": state,
            "oracle": {
                "object_xyz": snapshot_bridge.target_object_xyz(env),
                "is_grasped": bool("obj" in grasped),
                "task_success": bool(env._check_success()),
            },
        }

    # ---- rendering / perception ------------------------------------------

    def _render_depth(self, camera_key: str, size: int) -> np.ndarray:
        """Metric depth for one camera, TOP-DOWN (row 0 = image top).

        Copied from ``vendor/rpent/robocasa/env_server.py::render_camera``,
        including the sanitization, which is not optional:

            Sanitize the raw OpenGL normalized depth into [0,1]: replace NaN/inf
            (degenerate camera pose) then clip numerical overshoot. Otherwise an
            assertion inside get_real_depth_map crashes the whole env server
            process.

        The sim's OpenGL depth buffer is bottom-up; it is flipped here so it is
        pixel-aligned with the RGB we hand out (which this repo also flips, via
        ``flip_images``) and with robosuite's top-down camera transform matrix.
        """
        import robosuite.utils.camera_utils as CU

        env = self._require_env()
        name = _camera_names()[camera_key]
        _rgb, d = env.sim.render(
            width=size, height=size, camera_name=name, depth=True
        )
        d = np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=0.0)
        d = np.clip(d, 0.0, 1.0)
        if d.ndim == 3:
            depth = CU.get_real_depth_map(env.sim, d)[..., 0]
        else:
            depth = CU.get_real_depth_map(env.sim, d[..., None])[..., 0]
        return np.asarray(depth, dtype=np.float64)[::-1]  # -> top-down

    def _camera_meta(self, camera_key: str, size: int) -> dict[str, Any]:
        """Intrinsic / extrinsic / near-far for one camera.

        Copied from ``vendor/rpent/robocasa/env_server.py::get_camera_meta``; all
        four helpers exist in robosuite 1.5.1 (verified on the simulation host).
        """
        import robosuite.utils.camera_utils as CU

        env = self._require_env()
        name = _camera_names()[camera_key]
        K = CU.get_camera_intrinsic_matrix(env.sim, name, size, size)
        ext = CU.get_camera_extrinsic_matrix(env.sim, name)  # cam -> world
        model = env.sim.model
        extent = model.stat.extent
        return {
            "camera_name": name,
            "height": int(size),
            "width": int(size),
            "intrinsic": np.asarray(K, dtype=np.float64).tolist(),
            "extrinsic_cam2world": np.asarray(ext, dtype=np.float64).tolist(),
            "depth_near": float(model.vis.map.znear * extent),
            "depth_far": float(model.vis.map.zfar * extent),
        }

    def _pixel_to_world_matrix(self, camera_key: str, size: int) -> np.ndarray:
        """T_p2w: the inverse of robosuite's world->pixel transform.

        ``vendor/rpent/robocasa/env_server.py::get_camera_transform`` does exactly
        this (``np.linalg.inv(get_camera_transform_matrix(...))``); the transform
        is TOP-DOWN, which is why the depth map above is flipped to match.
        """
        import robosuite.utils.camera_utils as CU

        env = self._require_env()
        name = _camera_names()[camera_key]
        T = CU.get_camera_transform_matrix(env.sim, name, size, size)
        return np.asarray(np.linalg.inv(T), dtype=np.float64)

    def harness_oracle_grasp(self) -> dict[str, Any]:
        """HARNESS ONLY: the privileged grasp/success predicate.

        This WAS a model tool (``rex.py grasped`` -> ``read_grasp``). It is not
        one any more, and this is the single most important removal in the
        contract freeze.

        WHY: it bottoms out in ``env._check_grasp``, a MuJoCo contact query
        between the gripper geoms and the target body. The upstream docstring for
        the collector that computes it says these are "privileged labels for
        storage only; these never enter inference" -- so routing it to the model
        was using, at decision time, a signal the project itself declares
        off-limits at decision time. Concretely it broke two things:
          * it is a perfect, noiseless success oracle. An explorer could close
            the gripper, ask "did I get it?", and branch on the answer. A robot
            gets no such answer; a VLM gets no such answer. Any recipe learned
            against it is unrunnable at deployment.
          * it made the whole lift-and-check behaviour unnecessary, so the
            explorer never had to learn it -- and "fingers closed" vs "object
            actually lifted" is precisely the distinction a deployed policy has
            to make from proprioception plus images.

        The model must now judge a grasp the way a robot does: gripper width
        (proprioception, from ``read_state``) plus ``render`` before and after a
        ``lift``. The predicate stays here because the HARNESS still needs it --
        it is the scoring instrument for the paired trial, and scoring is
        allowed to be privileged as long as the policy never sees it.
        """
        env = self._require_env()
        from rpc.run_remote_robocasa_collect import _grasped_object_names

        return {
            "is_grasped": bool("obj" in set(_grasped_object_names(env))),
            "task_success": bool(env._check_success()),
        }

    def render(self) -> dict[str, Any]:
        """All three cameras' RGB at the FIXED render size + depth + params.

        RGB comes from ``cf_bench.vlm_correction_rollout.render_camera_512``
        (imported, not copied): an independent high-res offscreen render, decoupled
        from the env's configured 224x224 observation cameras -- ``env.sim.render``
        accepts any width/height regardless of how the observation cameras were set
        up at env creation.

        ``size`` is no longer a parameter (it is ``--render-size``, a launch-time
        property of the service, default 512): a camera has the resolution it has,
        and letting the model re-render the same instant at several resolutions is
        a free extra look that no robot gets. ``depth`` and ``main_camera`` went
        with it -- depth is always returned because ``unproject`` needs the same
        map, and there is only one depth camera.
        """
        from cf_bench.vlm_correction_rollout import render_camera_512

        env = self._require_env()
        size = int(self._render_size)
        main_camera = MAIN_CAMERA
        frames = render_camera_512(env, flip_images=self._cfg.flip_images, size=size)
        out: dict[str, Any] = {
            "size": size,
            "rgb": {key: np.ascontiguousarray(value) for key, value in frames.items()},
            "cameras": {
                key: self._camera_meta(key, size) for key in _camera_names()
            },
            "main_camera": main_camera,
        }
        out["depth"] = self._render_depth(main_camera, size)
        out["depth_camera"] = main_camera
        out["pixel_to_world"] = self._pixel_to_world_matrix(main_camera, size)
        return out

    def unproject_batch(
        self,
        rows: Sequence[int] | np.ndarray,
        cols: Sequence[int] | np.ndarray,
        camera: str = MAIN_CAMERA,
    ) -> dict[str, Any]:
        """Pixels -> world xyz, vectorized.

        Ported from ``vendor/rpent/robocasa/env_client.py::world_map``'s dense form
        (``world = T_p2w @ [col*z, row*z, z, 1]``, verified upstream to land within
        ~2cm of a ground-truth object), but computed only at the requested pixels
        instead of over the whole HxW grid -- an explorer asks about a handful of
        points, and a full 512x512 world map is 6MB of float64 per call.

        Rows/cols are TOP-DOWN image coordinates (row 0 = top), matching the RGB
        this service returns.

        HARNESS-ONLY, and not because it is privileged -- it is not. A model that
        can unproject a whole grid in one call is doing dense reconstruction, not
        the point-and-ask a VLM does; ``rex.py`` exposes only single-pixel
        ``unproject`` so that exploration costs a look per point.
        """
        size = int(self._render_size)
        z_map = self._render_depth(camera, size)
        T_p2w = self._pixel_to_world_matrix(camera, size)
        r = np.asarray(rows, dtype=np.int64).reshape(-1)
        c = np.asarray(cols, dtype=np.int64).reshape(-1)
        if r.shape != c.shape:
            raise ValueError(f"rows/cols length mismatch: {r.shape} vs {c.shape}")
        if r.size == 0:
            raise ValueError("no pixels requested")
        if (r < 0).any() or (c < 0).any() or (r >= size).any() or (c >= size).any():
            raise ValueError(f"pixel out of range for a {size}x{size} render")
        z = z_map[r, c]
        homog = np.stack([c * z, r * z, z, np.ones_like(z)], axis=-1)  # (N,4)
        world = homog @ T_p2w.T
        return {
            "camera": camera,
            "size": size,
            "rows": r.tolist(),
            "cols": c.tolist(),
            "depth": [float(v) for v in z],
            "world_xyz": np.asarray(world[:, :3], dtype=np.float64),
        }

    def unproject(
        self,
        row: int,
        col: int,
        camera: str = MAIN_CAMERA,
    ) -> dict[str, Any]:
        """Single pixel -> world xyz. Uses the fixed render size.

        RPent's single-pixel path (``env_client.world_xyz_at``) builds the entire
        HxWx3 world map and indexes one element out of it; this is the same maths
        applied to one pixel.
        """
        batch = self.unproject_batch([int(row)], [int(col)], camera=camera)
        world = np.asarray(batch["world_xyz"], dtype=np.float64)[0]
        return {
            "camera": batch["camera"],
            "size": batch["size"],
            "row": int(row),
            "col": int(col),
            "depth": float(batch["depth"][0]),
            "world_xyz": [float(v) for v in world],
        }

    # ---- motion -----------------------------------------------------------

    def move_to(
        self,
        xyz: Sequence[float],
        gripper: float | str | None = "hold",
    ) -> dict[str, Any]:
        """Closed-loop servo to a WORLD xyz. See ``primitives.move_to``.

        ``step_clip`` / ``max_steps`` / ``tol`` are frozen at their former
        defaults (see the module constants). They are controller internals -- a
        robot's servo gains are not a per-command choice -- and leaving them
        tunable meant a recipe mined at one clip was not a recipe at another,
        which makes the grid results incomparable across explorations.
        """
        env = self._require_env()
        self._spend_action("move_to")
        # End-effector position comes from proprioceptive robot0_eef_pos (on a real robot, forward kinematics from joint encoders),
        # not sim.data.site_xpos -- the latter is internal simulator state. The two are numerically equivalent.
        # The per-action translation limit is MAX_STEP_M, with one exception: when the target lies within ANCHOR_TOL_M
        # of "a point the end effector visited this episode", it is lifted to RECALL_MAX_M.
        #
        # Why the exception: human review kept pointing at the same thing -- what needs fixing is the moment of the "first miss":
        # reach a bit lower, shift a bit left. But after an empty grasp the policy pulls the hand back 30-40 cm,
        # and a 5 cm per-step limit times 8 actions does not cover half that distance, so every intervention was spent on the long
        # trip and never reached the spot to fix. In round two, 12 of 15 interventions used the full budget and were still 10+ cm short.
        #
        # Why not a "return to last landing point" primitive: the coordinates at the empty grasp were themselves wrong,
        # so returning exactly to that wrong point is pointless; the path back should also be chosen by the model (pressing down from
        # above-side rather than sliding along the counter, which knocks over the neighboring container). So what is lifted is the allowance, not an action:
        # target point, approach direction and number of steps are all written by the model.
        # Anchors can only be places the arm actually reached, so this retraces the path rather than teleporting.
        target = np.asarray(xyz, dtype=np.float64).reshape(3)
        cur = np.asarray(self._obs_value("robot0_eef_pos"), dtype=np.float64)
        delta = target - cur
        dist = float(np.linalg.norm(delta))
        anchors = [np.asarray(w["xyz"], dtype=np.float64) for w in self._waypoints]
        anchors += [np.asarray(e["eef"], dtype=np.float64) for e in self._grasp_events]
        near = min((float(np.linalg.norm(target - a)) for a in anchors), default=float("inf"))
        allowance = RECALL_MAX_M if near <= ANCHOR_TOL_M else MAX_STEP_M
        clipped_to = None
        if dist > allowance:
            target = cur + delta / dist * allowance
            clipped_to = round(allowance, 4)
        result = self._primitive_driver().move_to(
            np.asarray(target, dtype=np.float64),
            gripper=gripper,
            step_clip=MOVE_TO_STEP_CLIP,
            max_steps=MOVE_TO_MAX_STEPS * (3 if allowance > MAX_STEP_M else 1),
            tol=MOVE_TO_TOL,
        )
        return self._finish_action({
            **result, "step_count": int(self._step_count),
            "requested_dist_m": round(dist, 4), "clipped_to_m": clipped_to,
            "allowance_m": round(allowance, 3),
            "nearest_visited_point_m": (None if near == float("inf") else round(near, 3)),
        })

    def lift(
        self,
        dz: float,
        gripper: float | str | None = "hold",
    ) -> dict[str, Any]:
        """Move along world z by ``dz``. Servo tuning frozen, as for ``move_to``."""
        self._require_env()
        self._spend_action("lift")
        dz = float(np.clip(float(dz), -MAX_STEP_M, MAX_STEP_M))
        result = self._primitive_driver().lift(
            float(dz),
            gripper=gripper,
            step_clip=LIFT_STEP_CLIP,
            max_steps=LIFT_MAX_STEPS,
        )
        return self._finish_action({**result, "step_count": int(self._step_count)})

    def rotate(self, drot: Any, gripper: Any = "hold") -> dict[str, Any]:
        """Wrist rotation -- the 3 rotational DoF the policy has; the exploration side now has them too.

        The action vector is [dpos(3), drot(3), gripper, base(4), mode]; the policy fills all 6 arm DoF every step,
        while this primitive used to write only dpos, leaving the exploration side a notch weaker than what it is meant to correct
        (it could see an object was askew but had no tool to straighten it). Magnitude is clipped server-side, like translation.
        """
        self._require_env()
        self._spend_action("rotate")
        out = self._primitive_driver().rotate(
            np.asarray(drot, dtype=np.float64), gripper=gripper,
            max_rad=MAX_STEP_RAD, steps=ROTATE_SUBSTEPS,
        )
        self._refresh_observation()
        return self._finish_action({**out, "step_count": int(self._step_count)})

    def move_base(self, cmd: Any, steps: int = 6) -> dict[str, Any]:
        """Drive the mobile base: [forward/back, left/right, turn, lift], each clipped to +/-BASE_MAX_CMD, at most BASE_MAX_STEPS steps.

        The policy (pi0.5, natively 12-D) moves the base, so the VLM's action set must be just as wide.
        Magnitude and step count are clipped server-side, as for the arm.
        """
        self._require_env()
        self._spend_action("move_base")
        c = np.clip(np.asarray(cmd, dtype=np.float64).reshape(4), -BASE_MAX_CMD, BASE_MAX_CMD)
        n = int(np.clip(int(steps), 1, BASE_MAX_STEPS))
        out = self._primitive_driver().move_base(c, steps=n)
        self._refresh_observation()
        return self._finish_action({**out, "step_count": int(self._step_count)})

    def gripper(self, action: str = "close") -> dict[str, Any]:
        """Explicitly drive the fingers. ``action`` is "open" or "close".

        Note this is the EXPLICIT override path of ``_resolve_grip``: a sustained
        close command keeps driving the fingers shut, which is what you want to
        take a grasp but NOT what you want while carrying (that is what the
        ``gripper="hold"`` default of move_to/lift is for).
        """
        self._require_env()
        self._spend_action("gripper")
        driver = self._primitive_driver()
        if action == "open":
            result = driver.open_gripper(steps=GRIPPER_STEPS)
        elif action == "close":
            result = driver.close_gripper(steps=GRIPPER_STEPS)
        else:
            raise ValueError(f"gripper action must be 'open' or 'close', got {action!r}")
        return self._finish_action({**result, "action": action,
                                    "step_count": int(self._step_count)})

    # ---- Plan execution (v7: compress "one LLM round trip per primitive" into "one round trip per plan segment") -------
    #
    # v1 measurements: a median intervention took 15 tool rounds and 272 s, while the server executed all those actions
    # in under 10 s -- 96% of the time was the model rereading the context and rethinking between primitives.
    # But those rounds were not all waste: reading transcripts, the model really used feedback in only two places:
    # confirming after a lift that the object moved with the hand (the only grasp/no-grasp criterion), and changing course
    # on the spot when an action clearly failed. So this does not run "all 8 in one go"; it stops at a **checkpoint** or
    # an **anomaly** and returns the state and freshly rendered frames. Rounds dropped from 15 to 3-4,
    # with none of the places that need a look skipped.

    # Primitives allowed in a plan, with their parameter keys. Whatever rex's model-side interface has, this has.
    _PLAN_OPS = {
        "move_to": ("target", "gripper"),
        "nudge": ("dxyz", "gripper"),
        "lift": ("dz", "gripper"),
        "rotate": ("drot", "gripper"),
        "gripper": ("state",),
        # Handing control back to the robot's own policy for a few chunks is also an action. The judge used to have only
        # centimeter-level micro-actions, hence repeated servo_missed / closed_empty, while the policy can grasp fine from a decent
        # pose. After positioning, letting the policy do the final move often beats fine-tuning all the way by hand.
        "policy": ("chunks",),
        # Soft undo: return to the takeover pose and aperture. In non-rollback runs this is the only "start over";
        # in rollback rounds 33/70 interventions ended with an undo, while in non-rollback rounds the same judgement left 10+
        # actions in the scene and kept repairing until the quota ran out. retreat lets non-rollback runs also "hand back as found".
        "retreat": ("to",),
    }
    # Stop after these ops to let the model take a look. lift is the standard verification ("lift and see whether it
    # follows"); whether to stop after a close is covered by the hard closed_empty signal, avoiding an extra round trip.
    _PLAN_CHECKPOINT_OPS = ("lift", "policy", "retreat")
    # Max chunks the policy may run per policy step. Too many turns it into "the judge reruns the whole episode",
    # which is a different trajectory rather than a repair, and cannot be attributed.
    _PLAN_POLICY_MAX_CHUNKS = int(os.environ.get("PLAN_POLICY_MAX_CHUNKS", "2"))
    # Cap on the total chunks handed back to the policy in one intervention. First-round measurement: every chunk draws on the episode's
    # 500-step horizon; after handing back 8-10 chunks (128-160 steps) in an intervention, all four regressed
    # cells ended at the step limit, while old rounds on the same tasks finished in 192-416 steps. Cap 4 = 64 steps.
    _PLAN_POLICY_MAX_TOTAL = int(os.environ.get("PLAN_POLICY_MAX_TOTAL", "4"))
    # Runaway criterion: when the actual displacement exceeds the requested one by this factor and by this absolute amount, the servo
    # has failed at this pose (asked for 3 cm, moved 92 cm). Every later step in the segment would start from a scrambled pose,
    # so the segment stops on the spot. nudge used to not even report servo_ok and would keep executing into policy after a runaway.
    _OVERSHOOT_FACTOR = float(os.environ.get("PLAN_OVERSHOOT_FACTOR", "3.0"))
    _OVERSHOOT_MIN_CM = float(os.environ.get("PLAN_OVERSHOOT_MIN_CM", "10.0"))

    def _policy_step_gate(self, plan: Any) -> tuple[int, str] | None:
        """Gate on bare policy segments: handing back control requires a repair of one's own first.

        When this primitive first went live, many plan segments consisted solely of a policy step -- changing nothing,
        just handing control back. That is equivalent to not intervening, yet spends intervention quota and a plan segment; on the same
        tasks the treatment arm actually dropped from 20 to 16. So this is a hard requirement: the step immediately before a policy
        must be the model's own action. That blocks both "hand back at segment start" and two consecutive policy steps
        -- every hand-back must be earned by a repair.

        Returns (violating step index, explanation) or None. The whole plan is rejected before anything runs, never abandoned halfway.
        """
        for i, raw in enumerate(plan):
            if not isinstance(raw, dict) or str(raw.get("op")) != "policy":
                continue
            prev = plan[i - 1] if i > 0 else None
            prev_op = str(prev.get("op")) if isinstance(prev, dict) else None
            if prev_op is None:
                return (i, "a plan may not start with a policy step: hand control "
                           "back only after you have repaired something yourself. "
                           "Put the move_to / nudge / gripper / rotate that fixes "
                           "the fault first, then the policy step.")
            if prev_op == "policy":
                return (i, "two policy steps in a row are not allowed: each "
                           "hand-back has to be earned by a repair of your own in "
                           "between. If the previous hand-back did not help, change "
                           "something yourself before trying it again.")
        return None

    def _resolve_plan_target(self, target: Any) -> tuple[list[float], dict[str, Any]]:
        """Resolve targets in a plan to world coordinates.

        Two forms are allowed: ``{"xyz": [x, y, z]}``, or ``{"pixel": [row, col],
        "camera": "primary", "dz": 0.05}``. The latter is unprojected server-side plus a z offset --
        the model localizes by clicking pixels on rendered images and cannot be expected to compute 3-D coordinates up front.
        """
        if not isinstance(target, dict):
            raise ValueError("target must be an object with 'xyz' or 'pixel'")
        if "xyz" in target:
            xyz = [float(v) for v in target["xyz"]]
            if len(xyz) != 3:
                raise ValueError("target.xyz must have 3 numbers")
            return xyz, {"from": "xyz"}
        if "pixel" not in target:
            raise ValueError("target needs either 'xyz' or 'pixel'")
        row, col = (int(v) for v in target["pixel"])
        camera = str(target.get("camera") or MAIN_CAMERA)
        hit = self.unproject(row=row, col=col, camera=camera)
        # unproject returns the key world_xyz (not xyz). Reading the wrong key raises at the move_to step
        # a KeyError, aborting the whole plan -- and pixel targeting is exactly the form the model uses most.
        xyz = [float(v) for v in hit["world_xyz"]]
        dz = float(target.get("dz") or 0.0)
        xyz[2] += dz
        return xyz, {"from": "pixel", "pixel": [row, col], "camera": camera, "dz": dz}

    def execute_plan(self, plan: Any, max_steps: int = 8) -> dict[str, Any]:
        """Execute a plan in order, stopping and returning at a checkpoint or abort condition.

        Returns a per-step log (each step's op, resolved target, end-effector pose, gripper aperture, wrist contact force,
        whether the servo reached the target), the step it stopped at, why it stopped, the remaining unexecuted steps, and the
        state plus three-camera renders at the moment it stopped. From these the model decides whether to continue, replan, or finish.

        Abort conditions (stop on any; remaining steps are returned unchanged):
          * ``budget``      -- action budget exhausted, or the episode's step budget is used up (the judge's
                               own steps count too, see _raw_step; it stops before the step that would cross it)
          * ``closed_empty``-- gripper aperture below GRIP_EMPTY_W after closing, i.e. it grasped air;
                               later steps written assuming "already holding" are all meaningless
          * ``servo_missed``-- the move_to / lift servo did not get within tolerance (primitives' ok=False)
          * ``move_clipped``-- the target exceeded the per-action allowance and was clipped; where it ended up is not where the model wanted
          * ``contact``     -- wrist contact force above threshold
          * ``error``       -- invalid parameters or the primitive raised
        """
        # Lazy import, as in _proprio_state: this module pulls in robosuite, and a top-level import would slow startup.
        from rpc.run_remote_robocasa_collect import _gripper_aperture

        self._require_env()
        if not isinstance(plan, (list, tuple)) or not plan:
            raise ValueError("plan must be a non-empty list of steps")
        if len(plan) > int(max_steps):
            raise ValueError(f"plan has {len(plan)} steps; at most {int(max_steps)} are allowed")

        log: list[dict[str, Any]] = []
        stop_reason = "plan_complete"
        stopped_at: int | None = None

        for i, raw in enumerate(plan):
            # Run the whole plan through the bare-policy gate before executing any step: on violation it is returned unchanged,
            # with nothing executed. Abandoning halfway is harder to recover from than rejecting outright -- the model would get a half-changed
            # world and first have to work out which steps took effect.
            if i == 0:
                bad = self._policy_step_gate(plan)
                if bad is None:
                    try:
                        ep = self._episode or {}
                        bad = _repair_gate_check(
                            str(ep.get("task", "")), plan,
                            int(getattr(self, "_round_policy_chunks", 0)),
                            int(getattr(self, "_window_faults", 0)),
                            eef_pos=[float(v) for v in np.asarray(self._obs_value("robot0_eef_pos"))],
                            resolve_xyz=self._resolve_plan_target)
                    except Exception:
                        bad = None
                if bad is not None:
                    stop_reason, stopped_at = "error", -1
                    log.append({"step": bad[0], "op": str(plan[bad[0]].get("op")) if isinstance(plan[bad[0]], dict) else "policy", "error": bad[1]})
                    break
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
            left = self._episode_steps_left() if self._charge_clock else None
            if left is not None and left <= 0:
                stop_reason, stopped_at = "budget", i
                log.append({"step": i, "op": op,
                            "error": "not run: the episode's step budget is used up, your own moves included"})
                break
            eef_before = [float(v) for v in np.asarray(self._obs_value("robot0_eef_pos"))]
            entry: dict[str, Any] = {"step": i, "op": op, "eef_before": [round(v, 4) for v in eef_before]}
            steps_before = int(self._step_count)
            try:
                grip = raw.get("gripper", "hold")
                if op == "move_to":
                    xyz, how = self._resolve_plan_target(raw.get("target"))
                    entry["target"] = [round(v, 4) for v in xyz]
                    entry["target_from"] = how
                    res = self.move_to(xyz=xyz, gripper=grip)
                elif op == "nudge":
                    res = self.nudge(dxyz=raw["dxyz"], gripper=grip)
                    entry["dxyz"] = [round(float(v), 4) for v in raw["dxyz"]]
                elif op == "lift":
                    res = self.lift(dz=float(raw["dz"]), gripper=grip)
                    entry["dz"] = round(float(raw["dz"]), 4)
                elif op == "policy":
                    # Hand back to the embodied policy. harness_advance itself does not charge the budget (that is the harness advancing
                    # the experiment), but here it is an action the model chose to spend, so it is charged manually per chunk.
                    # The internal call does not go through _dispatch, so the privilege gate does not misfire -- running the frozen policy
                    # leaks no ground truth, and the treatment arm does the same thing in every window anyway.
                    n = max(1, min(int(raw.get("chunks", 1)), self._PLAN_POLICY_MAX_CHUNKS))
                    used_pc = int(getattr(self, "_round_policy_chunks", 0))
                    if used_pc + n > self._PLAN_POLICY_MAX_TOTAL:
                        n = max(0, self._PLAN_POLICY_MAX_TOTAL - used_pc)
                    if n <= 0:
                        stop_reason, stopped_at = "policy_cap", i
                        entry["error"] = (f"policy hand-back allowance for this intervention "
                                          f"({self._PLAN_POLICY_MAX_TOTAL} chunks) is used up; "
                                          f"finish the repair with your own steps or answer fixed")
                        log.append(entry); break
                    if self._round_budget is not None and self._round_used + n > self._round_budget:
                        n = max(0, int(self._round_budget) - int(self._round_used))
                    if n <= 0:
                        stop_reason, stopped_at = "budget", i
                        entry["error"] = "no budget left for a policy step"
                        log.append(entry); break
                    # The charge stays on: _raw_step counts each policy step as it runs and
                    # harness_advance writes back the same total at its end (no double count). A
                    # hand-back that raises part-way (policy server 500, render error) keeps the
                    # steps it ran charged. Its limit min(horizon, budget - committed) keeps it
                    # inside the budget, so _raw_step's refusal never fires in here.
                    adv = self.harness_advance(num_chunks=n, frame_size=128)
                    self._round_used += n
                    self._round_policy_chunks = used_pc + n
                    entry["policy_chunks_left"] = max(0, self._PLAN_POLICY_MAX_TOTAL - self._round_policy_chunks)
                    res = {"ok": True}
                    entry["chunks"] = n
                    entry["policy_done"] = bool(adv.get("done"))
                    entry["policy_task_success"] = bool(adv.get("task_success"))
                    entry["committed_timestep"] = adv.get("committed_timestep")
                elif op == "rotate":
                    res = self.rotate(drot=raw["drot"], gripper=grip)
                    entry["drot"] = [round(float(v), 4) for v in raw["drot"]]
                elif op == "retreat":
                    res = self.retreat(to=str(raw.get("to") or "takeover"))
                    entry["retreat"] = {k: res.get(k) for k in ("to", "fingers", "rot_before_deg", "rot_after_deg", "final_dist_m")}
                else:  # gripper
                    state = str(raw.get("state") or raw.get("action") or "close")
                    res = self.gripper(action=state)
                    entry["state"] = state
            except Exception as exc:  # also reached when a budget is spent (_spend_action raises RuntimeError, _raw_step EpisodeStepBudgetExhausted)
                entry["error"] = f"{type(exc).__name__}: {exc}"
                entry["sim_steps"] = int(self._step_count) - steps_before
                log.append(entry)
                stopped_at = i
                stop_reason = "budget" if "budget" in str(exc).lower() else "error"
                break

            width = float(_gripper_aperture(self._policy_obs))
            force = self._wrist_contact_force()
            eef_after = [float(v) for v in np.asarray(self._obs_value("robot0_eef_pos"))]
            entry.update({
                "eef_after": [round(v, 4) for v in eef_after],
                "moved_cm": round(float(np.linalg.norm(np.asarray(eef_after) - np.asarray(eef_before))) * 100.0, 2),
                "width_mm": round(width * 1000.0, 1),
                "wrist_contact_force": force,
                "servo_ok": bool(res.get("ok", True)),
                "final_dist_m": (round(float(res["final_dist"]), 4) if res.get("final_dist") is not None else None),
                "actions_used": int(self._round_used),
                # simulator steps this step took; in a judge window each of them came off the episode clock
                "sim_steps": int(self._step_count) - steps_before,
            })
            if res.get("clipped_to_m") is not None:
                entry["clipped_to_m"] = res["clipped_to_m"]
                entry["requested_dist_m"] = res.get("requested_dist_m")
            # How far this step "meant to go". nudge: the clipped norm; lift: dz; move_to: distance to the target
            # (the clipped allowance if clipped). Compare with moved_cm to tell whether the servo ran away.
            req_m = None
            if op == "nudge":
                req_m = res.get("applied_norm_m")
            elif op == "lift":
                req_m = min(abs(float(raw.get("dz") or 0.0)), MAX_STEP_M)
            elif op == "move_to":
                req_m = res.get("clipped_to_m") if res.get("clipped_to_m") is not None else res.get("requested_dist_m")
            elif op == "retreat":
                req_m = res.get("requested_dist_m")
            if req_m is not None:
                entry["requested_cm"] = round(float(req_m) * 100.0, 1)
                if entry["moved_cm"] > max(self._OVERSHOOT_FACTOR * entry["requested_cm"], self._OVERSHOOT_MIN_CM):
                    entry["overshoot"] = True
            log.append(entry)

            # --- Abort conditions, hardest to softest ---
            # The episode's step budget is used up (the judge's own steps count too): nothing more can run.
            left = self._episode_steps_left() if self._charge_clock else None
            if left is not None and left <= 0:
                stop_reason, stopped_at = "budget", i
                break
            if op == "gripper" and entry["state"] == "close" and width < _grip_empty_w_for(self._current_task()):
                stop_reason, stopped_at = "closed_empty", i
                break
            if entry.get("overshoot"):
                stop_reason, stopped_at = "overshoot", i
                break
            if entry.get("clipped_to_m") is not None:
                stop_reason, stopped_at = "move_clipped", i
                break
            if op in ("move_to", "lift", "retreat") and not entry["servo_ok"]:
                stop_reason, stopped_at = "servo_missed", i
                break
            if force == force and force > CONTACT_FORCE_ABORT_N:  # NaN does not trigger
                stop_reason, stopped_at = "contact", i
                break
            if self._round_budget is not None and self._round_used >= self._round_budget:
                stop_reason, stopped_at = "budget", i
                break
            if op in self._PLAN_CHECKPOINT_OPS or bool(raw.get("checkpoint")):
                stop_reason, stopped_at = "checkpoint", i
                break

        if stop_reason in _FAULT_STOPS:
            self._window_faults = int(getattr(self, "_window_faults", 0)) + 1
        remaining = list(plan[(stopped_at + 1):]) if stopped_at is not None else []
        out: dict[str, Any] = {
            "executed": log,
            "stop_reason": stop_reason,
            "stopped_at": stopped_at,
            "remaining": remaining,
            "remaining_count": len(remaining),
            "actions_used": int(self._round_used),
            "actions_budget": (int(self._round_budget) if self._round_budget is not None else None),
            "state": self._proprio_state(),
        }
        # Three-camera images at the stopping moment: the model relies on them next to confirm whether the object moved with the hand.
        try:
            frames = self.render()
            # render() returns the key "rgb", not "frames" -- this used to be wrong, and the model got not a single image at checkpoints,
            # only the per-step log. Whether "the object moved with the hand" after a lift can only be seen in images,
            # so missing images defeated the only verification point of plan-once.
            out["frames"] = frames.get("rgb")
            out["cameras"] = frames.get("cameras")
        except Exception as exc:
            out["render_error"] = f"{type(exc).__name__}: {exc}"
        return out

    def reset_step_counter(self) -> dict[str, Any]:
        """Zero the step counter ONLY.

        This does NOT reset the episode, does NOT re-restore the snapshot, and
        does NOT touch simulator state -- the env keeps standing exactly where it
        is. It exists so an explorer can measure "steps since I started this
        attempt" without the give-up-and-restart semantics of an episode reset
        (which this service does not offer at all: the anchor IS the starting
        state, and re-running it means calling restore_snapshot again).
        """
        self._require_env()
        previous = int(self._step_count)
        self._step_count = 0
        return {"step_count": 0, "previous_step_count": previous}

    # ---- policy -----------------------------------------------------------

    def call_cosmos(self, num_chunks: int = 2) -> dict[str, Any]:
        """Run the Cosmos policy closed-loop, best-of-K, from the CURRENT state.

        ``num_chunks`` is the ONLY parameter the model may set -- "how long do I
        let the policy drive before I look again". Three former parameters are
        now the service's own business:

        * ``seed`` -- REMOVED. A seed is not information a robot has, and a
          model that can name the RNG draw can reroll until lucky: same state,
          same plan, new sample, until the policy happens to succeed. That is
          not a transferable skill (nothing at deployment can redraw a
          trajectory on demand) and it confounds attribution -- we could no
          longer distinguish "this recovery pose makes the grasp work" from
          "the fourth draw worked". Seeds now come from this service's
          monotonic counter (``_next_cosmos_seed_base``), so runs stay
          reproducible as a sequence without any call naming a draw.
        * ``prompt`` -- REMOVED. The service already knows the current episode's
          instruction and sends it VERBATIM. Arbitrary prompts are not merely
          unfair, they 500: the policy only accepts instructions present in its
          precomputed T5 cache, so a rephrased prompt fails at the server rather
          than producing a worse rollout.
        * ``num_candidates`` -- FIXED at ``COSMOS_NUM_CANDIDATES`` (4). K is a
          deployment-side compute budget chosen by whoever runs the robot, not
          something the policy talks itself into mid-episode.

        ``policy_endpoint`` and ``capture_real_frames`` went too: the endpoint is
        a launch-time property of the service, and the real frames ARE the
        deliverable -- comparing what Cosmos imagined against what happened is
        how the model is supposed to notice "it thinks it grasped and it did
        not", now that the privileged grasp predicate is gone.

        One request per chunk asks for K candidates (``num_queries=K``, seeds
        ``base..base+K-1``, ``return_future_images=True``), the highest-value
        candidate's first 16 actions are committed, and the loop re-queries --
        the ``rpc/run_remote_robocasa_value_select_deploy.py`` pattern, factored
        into ``recovery_explore/cosmos_call.py`` so it is testable without an env.

        The policy side is DETERMINISTIC by construction: the seeds are ours, and
        ``cosmos_call.validate_candidate_response`` refuses a response whose seeds
        are not exactly the block we asked for -- so the same (prompt, seed, K,
        state) reproduces the same candidates.

        Returns ``gripper_width`` (proprioception), the per-chunk
        ``imagined_frames`` (the policy's predicted future images) and
        ``real_frames`` (what actually happened), and the selected candidate's
        ``value``.

        NOTE ON A DELIBERATE CONTRACT CHANGE: this used to also return
        ``is_grasped`` / ``task_success``, on the rationale that
        the post-rollout grasp judgement IS the success criterion. Those fields
        are gone. Keeping them would have re-opened, through this endpoint, the
        exact privileged contact query that was just removed from the model side
        (``harness_oracle_grasp``): the model could close the gripper, call
        ``cosmos`` with ``num_chunks=1``, and read a noiseless success oracle. The
        model judges the outcome from ``gripper_width`` plus ``render`` before and
        after a ``lift``, like a robot. The harness still scores the episode with
        ``harness_oracle_grasp`` / ``resume_episode``.
        """
        from rpc.run_remote_robocasa_collect import (
            TASK_MAX_STEPS,
            _gripper_aperture,
            _to_environment_action,
        )

        env = self._require_env()
        if not self._allow_cosmos:
            raise RuntimeError(
                "cosmos is disabled during this intervention window: fix the state with "
                "atomic actions (nudge/move-to/lift/gripper) and finish your turn; the "
                "harness hands control back to the policy afterwards."
            )
        self._spend_action("call_cosmos")
        # NOT gated on self._restored: that is set only by restore_snapshot, so
        # asserting on it made call_cosmos unusable from the make_failure path
        # (which is the path we actually use). Task and origin come from whichever
        # mode produced the current state -- see _current_task/_origin_timestep.
        if self._policy_obs is None:
            raise RuntimeError("no policy observation yet: call make_failure or restore_snapshot")
        client = self._policy_client()
        config = self._cosmos_config()
        num_chunks = int(num_chunks)
        if num_chunks < 1:
            raise ValueError("num_chunks must be >= 1")
        # The prompt is the episode's own instruction, verbatim; the seeds are
        # ours. See this method's docstring for why neither is a parameter.
        prompt = self._current_instruction()
        num_candidates = self._num_candidates()
        seed = self._next_cosmos_seed_base(num_chunks * num_candidates)
        action_dim = int(env.action_dim)
        # Episode budget is measured from the anchor's own timestep, exactly as
        # the rollout helpers in cf_bench do (origin_timestep + executed steps vs
        # TASK_MAX_STEPS[task]); the service's own counter is a separate,
        # user-resettable thing and deliberately not used for the budget.
        max_steps = int(self._task_max_steps(self._current_task()))
        committed = int(self._origin_timestep())

        def step_chunk(actions: np.ndarray) -> cosmos_call.ChunkResult:
            nonlocal committed
            executed = 0
            succeeded = False
            for action in np.asarray(actions, dtype=np.float64):
                if committed >= max_steps:
                    break
                self._raw_step(_to_environment_action(action, action_dim))
                committed += 1
                executed += 1
                if env._check_success():
                    succeeded = True
                    break
            # Real frames are ALWAYS captured: with the privileged grasp
            # predicate gone from the model side, the imagined-vs-real image pair
            # is the model's only evidence about what the policy actually did.
            from cf_bench.vlm_correction_rollout import render_camera_512

            frames = render_camera_512(
                env, flip_images=self._cfg.flip_images, size=self._render_size
            )
            assert self._policy_obs is not None
            return cosmos_call.ChunkResult(
                observation=self._policy_obs,
                steps=executed,
                stop=succeeded or committed >= max_steps,
                frames=frames,
            )

        result = cosmos_call.run_best_of_k_closed_loop(
            client,
            observation=self._policy_obs,
            task_description=str(prompt),
            num_chunks=int(num_chunks),
            num_candidates=int(num_candidates),
            base_seed=int(seed),
            step_chunk=step_chunk,
            # Must include an incrementing counter: in make_failure mode state_id is constant
            # (task-scripted), so calls from two different states with the same seed would produce the same
            # request_id with different payloads, and the policy service replies 409
            # "request_id was reused with a different payload" (observed).
            request_prefix=(
                f"{self._run_id}-{self._instance_nonce}-{self._state_id()}"
                f"-c{self._next_call_seq()}-s{int(seed)}"
            ),
            config=config,
        )
        # The policy just drove the arm somewhere else; the jacobian probe that
        # move_to calibrated is no longer trustworthy at the new pose.
        if self._primitives is not None:
            self._primitives.invalidate_calibration()

        assert self._policy_obs is not None
        return {
            # Proprioception only. is_grasped / task_success are NOT returned --
            # see the docstring; the harness reads them via harness_oracle_grasp.
            "gripper_width": float(_gripper_aperture(self._policy_obs)),
            "imagined_frames": result["imagined_frames"],
            "real_frames": result["real_frames"],
            "value": result["value"],
            "values": result["values"],
            "selected_index": result["selected_index"],
            "seeds": result["seeds"],
            "chunks": result["chunks"],
            "steps": result["steps"],
            "committed_timestep": committed,
            "max_steps": int(max_steps),
            "step_count": int(self._step_count),
        }

    # ---- full-episode harness loop ----------------------------------------
    #
    # WHY these live here and not in the model's toolbox: they ARE the
    # experiment. run_episode_to_failure produces the state under study by
    # running the real policy until the oracle says it genuinely failed a grasp;
    # resume_episode finishes the same episode afterwards and is the measurement
    # that answers "did the intervention actually rescue this episode". A driver
    # runs a PAIRED trial with them:
    #
    #     run_episode_to_failure(...)        -> snapshot at the failure
    #     resume_episode()                   -> CONTROL: success with no help
    #     reset_to_failure()                 -> rewind to the same instant
    #     <explorer acts through rex.py>     -> TREATMENT
    #     resume_episode()                   -> success after the intervention
    #
    # Both arms resume from the same episode clock, so the comparison is fair.

    def _policy_client(self) -> Any:
        from rpc.client import CosmosPolicyClient

        endpoint = self._policy_endpoint
        if not endpoint:
            raise ValueError("no policy endpoint: launch with --policy-endpoint")
        return CosmosPolicyClient(endpoint)

    def _policy_clock(self) -> dict[str, Any]:
        """Policy step clock: this episode's limit, used and remaining. All None when no state is loaded."""
        try:
            ep = getattr(self, "_episode", None)
            if ep:
                # The harness path sets only _episode, not _failure: _current_task() raises,
                # so in the first round s24p18's exec state all three fields were null. harness_advance
                # uses these two keys; reading the same place keeps the numbers consistent with the telemetry line.
                task = str(ep["task"])
                mx = int(ep.get("max_steps") or self._task_max_steps(task))
                used = int(ep["failure_committed_timestep"])
            else:
                mx = int(self._task_max_steps(self._current_task()))
                used = int(self._origin_timestep())
            return {"policy_steps_max": mx, "policy_steps_used": used,
                    "policy_steps_left": max(0, mx - used)}
        except Exception:
            return {"policy_steps_max": None, "policy_steps_used": None, "policy_steps_left": None}

    def _task_max_steps(self, task: str) -> int:
        from rpc.run_remote_robocasa_collect import TASK_MAX_STEPS

        return int(round(int(TASK_MAX_STEPS[task]) * self._step_budget_scale))

    def _num_candidates(self) -> int:
        from . import failure_factory as ff

        return int(ff.POLICY_FAMILY_NUM_CANDIDATES[self._policy_family])

    def _cosmos_config(self) -> "cosmos_call.CosmosCallConfig":
        from . import failure_factory as ff

        return cosmos_call.CosmosCallConfig(
            chunk_size=int(self._cfg.chunk_size),
            num_open_loop_steps=int(self._cfg.num_open_loop_steps),
            num_denoising_steps_action=int(self._cfg.num_denoising_steps_action),
            action_dim=int(ff.POLICY_FAMILY_ACTION_DIM[self._policy_family]),
        )

    def _seed_schedule_cfg(self, episode_index: int) -> Any:
        """A copy of the env cfg whose episode range covers ``episode_index``.

        ``deterministic_policy_seed`` validates ``episode_idx`` against
        ``cfg.num_trials_per_task``, and ``failure_factory.build_fresh_env`` builds
        a one-episode cfg (it creates exactly one env). Widening the range on a
        COPY keeps the seed schedule identical to an evaluation run's while
        leaving the env's own config untouched -- ``num_trials_per_task`` does not
        enter ``create_robocasa_env`` at all, only the seed validation.
        """
        from dataclasses import replace

        return replace(
            self._cfg,
            num_trials_per_task=max(
                int(self._cfg.num_trials_per_task), int(episode_index) + 1
            ),
        )

    @staticmethod
    def _new_expectation(
        *,
        request_id: str,
        query_index: int,
        origin_timestep: int,
        policy_seed: int,
        selected_seed: int,
        horizon: int,
        t0: Mapping[str, Any],
        plan_hash: str,
        first16_hash: str,
    ) -> Any:
        """Build the evidence tracker the oracle predicates expect.

        Field-for-field the same construction as
        ``run_remote_robocasa_cf_branch_collect.execute_plan``'s -- the tracker is
        what ``_attempt_evidence`` reads, so it has to be filled the same way or
        the predicate would be judging different evidence than an evaluation run.
        """
        from rpc.run_remote_robocasa_cf_branch_collect import PendingExpectation

        def number(value: Any) -> float | None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return float(value)

        return PendingExpectation(
            policy_request_id=str(request_id),
            query_index=int(query_index),
            origin_timestep=int(origin_timestep),
            policy_seed=int(policy_seed),
            selected_seed=int(selected_seed),
            execution_horizon=int(horizon),
            mode="normal",
            anchor_id=None,
            retry_ordinal=None,
            task_state_t0=dict(t0),
            task_state_t8=None,
            planned_action_sha256=str(plan_hash),
            planned_first16_sha256=str(first16_hash),
            min_target_object_eef_distance=number(t0.get("target_object_eef_distance")),
            min_control_eef_distance=number(t0.get("control_eef_distance")),
            control_contact_seen=t0.get("control_contact") is True,
            eef_position_t0=list(t0["eef_position"]),
            max_eef_displacement=0.0,
            min_gripper_aperture=float(t0["gripper_aperture"]),
            max_gripper_aperture=float(t0["gripper_aperture"]),
            executed_actions=[],
        )

    @staticmethod
    def _update_expectation(
        tracker: Any, privileged: Mapping[str, Any], relative_step: int
    ) -> None:
        """Port of ``execute_plan``'s ``update_tracker``, unchanged in substance."""
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
            float(np.linalg.norm(eef - np.asarray(tracker.eef_position_t0, dtype=np.float64))),
        )
        if relative_step == 8:
            tracker.task_state_t8 = dict(privileged)

    def harness_quit(self) -> dict[str, Any]:
        """HARNESS ONLY: exit the whole process after replying.

        In the three-tier topology the driver runs on the judge host with no direct access to the simulation host, yet ``harness_episode_begin``
        refuses to build another episode in a process that already holds an env -- each arm needs a fresh process. So the service
        exits itself, and a supervisor loop on the simulation host restarts it 3 seconds later.
        The half-second delay lets this reply go out first.
        """
        import threading
        threading.Timer(0.5, lambda: os._exit(0)).start()
        return {"ok": True, "pid": os.getpid()}

    def harness_release_env(self) -> dict[str, Any]:
        """HARNESS ONLY: drop the current episode so the next one can be built in the same process.

        Without this endpoint, every episode required restarting env_service and re-establishing port forwarding;
        startup alone took about a minute per episode, so 240 episodes took hours. The official collector already
        builds consecutive episodes in the same process (see the comment in _release_env), so this is safe.
        Use only where snapshot rollback is not needed (e.g. control arm only).
        """
        self._release_env()
        return {"ok": True}

    def _release_env(self) -> None:
        """Drop this process's env so another episode can be built in it.

        Used only when ``run_episode_to_failure`` did NOT trigger: the caller is
        meant to discard that episode and try another seed, and refusing to
        rebuild would force a whole new process per discarded episode. Rebuilding
        an env in-process is exactly what the official collectors do between
        episodes, so it is safe; what is NOT safe is keeping a stale snapshot
        around, hence every cached handle is cleared.
        """
        self.close()
        self._cfg = None
        self._env_key = None
        self._primitives = None
        self._raw_obs = None
        self._policy_obs = None
        self._failure = None
        self._restored = None
        self._anchor = None
        self._episode = None
        self._failure_origin_timestep = 0
        self._step_count = 0

    def run_episode_to_failure(
        self,
        task: str,
        seed: int = 901,
        episode_index: int = 0,
        max_chunks: int | None = None,
        obj_instance_split: str = "B",
        record_dir: str | None = None,
        stop_on_trigger: bool = True,
    ) -> dict[str, Any]:
        """HARNESS ONLY: run a REAL episode from t=0 until the policy fails a grasp.

        ``record_dir`` enables consistency instrumentation: for each chunk, record the imagined frame (t32) and three real frames,
        for offline exploring-distance computation. ``stop_on_trigger=False`` runs the full episode (for building datasets).

        Two ways to obtain the real side:
          A: snapshot after committing 16 steps -> dry-run the remaining 16 steps of the **same plan** -> grab frames -> roll back.
             Verbatim the same source as the exploring metric validated in cf_bench (same plan, both t32);
             the control protocol is unaffected (rollback 8 ms). Impossible on a real robot, calibration only.
          E: the real frame after re-querying normally and executing the next chunk's 16 steps. The trajectory is no longer the original plan;
             it asks "did reality reach the outcome the model expected 32 steps ago". Usable on a real robot.
        Frames always use the native resolution of the policy observation (same as the imagined frames); no separate 512 render.

        This is the high-fidelity alternative to ``make_failure``'s scripted miss.
        A fresh env is built and stabilized, then the Cosmos policy drives it
        chunk by chunk exactly the way the official evaluation loop does: one
        query per chunk (``num_queries=1``), the deterministic per-episode /
        per-query seed from ``deterministic_policy_seed``, the first
        ``num_open_loop_steps`` (16) actions committed, re-query.

        After every chunk the privileged t0/t8/t16 snapshots and the accumulated
        attempt evidence go to ``evaluate_initial_failure``. The failure trigger
        is ``eligible_failure == True`` -- for the PnP family that means the
        policy demonstrably ATTEMPTED a grasp (came within 5cm of the target with
        real gripper motion) and does not have the object. A terminal failure
        with no attempt is explicitly NOT a trigger; that is the whole point of
        the predicate, and it is why we let the oracle decide instead of watching
        ``is_grasped`` ourselves.

        On trigger the state is captured IN PROCESS (``failure_factory.capture``)
        and becomes the state the explorer works from; ``reset_to_failure``
        rewinds to it any number of times, hash- and observation-verified.

        If the episode succeeds, or runs out of chunks/steps without an eligible
        failure, this returns ``triggered: false`` with a reason and releases the
        env, so the caller can discard the episode and call again with another
        seed in the SAME process.
        """
        import math

        from rpc.run_remote_robocasa_cf_branch_collect import (
            _array_sha256,
            _attempt_evidence,
            _privileged_snapshot,
            deterministic_policy_seed,
        )
        from rpc.run_remote_robocasa_collect import TASK_MAX_STEPS, _to_environment_action
        from rpc.oracle_retry_predicates import evaluate_initial_failure

        from . import failure_factory as ff
        from .primitives import RecoveryPrimitives

        if self._env is not None:
            raise RuntimeError(
                "this process already holds an env; run_episode_to_failure builds a "
                "fresh one. Start another env_service on the next port, or discard "
                "the current episode first."
            )
        if task not in TASK_MAX_STEPS:
            raise ValueError(f"unsupported official RoboCasa task: {task!r}")
        client = self._policy_client()  # fail fast, before minutes of env build

        self._env, self._cfg = ff.build_fresh_env(
            task,
            seed=int(seed),
            episode_index=int(episode_index),
            obj_instance_split=str(obj_instance_split),
            run_id=self._run_id,
            output_dir=self._output_dir,
            policy_family=self._policy_family,
        )
        self._env_key = (str(task), int(seed), int(episode_index), str(obj_instance_split))
        self._primitives = RecoveryPrimitives(
            _DirectEnvHandle(self), action_dim=int(self._env.action_dim)
        )
        env = self._env
        # Empty actions until the scene settles -- the same thing the collectors
        # do at the top of every episode. stabilize() steps the env directly, so
        # re-read the service's caches afterwards.
        ff.stabilize(env, self._cfg)
        self._refresh_observation()
        self._step_count = 0

        instruction = str(env.get_ep_meta().get("lang") or "")
        config = self._cosmos_config()
        horizon = int(config.num_open_loop_steps)
        # Goes through step_budget_scale: harness_advance reads the value written here into episode["max_steps"];
        # it used to take TASK_MAX_STEPS directly, so --step-budget-scale never reached the harness loop.
        max_steps = int(self._task_max_steps(task))
        total_chunks = int(math.ceil(max_steps / horizon))
        chunk_budget = total_chunks if max_chunks is None else min(int(max_chunks), total_chunks)
        seed_cfg = self._seed_schedule_cfg(int(episode_index))
        action_dim = int(env.action_dim)

        def _flush_record(final_reason: str, task_success: bool) -> str | None:
            """Write this episode's frames and labels to disk. Instrumentation failures must never affect the main flow."""
            if not record_dir or not rec:
                return None
            try:
                d = Path(record_dir)
                d.mkdir(parents=True, exist_ok=True)
                stem = f"{task}_s{int(seed)}_ep{int(episode_index)}"
                np.savez_compressed(d / f"{stem}.npz", **rec)
                (d / f"{stem}.json").write_text(json.dumps({
                    "task": task, "seed": int(seed), "episode_index": int(episode_index),
                    "instruction": instruction, "reason": final_reason,
                    "task_success": bool(task_success),
                    "chunks": rec_meta,
                }, ensure_ascii=False, indent=1))
                return str(d / f"{stem}.npz")
            except Exception:
                return None

        rec: dict[str, Any] = {}
        rec_meta: list[dict[str, Any]] = []
        pending_e: list[tuple[int, str]] = []   # waits for the next chunk's t16 to serve as its t32_E
        REC_CAMS = ("primary_image", "wrist_image")

        def _obs_frames() -> dict[str, Any]:
            o = self._policy_obs or {}
            return {c: np.asarray(o[c]).copy() for c in REC_CAMS if c in o}

        committed = 0
        reason = "chunk_budget_exhausted"
        for chunk_index in range(chunk_budget):
            if committed >= max_steps:
                reason = "step_budget_exhausted"
                break
            assert self._policy_obs is not None
            observation = dict(self._policy_obs)
            t0 = _privileged_snapshot(env, task, observation)
            policy_seed = int(
                deterministic_policy_seed(
                    seed_cfg,
                    episode_idx=int(episode_index),
                    query_index=int(chunk_index),
                    retry=False,
                )
            )
            request_id = (
                f"{self._run_id}-{self._instance_nonce}-ep{int(episode_index):03d}-t{committed:04d}"
                f"-q{chunk_index:04d}-c{self._next_call_seq()}"
            )
            batch = cosmos_call.request_candidates(
                client,
                observation,
                instruction,
                base_seed=policy_seed,
                # The episode itself is the OFFICIAL single-query loop, not
                # best-of-K: the failure we hand the explorer has to be one the
                # deployed baseline would really have produced.
                num_candidates=1,
                request_id=request_id,
                config=config,
                return_future_images=bool(record_dir),
            )
            plan = np.asarray(batch.actions[0], dtype=np.float64)
            if record_dir:
                # batch.future_images is already the per-camera dict converted by request_candidates;
                # do not call future_images_to_frames again (it looks for the raw backend keys
                # future_image/future_wrist_image and returns empty for an already-converted dict -- hit this in practice).
                for cam, img in (batch.future_images or {}).items():
                    arr = np.asarray(img, dtype=np.uint8)
                    if arr.ndim == 4:      # (candidate, H, W, 3): K=1 in this loop
                        arr = arr[0]
                    rec[f"c{chunk_index:03d}_imagined_{cam}"] = arr
            first16 = np.asarray(plan[:horizon], dtype=np.float64)
            limit = min(horizon, max_steps - committed)
            tracker = self._new_expectation(
                request_id=request_id,
                query_index=chunk_index,
                origin_timestep=committed,
                policy_seed=policy_seed,
                selected_seed=int(batch.seeds[0]),
                horizon=limit,
                t0=t0,
                plan_hash=_array_sha256(plan),
                first16_hash=_array_sha256(first16),
            )
            privileged: dict[str, Any] = dict(t0)
            succeeded = False
            for index in range(limit):
                action = np.asarray(first16[index], dtype=np.float64)
                tracker.executed_actions.append(np.array(action, copy=True))
                self._raw_step(_to_environment_action(action, action_dim))
                committed += 1
                assert self._policy_obs is not None
                privileged = _privileged_snapshot(env, task, self._policy_obs)
                self._update_expectation(tracker, privileged, index + 1)
                if privileged["task_success"]:
                    succeeded = True
                    break
            if record_dir:
                # This chunk's t16 (also serves as the previous chunk's t32_E)
                f16 = _obs_frames()
                for cam, img in f16.items():
                    rec[f"c{chunk_index:03d}_actual16_{cam}"] = img
                for prev_idx, _ in pending_e:
                    for cam, img in f16.items():
                        rec[f"c{prev_idx:03d}_actual32E_{cam}"] = img
                pending_e = [(chunk_index, "waiting")]

                # A: snapshot -> dry-run the remaining 16 steps of the same plan -> grab frames -> roll back
                # This continuation is "diagnostic only", consistent with the existing convention in cf_branch_collect.py:2470:
                # no policy re-query, rolled back afterwards, control protocol unaffected.
                tail = np.asarray(plan[horizon: horizon + horizon], dtype=np.float64)
                if not succeeded and len(tail) > 0:
                    try:
                        mark = ff.capture(env, self._cfg, task=task, instruction=instruction,
                                          object_name=None, mode="t32A_mark", detail={})
                        for a in tail:
                            self._raw_step(_to_environment_action(
                                np.asarray(a, dtype=np.float64), action_dim))
                        for cam, img in _obs_frames().items():
                            rec[f"c{chunk_index:03d}_actual32A_{cam}"] = img
                        ff.restore(env, self._cfg, mark)
                        self._refresh_observation()
                        if self._primitives is not None:
                            self._primitives.invalidate_calibration()
                    except Exception as exc:  # instrumentation must never affect the main flow
                        rec_meta.append({"chunk": chunk_index, "t32A_error": repr(exc)[:200]})

            if succeeded:
                reason = "task_success"
                break
            if limit < horizon or tracker.task_state_t8 is None:
                # A censored chunk has no t16 (and maybe no t8) to judge; the
                # collectors skip the oracle call in exactly this case rather
                # than feeding it a short window.
                reason = "censored_chunk_at_step_budget"
                break
            evidence = _attempt_evidence(tracker)
            decision = evaluate_initial_failure(
                task,
                t0,
                tracker.task_state_t8,
                privileged,
                attempt_evidence=evidence,
                task_description=instruction,
            )
            if record_dir:
                rec_meta.append({
                    "chunk": int(chunk_index),
                    "committed": int(committed),
                    "eligible_failure": bool(decision["eligible_failure"]),
                    "reason": decision.get("reason"),
                    "is_grasped": bool(privileged.get("grasped_objects") and
                                       "obj" in set(privileged["grasped_objects"])),
                    "task_success": bool(privileged.get("task_success")),
                })
            if not decision["eligible_failure"]:
                continue
            if record_dir and not stop_on_trigger:
                # Dataset mode: record the trigger point but keep running the full episode to get episode-level labels
                rec_meta[-1]["first_trigger"] = True
                continue

            state = ff.capture(
                env,
                self._cfg,
                task=task,
                instruction=instruction,
                object_name=ff.object_display_name(env),
                mode="policy",
                detail={
                    "chunk_index": int(chunk_index),
                    "committed_timestep": int(committed),
                    "policy_seed": policy_seed,
                    "policy_request_id": request_id,
                    "decision": decision,
                    "attempt_evidence": evidence,
                },
            )
            self._failure = state
            self._failure_origin_timestep = int(committed)
            self._episode = {
                "task": str(task),
                "seed": int(seed),
                "episode_index": int(episode_index),
                "obj_instance_split": str(obj_instance_split),
                "instruction": instruction,
                "failure_committed_timestep": int(committed),
                "next_query_index": int(chunk_index) + 1,
                "max_steps": max_steps,
                "total_chunks": total_chunks,
            }
            self._refresh_observation()
            self._step_count = 0
            self._primitives.invalidate_calibration()
            return {
                "record_npz": _flush_record("eligible_failure", False),
                "triggered": True,
                "chunk_index": int(chunk_index),
                "committed_timestep": int(committed),
                "instruction": instruction,
                "task": str(task),
                "seed": int(seed),
                "episode_index": int(episode_index),
                "object_name": state.object_name,
                "decision": decision,
                "attempt_evidence": evidence,
                "step_count": int(self._step_count),
            }

        task_success = bool(env._check_success())
        # Nothing usable came out of this episode. Free the env so the caller can
        # immediately try another seed in this same process.
        _rec_path = _flush_record(reason, reason == "task_success")
        self._release_env()
        return {
            "record_npz": _rec_path,
            "triggered": False,
            "reason": "task_success" if task_success else reason,
            "task_success": task_success,
            "committed_timestep": int(committed),
            "task": str(task),
            "seed": int(seed),
            "episode_index": int(episode_index),
        }


    # ------------------------------------------------------------------
    #  Full-episode VLM supervision experiment (judge): start from t=0, hand to the judge every N chunks
    # ------------------------------------------------------------------

    def harness_episode_begin(
        self,
        task: str,
        seed: int = 901,
        episode_index: int = 0,
        obj_instance_split: str = "B",
        max_steps_extra: int = 0,
    ) -> dict[str, Any]:
        """HARNESS ONLY: build an episode and stop at t=0 without running the policy.

        Difference from ``run_episode_to_failure``: that one runs until the oracle declares failure,
        to "manufacture a failure state"; this one only builds the scene + settles + registers episode bookkeeping,
        after which the harness advances it segment by segment with ``harness_advance``, inserting the judge in between.
        """
        import math

        from rpc.run_remote_robocasa_collect import TASK_MAX_STEPS

        from . import failure_factory as ff
        from .primitives import RecoveryPrimitives

        if self._env is not None:
            raise RuntimeError(
                "this process already holds an env; start another env_service "
                "on the next port, or discard the current episode first."
            )
        if task not in TASK_MAX_STEPS:
            raise ValueError(f"unsupported official RoboCasa task: {task!r}")
        self._policy_client()  # fail fast, rather than finding the policy service down after building the env

        self._env, self._cfg = ff.build_fresh_env(
            task,
            seed=int(seed),
            episode_index=int(episode_index),
            obj_instance_split=str(obj_instance_split),
            run_id=self._run_id,
            output_dir=self._output_dir,
            policy_family=self._policy_family,
        )
        self._env_key = (str(task), int(seed), int(episode_index), str(obj_instance_split))
        self._primitives = RecoveryPrimitives(
            _DirectEnvHandle(self), action_dim=int(self._env.action_dim)
        )
        ff.stabilize(self._env, self._cfg)
        self._refresh_observation()
        self._step_count = 0

        instruction = str(self._env.get_ep_meta().get("lang") or "")
        horizon = int(self._cosmos_config().num_open_loop_steps)
        # ``max_steps_extra`` adds a few steps beyond the task's step limit (default 0). Every step in a judge window is
        # counted on the episode clock (_raw_step / _charge_clock), so the treatment arm gets no free sim steps and the
        # control arm needs no compensation.
        max_steps = self._task_max_steps(task) + max(0, int(max_steps_extra))
        self._episode = {
            "task": str(task),
            "seed": int(seed),
            "episode_index": int(episode_index),
            "obj_instance_split": str(obj_instance_split),
            "instruction": instruction,
            "failure_committed_timestep": 0,
            "next_query_index": 0,
            "max_steps": max_steps,
            "total_chunks": int(math.ceil(max_steps / horizon)),
        }
        self._failure_origin_timestep = 0
        self._act_log = []
        self._pose_hist = []
        self._pre_grasp_pose = None
        self._grasp_events = []
        self._waypoints = []
        self._window_snapshot = None
        self._charge_clock = False
        self._resets_used = 0
        self._resets_this_window = 0
        self._collateral_log = []
        return {
            "ok": True, "task": task, "seed": int(seed), "episode_index": int(episode_index),
            "instruction": instruction, "horizon": horizon, "max_steps": max_steps,
            "total_chunks": self._episode["total_chunks"],
            "policy_family": self._policy_family,
        }

    def harness_advance(
        self,
        num_chunks: int = 1,
        frame_size: int = 256,
        frame_cameras: Sequence[str] = ("primary", "secondary", "wrist"),
    ) -> dict[str, Any]:
        """HARNESS ONLY: advance the policy ``num_chunks`` chunks from the current state.

        Saves a low-resolution frame at the end of each chunk for the judge. Difference from ``call_cosmos``:
        does not count against the model's action budget (this is the harness advancing the experiment, not the model spending budget),
        and it advances the episode's ``next_query_index`` -- this really moves the episode forward,
        rather than resampling repeatedly from the same point.
        """
        from rpc.run_remote_robocasa_cf_branch_collect import deterministic_policy_seed
        from rpc.run_remote_robocasa_collect import (
            TASK_MAX_STEPS, _to_environment_action, _gripper_aperture,
        )
        from cf_bench.vlm_correction_rollout import render_camera_512
        from .primitives import (
            OSC_POS_SCALE, OSC_ROT_SCALE, GRIPPER_INDEX, BASE_MODE_INDEX,
        )

        if self._episode is None:
            raise RuntimeError("no episode: call harness_episode_begin first")
        env = self._require_env()
        assert self._policy_obs is not None
        episode = self._episode
        task = str(episode["task"])
        episode_index = int(episode["episode_index"])
        instruction = str(episode["instruction"])
        budget = int(episode.get("max_steps") or self._task_max_steps(task))
        client = self._policy_client()
        config = self._cosmos_config()
        horizon = int(config.num_open_loop_steps)
        seed_cfg = self._seed_schedule_cfg(episode_index)
        action_dim = int(env.action_dim)
        num_candidates = self._num_candidates()

        committed = int(episode["failure_committed_timestep"])
        query_index = int(episode["next_query_index"])
        succeeded = bool(env._check_success())
        frames: list[dict[str, Any]] = []
        frame_steps: list[int] = []
        executed = 0
        stopped = "chunks_done"

        def _grip_xyz() -> np.ndarray:
            # Use proprioceptive robot0_eef_pos, not sim.data.site_xpos: the latter is internal simulator
            # state that a real robot lacks. Must copy -- the observation array is overwritten by the next step, and keeping it as
            # "position at chunk start" would yield a displacement that is always 0.
            return np.array(self._obs_value("robot0_eef_pos"), dtype=np.float64)

        def _width() -> float:
            return float(_gripper_aperture(self._policy_obs))

        def _eef_quat() -> np.ndarray:
            return np.array(self._obs_value("robot0_eef_quat"), dtype=np.float64)

        def _eef_rpy_deg() -> list | None:
            # robosuite quaternions are (x, y, z, w); report roll/pitch/yaw in degrees so the judge can
            # read the wrist pose it would change with a rotate step. Telemetry only: never let it
            # break harness_advance (the control arm runs through here too).
            try:
                x, y, z, w = [float(v) for v in _eef_quat()]
            except Exception:
                return None
            roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
            pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
            yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            return [round(float(roll), 1), round(float(pitch), 1), round(float(yaw), 1)]

        def _joint_pos() -> list | None:
            try:
                return [round(float(v), 3) for v in np.asarray(self._obs_value("robot0_joint_pos")).reshape(-1)]
            except Exception:
                return None

        # Gripper state machine: only the one decidable state, "fully closed with nothing between the fingers", logging entry and exit.
        # It does not assert "holding" -- aperture cannot tell holding from open (see the threshold comment at the top of the module).
        # It records the start of each empty grasp: the step and the end-effector position. That is where "this one missed".
        empty_w = _grip_empty_w_for(task)
        empty_now = _width() < empty_w

        for _ in range(int(num_chunks)):
            if succeeded:
                stopped = "task_success"
                break
            if committed >= budget:
                stopped = "step_budget_exhausted"
                break
            if query_index >= int(episode["total_chunks"]) * 4:
                stopped = "seed_schedule_exhausted"
                break
            policy_seed = int(deterministic_policy_seed(
                seed_cfg, episode_idx=episode_index, query_index=query_index, retry=False))
            request_id = (
                f"{self._run_id}-{self._instance_nonce}-ep{episode_index:03d}-adv"
                f"-t{committed:04d}-q{query_index:04d}-c{self._next_call_seq()}"
            )
            batch = cosmos_call.request_candidates(
                client, dict(self._policy_obs), instruction,
                base_seed=policy_seed, num_candidates=num_candidates,
                request_id=request_id, config=config, return_future_images=False,
            )
            values = np.asarray(batch.values, dtype=np.float64)
            pick = int(np.argmax(values)) if values.size else 0
            plan = np.asarray(batch.actions[pick], dtype=np.float64)
            limit = min(horizon, budget - committed)
            mid = limit // 2
            seg = np.asarray(plan[:limit], dtype=np.float64)
            chunk_t0 = int(committed)
            chunk_eef0 = _grip_xyz()
            try:
                self._pose_hist.append({
                    "t": chunk_t0, "eef": [round(float(v), 4) for v in chunk_eef0],
                    "quat": [float(v) for v in np.asarray(self._obs_value("robot0_eef_quat"))],
                    "width_m": float(_width())})
                if len(self._pose_hist) > 400:
                    del self._pose_hist[:-400]
            except Exception:
                pass
            for index in range(limit):
                self._raw_step(_to_environment_action(
                    np.asarray(plan[index], dtype=np.float64), action_dim))
                committed += 1
                executed += 1
                w = _width()
                if not empty_now and w < empty_w:
                    empty_now = True
                    self._grasp_events.append(
                        {"index": len(self._grasp_events), "t": int(committed),
                         "eef": [round(float(v), 4) for v in _grip_xyz()],
                         "width_mm": round(w * 1000.0, 1), "event": "closed-empty"})
                    # Approach pose before the empty grasp: start of the chunk preceding the one where it closed empty, with the gripper open then.
                    try:
                        cand = [h for h in self._pose_hist
                                if h["t"] <= int(committed) - 16 and h["width_m"] > 0.03]
                        if not cand:
                            cand = [h for h in self._pose_hist if h["width_m"] > 0.03]
                        if cand:
                            h = cand[-1]
                            self._pre_grasp_pose = {"t": int(h["t"]), "eef": list(h["eef"]),
                                                    "quat": list(h["quat"]),
                                                    "width_mm": round(h["width_m"] * 1000.0, 1),
                                                    "miss_t": int(committed)}
                    except Exception:
                        pass
                elif empty_now and w > GRIP_NOTEMPTY_W:
                    empty_now = False
                    self._grasp_events.append(
                        {"index": len(self._grasp_events), "t": int(committed),
                         "eef": [round(float(v), 4) for v in _grip_xyz()],
                         "width_mm": round(w * 1000.0, 1), "event": "reopened"})
                # Also sample one frame inside the chunk: human review found "the drop happens inside the chunk and is only visible in the next frame";
                # sampling only at chunk end collapses the whole chunk into one static image.
                if index == mid - 1 and mid > 0:
                    shot = render_camera_512(env, flip_images=self._cfg.flip_images, size=int(frame_size))
                    frames.append({k: np.ascontiguousarray(shot[k]) for k in frame_cameras if k in shot})
                    frame_steps.append(int(committed))
                if env._check_success():
                    succeeded = True
                    break
            query_index += 1
            shot = render_camera_512(env, flip_images=self._cfg.flip_images, size=int(frame_size))
            frames.append({k: np.ascontiguousarray(shot[k]) for k in frame_cameras if k in shot})
            frame_steps.append(int(committed))
            # What this chunk's policy commanded, where the end effector actually went, and how wide the gripper is.
            # Commands are in action units, scaled by the OSC scale into meters/radians so the judge need not convert.
            eef1 = _grip_xyz()
            self._act_log.append({
                "chunk": int(query_index - 1),
                "t": [chunk_t0, int(committed)],
                "cmd_dpos_cm": [round(float(v) * OSC_POS_SCALE * 100.0, 1)
                                for v in seg[:, 0:3].sum(axis=0)],
                "cmd_drot_rad": [round(float(v) * OSC_ROT_SCALE, 2)
                                 for v in seg[:, 3:6].sum(axis=0)],
                "cmd_gripper": round(float(seg[:, GRIPPER_INDEX].mean()), 2),
                # Last 5 dims: accumulated base commands [forward/back, left/right, turn, lift] + mode (fraction of steps driving the base, >0)
                "cmd_base": ([round(float(v), 2) for v in seg[:, GRIPPER_INDEX + 1: BASE_MODE_INDEX].sum(axis=0)]
                             if seg.shape[1] > BASE_MODE_INDEX else None),
                "base_mode_frac": (round(float((seg[:, BASE_MODE_INDEX] > 0).mean()), 2)
                                   if seg.shape[1] > BASE_MODE_INDEX else None),
                "eef_to": [round(float(v), 4) for v in eef1],
                # wrist orientation at the end of the chunk (proprioception, degrees)
                "eef_rpy_deg": _eef_rpy_deg(),
                "moved_cm": round(float(np.linalg.norm(eef1 - chunk_eef0)) * 100.0, 1),
                "width_mm": round(_width() * 1000.0, 1),
            })
            self._waypoints.append({"t": int(committed),
                                    "xyz": [round(float(v), 4) for v in eef1]})

        episode["failure_committed_timestep"] = int(committed)
        episode["next_query_index"] = int(query_index)
        self._failure_origin_timestep = int(committed)
        # Ground truth: object world coordinates + whether it is grasped. The harness side uses it to mark objects on the replay page,
        # so human reviewers need not search the image by eye. The judge never gets these fields (it receives only the overview image).
        try:
            from . import snapshot_bridge as _sb
            from rpc.run_remote_robocasa_collect import _grasped_object_names as _gon
            truth = {"object_xyz": [float(v) for v in _sb.target_object_xyz(env)],
                     "is_grasped": bool("obj" in set(_gon(env)))}
        except Exception as exc:
            truth = {"error": repr(exc)[:120]}
        if self._primitives is not None:
            self._primitives.invalidate_calibration()
        return {
            "task_success": bool(succeeded or env._check_success()),
            "committed_timestep": int(committed),
            # This episode's policy step limit. harness.py uses it to write "used X / limit Y" in telemetry.
            "max_steps": int(budget),
            "next_query_index": int(query_index),
            "steps_executed": int(executed),
            "chunks_done": len(frames),
            "stopped": stopped,
            "truth": truth,
            # Proprioceptive history, visible to the judge (unlike truth, which is only for post-hoc harness checks).
            "telemetry": {
                # whole episode (a 900-step episode is ~60 chunks); harness.py shows the last TEL_CHUNKS
                # as a table and samples the earlier ones into a trajectory summary.
                "chunks": self._act_log[-120:],
                "grasp_attempts": self._grasp_events,
                "pre_grasp_pose": (dict(self._pre_grasp_pose) if self._pre_grasp_pose else None),
                "eef_now": [round(float(v), 4) for v in _grip_xyz()],
                "eef_quat_now": ([round(float(v), 4) for v in _eef_quat()] if _eef_rpy_deg() is not None else None),
                "eef_rpy_now_deg": _eef_rpy_deg(),
                "joint_pos_now": _joint_pos(),
                "width_mm": round(_width() * 1000.0, 1),
            },
            "done": bool(succeeded or committed >= budget or stopped != "chunks_done"),
            "frames": frames,
            "frame_steps": frame_steps,
        }

    def resume_episode(
        self,
        max_steps: int | None = None,
        record_frames: bool = False,
        frame_size: int = 256,
        frame_cameras: Sequence[str] = ("primary", "secondary", "wrist"),
    ) -> dict[str, Any]:
        """HARNESS ONLY: finish the episode from wherever the explorer left it.

        Continues the SAME episode with the same policy and the same
        deterministic seed schedule, from the CURRENT simulator state, until
        ``env._check_success()`` or the episode's step budget
        (``TASK_MAX_STEPS[task]``, overridable with ``max_steps`` for a shorter
        probe) runs out.

        THE BUDGET IS THE FAILURE'S OWN EPISODE CLOCK, not the explorer's step
        counter: both arms of a paired trial resume from the timestep the failure
        was captured at, so whatever the explorer spent on the intervention does
        not shorten the treatment arm's runway relative to the control arm's.
        Charging the intervention to the episode budget would make the comparison
        measure "how many steps did the explorer waste" instead of "did the
        intervention help". (The judge experiment is the exception: a window opened
        by harness_open_action_budget on a harness_episode_begin episode charges
        every step the judge takes to that episode's clock, see ``_raw_step``.)

        This is the measurement that answers "did the intervention actually
        rescue the episode".
        """
        from rpc.run_remote_robocasa_cf_branch_collect import deterministic_policy_seed
        from rpc.run_remote_robocasa_collect import TASK_MAX_STEPS, _to_environment_action

        if self._episode is None:
            raise RuntimeError(
                "no episode to resume: resume_episode continues an episode started "
                "by run_episode_to_failure (a scripted make_failure has no episode)"
            )
        env = self._require_env()
        assert self._policy_obs is not None
        episode = self._episode
        task = str(episode["task"])
        episode_index = int(episode["episode_index"])
        instruction = str(episode["instruction"])
        budget = int(max_steps) if max_steps is not None else int(
            episode.get("max_steps") or self._task_max_steps(task))
        client = self._policy_client()
        config = self._cosmos_config()
        horizon = int(config.num_open_loop_steps)
        seed_cfg = self._seed_schedule_cfg(episode_index)
        action_dim = int(env.action_dim)

        committed = int(self._origin_timestep())
        query_index = int(episode["next_query_index"])
        executed_total = 0
        succeeded = bool(env._check_success())
        # rounds experiment: save a low-resolution frame after each chunk; the harness stitches them into a keyframe image for the model,
        # so it sees "what actually happened after handing back to the policy" (outcomes like lifted-then-dropped that local predicates miss).
        frames: list[dict[str, Any]] = []
        frame_steps: list[int] = []
        eef_track: list[list[float]] = []

        def _track() -> None:
            try:
                eef_track.append([round(float(v), 5)
                                  for v in np.asarray(env.sim.data.site_xpos[
                                      env.sim.model.site_name2id("gripper0_right_grip_site")],
                                      dtype=np.float64)])
            except Exception:
                try:
                    eef_track.append([round(float(v), 5) for v in
                                      np.asarray(self._obs_value("robot0_eef_pos"), dtype=np.float64)])
                except Exception:
                    pass

        def _snap() -> None:
            if not record_frames:
                return
            from cf_bench.vlm_correction_rollout import render_camera_512
            shot = render_camera_512(env, flip_images=self._cfg.flip_images, size=int(frame_size))
            frames.append({k: np.ascontiguousarray(shot[k]) for k in frame_cameras if k in shot})
            frame_steps.append(int(committed))

        _snap()
        _track()
        while not succeeded and committed < budget:
            assert self._policy_obs is not None
            policy_seed = int(
                deterministic_policy_seed(
                    seed_cfg,
                    episode_idx=episode_index,
                    query_index=query_index,
                    retry=False,
                )
            )
            # The call sequence number is NOT decoration: a paired trial resumes
            # twice from the same (episode, timestep, query_index), and the
            # generation server answers a reused request_id carrying a different
            # payload with 409 "request_id was reused with a different payload".
            request_id = (
                f"{self._run_id}-{self._instance_nonce}-ep{episode_index:03d}-resume-t{committed:04d}"
                f"-q{query_index:04d}-c{self._next_call_seq()}"
            )
            batch = cosmos_call.request_candidates(
                client,
                dict(self._policy_obs),
                instruction,
                base_seed=policy_seed,
                num_candidates=1,
                request_id=request_id,
                config=config,
                return_future_images=False,
            )
            plan = np.asarray(batch.actions[0], dtype=np.float64)
            limit = min(horizon, budget - committed)
            for index in range(limit):
                self._raw_step(
                    _to_environment_action(
                        np.asarray(plan[index], dtype=np.float64), action_dim
                    )
                )
                committed += 1
                executed_total += 1
                if env._check_success():
                    succeeded = True
                    break
            _snap()
            _track()
            query_index += 1
            if query_index >= int(episode["total_chunks"]) * 4:
                # deterministic_policy_seed validates query_index against the
                # schedule; stop rather than walk off the end of it.
                break
        if self._primitives is not None:
            self._primitives.invalidate_calibration()
        # ``next_query_index`` is deliberately NOT advanced. A paired trial calls
        # resume_episode twice from the same captured failure (control, then
        # treatment), and both arms must walk the SAME deterministic seed
        # schedule -- letting the control arm move the cursor would hand the
        # treatment arm a different policy sample and make the two runs
        # incomparable. Only the request_id differs between the arms (via the
        # call-sequence counter), which is what the generation server needs.
        out: dict[str, Any] = {
            "task_success": bool(succeeded or env._check_success()),
            "steps_executed": int(executed_total),
            "committed_timestep": int(committed),
            "task": task,
            "step_count": int(self._step_count),
        }
        if record_frames:
            out["frames"] = frames
            out["frame_steps"] = frame_steps
        if len(eef_track) >= 2:
            arr = np.asarray(eef_track, dtype=np.float64)
            seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
            tail = seg[-3:] if len(seg) >= 3 else seg
            out["eef_path_len_m"] = round(float(seg.sum()), 4)
            out["eef_move_last3_chunks_m"] = round(float(tail.sum()), 4)
            out["eef_track"] = [list(map(float, p)) for p in arr]
            # Last 3 chunks (cosmos 48 steps / pi05 75 steps) moved < 1 cm: treated as stalled;
            # more step budget is pointless here -- the policy was not cut off, it stopped moving on its own.
            out["stalled_at_end"] = bool(tail.sum() < 0.01)
        out["hit_step_budget"] = bool(committed >= budget and not succeeded)
        return out


# ---------------------------------------------------------------------------
#  Launcher
# ---------------------------------------------------------------------------


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def pick_port(host: str, port: int, tries: int = 16) -> int:
    """First free port at or after ``port`` -- start at 8410, move up if taken."""
    for candidate in range(int(port), int(port) + int(tries)):
        if _port_is_free(host, candidate):
            return candidate
    raise RuntimeError(f"no free port in [{port}, {port + tries})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--port-tries",
        type=int,
        default=16,
        help="how many consecutive ports to try before giving up (8410 then 8411, ...)",
    )
    parser.add_argument("--anchors-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="recovery-explore")
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="call log (one JSON line per RPC); default <output-dir>/env_service_calls.jsonl",
    )
    parser.add_argument(
        "--policy-endpoint",
        default=None,
        help="Cosmos generation server, e.g. http://127.0.0.1:9000 (used by call_cosmos)",
    )
    parser.add_argument("--render-size", type=int, default=DEFAULT_RENDER_SIZE)
    parser.add_argument(
        "--step-budget-scale", type=float, default=1.0,
        help="multiplier on TASK_MAX_STEPS. Scaled results are not comparable to the official benchmark, only within the batch.",
    )
    parser.add_argument(
        "--policy-family", default="cosmos", choices=["cosmos", "pi05"],
        help="policy family: sets chunk length (32/16 vs 50/25), action dimension (7 vs 12), best-of-K (4 vs 1)",
    )
    parser.add_argument(
        "--debug-resets", type=int, default=0,
        help="max model-side rollbacks (debug only). 0 = off, the model has no reset ability.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="physical CUDA ordinal to pin MuJoCo EGL rendering and torch to",
    )
    parser.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch the parent process via the stdin pipe and exit when it dies",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cuda_device is not None:
        # Copied from vendor/rpent/robocasa/env_server.py's main, comment and all,
        # because the reasoning is the non-obvious part:
        #
        #   Deliberately do NOT set CUDA_VISIBLE_DEVICES. robosuite asserts at
        #   import time that ``MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES``
        #   (substring check), which assumes the EGL index equals the CUDA ordinal
        #   and crashes on multi-GPU boxes where the EGL order differs. That
        #   assertion is gated on ``CUDA_VISIBLE_DEVICES != ""``, so leaving it
        #   unset skips it in this process and in any spawned render worker (which
        #   inherits the env). Pin the two backends directly instead:
        #     - MuJoCo render device <- MUJOCO_EGL_DEVICE_ID (configure_egl_device)
        #     - torch default device <- torch.cuda.set_device(N)
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        if previous is not None:
            print(
                f"CUDA_VISIBLE_DEVICES={previous} is set; clearing it and pinning via "
                f"MUJOCO_EGL_DEVICE_ID + torch.cuda.set_device({args.cuda_device}) instead",
                file=sys.stderr,
                flush=True,
            )
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        configure_egl_device(args.cuda_device)
        import torch

        torch.cuda.set_device(args.cuda_device)

    output_dir = Path(args.output_dir)
    log_path = Path(args.log) if args.log else output_dir / "env_service_calls.jsonl"
    port = pick_port(args.host, args.port, args.port_tries)

    service = RecoveryEnvService(
        anchors_jsonl=args.anchors_jsonl,
        output_dir=output_dir,
        run_id=args.run_id,
        log_path=log_path,
        policy_endpoint=args.policy_endpoint,
        render_size=args.render_size,
        debug_resets=args.debug_resets,
        policy_family=args.policy_family,
        step_budget_scale=args.step_budget_scale,
    )
    service.serve(
        transport="http", host=args.host, port=port, parent_watch=args.parent_watch
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
