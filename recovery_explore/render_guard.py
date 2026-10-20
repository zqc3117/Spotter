"""GL render guard for the RoboCasa env service (``recovery_explore.env_service``).

Root cause of the "noise / black observation frames" bug (found 2026-10-12)
--------------------------------------------------------------------------
robosuite's ``MujocoEnv.reset()`` with ``hard_reset=True`` destroys the previous
``MjSim`` only when ``renderer == "mujoco"``.  robocasa's ``Kitchen`` envs use
``renderer == "mjviewer"``, so the MjSim built by the constructor -- together with
its ``MjRenderContextOffscreen`` and that context's own EGL context -- is never
freed: after the first ``env.reset()`` it survives as an unreachable reference
cycle (``sim._render_context_offscreen`` <-> ``ctx.sim``).

Python's cyclic GC collects that cycle at some later gen-2 collection, i.e. at a
moment set by the process's allocation pattern (scene size, RPC traffic, policy
chunking), which is why the same cells fail run after run.  When it happens,
``MjRenderContext.__del__`` runs while the LIVE env's EGL context is current:

* ``mjr_freeContext`` deletes GL object names (FBOs, textures, ...) in the
  CURRENT context, i.e. the live one, and
* ``EGLGLContext.free()`` ends with ``eglReleaseThread()``, which unbinds the
  live context from the thread.

From then on every ``mjr_render`` draws into nothing and ``mjr_readPixels``
returns whatever memory ``np.empty`` recycled: RGB noise, black, black with a
noisy band, or an older frame.  All three cameras and the 224-px policy
observations are affected, so the policy flails and the episode fails.

Bisecting it showed that freeing the leaked context with ITSELF current and re-binding the live one
afterwards leaves rendering intact.

What this module does (all default-on; ``RENDER_GUARD=0`` disables everything,
``RENDER_GUARD_VERBOSE=1`` logs per-render statistics and GL state):

1. ``install()`` -- root-cause fix.  Patches ``MjRenderContext.__del__`` so a
   context is freed with itself current and the previously current context is
   restored afterwards, and patches ``MjSim.render`` so the live context is
   re-bound if the thread lost it.
2. ``reap_leaked_contexts(env)`` -- right after an env is built, run the cyclic
   GC so the leaked pair dies now, at a known safe point, instead of mid-episode.
3. ``guarded_render(...)`` / ``frames_look_bad(...)`` -- cheap image statistics
   on every frame handed out.  A frame with the failure signature is re-rendered
   with the context re-bound; if still bad the ``MjrContext`` is rebuilt; if
   still bad the whole offscreen context is recreated.  Every event is logged.

Pure rendering concern: nothing here steps the simulation or touches seeds,
actions or step budgets.
"""
from __future__ import annotations

import gc
import os
import sys
import time
from typing import Any, Callable

import numpy as np

ENABLED = os.environ.get("RENDER_GUARD", "1") not in ("0", "false", "no", "off")
VERBOSE = os.environ.get("RENDER_GUARD_VERBOSE", "0") not in ("0", "", "false", "no", "off")
# RENDER_GUARD_REAP=0 keeps layer 2 off (diagnostics: the leaked context is then collected whenever the
# GC gets to it, and the safe __del__ logs the moment -- which is the moment frames used to go bad).
REAP = os.environ.get("RENDER_GUARD_REAP", "1") not in ("0", "false", "no", "off")

# Failure signature, from scanning the frames of 10 runs: natural 256-px frames have a mean
# |horizontal neighbour difference| (grey) below 8; garbage frames 20-60; black frames mean < 5.
NDIFF_BAD = 16.0
MEAN_BLACK = 5.0
DARK_FRAC_BAD = 0.9
# A legitimately dark frame (wrist camera against a dark surface) must not make the guard rebuild
# the GL context every step: once a full recreation did not help, hold off for this long.
RECREATE_COOLDOWN_S = 30.0

_installed = False
_counters: dict[str, int] = {"rebound": 0, "rerender": 0, "rebuild_mjr": 0, "recreate_ctx": 0,
                             "unresolved": 0, "reaped": 0}
_last_recreate = 0.0
_log_fn: Callable[[dict[str, Any]], None] | None = None


def _say(msg: str) -> None:
    print(f"[render_guard] {msg}", file=sys.stderr, flush=True)


def set_logger(fn: Callable[[dict[str, Any]], None] | None) -> None:
    """Optional structured sink (the service's call log); stderr always gets a line too."""
    global _log_fn
    _log_fn = fn


def _emit(event: dict[str, Any]) -> None:
    event = {"render_guard": True, "ts": time.time(), **event}
    _say(" ".join(f"{k}={v}" for k, v in event.items() if k not in ("render_guard", "ts")))
    if _log_fn is not None:
        try:
            _log_fn(event)
        except Exception:
            pass


def counters() -> dict[str, int]:
    return dict(_counters)


# ---------------------------------------------------------------- EGL helpers

def _egl():
    from mujoco.egl import egl_ext as EGL
    return EGL


def _addr(ctx: Any) -> int:
    """Address of a PyOpenGL EGLContext handle; 0 for EGL_NO_CONTEXT / None."""
    if ctx is None:
        return 0
    try:
        return int(ctx.address)
    except Exception:
        return 0


def current_context_addr() -> int:
    try:
        return _addr(_egl().eglGetCurrentContext())
    except Exception:
        return 0


def _gl_ctx_addr(render_ctx: Any) -> int:
    gl = getattr(render_ctx, "gl_ctx", None)
    return _addr(getattr(gl, "_context", None)) if gl is not None else 0


def _make_current_addr(addr_ctx: Any) -> bool:
    """Bind a raw EGLContext handle (used to restore the previously current context)."""
    try:
        import robosuite.renderers.context.egl_context as egl_mod
        EGL = _egl()
        return bool(EGL.eglMakeCurrent(egl_mod.EGL_DISPLAY, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, addr_ctx))
    except Exception:
        return False


def ensure_current(render_ctx: Any, *, why: str = "") -> bool:
    """Re-bind ``render_ctx``'s EGL context if the thread lost it. Returns True if it had to."""
    gl = getattr(render_ctx, "gl_ctx", None)
    if gl is None or getattr(gl, "_context", None) is None:
        return False
    if current_context_addr() == _gl_ctx_addr(render_ctx):
        return False
    gl.make_current()
    _counters["rebound"] += 1
    _emit({"event": "rebound_lost_egl_context", "why": why, "n": _counters["rebound"]})
    return True


# ---------------------------------------------------------------- 1. root-cause patch

def _safe_render_context_del(self: Any) -> None:
    """Replacement for robosuite ``MjRenderContext.__del__``.

    Frees the MjrContext and the EGL context with THIS context current (so ``mjr_freeContext``
    deletes this context's GL objects, not the live one's) and re-binds whatever was current
    before (``EGLGLContext.free`` ends with ``eglReleaseThread``, which unbinds everything).
    """
    d = self.__dict__
    con, gl = d.get("con"), d.get("gl_ctx")
    prev = None
    mine = 0
    switched = False
    try:
        EGL = _egl()
        cur = EGL.eglGetCurrentContext()
        prev = cur if _addr(cur) else None
        mine = _gl_ctx_addr(self)
        if gl is not None and mine and _addr(prev) != mine:
            gl.make_current()
            switched = True
    except Exception:
        pass
    try:
        if con is not None:
            con.free()
    except Exception:
        pass
    try:
        if gl is not None:
            gl.free()
    except Exception:
        pass
    if prev is not None and _addr(prev) != mine:
        # Only when another context was live; after eglReleaseThread it must be re-bound.
        _make_current_addr(prev)
    if switched or mine:
        _counters["reaped"] += 1
        _emit({"event": "freed_render_context", "ctx": hex(mine), "switched": switched,
               "restored": hex(_addr(prev)), "current": hex(current_context_addr())})
    for key in ("con", "gl_ctx", "scn", "cam", "vopt", "pert"):
        d.pop(key, None)


def install() -> bool:
    """Patch robosuite's render context / MjSim.render. Idempotent; no-op with RENDER_GUARD=0."""
    global _installed
    if _installed or not ENABLED:
        return _installed
    try:
        from robosuite.utils import binding_utils as bu
    except Exception as exc:  # off the simulation host
        _say(f"not installed (robosuite unavailable: {exc!r})")
        return False
    bu.MjRenderContext.__del__ = _safe_render_context_del

    orig_render = bu.MjSim.render

    def render(self, width=None, height=None, *, camera_name=None, depth=False,
               mode="offscreen", device_id=-1, segmentation=False):
        ctx = getattr(self, "_render_context_offscreen", None)
        if ctx is not None:
            try:
                ensure_current(ctx, why="MjSim.render")
            except Exception as exc:
                _say(f"ensure_current failed: {exc!r}")
        return orig_render(self, width, height, camera_name=camera_name, depth=depth,
                           mode=mode, device_id=device_id, segmentation=segmentation)

    bu.MjSim.render = render
    _installed = True
    _say("installed (safe MjRenderContext.__del__, context re-bind in MjSim.render)")
    return True


# ---------------------------------------------------------------- 2. reap at a safe point

def reap_leaked_contexts(env: Any) -> int:
    """Collect the leaked construction-time sim/context now, with the live context protected.

    Call right after an env is built (``robosuite.make`` + ``env.reset()``). Returns the number
    of stale ``MjRenderContext`` objects found before collection.
    """
    if not ENABLED or not REAP:
        return 0
    install()
    try:
        from robosuite.utils.binding_utils import MjRenderContext
    except Exception:
        return 0
    live = getattr(getattr(env, "sim", None), "_render_context_offscreen", None)
    # Count only -- do not keep a reference (a loop variable left alive here would keep the cycle
    # reachable and make the gc.collect() below a no-op).
    n = sum(1 for o in gc.get_objects() if isinstance(o, MjRenderContext) and o is not live)
    before = _counters["reaped"]
    gc.collect()
    if live is not None:
        ensure_current(live, why="reap")
    if n or VERBOSE:
        _emit({"event": "reaped_leaked_contexts", "stale": n, "freed": _counters["reaped"] - before,
               "live_ctx": hex(_gl_ctx_addr(live)), "current": hex(current_context_addr())})
    return n


# ---------------------------------------------------------------- 3. frame statistics

def frame_stats(img: np.ndarray) -> tuple[float, float, float]:
    """(mean grey, mean |horizontal neighbour diff| of grey, fraction of near-black pixels)."""
    a = np.asarray(img)
    if a.ndim == 3:
        g = a.astype(np.float32).mean(axis=2)
        dark = float((a.max(axis=2) < 8).mean())
    else:
        g = a.astype(np.float32)
        dark = float((g < 8).mean())
    nd = float(np.abs(g[:, 1:] - g[:, :-1]).mean()) if g.shape[1] > 1 else 0.0
    return float(g.mean()), nd, dark


def looks_bad(img: np.ndarray) -> bool:
    m, nd, dark = frame_stats(img)
    return m < MEAN_BLACK or dark > DARK_FRAC_BAD or nd > NDIFF_BAD


def frames_look_bad(frames: dict[str, np.ndarray]) -> list[str]:
    return [k for k, v in frames.items() if looks_bad(v)]


def _stats_str(frames: dict[str, np.ndarray]) -> str:
    return " ".join(f"{k}:m{m:.0f}/nd{nd:.1f}" for k, (m, nd, _d) in
                    ((k, frame_stats(v)) for k, v in frames.items()))


# ---------------------------------------------------------------- 3b. recovery ladder

def rebuild_mjr_context(env: Any) -> None:
    """Rebuild only the MjrContext (FBOs, textures) inside the live EGL context."""
    ctx = env.sim._render_context_offscreen
    ensure_current(ctx, why="rebuild_mjr")
    try:
        ctx.con.free()
    except Exception:
        pass
    ctx._set_mujoco_context_and_buffers()


def recreate_offscreen_context(env: Any) -> None:
    """Free the live offscreen context (safely) and build a new one for the same sim."""
    from robosuite.utils.binding_utils import MjRenderContextOffscreen
    sim = env.sim
    old = sim._render_context_offscreen
    sim._render_context_offscreen = None
    if old is not None:
        _safe_render_context_del(old)
    new = MjRenderContextOffscreen(sim, device_id=getattr(env, "render_gpu_device_id", -1))
    # same visibility flags as robosuite's _reset_internal
    new.vopt.geomgroup[0] = 1 if getattr(env, "render_collision_mesh", False) else 0
    new.vopt.geomgroup[1] = 1 if getattr(env, "render_visual_mesh", True) else 0
    if sim._render_context_offscreen is not new:  # __init__ registers itself; be explicit anyway
        sim._render_context_offscreen = new
    ensure_current(new, why="recreate")


def guarded_render(env: Any, render: Callable[[], dict[str, np.ndarray]], *,
                   where: str, meta: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """Run ``render()`` (any callable returning {camera: HxWx3 uint8}); repair and retry on garbage.

    Ladder: re-bind context + re-render -> rebuild MjrContext -> recreate the offscreen context.
    The last frames rendered are returned even if still flagged (a legitimately dark frame is not
    an error); every stage is logged with ``where`` and ``meta`` (episode / timestep).
    """
    global _last_recreate
    frames = render()
    if not ENABLED:
        return frames
    bad = frames_look_bad(frames)
    if VERBOSE:
        _emit({"event": "render", "where": where, "stats": _stats_str(frames), "bad": bad,
               "current": hex(current_context_addr()), **(meta or {})})
    if not bad:
        return frames
    ctx = env.sim._render_context_offscreen
    base = {"where": where, "cameras": bad, "stats": _stats_str(frames), **(meta or {})}
    rebound = ensure_current(ctx, why=where)
    frames = render()
    bad2 = frames_look_bad(frames)
    if not bad2:
        _counters["rerender"] += 1
        _emit({"event": "recovered", "stage": "rerender", "rebound": rebound, **base})
        return frames
    try:
        rebuild_mjr_context(env)
        frames = render()
        bad3 = frames_look_bad(frames)
    except Exception as exc:
        bad3 = bad2
        _emit({"event": "rebuild_mjr_failed", "error": repr(exc)[:200], **base})
    if not bad3:
        _counters["rebuild_mjr"] += 1
        _emit({"event": "recovered", "stage": "rebuild_mjr_context", **base})
        return frames
    now = time.monotonic()
    if now - _last_recreate < RECREATE_COOLDOWN_S:
        _counters["unresolved"] += 1
        _emit({"event": "still_flagged_in_cooldown", "stats_after": _stats_str(frames), **base})
        return frames
    _last_recreate = now
    try:
        recreate_offscreen_context(env)
        frames = render()
        bad4 = frames_look_bad(frames)
    except Exception as exc:
        bad4 = bad3
        _emit({"event": "recreate_ctx_failed", "error": repr(exc)[:200], **base})
    if not bad4:
        _counters["recreate_ctx"] += 1
        _emit({"event": "recovered", "stage": "recreate_offscreen_context", **base})
    else:
        _counters["unresolved"] += 1
        _emit({"event": "unresolved_after_recreate", "stats_after": _stats_str(frames), **base})
    return frames


def obs_images(raw_obs: Any) -> dict[str, np.ndarray]:
    """The camera images of a robosuite observation dict (keys ending in ``_image``)."""
    if not raw_obs:
        return {}
    return {k: v for k, v in raw_obs.items() if k.endswith("_image") and getattr(v, "ndim", 0) == 3}
