"""Persistent RoboDojo (Isaac Sim 5.1 / Isaac Lab, ARX X5 bimanual) environment
service -- ONE process, ONE Isaac Lab env (num_envs=1), HTTP-RPC.

MILESTONE 1 = CONTROL ARM ONLY.
    The policy runs alone; there is no judge intervention. Everything the
    ``judge_driver_v7.sh`` control path touches is implemented:

        harness_env_meta / harness_episode_begin / harness_advance /
        harness_release_env / harness_quit / reset_step_counter /
        harness_read_state_privileged

    Every intervention method (``retreat``, ``execute_plan``,
    ``harness_open_action_budget``, ``reset_to_failure``, ``make_failure``,
    ``restore_snapshot``, the movement primitives, ...) is REGISTERED but
    raises :class:`NotImplementedError` with a message naming the milestone.
    That is deliberate: the control path works end-to-end, and any accidental
    treatment-arm call fails loudly at the RPC boundary instead of silently
    degrading into "the policy just kept running".

WHY a sibling module and not a flag on ``env_service.py``
    ``env_service.py`` is RoboCasa/MuJoCo all the way down: EGL device pinning,
    ``robosuite`` observation keys, ``_check_success``, OSC action scaling,
    ``cf_bench`` env construction. RoboDojo shares none of that -- it is Isaac
    Lab, a 14-dim joint-position action dict, a layout-id episode identity and a
    websocket policy protocol. What the two DO share is the wire contract that
    ``cli/harness.py`` and the bash driver speak, and that is reproduced here
    field for field so **neither the CLI nor the driver needs a single change**
    for the control path. Pick the benchmark by picking the module:

        python -m recovery_explore.env_service            # RoboCasa
        python -m recovery_explore.env_service_robodojo   # RoboDojo

IMPORTABLE WITHOUT ISAAC
    Nothing at module scope imports ``isaaclab``, ``omni``, ``torch`` or the
    RoboDojo tree. Every such import lives inside a function, so this file can
    be linted, unit-tested and read on a laptop. The first call that actually
    needs the simulator is ``harness_episode_begin`` (or ``--selftest``).

GPU SELECTION (Isaac is NOT MuJoCo)
    ``env_service.py`` deliberately leaves ``CUDA_VISIBLE_DEVICES`` unset and
    pins MuJoCo's EGL device instead, because robosuite asserts on the
    relationship between the two. RoboDojo does the opposite and we follow
    RoboDojo: ``scripts/eval_policy.sh:106`` exports
    ``CUDA_VISIBLE_DEVICES=<device_id>`` and every downstream consumer then
    hardcodes ordinal 0 (e.g. the capture buffers at
    ``env/camera_manager/capture/tiled_capture_manager.py:130`` allocate on
    ``"cuda:0"``). ``--cuda-device N`` therefore sets ``CUDA_VISIBLE_DEVICES=N``
    **before** the Isaac app is launched, and Isaac Lab's ``--device`` stays at
    its default ``cuda:0``, which is then the physical card N. Setting both
    would double-remap.

EPISODE IDENTITY  (task, seed, episode_index)  ->  RoboDojo layout id
    RoboCasa derives an episode from (task, seed, episode_index). RoboDojo does
    not have a seed in that sense at all: ``SeedManager.init_eval``
    (``env/seed_manager/seed_manager.py:27-56``) enumerates the pre-generated
    layout files ``Assets/Eval_Layout/RoboDojo/<config_name>/<eval_seed>/
    <task>_<k>.json`` and calls the *index* k the "seed"; ``EvalEnv.reset``
    then feeds that index back through ``get_seed_scene_info``. So an episode
    is one layout id in ``range(n_layouts)``.

    The mapping used here (documented, no per-task or per-seed table anywhere):

        eval_seed (the layout SET) := ``--layout-seed`` (default 0)
        layout_id (the EPISODE)    := (seed + episode_index) mod n_layouts

    ``seed + episode_index`` is exactly what the driver already varies, and the
    modulo is what makes RoboCasa-sized seeds (901, 902, ...) land inside a
    25/50-layout task. ``--episode-id-mode strict`` turns the modulo off and
    raises instead, for runs that want the raw index. ``n_layouts`` is read from
    the env's own ``seed_manager`` -- never hardcoded.

STEP BUDGET
    RoboDojo's per-task limit is ``self.step_lim``, set in each task class
    (e.g. ``task/RoboDojo/tasks/general_pickup.py:9`` -> 200,
    ``task/RoboDojo/tasks/fasten_screws.py:11`` -> 1900). It counts
    ``take_action`` calls -- policy control steps, not physics steps. One
    control step expands to ``obs_manager.collect_interval`` physics steps
    (``eval_env.py:500``), which for the shipped config is
    ``1 / (dt * collect_freq) = 1 / (0.004 * 25) = 10``.
    ``step_limit = round(step_lim * step_budget_scale) + max_steps_extra`` and
    the service writes that number back onto ``env.step_lim`` so the env's own
    clamp (``eval_env.py:368``) agrees with ours in both directions.

Transport, response envelope and call log are the ones in
``recovery_explore/rpc_transport.py``, identical to ``env_service.py``:
POST /call with ``{"method","args","kwargs"}``, reply
``{"ok": true, "result": ...}`` or ``{"ok": false, "error": ...}``, and one
JSON line per business call appended to ``<output-dir>/env_service_calls.jsonl``.
``MainThreadServeMixin`` is mandatory for the same reason as on RoboCasa: the
renderer context is thread-bound.

See ``ENV_SERVICE_ROBODOJO_NOTES.md`` next to this file for the citations, the
observation/action contract, what is still TODO pending a pod run, and the
smoke-test command sequence.
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
import os
import socket
import struct
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # package import (normal: python -m recovery_explore.env_service_robodojo)
    from recovery_explore.rpc_transport import MainThreadServeMixin, RpcFacade
except ImportError:  # pragma: no cover - direct `python env_service_robodojo.py`
    from rpc_transport import MainThreadServeMixin, RpcFacade  # type: ignore[no-redef]


DEFAULT_PORT = 8460  # 8410 is env_service.py's; keep the two families apart
DEFAULT_ENV_CFG_TYPE = "arx_x5"
DEFAULT_POLICY_NAME = "Pi_05"
DEFAULT_ACTION_TYPE = "joint"

#: RoboDojo camera key -> the camera name ``cli/harness.py`` already knows.
#: harness.py's ``write_montage`` builds one montage row per camera and looks
#: for exactly ("primary", "secondary", "wrist"), so translating here is what
#: lets the CLI stay untouched. cam_head is the scene camera (the "head" view),
#: the two wrist cameras are the close-ups.
CAMERA_ALIASES: dict[str, str] = {
    "cam_head": "primary",
    "cam_left_wrist": "secondary",
    "cam_right_wrist": "wrist",
}

#: Frames written per chunk, in montage order. Head + the two wrists.
FRAME_CAMERAS: tuple[str, ...] = ("primary", "secondary", "wrist")

#: Size (px, square) of the per-chunk review frames. Fixed, like
#: env_service.py's render size: a camera has the resolution it has, and a
#: tunable would make two batches of frames incomparable.
FRAME_SIZE = 256

#: Action-chunk length assumed BEFORE the policy has ever answered, only so
#: ``harness_episode_begin`` can report ``total_chunks``. The real value is
#: ``len(actions)`` from the first ``get_action`` and overwrites this for the
#: rest of the episode. See the notes: openpi pi0.5 chunks are 50 long in the
#: configs we have seen, but this has NOT been confirmed against the RoboDojo
#: checkpoint, so it is a CLI flag rather than a constant.
DEFAULT_POLICY_CHUNK_HINT = 50

#: Normalised gripper opening at or below which the fingers count as shut on
#: nothing. RoboCasa uses a finger gap in metres; this robot reports only the
#: normalised 0..1 opening, so the threshold lives in that unit.
# measured finger opening; the ARX X5 fingers bottom out at 0.185 (probe 2026-10-20), so 0.20 = met on nothing
EMPTY_CLOSURE_NORM = float(os.environ.get("RD_GRIP_EMPTY", "0.20"))


# Milestone 2 (the treatment arm) lives in its own module and mixes in below.
# Imported HERE, after the constants, and not at the top of the file: the mixin
# reads FRAME_SIZE from this module, so importing it earlier would hand it a
# half-initialised module and a silently different frame size.
try:
    from recovery_explore.robodojo_intervention import (
        INTERVENE_CHARGES_STEP_LIM,
        RoboDojoInterventionMixin,
    )
except ImportError:  # pragma: no cover - direct `python env_service_robodojo.py`
    from robodojo_intervention import (  # type: ignore[no-redef]
        INTERVENE_CHARGES_STEP_LIM,
        RoboDojoInterventionMixin,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _summarize(value: Any, *, depth: int = 0) -> Any:
    """Shrink a value for the call log (same intent as env_service._summarize).

    Arrays and long sequences become shape/length descriptors: the log is an
    audit trail, not a data dump, and a 14x50 action chunk per line would bury
    the call history.
    """
    if depth > 3:
        return "<...>"
    if isinstance(value, np.ndarray):
        return {"__array__": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(k): _summarize(v, depth=depth + 1) for k, v in list(value.items())[:32]}
    if isinstance(value, (list, tuple)):
        if len(value) > 16:
            return {"__seq__": len(value), "head": _summarize(list(value[:4]), depth=depth + 1)}
        return [_summarize(v, depth=depth + 1) for v in value]
    if isinstance(value, (str, bytes)):
        text = value if isinstance(value, str) else value.decode("utf-8", "replace")
        return text if len(text) <= 200 else text[:200] + "..."
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return repr(value)[:200]


def _write_png(path: Path, image: np.ndarray) -> None:
    """Zero-dependency PNG writer for an HxWx3 uint8 array.

    Same encoder as ``cli/rex.write_png`` (filter 0, colour type 2), kept local
    so the service never needs the CLI on its PYTHONPATH.
    """
    arr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 uint8, got {arr.shape}")
    height, width = arr.shape[:2]
    raw = bytearray()
    for row in arr:
        raw.append(0)
        raw.extend(row.tobytes())

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


def _resize_nearest(image: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize to ``size x size``, numpy only.

    Review frames only; the POLICY never sees a resized image from here (the
    policy server does its own preprocessing on the full-resolution frame we
    forward verbatim -- see ``_policy_observation``). Nearest keeps this
    dependency-free and, for a thumbnail a human or a VLM glances at, the
    interpolation kernel is not the thing that matters.
    """
    arr = np.asarray(image)
    height, width = arr.shape[:2]
    if (height, width) == (size, size):
        return np.ascontiguousarray(arr)
    rows = (np.arange(size) * (height / size)).astype(np.int64).clip(0, height - 1)
    cols = (np.arange(size) * (width / size)).astype(np.int64).clip(0, width - 1)
    return np.ascontiguousarray(arr[rows][:, cols])


def _as_rgb_uint8(image: Any) -> np.ndarray:
    """Coerce whatever the capture manager handed us into HxWx3 uint8.

    ``ObsManager.get_obs`` already slices RGBA down to ``[:, :, :3]``
    (``env/observation_manager/obs_manager.py:122-125``), but the dtype is only
    uint8 by convention and a torch/warp tensor can arrive instead of an
    ndarray, so normalise defensively rather than trust it.
    """
    arr = image
    if hasattr(arr, "detach"):  # torch tensor
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        raise ValueError(f"expected an image, got shape {arr.shape}")
    arr = arr[:, :, :3]
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    return np.ascontiguousarray(arr.astype(np.uint8))


def _to_list(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return [float(v) for v in np.asarray(value).reshape(-1)]


class _NotBuilt(RuntimeError):
    pass


# ---------------------------------------------------------------------------
#  Policy transport
# ---------------------------------------------------------------------------


_VPP2_CLIENT_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def _vpp2_tag(obs: Mapping[str, Any] | None = None) -> dict:
    """VPP2/deploy.py's client tagging: one server keeps one history per _vpp2_client_id."""
    tagged = dict(obs or {})
    tagged["_vpp2_client_id"] = _VPP2_CLIENT_ID
    return tagged


class _EEChunk(list):
    """get_action's EE reply (a list of per-step action dicts) with the ndarray bits harness_advance reads."""

    ndim = 2

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self), 16)


class _PolicyClient:
    """Thin façade over the two ways to reach a RoboDojo policy server.

    ``xpl_ws`` (DEFAULT, and what the official evaluation actually does)
        XPolicyLab's ``client_server.ws.model_client.WsModelClient``: a
        msgpack-over-websocket RPC whose verbs are ``reset`` / ``update_obs`` /
        ``get_action``. ``eval_env.py:188-196`` constructs exactly this client
        and ``XPolicyLab/policy/Pi_05/deploy.py:1-17`` is the whole client-side
        loop. The **raw RoboDojo observation dict goes over the wire unchanged**
        -- all the pi0.5 preprocessing (camera name mapping, CHW/uint8
        conversion, state packing, prompt) happens server-side in
        ``XPolicyLab/policy/Pi_05/model.py::encode_obs``. That is why this path
        needs no image preprocessing here at all.

    ``openpi_ws``
        ``openpi_client.WebsocketClientPolicy(host, port).infer(obs)`` against a
        bare openpi serving process. openpi expects the ALREADY-ENCODED
        observation (``state`` / ``images`` / ``prompt``), so this path
        replicates ``Pi_05/model.py::encode_obs`` locally. Provided because the
        milestone brief named it; NOT the path the RoboDojo repo takes, and it
        has not been exercised. Prefer ``xpl_ws`` unless you know the server on
        the other end is bare openpi.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        protocol: str,
        run_id: str,
        task_name: str,
        action_type: str,
        env_cfg_type: str,
    ) -> None:
        self.protocol = protocol
        self.endpoint = endpoint
        self._task_name = task_name
        self._action_type = action_type
        self._env_cfg_type = env_cfg_type
        self._impl: Any = None
        self._dim_info: Any = None
        host, port = _split_endpoint(endpoint)
        if protocol == "xpl_ws":
            from client_server.ws.model_client import WsModelClient  # lazy

            self._impl = WsModelClient(
                url=f"ws://{host}:{port}",
                evaluation_id=run_id,
                trial_id=f"{task_name}-{run_id}",
                action_case_id=f"{task_name}_case",
                repeat_index=None,
            )
        elif protocol == "openpi_ws":
            from openpi_client.websocket_client_policy import (  # lazy
                WebsocketClientPolicy,
            )
            from XPolicyLab.utils.process_data import get_robot_action_dim_info

            self._impl = WebsocketClientPolicy(host=host, port=port)
            self._dim_info = get_robot_action_dim_info(env_cfg_type)
        else:  # pragma: no cover - argparse constrains this
            raise ValueError(f"unknown policy protocol {protocol!r}")

    # -- verbs ---------------------------------------------------------------

    def reset(self) -> None:
        if self.protocol == "xpl_ws":
            self._impl.call(func_name="reset")
            if self._action_type == "ee":
                self._impl.call(func_name="reset_client", obs=_vpp2_tag())
        # openpi's WebsocketClientPolicy is stateless per infer(); nothing to do.

    def act(self, obs: Mapping[str, Any]) -> np.ndarray:
        """One inference. Returns the chunk as (T, 14) in packed order.

        Packed order is the one ``XPolicyLab/utils/process_data.py:119-134``
        documents for a dual-arm robot:
        ``[left_arm(6), left_ee(1), right_arm(6), right_ee(1)]``.
        """
        if self.protocol == "xpl_ws":
            if self._action_type == "ee":
                self._impl.call(func_name="update_obs", obs=_vpp2_tag(obs))
                return _EEChunk(self._impl.call(func_name="get_action", obs=_vpp2_tag()))
            self._impl.call(func_name="update_obs", obs=dict(obs))
            actions = self._impl.call(func_name="get_action")
            return _pack_action_chunk(actions)

        encoded = _encode_obs_for_openpi(obs, self._action_type, self._dim_info)
        result = self._impl.infer(encoded)
        return np.asarray(result["actions"], dtype=np.float64)

    def observe(self, obs: Mapping[str, Any]) -> None:
        """Push an observation WITHOUT asking for an action.

        ``Pi_05/deploy.py:16-17`` does this between the steps of a chunk. It
        does not change what is executed -- the chunk was already decided -- but
        it does change the server's own observation window, so mirroring it
        keeps our rollout bit-identical in structure to the official one.
        """
        if self.protocol == "xpl_ws":
            self._impl.call(func_name="update_obs", obs=_vpp2_tag(obs) if self._action_type == "ee" else dict(obs))

    def close(self) -> None:
        try:
            close = getattr(self._impl, "close", None)
            if callable(close):
                close()
        except Exception:
            pass


def _split_endpoint(endpoint: str) -> tuple[str, int]:
    """``[ws://|http://]host:port`` -> ``(host, port)``.

    The driver and ``launch_svc.sh`` hand around ``http://IP:PORT`` for the
    RoboCasa policy server; accepting that spelling means the launcher needs no
    special case for RoboDojo.
    """
    rest = endpoint
    if "://" in rest:
        rest = rest.split("://", 1)[1]
    host, _, port = rest.partition(":")
    if not host or not port:
        raise ValueError(f"policy endpoint must be [scheme://]host:port, got {endpoint!r}")
    return host, int(port)


def _pack_action_chunk(actions: Any) -> np.ndarray:
    """Normalise ``get_action``'s reply into an (T, 14) float array.

    ``Pi_05/model.py:139-149`` returns ``unpack_robot_state(...)`` with
    ``source_type="obs"`` on a 2-D array, i.e. a LIST of per-step dicts keyed
    ``left_arm_joint_state`` / ``left_ee_joint_state`` / ``right_arm_joint_state``
    / ``right_ee_joint_state``. Other policies in XPolicyLab return a plain
    (T, D) array. Accept both, and re-pack dicts in the documented order so the
    rest of this module only ever handles one shape.
    """
    if isinstance(actions, Mapping):
        actions = [actions]
    if isinstance(actions, (list, tuple)) and actions and isinstance(actions[0], Mapping):
        rows = []
        for step in actions:
            rows.append(
                _to_list(step["left_arm_joint_state"])
                + _to_list(step["left_ee_joint_state"])
                + _to_list(step["right_arm_joint_state"])
                + _to_list(step["right_ee_joint_state"])
            )
        return np.asarray(rows, dtype=np.float64)
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _unpack_action_row(row: np.ndarray) -> dict[str, np.ndarray]:
    """(14,) -> the action dict ``EvalEnv.take_action`` validates.

    Dimensions come from ``env_cfg/robot/_robot_info.json``
    (``dual_x5: arm_dim [6,6], ee_dim [1,1]``) and the key names from
    ``EvalEnv.validate_action_dict`` (``eval_env.py:595-608``), which is the
    authority: it rejects any unexpected key outright.
    """
    if isinstance(row, Mapping):  # an EE16 step from VPP2 goes to take_action unchanged
        return {k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in row.items()}
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    if row.shape[0] != 14:
        raise ValueError(f"expected a 14-dim joint action, got {row.shape[0]}")
    return {
        "left_arm_joint_state": row[0:6],
        "left_ee_joint_state": row[6:7],
        "right_arm_joint_state": row[7:13],
        "right_ee_joint_state": row[13:14],
    }


def _encode_obs_for_openpi(obs: Mapping[str, Any], action_type: str, dim_info: Any) -> dict:
    """Local copy of ``Pi_05/model.py::encode_obs`` for the bare-openpi path.

    Only used when ``--policy-protocol openpi_ws``. On the default ``xpl_ws``
    path the server does this itself and we must NOT pre-encode.
    """
    from XPolicyLab.utils.process_data import pack_robot_state

    def image(names: Sequence[str]) -> np.ndarray:
        vision = obs.get("vision", {})
        for name in names:
            if name in vision:
                data = vision[name]
                raw = data.get("color", data.get("rgb")) if isinstance(data, Mapping) else data
                if raw is not None:
                    return np.transpose(_as_rgb_uint8(raw), (2, 0, 1))  # -> CHW uint8
        raise KeyError(f"no image among {list(names)}; have {list(obs.get('vision', {}))}")

    return {
        "state": pack_robot_state(obs, action_type, dim_info, source_type="obs").astype(np.float32),
        "images": {
            "cam_high": image(("cam_high", "cam_head", "head_camera", "top_camera")),
            "cam_left_wrist": image(("cam_left_wrist", "left_camera", "left_wrist")),
            "cam_right_wrist": image(("cam_right_wrist", "right_camera", "right_wrist")),
        },
        "prompt": obs.get("instruction"),
    }


# ---------------------------------------------------------------------------
#  The service
# ---------------------------------------------------------------------------


def _with_depth(cam_cfg: dict) -> dict:
    """RD_DEPTH=1: give every camera a distance_to_image_plane annotator (the Gemini 345Lg head camera and the
    wrist cameras are RGB-D devices). ObsManager pops it unless observation.vision.depth is on, so the policy's
    observation is unchanged; the grasp tool switches depth on for its own single read."""
    if os.environ.get("RD_DEPTH", "0") != "1":
        return cam_cfg
    ann = cam_cfg.get("annotator", {})
    for name, block in ann.items():
        if name == "common" or not isinstance(block, dict):
            continue
        block.setdefault("distance_to_image_plane_capture", {"type": "distance_to_image_plane", "device": "cpu"})
    return cam_cfg


class RoboDojoEnvService(RoboDojoInterventionMixin, MainThreadServeMixin, RpcFacade):
    """Holds exactly one Isaac Lab RoboDojo env (num_envs=1) for one task."""

    #: Still unimplemented after milestone 2. Every one of these belongs to the
    #: RoboCasa *failure-factory* line of work (generate a failure state, rewind
    #: an episode to it, replay from it) which the judge experiment's treatment
    #: arm does not use -- it intervenes mid-trajectory and rewinds only its own
    #: actions, which is ``reset_window``. Registered so an accidental call is a
    #: clean RPC error rather than "unknown RPC method".
    _UNIMPLEMENTED_METHODS: tuple[str, ...] = (
        "make_failure",
        "reset_to_failure",
        "run_episode_to_failure",
        "resume_episode",
        "harness_round_begin",
        "harness_oracle_grasp",   # RoboCasa grasp predicate; use _oracle() here
        "call_cosmos",            # no Cosmos world model wired for RoboDojo
        "unproject_batch",
        "env_meta",
        "debug_reset",
    )

    def __init__(
        self,
        *,
        output_dir: Path,
        run_id: str,
        log_path: Path,
        policy_endpoint: str | None = None,
        policy_family: str = "pi05",
        policy_protocol: str = "xpl_ws",
        policy_name: str = DEFAULT_POLICY_NAME,
        policy_chunk_hint: int = DEFAULT_POLICY_CHUNK_HINT,
        action_type: str = DEFAULT_ACTION_TYPE,
        env_cfg_type: str = DEFAULT_ENV_CFG_TYPE,
        robodojo_root: Path | None = None,
        layout_seed: int = 0,
        episode_id_mode: str = "modulo",
        step_budget_scale: float = 1.0,
        cuda_device: int | None = None,
    ) -> None:
        # ABSOLUTE, and resolved HERE -- before anything can chdir.
        # ``_ensure_app`` does ``os.chdir(robodojo_root)`` because RoboDojo
        # resolves Assets/ and eval_result/ relatively. A relative output dir
        # created at startup against the repo checkout then re-resolves against
        # the RoboDojo tree on the first log write, and the service dies with
        # FileNotFoundError on env_service_calls.jsonl. It only looked fine
        # while a previous run had happened to leave a same-named directory
        # under the RoboDojo tree; every fresh run-id hit it.
        self._output_dir = Path(output_dir).resolve()
        self._run_id = str(run_id)
        self._log_path = Path(log_path).resolve()
        self._policy_endpoint = policy_endpoint
        self._policy_family = str(policy_family)
        self._policy_protocol = str(policy_protocol)
        self._policy_name = str(policy_name)
        self._policy_chunk_hint = int(policy_chunk_hint)
        self._action_type = str(action_type)
        self._env_cfg_type = str(env_cfg_type)
        self._robodojo_root = Path(robodojo_root) if robodojo_root else None
        self._layout_seed = int(layout_seed)
        if episode_id_mode not in ("modulo", "strict"):
            raise ValueError("episode_id_mode must be 'modulo' or 'strict'")
        self._episode_id_mode = str(episode_id_mode)
        if not 0.1 <= float(step_budget_scale) <= 5.0:
            raise ValueError("step_budget_scale must be in [0.1, 5.0]")
        self._step_budget_scale = float(step_budget_scale)
        self._cuda_device = cuda_device

        self._app: Any = None          # Isaac ``simulation_app``
        self._env: Any = None          # the EvalEnv instance
        self._env_task: str | None = None
        self._client: _PolicyClient | None = None
        self._episode: dict[str, Any] | None = None
        self._step_count = 0
        self._call_seq = 0
        self._act_log: list[dict[str, Any]] = []
        #: Empty-closure history + the per-arm latch that turns a continuous
        #: opening into discrete "the fingers just met on nothing" events.
        self._empty_closures: list[dict[str, Any]] = []
        self._empty_now: dict[str, bool] = {"left": False, "right": False}
        self._instance_nonce = f"{os.getpid():05d}{int(time.time()) % 100000:05d}"

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        # Treatment-arm bookkeeping (budget, judge lock, snapshots, arm choice).
        # Must run before super().__init__() -> _register_rpc().
        self._init_intervention_state()
        super().__init__()

    # ---- RPC plumbing -----------------------------------------------------

    def _register_rpc(self) -> None:
        self._rpc.update(
            {
                "harness_env_meta": self.harness_env_meta,
                "harness_episode_begin": self.harness_episode_begin,
                "harness_advance": self.harness_advance,
                "harness_release_env": self.harness_release_env,
                "harness_quit": self.harness_quit,
                "reset_step_counter": self.reset_step_counter,
                "harness_read_state_privileged": self.harness_read_state_privileged,
                "debug_depth_probe": self.debug_depth_probe,
                "debug_gripper_geom": self.debug_gripper_geom,
                "debug_objects": self.debug_objects,
                "debug_obs_drift": self.debug_obs_drift,
                "debug_ik_probe": self.debug_ik_probe,
                "debug_estimate_truth": self.debug_estimate_truth,
                "debug_camera_check": self.debug_camera_check,
            }
        )
        # Milestone 2: the treatment arm. One table, defined next to the code
        # that implements it, so "what can the judge call" has a single answer.
        self._rpc.update(self._intervention_rpc())
        for name in self._UNIMPLEMENTED_METHODS:
            self._rpc[name] = self._make_stub(name)

    def _make_stub(self, name: str):
        def _stub(*_args: Any, **_kwargs: Any) -> Any:
            raise NotImplementedError(
                f"{name}() is not implemented on the RoboDojo service. Both arms "
                "work: the control path (harness_episode_begin / harness_advance / "
                "harness_read_state_privileged / harness_release_env / harness_quit) "
                "and the treatment path (harness_judge_lock, harness_open_action_budget, "
                "execute_plan and the primitives, reset_window, harness_close_action_budget). "
                f"{name} belongs to the RoboCasa failure-factory line of work, which this "
                "benchmark does not use -- see robodojo_intervention.py's module docstring."
            )

        _stub.__name__ = f"stub_{name}"
        return _stub

    def _log(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, default=str, sort_keys=True)
        with self._log_path.open("a") as stream:
            stream.write(line + "\n")

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        """Time every business call, append one JSON log line. Mirrors
        ``env_service._dispatch`` including the ``healthz`` exemption (the
        readiness poll would otherwise bury the call history).

        There is no privileged-method gate here because milestone 1 has no
        model-facing surface at all: the only caller is ``cli/harness.py``.
        The gate comes back with the intervention methods in milestone 2.
        """
        if method == "healthz":
            return super()._dispatch(method, args, kwargs)
        # Action budget open = the model is driving; judge lock on = the model
        # is looking. Either way every privileged method is refused, so a judge
        # with a shell cannot curl the success predicate it is being tested on.
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
            "anchor_id": None,  # RoboCasa-only concept; kept so the two logs parse alike
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

    def _next_call_seq(self) -> int:
        self._call_seq += 1
        return self._call_seq

    def close(self) -> None:
        self._teardown_env()
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._app is not None:
            try:
                self._app.close()
            except Exception:
                pass
            self._app = None

    # ---- Isaac bring-up ---------------------------------------------------

    def _ensure_app(self) -> Any:
        """Launch the Omniverse app ONCE per process, headless.

        Ordering here is load-bearing and copied from
        ``src/eval_client/main.py:123-132``:

        1. ``CUDA_VISIBLE_DEVICES`` must be set before anything CUDA-aware is
           imported (see the module docstring on GPU selection).
        2. ``AppLauncher`` must run before ANY other ``isaaclab`` /
           ``omni.*`` / RoboDojo-env import -- those modules only exist once
           Kit has booted.
        3. ``add_zero_delay_kit_args`` (``env/camera_manager/capture/
           render_sync.py``) must be applied to the launcher args or camera
           frames lag physics by a frame, which silently feeds the policy a
           stale image.
        """
        if self._app is not None:
            return self._app

        if self._robodojo_root is not None:
            root = str(self._robodojo_root)
            for entry in (root, os.path.join(root, "XPolicyLab")):
                if entry not in sys.path:
                    sys.path.insert(0, entry)
            os.chdir(root)  # RoboDojo resolves Assets/ and eval_result/ relatively

        # Isaac Sim 5.1 / Isaac Lab 2.x (the official RoboDojo tree), same as scripts/eval_policy.sh:
        # pick the GPU with CUDA_VISIBLE_DEVICES and let Isaac use cuda:0 -- on the 580 node the
        # renderer follows it (Kit's GPU table shows the visible card as Active). The Isaac 6 /
        # Isaac Lab 3.0 workarounds (no CUDA_VISIBLE_DEVICES, renderer pinning through kit args,
        # aliased --headless/--enable_cameras, an explicit experience file) do not apply here.
        if self._cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(int(self._cuda_device))
        self._isaac_device = "cuda:0"
        print(f"[robodojo-svc] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
        os.environ.setdefault(
            "ROBODOJO_RUN_ID", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

        from isaaclab.app import AppLauncher  # lazy: needs the Isaac install

        parser = argparse.ArgumentParser(add_help=False)
        AppLauncher.add_app_launcher_args(parser)
        app_args = parser.parse_args(
            [
                "--headless",
                "--enable_cameras",
                "--kit_args",
                " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera",
            ]
        )

        from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

        add_zero_delay_kit_args(app_args)
        launcher = AppLauncher(app_args)
        self._app = launcher.app
        return self._app

    def _build_env(self, task: str) -> Any:
        """Assemble the OmegaConf tree and build the EvalEnv for ``task``.

        A faithful reduction of ``src/eval_client/main.py::main``
        (``main.py:254-346``) to the single-env case: same yaml files, same
        ``process_randomization`` / ``process_config`` passes, same deploy_cfg
        shape. Reusing ``create_eval_env`` rather than re-deriving the task
        class matters -- it is what wires up ObsManager, SeedManager, the
        reward manager and the policy ``model_client``.
        """
        self._ensure_app()

        from omegaconf import OmegaConf

        from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
        from src.eval_client.eval_env import create_eval_env
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        import importlib

        task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
        benchmark_path = os.path.join(ROOT_DIR, "task", BENCHMARK)

        eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, self._env_cfg_type + ".yml"))
        eval_cfg["task_name"] = task
        eval_cfg["num_envs"] = 1
        # RoboDojo 的 scripts/eval_policy.sh 会拿 device_id 去设 CUDA_VISIBLE_DEVICES；
        # 我们不走那个脚本（直接在进程内建环境），所以这里把物理序号原样传下去即可。
        eval_cfg["device_id"] = 0  # CUDA_VISIBLE_DEVICES already selects the card
        # num_envs is forced to 1, so batch inference is irrelevant; saying so
        # explicitly keeps EvalEnv on the single-env code path.
        eval_cfg["eval_batch"] = False
        eval_cfg["policy_name"] = self._policy_name
        eval_cfg["additional_info"] = f"recovery_explore={self._run_id}"
        eval_cfg["seed"] = self._layout_seed
        eval_cfg["physx_monitor_enabled"] = False

        host, port = _split_endpoint(self._policy_endpoint or "127.0.0.1:6000")
        deploy_cfg = {
            "policy_name": self._policy_name,
            "host": host,
            "port": port,
            "protocol": "ws",
            "policy_server_url": f"ws://{host}:{port}",
            "evaluation_id": os.environ["ROBODOJO_RUN_ID"],
            "trial_id": f"{task}-{os.environ['ROBODOJO_RUN_ID']}",
            "action_case_id": f"{task}_case",
            "repeat_index": None,
        }

        cfg = OmegaConf.create(
            {
                "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml")),
                "scene": load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", eval_cfg["config"]["scene"] + ".yml")),
                "camera": _with_depth(load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", eval_cfg["config"]["camera"] + ".yml"))),
                "robot": load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", eval_cfg["config"]["robot"] + ".yml")),
                "task_env": load_yaml(
                    task_registry.task_config_path(os.path.join(benchmark_path, "config"), task)
                ),
                "eval_cfg": eval_cfg,
                "deploy_cfg": deploy_cfg,
            }
        )
        OmegaConf.update(cfg, "sim.scene.num_envs", 1, force_add=True)
        OmegaConf.update(cfg, "eval_cfg.num_envs", 1, force_add=True)
        cfg = process_randomization(cfg)
        cfg, _native_eval_num = process_config(cfg, task_name=task)
        eval_cfg["eval_num"] = 1  # we drive episode selection ourselves
        OmegaConf.update(
            cfg,
            "camera.default_frequency",
            eval_cfg["observation"].get("collect_freq", 0),
            force_add=True,
        )
        cfg.sim.seed = [0]

        env = create_eval_env(cfg, self._app)
        self._env = env
        self._env_task = task
        return env

    def _teardown_env(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        self._env = None
        self._env_task = None
        self._episode = None
        self._act_log = []
        self._step_count = 0

    def _require_env(self) -> Any:
        if self._env is None:
            raise _NotBuilt(
                "no env: the harness must call harness_episode_begin first"
            )
        return self._env

    def _require_episode(self) -> dict[str, Any]:
        if self._episode is None:
            raise _NotBuilt("no episode: call harness_episode_begin first")
        return self._episode

    def _policy(self) -> _PolicyClient:
        """The policy handle.

        On the default ``xpl_ws`` protocol this is the client ``EvalEnv``
        already built for us (``eval_env.py:188``) -- reusing it is not an
        optimisation but a correctness point: ``EvalEnv.reset`` itself calls
        ``model_client.call("reset")`` (``eval_env.py:245``), so a second,
        separate connection would have a different trial identity and a
        different server-side observation window from the one the env reset.
        """
        if self._policy_endpoint is None:
            raise RuntimeError("--policy-endpoint is required to advance the policy")
        if self._client is None:
            env = self._env
            if self._policy_protocol == "xpl_ws" and env is not None and getattr(env, "model_client", None):
                self._client = _PolicyClient.__new__(_PolicyClient)
                self._client.protocol = "xpl_ws"
                self._client.endpoint = self._policy_endpoint
                self._client._impl = env.model_client
                self._client._dim_info = None
                self._client._task_name = self._env_task or ""
                self._client._action_type = self._action_type
                self._client._env_cfg_type = self._env_cfg_type
            else:
                self._client = _PolicyClient(
                    endpoint=self._policy_endpoint,
                    protocol=self._policy_protocol,
                    run_id=self._run_id,
                    task_name=self._env_task or "unknown",
                    action_type=self._action_type,
                    env_cfg_type=self._env_cfg_type,
                )
        return self._client

    # ---- episode identity -------------------------------------------------

    def _layout_id(self, seed: int, episode_index: int) -> tuple[int, int]:
        """(seed, episode_index) -> (layout_id, n_layouts). See the module docstring."""
        env = self._require_env()
        n_layouts = len(env.seed_manager.seed_list)
        if n_layouts <= 0:
            raise RuntimeError(
                "SeedManager found no layout files for this task; check "
                "Assets/Eval_Layout/<benchmark>/<config>/<layout-seed>/"
            )
        raw = int(seed) + int(episode_index)
        if self._episode_id_mode == "strict":
            if not 0 <= raw < n_layouts:
                raise ValueError(
                    f"seed+episode_index={raw} is outside this task's "
                    f"{n_layouts} layouts and --episode-id-mode=strict"
                )
            return raw, n_layouts
        return raw % n_layouts, n_layouts

    # ---- observation / telemetry -----------------------------------------

    def _observation(self) -> dict[str, Any]:
        """One raw RoboDojo observation for env 0.

        ``EvalEnv.get_obs`` (``eval_env.py:266-267``) is
        ``get_obs_batch([0])[0]``: it renders, reads the cameras and the robot
        state, and appends the frame to the episode video writers. Keys:
        ``vision`` (per camera -> ``color``/``shape``), ``state``, ``action``,
        ``instruction``, ``additional_info``, ``data_format_version``,
        ``env_idx``.
        """
        return self._require_env().get_obs()

    def _proprio_state(self, obs: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Proprioception, RoboDojo flavour. The analogue of
        ``env_service._proprio_state``.

        14 joint positions in the SAME packed order the policy speaks, plus the
        two gripper scalars broken out, plus the end-effector poses when the env
        exposes them (``world_ee_state: true`` in ``env_cfg/arx_x5.yml:13``
        makes ObsManager publish ``left_ee_pose`` / ``right_ee_pose``, each a
        7-vector: xyz + wxyz quaternion).
        """
        obs = obs if obs is not None else self._observation()
        state = obs.get("state", {}) or {}
        measured = self._grip_measured()
        left_arm = _to_list(state.get("left_arm_joint_state"))
        right_arm = _to_list(state.get("right_arm_joint_state"))
        left_ee = _to_list(state.get("left_ee_joint_state"))
        right_ee = _to_list(state.get("right_ee_joint_state"))
        episode = self._episode or {}
        env = self._env
        return {
            # 14-dim, packed [left_arm(6), left_ee(1), right_arm(6), right_ee(1)]
            "joint_positions": left_arm + left_ee + right_arm + right_ee,
            "left_arm_joint_state": left_arm,
            "right_arm_joint_state": right_arm,
            # Normalised 0..1 opening, the same scalar the policy commands
            # (ObsManager rescales the raw joint back through gripper_scale --
            # obs_manager.py:185-212).
            # 2026-10-20: the observation's ee_joint_state is the controller's last (rate-limited)
            # COMMAND, not the finger position, so it cannot tell a held object from an empty close.
            # "gripper" is now the measured finger joint (the encoder a real ARX X5 reports),
            # normalised the same way; "gripper_cmd" keeps the command echo.
            "gripper": {
                "left": measured.get("left", left_ee[0] if left_ee else None),
                "right": measured.get("right", right_ee[0] if right_ee else None),
            },
            "gripper_cmd": {
                "left": left_ee[0] if left_ee else None,
                "right": right_ee[0] if right_ee else None,
            },
            "left_ee_pose": _to_list(state.get("left_ee_pose")) or None,
            "right_ee_pose": _to_list(state.get("right_ee_pose")) or None,
            "step_count": int(self._step_count),
            "control_steps_used": int(getattr(env, "take_action_cnt", [0])[0]) if env is not None else 0,
            "policy_steps_max": int(episode.get("max_steps") or 0) or None,
            "policy_steps_used": self._steps_used_now(episode, env),
            "policy_steps_left": (
                max(0, int(episode["max_steps"]) - self._steps_used_now(episode, env))
                if episode.get("max_steps")
                else None
            ),
            "recent_chunks": list(self._act_log[-6:]),
        }

    def _grip_measured(self) -> dict[str, float]:
        """Measured finger opening per arm, 0 (shut) .. 1 (open): the gripper joint position
        (proprioception) mapped through gripper_scale exactly like ObsManager maps the command."""
        out: dict[str, float] = {}
        env = self._env
        rm = getattr(env, "robot_manager", None) if env is not None else None
        if rm is None:
            return out
        try:
            for robot in rm.robot_list:
                side = rm.process_name(robot.arm_name).split("_")[0]
                if side not in ("left", "right") or getattr(robot, "ee_type", "gripper") != "gripper":
                    continue
                val = float(np.asarray(rm.get_end_effector_real_val(robot, env_idx_list=[0])[0]).reshape(-1)[0])
                s0, s1 = float(robot.gripper_scale[0]), float(robot.gripper_scale[1])
                norm = (val - s0) / (s1 - s0) if robot.gripper_move["sign"] == 1 else (s1 - val) / (s1 - s0)
                out[side] = float(np.clip(norm, 0.0, 1.0))
        except Exception as exc:  # never break the episode over a telemetry field
            print(f"[robodojo-svc] measured gripper unavailable: {exc!r}", flush=True)
        return out

    def _oracle(self) -> dict[str, Any]:
        """Privileged outcome: success, partial score, end flag.

        Success is ``EvalEnv.is_episode_end`` territory
        (``eval_env.py:841-867``): a reward of ~1.0 latches
        ``success[i]=True``; hitting ``step_lim`` latches failure. The partial
        score is ``reward_manager.get_score()`` in percent, and ``run_eval``
        (``eval_env.py:795-817``) scores an episode 1.0 on success, else
        ``process_score/100``. Reproduced here so a single episode reports the
        same number the batch harness would have.
        """
        env = self._require_env()
        success = bool(env.success[0]) and bool(env.end_flag[0])
        # ``hasattr(env, "get_score")`` is a per-TASK property, not a global
        # one: only 35 of the 54 task classes define a staged score (the list
        # is ``task/RoboDojo/tasks/*.py``). ``general_pickup`` and the rest are
        # pass/fail, and ``run_eval`` scores their failures 0.0
        # (``eval_env.py:901-905``) rather than skipping them. Returning None
        # there would make the 420-episode mean uncomputable, so the None is
        # kept only in ``process_score`` (where it honestly means "this task has
        # no partial credit") and ``episode_score`` is always a number.
        score_pct: float | None = None
        try:
            if hasattr(env, "get_score"):
                score_pct = float(env.reward_manager.get_score()[0])
        except Exception:
            score_pct = None
        if success:
            episode_score = 1.0
        elif score_pct is not None:
            episode_score = score_pct / 100.0
        else:
            episode_score = 0.0
        return {
            "task_success": success,
            # 0..1, exactly ``run_eval``'s per-episode contribution; the
            # leaderboard number is mean(episode_score) * 100.
            "episode_score": episode_score,
            # 0..100 (RoboDojo's own units) or None when the task has no
            # staged score at all. NOT the same scale as episode_score.
            "process_score": score_pct,
            "has_process_score": bool(hasattr(env, "get_score")),
            "end_flag": bool(env.end_flag[0]),
            "control_steps_used": int(env.take_action_cnt[0]),
            "step_limit": int(env.step_lim),
        }

    # ---- harness surface --------------------------------------------------

    def harness_env_meta(self) -> dict[str, Any]:
        """HARNESS ONLY: full bookkeeping for this process.

        Field names match ``env_service.harness_env_meta`` wherever the concept
        exists on both benchmarks, so the driver's ``jget`` calls and the
        readiness probe (``judge_driver_v7.sh:150,162`` just check that ``meta``
        answers) work unchanged. RoboCasa-only fields (``anchor_id``,
        ``object_name``, ``state_sha256``) are present and ``None`` rather than
        missing -- an absent key and a null key read differently downstream.
        """
        episode = self._episode or {}
        return {
            "benchmark": "RoboDojo",
            "run_id": self._run_id,
            "anchors_jsonl": None,
            "anchor_id": None,
            "state_id": f"{self._instance_nonce}-{self._call_seq}",
            "mode": "control" if self._episode else None,
            "task": self._env_task,
            "instruction": episode.get("instruction"),
            "object_name": None,
            "state_sha256": None,
            "origin_timestep": int(episode.get("committed_timestep") or 0) if episode else None,
            "episode": dict(episode) if episode else None,
            "step_count": int(self._step_count),
            "render_size": FRAME_SIZE,
            "policy_endpoint": self._policy_endpoint,
            "policy_family": self._policy_family,
            "policy_protocol": self._policy_protocol,
            "policy_name": self._policy_name,
            "step_budget_scale": self._step_budget_scale,
            "action_dim": 14,
            "action_type": self._action_type,
            "env_cfg_type": self._env_cfg_type,
            "layout_seed": self._layout_seed,
            "episode_id_mode": self._episode_id_mode,
            "cuda_device": self._cuda_device,
            "isaac_running": self._app is not None,
            # Milestone marker: a caller that expects interventions can detect
            # this service for what it is instead of discovering it method by
            # method.
            "milestone": 2,
            "supports_intervention": True,
            "round_index": int(self._round_index),
            "round_budget": self._round_budget,
            "round_used": int(self._round_used),
            "resets_used": int(self._resets_used),
            "intervene_control_steps": int(self._intervene_steps),
            "intervene_charges_step_lim": bool(
                INTERVENE_CHARGES_STEP_LIM
            ),
        }

    def harness_episode_begin(
        self,
        task: str,
        seed: int = 0,
        episode_index: int = 0,
        obj_instance_split: str | None = None,
        max_steps_extra: int = 0,
    ) -> dict[str, Any]:
        """HARNESS ONLY: build/reset one episode and stop before the first action.

        Same signature as the RoboCasa one so ``cli/harness.py episode-begin``
        needs no change. ``obj_instance_split`` is a RoboCasa concept (which
        half of the object asset pool a scene may draw from); RoboDojo picks its
        objects inside the pre-generated layout file, so the argument is
        accepted and echoed but has no effect. Rejecting it instead would break
        the driver for no gain.

        The env is built on the first call and REUSED afterwards, via
        ``close()`` + ``reset(seed=[layout_id])`` -- the same reuse the official
        loop does between batches (``main.py:439-444``). A request for a
        DIFFERENT task is refused rather than silently rebuilt: the task class
        is baked into the env at construction, rebuilding Isaac in-process is
        not a supported path, and the driver already gets a fresh process per
        arm via ``harness_quit`` + ``sim_lane.sh``.
        """
        started = time.monotonic()
        if self._env is not None and self._env_task != task:
            raise RuntimeError(
                f"this process holds an env for task {self._env_task!r}; "
                f"RoboDojo bakes the task class into the env, so switching to "
                f"{task!r} needs a fresh process. Call harness_quit and let "
                "sim_lane.sh relaunch."
            )
        if self._env is None:
            self._build_env(task)
        else:
            # Official between-episode reuse: drop the scene, keep the app.
            self._env.close()

        env = self._env
        layout_id, n_layouts = self._layout_id(seed, episode_index)

        env.reset(seed=[layout_id])
        if self._action_type == "ee" and self._policy_endpoint is not None:
            self._policy().reset()  # tagged reset_client: drop this lane's VPP2 history
        # ``run_eval`` primes the reward/score machinery before handing control
        # to the policy (``eval_env.py:772-774``); skipping it leaves the
        # first ``is_episode_end`` comparing against an uninitialised state.
        env.run_reward()
        if hasattr(env, "get_score"):
            env.get_score()

        native_step_lim = int(env.step_lim)
        max_steps = int(round(native_step_lim * self._step_budget_scale)) + max(0, int(max_steps_extra))
        # Write the budget back so the env's own clamp agrees with ours in both
        # directions -- ``take_action_batch`` drops actions once
        # ``take_action_cnt == step_lim`` (``eval_env.py:368``), and
        # ``is_episode_end`` latches failure at the same number.
        env.step_lim = max_steps

        obs = self._observation()
        instruction = str(obs.get("instruction") or "")
        horizon = int(self._policy_chunk_hint)

        self._step_count = 0
        try:  # rd16: where each hand starts (re-stage target at hand-back)
            self._home_pose = {sd: self._ee_pose(sd, obs) for sd in ("left", "right")}
        except Exception:
            self._home_pose = {}
        self._act_log = []
        self._empty_closures = []
        self._empty_now = {"left": False, "right": False}
        self._episode = {
            "task": str(task),
            "seed": int(seed),
            "episode_index": int(episode_index),
            "obj_instance_split": obj_instance_split,
            "layout_id": int(layout_id),
            "n_layouts": int(n_layouts),
            "instruction": instruction,
            "committed_timestep": 0,
            "next_query_index": 0,
            "max_steps": int(max_steps),
            "native_step_lim": native_step_lim,
            "horizon": horizon,
            "horizon_confirmed": False,
            "total_chunks": int(math.ceil(max_steps / max(1, horizon))),
            "frames_written": 0,
        }
        # Episode-scoped output directory, mirroring env_service's layout:
        # <output-dir>/<task>_s<seed>_ep<index>/
        self._episode_dir().mkdir(parents=True, exist_ok=True)

        return {
            "ok": True,
            "task": str(task),
            "seed": int(seed),
            "episode_index": int(episode_index),
            "obj_instance_split": obj_instance_split,
            "layout_id": int(layout_id),
            "n_layouts": int(n_layouts),
            "instruction": instruction,
            "horizon": horizon,
            "horizon_confirmed": False,
            "max_steps": int(max_steps),
            "native_step_lim": native_step_lim,
            "total_chunks": int(self._episode["total_chunks"]),
            "policy_family": self._policy_family,
            "build_s": round(time.monotonic() - started, 2),
        }

    def _episode_dir(self) -> Path:
        episode = self._require_episode()
        return self._output_dir / (
            f"{episode['task']}_s{int(episode['seed'])}_ep{int(episode['episode_index'])}"
        )

    def _frames_from_obs(self, obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Per-chunk review frames: head + both wrists, aliased and thumbnailed."""
        vision = obs.get("vision", {}) or {}
        out: dict[str, np.ndarray] = {}
        for cam_key, alias in CAMERA_ALIASES.items():
            data = vision.get(cam_key)
            if data is None:
                continue
            raw = data.get("color") if isinstance(data, Mapping) else data
            if raw is None:
                continue
            out[alias] = _resize_nearest(_as_rgb_uint8(raw), FRAME_SIZE)
        return {alias: out[alias] for alias in FRAME_CAMERAS if alias in out}

    def harness_advance(self, num_chunks: int = 1, stop_on_grasp: bool = False, lift_thresh: float = 0.04) -> dict[str, Any]:
        """HARNESS ONLY: let the policy run ``num_chunks`` action chunks.

        One chunk, exactly as ``XPolicyLab/policy/Pi_05/deploy.py`` does it:

            obs = env.get_obs()
            client.update_obs(obs); actions = client.get_action()
            for each action in the chunk:
                env.take_action(action)        # 1 control step = 10 physics steps
                stop if the episode ended or this was the last action
                client.update_obs(env.get_obs())

        The official client executes the WHOLE chunk before asking again --
        there is no ``n_action_steps`` truncation on this path (contrast the
        RoboCasa pi0.5 server, where the caller executes 25 of 50). It only
        re-reads the observation between steps, which updates the server's
        window without triggering inference.

        Stops early on success, on the step budget, or when the env latches
        ``end_flag``. Returns the SAME field set as
        ``env_service.harness_advance`` so ``cli/harness.py advance`` works
        unmodified: task_success / committed_timestep / max_steps /
        next_query_index / steps_executed / chunks_done / stopped / truth /
        telemetry / done / frames / frame_steps.
        """
        # the policy takes over the fingers again: forget the judge's own gripper command
        self._grip_cmd_issued = {}
        episode = self._require_episode()
        env = self._require_env()
        client = self._policy()

        budget = int(episode["max_steps"])
        committed = int(episode["committed_timestep"])
        if INTERVENE_CHARGES_STEP_LIM:          # rdv5: the judge's charged steps are on the same clock
            committed = max(committed, int(env.take_action_cnt[0]))
        query_index = int(episode["next_query_index"])
        executed = 0
        frames: list[dict[str, np.ndarray]] = []
        frame_steps: list[int] = []
        stopped = "chunks_done"
        episode_dir = self._episode_dir()

        oracle = self._oracle()
        succeeded = bool(oracle["task_success"])

        _gtrack = {}
        grasp_hit = False
        for _ in range(int(num_chunks)):
            if grasp_hit:
                stopped = "grasp_detected"
                break
            if succeeded:
                stopped = "task_success"
                break
            if bool(env.end_flag[0]):
                stopped = "episode_end"
                break
            if committed >= budget:
                stopped = "step_budget_exhausted"
                break

            chunk_t0 = int(committed)
            obs = self._observation()
            chunk = client.act(obs)
            if chunk.ndim != 2 or chunk.shape[1] != (16 if self._action_type == "ee" else 14):
                raise RuntimeError(
                    f"policy returned an action chunk of shape {chunk.shape}; "
                    "expected (T, 14) for the dual ARX X5 (arm_dim [6,6], ee_dim [1,1])"
                )
            if not episode["horizon_confirmed"]:
                # First real answer wins over the CLI hint; total_chunks is only
                # ever an estimate for progress reporting.
                episode["horizon"] = int(chunk.shape[0])
                episode["horizon_confirmed"] = True
                episode["total_chunks"] = int(math.ceil(budget / max(1, chunk.shape[0])))

            limit = min(int(chunk.shape[0]), budget - committed)
            start_proprio = self._proprio_state(obs)
            joints_start = start_proprio["joint_positions"]
            ee_start = {
                "left": start_proprio.get("left_ee_pose"),
                "right": start_proprio.get("right_ee_pose"),
            }
            for index in range(limit):
                env.take_action(_unpack_action_row(chunk[index]))
                committed += 1
                executed += 1
                self._step_count += 1
                if bool(env.end_flag[0]):
                    break
                if index + 1 == limit:
                    break
                # Mirror the official client's mid-chunk observation push.
                _o = self._observation()
                client.observe(_o)
                if stop_on_grasp:
                    _pp = self._proprio_state(_o)
                    for _sd in ("left", "right"):
                        _pose = _pp.get(f"{_sd}_ee_pose"); _g = (_pp.get("gripper") or {}).get(_sd)
                        if _pose is None or _g is None:
                            continue
                        _z = float(_pose[2]); _tr = _gtrack.setdefault(_sd, {"start": _z, "min": _z, "peak": _z})
                        if _z < _tr["min"]:
                            _tr["min"] = _z; _tr["peak"] = _z
                        else:
                            _tr["peak"] = max(_tr["peak"], _z)
                        if (_tr["start"] - _tr["min"]) >= 0.06 and (_tr["peak"] - _tr["min"]) >= lift_thresh \
                                and EMPTY_CLOSURE_NORM < float(_g) < 0.6:
                            grasp_hit = True
                            _gtrack["hit_arm"] = _sd
                    if grasp_hit:
                        break

            query_index += 1
            end_obs = self._observation()
            shot = self._frames_from_obs(end_obs)
            frames.append(shot)
            frame_steps.append(int(committed))
            for alias, image in shot.items():
                _write_png(
                    episode_dir / f"c{query_index - 1:03d}_t{committed:05d}_{alias}.png",
                    image,
                )
            episode["frames_written"] = int(episode["frames_written"]) + len(shot)

            proprio = self._proprio_state(end_obs)
            joints_end = proprio["joint_positions"]
            delta = [
                round(float(b) - float(a), 4)
                for a, b in zip(joints_start, joints_end)
            ] if len(joints_start) == len(joints_end) else []
            oracle = self._oracle()
            succeeded = bool(oracle["task_success"])
            # Per-arm Cartesian travel over the chunk. The judge reads this
            # rather than the joint delta: "the hand went 4 cm" is legible in
            # the same picture it is looking at, 6 joint angles are not.
            moved_cm = []
            for side in ("left", "right"):
                a, b = ee_start.get(side), proprio.get(f"{side}_ee_pose")
                moved_cm.append(
                    round(float(np.linalg.norm(
                        np.asarray(b[:3], dtype=np.float64) - np.asarray(a[:3], dtype=np.float64)
                    )) * 100.0, 2) if (a and b) else None
                )
            # Tracking error = where the last commanded joint target was vs the
            # joint state that actually came back. On a position-controlled arm
            # that is the stall signal; there is no force sensor to ask instead.
            commanded = _unpack_action_row(chunk[limit - 1])
            track_err = None
            try:
                want = np.concatenate([
                    commanded["left_arm_joint_state"], commanded["left_ee_joint_state"],
                    commanded["right_arm_joint_state"], commanded["right_ee_joint_state"],
                ]).astype(np.float64)
                got = np.asarray(joints_end, dtype=np.float64)
                if want.shape == got.shape:
                    track_err = round(float(np.max(np.abs(want - got))), 3)
            except Exception:
                track_err = None
            grip_now = proprio["gripper"]
            entry = {
                "chunk": int(query_index - 1),
                "t": [chunk_t0, int(committed)],
                "steps": int(limit),
                "joint_delta": delta,
                "joints_end": [round(float(v), 4) for v in joints_end],
                "gripper": grip_now,
                "grip": [
                    round(float(grip_now["left"]), 3) if grip_now.get("left") is not None else None,
                    round(float(grip_now["right"]), 3) if grip_now.get("right") is not None else None,
                ],
                "moved_cm": moved_cm,
                "track_err_rad": track_err,
                "left_ee_pose": proprio["left_ee_pose"],
                "right_ee_pose": proprio["right_ee_pose"],
                "task_success": bool(oracle["task_success"]),
                "score": oracle["process_score"],
            }
            self._act_log.append(entry)
            # "Shut all the way with nothing between the fingers" is the ONLY
            # gripper state this robot can identify: the opening cannot tell
            # a held object from an open hand. Record each entry into it.
            for side in ("left", "right"):
                opening = grip_now.get(side)
                pose = proprio.get(f"{side}_ee_pose")
                if opening is None or pose is None:
                    continue
                was_empty = self._empty_now.get(side, False)
                now_empty = float(opening) <= EMPTY_CLOSURE_NORM
                self._empty_now[side] = now_empty
                if now_empty and not was_empty:
                    self._empty_closures.append({
                        "t": int(committed),
                        "arm": side,
                        "eef": [round(float(v), 4) for v in pose[:3]],
                        "opening": round(float(opening), 3),
                    })
                    del self._empty_closures[:-20]
            if len(self._act_log) > 400:
                del self._act_log[:-400]
            if succeeded:
                stopped = "task_success"
                break

        episode["committed_timestep"] = int(committed)
        episode["next_query_index"] = int(query_index)
        oracle = self._oracle()
        succeeded = bool(oracle["task_success"])
        # One observation read for the whole telemetry block: ``_proprio_state``
        # with no argument renders the env again, and the previous version
        # called it three times per advance.
        final_proprio = self._proprio_state()

        return {
            "task_success": bool(succeeded),
            "committed_timestep": int(committed),
            "max_steps": int(budget),
            "next_query_index": int(query_index),
            "steps_executed": int(executed),
            "chunks_done": len(frames),
            "stopped": stopped,
            # Privileged; cli/harness.py pops this before emitting so it never
            # reaches a judge's stdout. Kept for post-hoc scoring.
            "truth": {
                "task_success": bool(succeeded),
                "episode_score": oracle["episode_score"],
                "process_score": oracle["process_score"],
                "layout_id": int(episode["layout_id"]),
            },
            "telemetry": {
                # ``schema`` routes cli/harness.py to the RoboDojo renderer.
                # The RoboCasa table's columns (commanded dxyz in cm, aperture
                # in mm) do not exist on a bimanual joint-position robot, and
                # rendering it would KeyError mid-window.
                "schema": "robodojo",
                "chunks": self._act_log[-12:],
                # Measured from travel, not a constant -- the policy uses a
                # different hand on different layouts of the same task.
                "active_arm": self._active_arm(),
                "eef_now": {
                    "left": final_proprio["left_ee_pose"],
                    "right": final_proprio["right_ee_pose"],
                },
                "opening_now": {
                    side: (round(float(v), 3) if v is not None else None)
                    for side, v in final_proprio["gripper"].items()
                },
                "empty_closures": list(self._empty_closures[-8:]),
                "joints_now": final_proprio["joint_positions"],
                "horizon": int(episode["horizon"]),
                "horizon_confirmed": bool(episode["horizon_confirmed"]),
            },
            "done": bool(
                succeeded
                or committed >= budget
                or bool(env.end_flag[0])
                or stopped != "chunks_done"
            ),
            "frames": frames,
            "frame_steps": frame_steps,
            "frames_dir": str(episode_dir.resolve()),
        }

    def harness_read_state_privileged(self) -> dict[str, Any]:
        """HARNESS ONLY: proprioception PLUS the outcome oracle.

        The RoboDojo analogue of ``env_service.harness_read_state_privileged``.
        RoboDojo has no ``is_grasped`` predicate and no single target-object
        pose (many tasks have several objects and a staged score), so the oracle
        block carries ``task_success`` and the partial ``score`` instead.
        """
        state = self._proprio_state()
        return {"state": state, "oracle": self._oracle()}

    @staticmethod
    def _steps_used_now(episode, env) -> int:
        """Episode clock incl. the judge's own steps when those are charged (rdv5)."""
        used = int(episode.get("committed_timestep") or 0)
        if INTERVENE_CHARGES_STEP_LIM and env is not None:
            used = max(used, int(getattr(env, "take_action_cnt", [0])[0]))
        return used

    def reset_step_counter(self) -> dict[str, Any]:
        """Zero the service's own step counter ONLY.

        Does not touch the simulator, the episode budget, or
        ``env.take_action_cnt`` -- exactly the semantics of the RoboCasa one.
        The episode clock the driver reads is ``committed_timestep``, which is
        untouched here on purpose.
        """
        self._require_env()
        previous = int(self._step_count)
        self._step_count = 0
        return {"step_count": 0, "previous_step_count": previous}

    def harness_release_env(self) -> dict[str, Any]:
        """HARNESS ONLY: drop the episode so the next one can be built here.

        Keeps the Isaac app AND the policy connection alive -- booting Kit is
        the expensive part (tens of seconds) and there is no reason to pay it
        per episode. The scene itself is torn down by ``EvalEnv.close()``, which
        is what the official loop does between batches.
        """
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        self._episode = None
        self._act_log = []
        self._step_count = 0
        return {"ok": True, "app_alive": self._app is not None}

    def harness_quit(self) -> dict[str, Any]:
        """HARNESS ONLY: answer, then exit the whole process.

        Same contract and same reason as the RoboCasa one: the driver needs a
        brand-new process per arm and has no kubectl, so the service exits and
        ``ops/podlib/sim_lane.sh`` relaunches it three seconds later. ``os._exit``
        rather than a clean shutdown because Kit does not reliably return from
        ``simulation_app.close()`` once cameras are live.
        """
        import threading

        threading.Timer(0.5, lambda: os._exit(0)).start()
        return {"ok": True, "pid": os.getpid()}

    # ---- selftest ---------------------------------------------------------

    def selftest(self, task: str, chunks: int = 3, chunk_len: int = 10) -> dict[str, Any]:
        """Build one episode and drive it with RANDOM actions. No policy server.

        The point is to separate "Isaac + this module's env assembly works" from
        "the policy server works", which on a fresh pod are two different
        failures with the same symptom (advance hangs). Prints per-phase
        timings. Actions are uniform noise around the current joint state, so
        the robot twitches rather than slams into a limit.
        """
        timings: dict[str, float] = {}
        t0 = time.monotonic()
        self._build_env(task)
        timings["build_env_s"] = round(time.monotonic() - t0, 2)

        t0 = time.monotonic()
        # seed=0, episode_index=0 -> layout 0; no env rebuild, we just built it.
        self._episode = None
        env = self._env
        layout_id, n_layouts = self._layout_id(0, 0)
        env.reset(seed=[layout_id])
        env.run_reward()
        if hasattr(env, "get_score"):
            env.get_score()
        timings["reset_s"] = round(time.monotonic() - t0, 2)

        t0 = time.monotonic()
        obs = self._observation()
        timings["first_obs_s"] = round(time.monotonic() - t0, 2)
        vision_shapes = {
            key: list(np.asarray(_as_rgb_uint8(value["color"])).shape)
            for key, value in (obs.get("vision") or {}).items()
            if isinstance(value, Mapping) and "color" in value
        }

        rng = np.random.default_rng(0)
        base = np.asarray(
            _to_list(obs["state"].get("left_arm_joint_state"))
            + _to_list(obs["state"].get("left_ee_joint_state"))
            + _to_list(obs["state"].get("right_arm_joint_state"))
            + _to_list(obs["state"].get("right_ee_joint_state")),
            dtype=np.float64,
        )
        step_times: list[float] = []
        steps = 0
        for _ in range(int(chunks)):
            for _ in range(int(chunk_len)):
                if bool(env.end_flag[0]):
                    break
                action = base + rng.normal(0.0, 0.01, size=base.shape)
                action[6] = float(np.clip(action[6], 0.0, 1.0))
                action[13] = float(np.clip(action[13], 0.0, 1.0))
                t0 = time.monotonic()
                env.take_action(_unpack_action_row(action))
                step_times.append(time.monotonic() - t0)
                steps += 1
            self._observation()

        report = {
            "ok": True,
            "task": task,
            "layout_id": layout_id,
            "n_layouts": n_layouts,
            "native_step_lim": int(env.step_lim),
            "instruction": obs.get("instruction"),
            "state_keys": sorted((obs.get("state") or {}).keys()),
            "vision_cameras": sorted((obs.get("vision") or {}).keys()),
            "vision_shapes": vision_shapes,
            "control_steps": steps,
            "mean_control_step_s": round(float(np.mean(step_times)), 4) if step_times else None,
            "total_control_s": round(float(np.sum(step_times)), 2) if step_times else 0.0,
            **timings,
            **self._oracle(),
        }
        print(json.dumps(report, indent=2, default=str), flush=True)
        return report


# ---------------------------------------------------------------------------
#  CLI
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
    for candidate in range(int(port), int(port) + int(tries)):
        if _port_is_free(host, candidate):
            return candidate
    raise RuntimeError(f"no free port in [{port}, {port + tries})")


def build_parser() -> argparse.ArgumentParser:
    """Every flag ``env_service.build_parser`` has, plus the RoboDojo-only ones.

    The shared flags keep the same names, types and meanings so
    ``recovery_explore/launch_svc.sh`` and ``ops/podlib/sim_lane.sh`` can launch
    this module by swapping the ``-m`` target and nothing else.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # -- shared with env_service.py ----------------------------------------
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--port-tries", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="recovery-explore-robodojo")
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="call log (one JSON line per RPC); default <output-dir>/env_service_calls.jsonl",
    )
    parser.add_argument(
        "--policy-endpoint",
        default=None,
        help="policy websocket server, e.g. ws://10.0.0.7:6000 (http:// and bare "
        "host:port are accepted too)",
    )
    parser.add_argument(
        "--policy-family",
        default="pi05",
        choices=["pi05"],
        help="RoboDojo milestone 1 serves pi0.5 only; the flag exists so the "
        "launcher can pass it through unchanged",
    )
    parser.add_argument(
        "--step-budget-scale",
        type=float,
        default=1.0,
        help="multiplier on the task's own step_lim. Anything but 1.0 makes the "
        "batch comparable only to itself, never to the published benchmark.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="physical CUDA ordinal, passed to Isaac as --device cuda:N. NOT exported as "
             "CUDA_VISIBLE_DEVICES (that makes Omniverse skip every GPU). Before "
        "Isaac boots; Isaac then stays on its default cuda:0.",
    )
    parser.add_argument("--parent-watch", action="store_true")
    # Accepted and ignored, so a launcher script can pass the RoboCasa flag set
    # verbatim. Ignoring loudly (a printed note at startup) beats erroring.
    parser.add_argument("--debug-resets", type=int, default=0)
    parser.add_argument("--render-size", type=int, default=FRAME_SIZE)
    parser.add_argument("--anchors-jsonl", type=Path, default=None)

    # -- RoboDojo-only ------------------------------------------------------
    parser.add_argument(
        "--robodojo-root",
        type=Path,
        default=Path(os.environ.get("ROBODOJO_ROOT", "/workspace/RoboDojo")),
        help="checkout root; prepended to sys.path (with its XPolicyLab/) and "
        "chdir'd into, because RoboDojo resolves Assets/ relatively",
    )
    parser.add_argument("--env-cfg-type", default=DEFAULT_ENV_CFG_TYPE)
    parser.add_argument("--policy-name", default=DEFAULT_POLICY_NAME)
    parser.add_argument(
        "--policy-protocol",
        default="xpl_ws",
        choices=["xpl_ws", "openpi_ws"],
        help="xpl_ws = XPolicyLab WsModelClient (what the official eval uses); "
        "openpi_ws = bare openpi WebsocketClientPolicy (untested)",
    )
    parser.add_argument("--action-type", default=DEFAULT_ACTION_TYPE, choices=["joint", "ee"])
    parser.add_argument(
        "--policy-chunk",
        type=int,
        default=DEFAULT_POLICY_CHUNK_HINT,
        help="assumed action-chunk length, used ONLY for total_chunks before the "
        "policy has answered once; the real length overwrites it",
    )
    parser.add_argument(
        "--layout-seed",
        type=int,
        default=0,
        help="which pre-generated layout SET to use "
        "(Assets/Eval_Layout/<bench>/<config>/<layout-seed>/)",
    )
    parser.add_argument(
        "--episode-id-mode",
        default="modulo",
        choices=["modulo", "strict"],
        help="how (seed + episode_index) becomes a layout id",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run one episode of --selftest-task with random actions and no "
        "policy server, print timings, exit",
    )
    parser.add_argument("--selftest-task", default=None)
    parser.add_argument("--selftest-chunks", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.debug_resets:
        print(
            "[robodojo-svc] --debug-resets is a RoboCasa concept (model-side rewind); "
            "ignored on this service, which has no rewind at milestone 1",
            file=sys.stderr,
            flush=True,
        )
    if args.anchors_jsonl is not None:
        print(
            "[robodojo-svc] --anchors-jsonl ignored: RoboDojo has no snapshot anchors",
            file=sys.stderr,
            flush=True,
        )

    output_dir = Path(args.output_dir)
    log_path = Path(args.log) if args.log else output_dir / "env_service_calls.jsonl"

    service = RoboDojoEnvService(
        output_dir=output_dir,
        run_id=args.run_id,
        log_path=log_path,
        policy_endpoint=args.policy_endpoint,
        policy_family=args.policy_family,
        policy_protocol=args.policy_protocol,
        policy_name=args.policy_name,
        policy_chunk_hint=args.policy_chunk,
        action_type=args.action_type,
        env_cfg_type=args.env_cfg_type,
        robodojo_root=args.robodojo_root,
        layout_seed=args.layout_seed,
        episode_id_mode=args.episode_id_mode,
        step_budget_scale=args.step_budget_scale,
        cuda_device=args.cuda_device,
    )

    if args.selftest:
        if not args.selftest_task:
            raise SystemExit("--selftest needs --selftest-task <task_name>")
        try:
            service.selftest(args.selftest_task, chunks=args.selftest_chunks)
        finally:
            service.close()
        return 0

    port = pick_port(args.host, args.port, args.port_tries)
    service.serve(transport="http", host=args.host, port=port, parent_watch=args.parent_watch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
