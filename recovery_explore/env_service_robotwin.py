"""Persistent RoboTwin 2.0 (SAPIEN 3, dual-arm aloha-agilex) environment
service -- ONE process, ONE task env at a time, HTTP-RPC.

The RoboTwin sibling of ``env_service_robodojo.py``: same wire contract
(``rpc_transport``: POST /call, ``{"ok": true, "result": ...}``, one JSON line
per business call in ``<output-dir>/env_service_calls.jsonl``), same harness
verbs, same return field sets, so ``cli/harness.py`` and the driver speak to it
unchanged. Pick the benchmark by picking the module:

    python -m recovery_explore.env_service            # RoboCasa
    python -m recovery_explore.env_service_robodojo   # RoboDojo
    python -m recovery_explore.env_service_robotwin   # RoboTwin 2.0

CONTROL ARM (this file)
    harness_env_meta / harness_episode_begin / harness_advance /
    harness_release_env / harness_quit / reset_step_counter /
    harness_read_state_privileged / harness_oracle_grasp / harness_state_digest

TREATMENT ARM
    Lives in ``recovery_explore/robotwin_intervention.py`` (another agent's
    file), mixed in FIRST in the class bases exactly like RoboDojo. Until it
    exists, the import falls back to an empty class and every treatment verb is
    registered as a stub raising ``NotImplementedError``. The host methods the
    mixin needs (``_require_env``, ``_require_episode``, ``_observation``,
    ``_proprio_state``, ``_oracle``, ``_policy``, ``_episode_dir``,
    ``_frames_from_obs``, ``_log``, ``_active_arm``, ``_act_log``,
    ``_step_count``, ``_episode``, ``_env``) all exist here; the list is pinned
    by ``tests/test_robotwin_env_service.py``.

IMPORTABLE WITHOUT SAPIEN
    Nothing at module scope imports ``sapien``, ``torch``, ``mplib`` or the
    RoboTwin tree. The first call that needs the simulator is
    ``harness_episode_begin`` (or ``--selftest``).

OFFICIAL LOOP THIS REPRODUCES  (RoboTwin ``script/eval_policy.py`` +
``policy/pi05/deploy_policy.py``)::

    TASK_ENV.setup_demo(now_ep_num, seed=now_seed, is_test=True, **args)
    results = generate_episode_descriptions(task, [episode_info], 100)
    instruction = np.random.choice(results[0]["unseen"]); set_instruction(...)
    reset_model(model)
    while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
        observation = TASK_ENV.get_obs()
        eval(TASK_ENV, model, observation):          # == ONE harness chunk
            model.update_observation_window(*encode_obs(observation))
            actions = model.get_action()[:pi0_step]
            for action in actions:
                TASK_ENV.take_action(action)
                observation = TASK_ENV.get_obs(); update window
        if TASK_ENV.eval_success: break
    TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

    One ``harness_advance`` chunk == exactly one ``eval`` call. We keep the
    per-action ``get_obs`` too, although the policy only ever consumes the
    observation taken at the START of the next chunk (``update_observation_window``
    OVERWRITES the window; pi0.5 has no history). Two reasons it still matters:
    (1) ``get_obs -> _update_render`` draws from the global numpy RNG when
    ``crazy_random_light`` is on (demo_randomized, 2% of episodes), so skipping
    it would shift the lighting of every later frame the policy sees; (2) it
    keeps ``TASK_ENV.now_obs`` exactly as the official loop leaves it. We stop
    iterating the chunk as soon as ``eval_success`` latches or the step limit is
    hit: from there on the official loop's ``take_action`` is a no-op and its
    ``get_obs`` calls only touch state after the episode is over.

EPISODE IDENTITY  (task, seed, episode_index)  ->  RoboTwin seed
    ``seed`` is the official eval seed (``script/eval_policy.py --seed``), which
    selects the SEED SET; ``episode_index`` is the index of the valid
    (expert-checked) seed inside that set. We do NOT run the expert planner per
    episode: the valid seeds and their ``episode_info`` (the placeholder values
    the instruction generator needs, which the official loop gets from
    ``play_once``) come from the precomputed caches
    ``<seed-cache-dir>/<task_config>_seed<seed>_n100.json``
    (``tasks[task][episode_index] = {"seed": int, "episode_info": {...}}``).

    NOTE the caches were built by the LingBot pipeline with
    ``seed_base = 10000`` (``st_seed = 10000 * (1 + seed)``), NOT the official
    ``100000 * (1 + seed)``. Same procedure, different starting seed: the
    episodes are official-procedure episodes, but not the official leaderboard
    episode list. ``--seed-source expert`` runs the official expert check live
    from ``--st-seed-base * (1 + seed)`` instead (slow, memoised per process in
    ``<output-dir>/expert_seeds_*.json``).

INSTRUCTION DETERMINISM (the one deliberate deviation)
    ``generate_episode_descriptions`` shuffles with Python's ``random`` module,
    which the official loop never seeds, so the official instruction for a
    given seed is not reproducible across processes. Control and treatment arms
    of one episode must see the same instruction, so we ``random.seed(<robotwin
    seed>)`` immediately before calling it. ``np.random.choice`` then runs on
    the numpy RNG exactly as officially (``setup_demo`` seeds it with the
    RoboTwin seed and the draw happens right after ``setup_demo``).

STEP BUDGET
    ``_eval_step_limit.yml`` via ``TASK_ENV.step_lim`` (counts ``take_action``
    calls). ``max_steps = round(step_lim * step_budget_scale) + max_steps_extra``
    and it is written back onto ``TASK_ENV.step_lim`` so the env's own clamp in
    ``take_action`` agrees with ours.

GPU / RENDERING
    SAPIEN picks its Vulkan device from the CUDA device, so ``--cuda-device N``
    exports ``CUDA_VISIBLE_DEVICES=N`` (plus the LingBot pipeline's
    ``MESA_VK_DEVICE_SELECT=<uuid>`` pin, see
    ``/workspace/lingbot-va/evaluation/robotwin/vulkan_env.py``) BEFORE sapien
    is imported. On the RTX 5090 the official OIDN denoiser fails
    ("OIDN Error: invalid handle", measured 2026-09-19); the runtime tree's
    ``_base_task.py`` honours ``ROBOTWIN_RT_DENOISER`` and ``launch_svc.sh``
    defaults it to ``optix`` exactly like the LingBot 5090 eval
    (``script/run_checkpoint_eval_6gpu.sh``). Recorded in ``harness_env_meta``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import socket
import struct
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # package import (normal: python -m recovery_explore.env_service_robotwin)
    from recovery_explore.rpc_transport import MainThreadServeMixin, RpcFacade
except ImportError:  # pragma: no cover - direct `python env_service_robotwin.py`
    from rpc_transport import MainThreadServeMixin, RpcFacade  # type: ignore[no-redef]


DEFAULT_PORT = 8480  # 8410 RoboCasa, 8460 RoboDojo; keep the families apart
DEFAULT_TASK_CONFIG = "demo_randomized"
TASK_CONFIGS = ("demo_clean", "demo_randomized")
#: Where the RoboTwin tree and the seed caches are. launch_svc.sh passes both explicitly
#: (--robotwin-root from ROBOTWIN_ROOT / ROBOTWIN_RUNTIME, --seed-cache from ROBOTWIN_SEED_CACHE);
#: these defaults only matter when the module is started by hand.
DEFAULT_ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT") or "RoboTwin"
DEFAULT_SEED_CACHE_DIR = os.environ.get("ROBOTWIN_SEED_CACHE_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "episode_sets", "robotwin_seed_cache")
#: ``policy/pi05/deploy_policy.yml``: ``instruction_type: unseen``, ``pi0_step: 50``.
DEFAULT_INSTRUCTION_TYPE = "unseen"
DEFAULT_PI0_STEP = 50
#: ``script/eval_policy.py``: ``test_num = 100`` is passed as max_descriptions.
OFFICIAL_TEST_NUM = 100
#: ``script/eval_policy.py``: ``st_seed = 100000 * (1 + seed)``.
OFFICIAL_ST_SEED_BASE = 100000

#: RoboTwin camera -> the camera name ``cli/harness.py`` knows. Same convention
#: as RoboDojo: the scene ("head") camera is primary, left wrist is secondary,
#: right wrist is "wrist". ``front_camera`` is rendered by RoboTwin but not
#: used by the policy; it is not shown either.
CAMERA_ALIASES: dict[str, str] = {
    "head_camera": "primary",
    "left_camera": "secondary",
    "right_camera": "wrist",
}
FRAME_CAMERAS: tuple[str, ...] = ("primary", "secondary", "wrist")
FRAME_SIZE = 256

#: Action-chunk length assumed before the policy has answered once (only for
#: ``total_chunks``); the first real reply's ``actions.shape[0]`` (capped at
#: ``pi0_step``) overwrites it. The pi0.5 RoboTwin checkpoint has
#: action_horizon=50 (metadata.pt), which with pi0_step=50 is 50 per chunk.
DEFAULT_POLICY_CHUNK_HINT = 50

#: Measured, normalised (0 shut .. 1 open) finger opening at or below which a
#: closure counts as "met on nothing". Normalised by the finger joint's own limits
#: ([0, 0.04765] m, see _measured), so 0.05 is ~2.4 mm; real holds read >= ~0.08.
EMPTY_CLOSURE_NORM = float(os.environ.get("RT_GRIP_EMPTY", "0.05"))
#: Commanded opening at or below which the gripper is being asked to shut.
GRIP_CMD_SHUT = float(os.environ.get("RT_GRIP_CMD_SHUT", "0.5"))


# Treatment arm: another agent's module. Imported here, after the constants,
# for the same reason as RoboDojo (the mixin may read FRAME_SIZE from us).
try:
    from recovery_explore.robotwin_intervention import RobotwinInterventionMixin
except ImportError:  # pragma: no cover - depends on whether the mixin landed
    try:
        from robotwin_intervention import RobotwinInterventionMixin  # type: ignore[no-redef]
    except ImportError:
        class RobotwinInterventionMixin:  # type: ignore[no-redef]
            """Placeholder until ``robotwin_intervention.py`` exists."""


#: Every treatment verb the RoboDojo mixin registers. Whatever the RoboTwin
#: mixin does not provide is registered as a NotImplementedError stub, so an
#: accidental treatment call fails loudly at the RPC boundary.
INTERVENTION_METHODS: tuple[str, ...] = (
    "harness_open_action_budget",
    "harness_close_action_budget",
    "harness_judge_lock",
    "harness_judge_unlock",
    "harness_round_status",
    "reset_window",
    "execute_plan",
    "move_to",
    "nudge",
    "lift",
    "rotate",
    "gripper",
    "retreat",
    "read_state",
    "render",
    "unproject",
    "restore_snapshot",
    "harness_checkpoint",
)

#: RoboCasa failure-factory verbs; not part of the judge experiment.
UNIMPLEMENTED_METHODS: tuple[str, ...] = (
    "make_failure",
    "reset_to_failure",
    "run_episode_to_failure",
    "resume_episode",
    "harness_round_begin",
    "call_cosmos",
    "unproject_batch",
    "env_meta",
    "debug_reset",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _summarize(value: Any, *, depth: int = 0) -> Any:
    """Shrink a value for the call log (same as the RoboDojo service)."""
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
    """Zero-dependency PNG writer for an HxWx3 uint8 array (as RoboDojo's)."""
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
    """Nearest-neighbour resize to ``size x size`` (review frames only; the
    policy always gets the untouched 240x320 camera image)."""
    arr = np.asarray(image)
    height, width = arr.shape[:2]
    if (height, width) == (size, size):
        return np.ascontiguousarray(arr)
    rows = (np.arange(size) * (height / size)).astype(np.int64).clip(0, height - 1)
    cols = (np.arange(size) * (width / size)).astype(np.int64).clip(0, width - 1)
    return np.ascontiguousarray(arr[rows][:, cols])


def _as_rgb_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
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
    return [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1)]


def _rss_mb() -> float | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        return None
    return None


class _NotBuilt(RuntimeError):
    pass


# ---------------------------------------------------------------------------
#  Official policy-side encoding (verbatim)
# ---------------------------------------------------------------------------


# Copied VERBATIM from RoboTwin ``policy/pi05/deploy_policy.py::encode_obs``
# (the file cannot be imported in the simulator venv: its module scope does
# ``from pi_model import *``, which imports jax/openpi). The unit test compares
# this body against that file whenever the RoboTwin tree is present.
# Order is head, RIGHT wrist, LEFT wrist -- ``PI0.update_observation_window``
# unpacks it as ``img_front, img_right, img_left``.
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def build_policy_request(observation: Mapping[str, Any], instruction: str, reset: bool) -> dict:
    """The websocket request of the pi0.5 policy protocol for this project.

    ``{"input_rgb_arr": [head, right, left] uint8 HxWx3, "input_state":
    float32[14], "instruction": str, "reset": bool}``. The server applies what
    ``PI0.update_observation_window`` does (HWC->CHW, the cam_high /
    cam_left_wrist / cam_right_wrist mapping, prompt), so we send exactly what
    ``encode_obs`` returns.
    """
    input_rgb_arr, input_state = encode_obs(observation)
    return {
        "input_rgb_arr": [np.ascontiguousarray(np.asarray(img, dtype=np.uint8)) for img in input_rgb_arr],
        "input_state": np.asarray(input_state, dtype=np.float32).reshape(-1),
        "instruction": str(instruction),
        "reset": bool(reset),
    }


def _split_endpoint(endpoint: str) -> tuple[str, int]:
    """``[ws://|http://]host:port`` -> ``(host, port)``."""
    rest = endpoint
    if "://" in rest:
        rest = rest.split("://", 1)[1]
    rest = rest.rstrip("/")
    host, _, port = rest.rpartition(":")
    if not host or not port:
        raise ValueError(f"policy endpoint must be [scheme://]host:port, got {endpoint!r}")
    return host, int(port)


class _PolicyClient:
    """openpi ``WebsocketClientPolicy`` with a bounded connect.

    ``WebsocketClientPolicy.__init__`` retries a refused connection FOREVER
    (``_wait_for_server``), which would hang the RPC thread of a lane whose
    policy replica died. We probe the TCP port first and fail the RPC instead.
    """

    def __init__(self, endpoint: str, connect_timeout_s: float = 30.0) -> None:
        self.endpoint = endpoint
        host, port = _split_endpoint(endpoint)
        deadline = time.monotonic() + float(connect_timeout_s)
        while True:
            try:
                with socket.create_connection((host, port), timeout=5.0):
                    break
            except OSError as exc:
                if time.monotonic() > deadline:
                    raise ConnectionError(f"policy server {host}:{port} unreachable: {exc}") from exc
                time.sleep(1.0)
        from openpi_client.websocket_client_policy import WebsocketClientPolicy  # lazy

        self._impl = WebsocketClientPolicy(host=host, port=port)
        try:
            self.metadata = self._impl.get_server_metadata()
        except Exception:
            self.metadata = None

    def infer(self, request: Mapping[str, Any]) -> np.ndarray:
        result = self._impl.infer(dict(request))
        actions = np.asarray(result["actions"], dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None, :]
        return actions

    def reset(self) -> None:
        """openpi's client is stateless; the episode reset travels as the
        ``reset`` flag of the next request (see _reset_policy_after_rewind)."""

    def close(self) -> None:
        ws = getattr(self._impl, "_ws", None)
        try:
            if ws is not None:
                ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
#  Seed cache
# ---------------------------------------------------------------------------


def seed_cache_path(cache_dir: str | Path, task_config: str, seed: int) -> Path:
    return Path(cache_dir) / f"{task_config}_seed{int(seed)}_n100.json"


def load_seed_cache(path: str | Path, task_config: str) -> dict[str, Any]:
    """Load and validate one seed-cache file.

    Schema (LingBot ``eval_seed_cache``, version 1): ``{"version": 1,
    "task_config": str, "seed_base": int, "per_task": int, "tasks": {task:
    [{"seed": int, "episode_info": {placeholder: value}}, ...]}}``. Entries are
    in seed order and every one passed the official expert check
    (``setup_demo`` + ``play_once`` + ``plan_success and check_success``).
    """
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), dict):
        raise ValueError(f"{path}: not a seed cache (no 'tasks' mapping)")
    cfg = payload.get("task_config")
    if cfg is not None and cfg != task_config:
        raise ValueError(
            f"{path} was built for task_config={cfg!r}, this service runs {task_config!r}"
        )
    for task, entries in payload["tasks"].items():
        if not isinstance(entries, list):
            raise ValueError(f"{path}: tasks[{task!r}] is not a list")
        seeds = [int(e["seed"]) for e in entries]
        if seeds != sorted(seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f"{path}: tasks[{task!r}] seeds are not strictly increasing")
    payload["_path"] = str(path)
    payload["_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def cache_entry(payload: Mapping[str, Any], task: str, episode_index: int) -> dict[str, Any]:
    entries = payload["tasks"].get(task)
    if not entries:
        raise KeyError(f"seed cache {payload.get('_path')} has no entries for task {task!r}")
    k = int(episode_index)
    if not 0 <= k < len(entries):
        raise IndexError(
            f"episode_index={k} is outside the {len(entries)} cached valid seeds for {task!r}"
        )
    entry = entries[k]
    info = entry.get("episode_info")
    if not isinstance(info, dict):
        raise ValueError(f"cache entry {k} for {task!r} has no episode_info dict")
    return {"seed": int(entry["seed"]), "episode_info": dict(info), "cache_index": k}


def scene_digest(env: Any) -> tuple[dict[str, Any], str]:
    """Fingerprint of a live RoboTwin env: every SAPIEN entity pose, every
    articulation's qpos/qvel, and the evaluator's ``take_action_cnt`` /
    ``eval_success``. Returns (per-field summary, sha256 over values rounded to
    1e-6). Shared by ``harness_state_digest`` and the parity runner
    (``ops/robotwin/env_official_loop.py``)."""
    digest: dict[str, Any] = {}
    hasher = hashlib.sha256()

    def add(key: str, values: Any) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        digest[key] = {
            "sum": round(float(np.nansum(arr)), 6),
            "absmax": round(float(np.nanmax(np.abs(arr))) if arr.size else 0.0, 6),
            "shape": list(arr.shape),
        }
        hasher.update(key.encode())
        hasher.update(np.round(arr, 6).tobytes())

    scene = env.scene
    for i, ent in enumerate(scene.get_entities()):
        pose = ent.get_pose() if hasattr(ent, "get_pose") else ent.pose
        add(f"entity/{i:03d}:{ent.name}/pose", list(pose.p) + list(pose.q))
    for i, art in enumerate(scene.get_all_articulations()):
        add(f"articulation/{i}:{art.name}/qpos", art.get_qpos())
        add(f"articulation/{i}:{art.name}/qvel", art.get_qvel())
    add("bookkeeping/take_action_cnt", [int(env.take_action_cnt)])
    add("bookkeeping/eval_success", [float(bool(env.eval_success))])
    return digest, hasher.hexdigest()


# ---------------------------------------------------------------------------
#  GPU pinning (before sapien/torch are imported)
# ---------------------------------------------------------------------------


def configure_gpu(cuda_device: int | None) -> dict[str, str]:
    """Pin CUDA + Vulkan to one physical GPU, the LingBot way.

    Mirrors ``lingbot-va/evaluation/robotwin/vulkan_env.py::
    configure_robotwin_vulkan`` (the pipeline whose RoboTwin numbers the user
    verified): ``CUDA_VISIBLE_DEVICES=N`` (SAPIEN's renderer follows the CUDA
    device), plus ``MESA_VK_DEVICE_SELECT=<gpu uuid>`` through the Mesa
    device-select layer so Vulkan enumeration cannot pick another card.
    """
    if cuda_device is None:
        return {}
    gpu = str(int(cuda_device))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["ROBOTWIN_VULKAN_GPU"] = gpu
    os.environ["NVIDIA_VISIBLE_DEVICES"] = gpu
    if not os.environ.get("VK_ICD_FILENAMES"):
        for candidate in (
            "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "/etc/vulkan/icd.d/nvidia_icd.json",
            os.environ.get("VK_ICD_FALLBACK", ""),
        ):
            if os.path.isfile(candidate):
                os.environ["VK_ICD_FILENAMES"] = candidate
                break
    os.environ.setdefault("VK_LOADER_LAYERS_ENABLE", "VK_LAYER_MESA_device_select")
    if not os.environ.get("MESA_VK_DEVICE_SELECT"):
        try:
            uuid = subprocess.check_output(
                ["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=30,
            ).strip()
            if uuid:
                os.environ["MESA_VK_DEVICE_SELECT"] = uuid
        except Exception:
            pass
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "VK_ICD_FILENAMES",
        "VK_LOADER_LAYERS_ENABLE",
        "MESA_VK_DEVICE_SELECT",
        "ROBOTWIN_RT_DENOISER",
    )
    return {k: os.environ.get(k, "") for k in keys}


# ---------------------------------------------------------------------------
#  The service
# ---------------------------------------------------------------------------


class _RobotwinHostDefaults:
    """Host-side defaults for everything the intervention mixin may REPLACE.

    They live in their own base class, placed AFTER ``RobotwinInterventionMixin``
    in the MRO, because a method defined on ``RobotwinEnvService`` itself would
    shadow the mixin's version. Without the mixin these make the control arm
    complete on their own (privileged gate, digest, active arm).
    """

    #: Refused while the model holds the action budget or the judge turn is
    #: locked. The RoboDojo list; the mixin may override it.
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

    # ---- intervention hooks (defaults; the mixin overrides) ---------------

    def _init_intervention_state(self) -> None:
        """Default treatment-arm bookkeeping, so meta/_dispatch work without
        the mixin. ``RobotwinInterventionMixin`` is first in the MRO and
        replaces this with the real one."""
        self._round_budget: int | None = None
        self._round_used = 0
        self._round_index = 0
        self._judge_locked = False
        self._resets_used = 0
        self._intervene_steps = 0
        self._default_arm = os.environ.get("RT_DEFAULT_ARM", "right")

    def _intervention_rpc(self) -> dict[str, Any]:
        return {}

    def _active_arm(self) -> str:
        """Which hand is doing the work: the one with more cumulative
        end-effector travel this episode; ``RT_DEFAULT_ARM`` before anything
        moved."""
        travel = self._travel_cm
        if max(travel.values()) <= 0.0:
            return str(getattr(self, "_default_arm", "right"))
        return "left" if travel["left"] > travel["right"] else "right"

    def harness_state_digest(self) -> dict[str, Any]:
        """HARNESS ONLY: comparable fingerprint of the simulated world.

        Every SAPIEN entity pose + every articulation's qpos/qvel, plus the
        evaluator's own bookkeeping. ``sha256`` over the values rounded to 1e-6
        answers "same world or not"; per-field sums say by how much.
        """
        env = self._require_env()
        digest, sha = scene_digest(env)
        obs = env.now_obs if getattr(env, "now_obs", None) else None
        ep = (obs or {}).get("endpose", {}) or {}
        return {
            "scene": digest,
            "sha256": sha,
            "left_ee_pose": _to_list(ep.get("left_endpose")) or None,
            "right_ee_pose": _to_list(ep.get("right_endpose")) or None,
            "take_action_cnt": int(env.take_action_cnt),
            "eval_success": bool(env.eval_success),
            "step_count": int(self._step_count),
            "committed_timestep": int((self._episode or {}).get("committed_timestep") or 0),
            "oracle": self._oracle(),
        }


class RobotwinEnvService(
    RobotwinInterventionMixin, _RobotwinHostDefaults, MainThreadServeMixin, RpcFacade
):
    """Holds one RoboTwin task env (``Base_Task`` subclass) and runs episodes."""

    #: Harness-only verbs this host adds on top of whatever privileged list the
    #: mixin declares (the mixin's list does not know about them).
    _HOST_PRIVILEGED_METHODS = frozenset({"harness_world_digest", "harness_oracle_grasp",
                                          "harness_state_digest"})

    #: Read by the intervention mixin (its own defaults differ; ours follow the
    #: RoboDojo / cli/harness.py montage convention: primary/secondary/wrist).
    CAMERA_ALIASES = CAMERA_ALIASES
    FRAME_SIZE = FRAME_SIZE
    #: ``_act_log[*]["{side}_ee_pose"]`` is ``obs["endpose"]`` =
    #: ``Robot.get_*_ee_pose()`` = ``_trans_endpose(is_endpose=False)``, i.e.
    #: the robot's "ee" point, not the TCP.
    ACT_LOG_POSE_POINT = "ee"

    def __init__(
        self,
        *,
        output_dir: Path,
        run_id: str,
        log_path: Path,
        policy_endpoint: str | None = None,
        policy_family: str = "pi05",
        policy_chunk_hint: int = DEFAULT_POLICY_CHUNK_HINT,
        pi0_step: int = DEFAULT_PI0_STEP,
        robotwin_root: Path | None = None,
        task_config: str = DEFAULT_TASK_CONFIG,
        instruction_type: str = DEFAULT_INSTRUCTION_TYPE,
        seed_source: str = "cache",
        seed_cache: Path | None = None,
        seed_cache_dir: Path | None = None,
        st_seed_base: int = OFFICIAL_ST_SEED_BASE,
        step_budget_scale: float = 1.0,
        cuda_device: int | None = None,
        telemetry_schema: str = "robotwin",
    ) -> None:
        self._output_dir = Path(output_dir).resolve()
        self._run_id = str(run_id)
        self._log_path = Path(log_path).resolve()
        self._policy_endpoint = policy_endpoint
        self._policy_family = str(policy_family)
        self._policy_chunk_hint = int(policy_chunk_hint)
        self._pi0_step = int(pi0_step)
        self._robotwin_root = Path(robotwin_root or DEFAULT_ROBOTWIN_ROOT).resolve()
        if task_config not in TASK_CONFIGS:
            raise ValueError(f"task_config must be one of {TASK_CONFIGS}, got {task_config!r}")
        self._task_config = str(task_config)
        self._instruction_type = str(instruction_type)
        if seed_source not in ("cache", "expert"):
            raise ValueError("seed_source must be 'cache' or 'expert'")
        self._seed_source = seed_source
        self._seed_cache_file = Path(seed_cache) if seed_cache else None
        self._seed_cache_dir = Path(seed_cache_dir or DEFAULT_SEED_CACHE_DIR)
        self._st_seed_base = int(st_seed_base)
        if not 0.1 <= float(step_budget_scale) <= 5.0:
            raise ValueError("step_budget_scale must be in [0.1, 5.0]")
        self._step_budget_scale = float(step_budget_scale)
        self._cuda_device = cuda_device
        if telemetry_schema not in ("robotwin", "robodojo"):
            raise ValueError("telemetry_schema must be 'robotwin' or 'robodojo'")
        self._telemetry_schema = telemetry_schema

        self._runtime_ready = False
        self._task_args_cache: dict[str, Any] = {}
        self._generate_descriptions: Any = None
        self._env: Any = None
        self._env_task: str | None = None
        self._env_live = False  # a scene built by setup_demo and not yet closed
        self._client: _PolicyClient | None = None
        self._episode: dict[str, Any] | None = None
        self._episodes_run = 0
        self._step_count = 0
        self._call_seq = 0
        self._act_log: list[dict[str, Any]] = []
        self._empty_closures: list[dict[str, Any]] = []
        self._empty_now: dict[str, bool] = {"left": False, "right": False}
        self._travel_cm: dict[str, float] = {"left": 0.0, "right": 0.0}
        self._seed_caches: dict[int, dict[str, Any]] = {}
        self._expert_seeds: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self._instance_nonce = f"{os.getpid():05d}{int(time.time()) % 100000:05d}"
        self._gpu_env: dict[str, str] = {}

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)
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
                "harness_oracle_grasp": self.harness_oracle_grasp,
                "harness_state_digest": self.harness_state_digest,
                "harness_world_digest": self.harness_world_digest,
            }
        )
        self._rpc.update(self._intervention_rpc())
        for name in INTERVENTION_METHODS + UNIMPLEMENTED_METHODS:
            if name not in self._rpc:
                self._rpc[name] = self._make_stub(name)

    def _make_stub(self, name: str):
        def _stub(*_args: Any, **_kwargs: Any) -> Any:
            raise NotImplementedError(
                f"{name}() is not implemented on the RoboTwin service. The control path "
                "(harness_episode_begin / harness_advance / harness_read_state_privileged / "
                "harness_oracle_grasp / harness_state_digest / harness_release_env / "
                "harness_quit) works; treatment verbs come from "
                "recovery_explore/robotwin_intervention.py when it is present."
            )

        _stub.__name__ = f"stub_{name}"
        return _stub

    def _log(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, default=str, sort_keys=True)
        with self._log_path.open("a") as stream:
            stream.write(line + "\n")

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if method == "healthz":
            return super()._dispatch(method, args, kwargs)
        if (self._round_budget or self._judge_locked) and (
            method in self._PRIVILEGED_METHODS or method in self._HOST_PRIVILEGED_METHODS
        ):
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
            "anchor_id": None,
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
        self._close_scene()
        if self._client is not None:
            self._client.close()
            self._client = None

    # ---- RoboTwin runtime -------------------------------------------------

    def _ensure_runtime(self) -> None:
        """chdir into the RoboTwin tree and make its imports resolvable.

        RoboTwin resolves ``./task_config``, ``./assets`` and
        ``./description`` relative to the CWD (``script/eval_policy.py`` does
        ``sys.path.append("./")`` and is run from the tree root), so we do the
        same. Output paths were made absolute in ``__init__``.
        """
        if self._runtime_ready:
            return
        root = self._robotwin_root
        if not (root / "envs" / "_base_task.py").is_file():
            raise RuntimeError(f"{root} is not a RoboTwin tree (no envs/_base_task.py)")
        if not (root / "assets" / "embodiments").is_dir():
            raise RuntimeError(f"{root}/assets has no embodiments/ (assets missing or dangling symlink)")
        os.chdir(root)
        for entry in (str(root / "envs" / "curobo" / "src"), str(root)):
            if os.path.isdir(entry) and entry not in sys.path:
                sys.path.insert(0, entry)
        for entry in (str(root / "policy"), str(root / "description" / "utils"),
                      str(root / "policy" / "pi05" / "packages" / "openpi-client" / "src")):
            if os.path.isdir(entry) and entry not in sys.path:
                sys.path.append(entry)
        from generate_episode_instructions import generate_episode_descriptions  # lazy

        self._generate_descriptions = generate_episode_descriptions
        self._runtime_ready = True

    def _task_args(self, task: str) -> dict[str, Any]:
        """The ``args`` dict ``script/eval_policy.py::main`` builds, verbatim
        in content, minus the video writer (``eval_video_log`` would spawn an
        ffmpeg per episode; it has no effect on the simulation)."""
        import yaml  # lazy
        from envs import CONFIGS_PATH  # lazy

        with open(f"./task_config/{self._task_config}.yml", "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)
        args["task_name"] = task
        args["task_config"] = self._task_config
        args["ckpt_setting"] = self._run_id
        embodiment_type = args.get("embodiment")
        with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
            embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
        with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
            camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        head_camera_type = args["camera"]["head_camera_type"]
        args["head_camera_h"] = camera_config[head_camera_type]["h"]
        args["head_camera_w"] = camera_config[head_camera_type]["w"]

        def robot_file(kind: str) -> str:
            path = embodiment_types[kind]["file_path"]
            if path is None:
                raise RuntimeError("No embodiment files")
            return path

        def robot_config(path: str) -> dict:
            with open(os.path.join(path, "config.yml"), "r", encoding="utf-8") as f:
                return yaml.load(f.read(), Loader=yaml.FullLoader)

        if len(embodiment_type) == 1:
            args["left_robot_file"] = robot_file(embodiment_type[0])
            args["right_robot_file"] = robot_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = robot_file(embodiment_type[0])
            args["right_robot_file"] = robot_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        else:
            raise RuntimeError("embodiment items should be 1 or 3")
        args["left_embodiment_config"] = robot_config(args["left_robot_file"])
        args["right_embodiment_config"] = robot_config(args["right_robot_file"])
        args["policy_name"] = self._policy_family
        args["eval_mode"] = True
        args["eval_video_log"] = False
        args.pop("eval_video_save_dir", None)
        return args

    def _ensure_task_env(self, task: str) -> Any:
        """One ``Base_Task`` subclass instance per task, reused across episodes
        like the official loop reuses ``TASK_ENV``.

        On a task switch the ``Robot`` (and with it the cuRobo planners, which
        are the expensive, GPU-resident part: ~50 s and ~2 GB) is carried over
        to the new instance; ``Base_Task.load_robot`` then takes its own
        ``robot.reset(scene)`` branch, the same one every later episode of a
        single-task official run takes.
        """
        self._ensure_runtime()
        if self._env is not None and self._env_task == task:
            return self._env
        import importlib  # lazy

        module = importlib.import_module(f"envs.{task}")
        env_cls = getattr(module, task, None)
        if env_cls is None:
            raise ValueError(f"RoboTwin has no task {task!r}")
        new_env = env_cls()
        if self._env is not None:
            self._close_scene()
            if hasattr(self._env, "robot"):
                new_env.robot = self._env.robot
        self._env = new_env
        self._env_task = task
        return new_env

    def _close_scene(self) -> None:
        """``TASK_ENV.close_env(clear_cache=...)`` with the official cadence
        (``clear_cache`` every ``clear_cache_freq`` episodes)."""
        if self._env is None or not self._env_live:
            return
        try:
            freq = max(1, int(self._task_args_cache.get("clear_cache_freq", 5)))
        except Exception:
            freq = 5
        # Official: close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        # where succ_seed already counts the episode being closed.
        try:
            self._env.close_env(clear_cache=((self._episodes_run + 1) % freq == 0))
        except Exception:
            pass
        self._env_live = False

    def _require_env(self) -> Any:
        if self._env is None or not self._env_live:
            raise _NotBuilt("no env: the harness must call harness_episode_begin first")
        return self._env

    def _require_episode(self) -> dict[str, Any]:
        if self._episode is None:
            raise _NotBuilt("no episode: call harness_episode_begin first")
        return self._episode

    def _reset_policy_after_rewind(self) -> None:
        """Called by the intervention mixin after a rewind: the next chunk's
        request carries ``reset=True`` (the protocol's episode-start flag), so
        the server drops any per-episode state exactly as at episode start."""
        if self._episode is not None:
            self._episode["policy_calls"] = 0

    def _policy(self) -> _PolicyClient:
        if self._policy_endpoint is None:
            raise RuntimeError("--policy-endpoint is required to advance the policy")
        if self._client is None:
            self._client = _PolicyClient(self._policy_endpoint)
        return self._client

    # ---- episode identity -------------------------------------------------

    def _seed_cache(self, seed: int) -> dict[str, Any]:
        seed = int(seed)
        if seed not in self._seed_caches:
            path = self._seed_cache_file or seed_cache_path(self._seed_cache_dir, self._task_config, seed)
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"no seed cache for task_config={self._task_config} seed={seed}: {path}. "
                    "Only seed 0 caches exist; use --seed-source expert for other seeds."
                )
            self._seed_caches[seed] = load_seed_cache(path, self._task_config)
        return self._seed_caches[seed]

    def _resolve_episode(self, task: str, seed: int, episode_index: int) -> dict[str, Any]:
        if self._seed_source == "cache":
            payload = self._seed_cache(seed)
            entry = cache_entry(payload, task, episode_index)
            entry.update(
                {
                    "seed_source": "cache",
                    "seed_cache": payload["_path"],
                    "seed_cache_sha256": payload["_sha256"],
                    "seed_base": payload.get("seed_base"),
                }
            )
            return entry
        return self._expert_entry(task, seed, episode_index)

    def _expert_entry(self, task: str, seed: int, episode_index: int) -> dict[str, Any]:
        """The official expert check, live: scan seeds from
        ``st_seed_base * (1 + seed)``, keep those where ``play_once`` plans and
        succeeds, return the ``episode_index``-th. Memoised to disk."""
        key = (task, int(seed))
        memo_path = self._output_dir / f"expert_seeds_{self._task_config}_{task}_seed{int(seed)}.json"
        if key not in self._expert_seeds:
            self._expert_seeds[key] = (
                json.loads(memo_path.read_text())["entries"] if memo_path.is_file() else []
            )
        found = self._expert_seeds[key]
        env = self._ensure_task_env(task)
        from envs.utils.create_actor import UnStableError  # lazy

        args = self._task_args(task)
        now_seed = (int(found[-1]["seed"]) + 1) if found else self._st_seed_base * (1 + int(seed))
        while len(found) <= int(episode_index):
            self._close_scene()
            ok = False
            info: dict = {}
            try:
                env.setup_demo(now_ep_num=len(found), seed=now_seed, is_test=True,
                               **{**copy.deepcopy(args), "render_freq": 0})
                self._env_live = True
                info = env.play_once()
                env.close_env()
                self._env_live = False
                ok = bool(env.plan_success and env.check_success())
            except UnStableError:
                ok = False
            except Exception as exc:  # official: "error occurs !" and skip
                print(f"[robotwin-svc] expert check seed {now_seed}: {exc}", flush=True)
                ok = False
            finally:
                if self._env_live:
                    try:
                        env.close_env()
                    except Exception:
                        pass
                    self._env_live = False
            if ok:
                found.append({"seed": int(now_seed), "episode_info": dict(info.get("info") or {})})
                memo_path.write_text(json.dumps({"task": task, "seed": int(seed),
                                                 "st_seed_base": self._st_seed_base,
                                                 "entries": found}))
            now_seed += 1
        entry = found[int(episode_index)]
        return {
            "seed": int(entry["seed"]),
            "episode_info": dict(entry["episode_info"]),
            "cache_index": int(episode_index),
            "seed_source": "expert",
            "seed_cache": str(memo_path),
            "seed_cache_sha256": None,
            "seed_base": self._st_seed_base,
        }

    def _choose_instruction(self, task: str, robotwin_seed: int, episode_info: Mapping[str, Any]) -> str:
        """``generate_episode_descriptions`` + ``np.random.choice`` exactly as
        ``script/eval_policy.py``, with Python's ``random`` seeded first (see
        the module docstring)."""
        random.seed(int(robotwin_seed))
        results = self._generate_descriptions(task, [dict(episode_info)], OFFICIAL_TEST_NUM)
        return str(np.random.choice(results[0][self._instruction_type]))

    # ---- observation / telemetry -----------------------------------------

    def _observation(self) -> dict[str, Any]:
        """``TASK_ENV.get_obs()``: renders all cameras, reads joint targets and
        end poses. Keys: ``observation`` (per camera ``rgb`` 240x320x3 uint8 +
        intrinsics/extrinsics), ``joint_action`` (``left_arm``,
        ``left_gripper``, ``right_arm``, ``right_gripper``, ``vector`` =
        14-dim), ``endpose`` (``left_endpose``/``right_endpose`` =
        [x,y,z,qw,qx,qy,qz], ``left_gripper``/``right_gripper``),
        ``pointcloud``."""
        return self._require_env().get_obs()

    def _measured(self) -> dict[str, Any]:
        """Joint state as SIMULATED, not as commanded.

        ``obs["joint_action"]`` (what the policy sees) reports the joint DRIVE
        TARGETS (``Robot.get_left_arm_jointState`` reads
        ``joint.get_drive_target()``) and the COMMANDED gripper value, so it
        cannot show an arm that is being held back or fingers that closed on
        nothing. This reads the articulation qpos instead
        (``get_*_arm_real_jointState``) and normalises the measured base finger
        joint by the joint's OWN limits (aloha: [0, 0.04765] m), exactly like
        ``RobotwinInterventionMixin._grip_norm``.

        It used to go through ``gripper_scale`` ([-0.01, 0.045]), whose range
        extends past the physical stop: fingers shut on NOTHING read 0.182, never
        reached ``EMPTY_CLOSURE_NORM`` (0.05), so no empty closure was ever
        reported and the judge read every missed grasp as "holding (0.182)".
        rtj12 (2026-09-19): all 4 failed treatment episodes sat at exactly 0.182
        while real holds read 0.25-0.48.
        """
        robot = self._require_env().robot
        out: dict[str, Any] = {}
        for side in ("left", "right"):
            entity = getattr(robot, f"{side}_entity")
            qpos = np.asarray(entity.get_qpos(), dtype=np.float64)
            active = entity.get_active_joints()
            arm = [float(qpos[active.index(j)]) for j in getattr(robot, f"{side}_arm_joints")]
            base = getattr(robot, f"{side}_gripper")[0][0]
            try:
                lo, hi = (float(v) for v in np.asarray(base.get_limits(), dtype=np.float64).reshape(-1)[:2])
                if not (np.isfinite(lo) and np.isfinite(hi) and hi - lo > 1e-9):   # no usable limits
                    lo, hi = (float(v) for v in getattr(robot, f"{side}_gripper_scale")[:2])
                opening = (float(qpos[active.index(base)]) - lo) / (hi - lo)
            except Exception:
                opening = None
            out[f"{side}_arm"] = arm
            out[f"{side}_opening"] = None if opening is None else float(np.clip(opening, 0.0, 1.0))
        return out

    def _proprio_state(self, obs: Mapping[str, Any] | None = None) -> dict[str, Any]:
        obs = obs if obs is not None else self._observation()
        ja = obs.get("joint_action", {}) or {}
        ep = obs.get("endpose", {}) or {}
        vector = _to_list(ja.get("vector"))
        measured: dict[str, Any] = {}
        try:
            measured = self._measured()
        except Exception:
            measured = {}
        episode = self._episode or {}
        env = self._env
        left_pose = _to_list(ep.get("left_endpose")) or None
        right_pose = _to_list(ep.get("right_endpose")) or None
        return {
            # 14-dim [left_arm(6), left_grip(1), right_arm(6), right_grip(1)],
            # exactly the policy's input_state (drive targets + commanded grip).
            "joint_positions": vector,
            "left_arm_joint_state": _to_list(ja.get("left_arm")),
            "right_arm_joint_state": _to_list(ja.get("right_arm")),
            # Commanded 0..1 opening (1 = open), what the policy reads.
            "gripper": {
                "left": float(ja["left_gripper"]) if ja.get("left_gripper") is not None else None,
                "right": float(ja["right_gripper"]) if ja.get("right_gripper") is not None else None,
            },
            # Simulated state (see _measured).
            "measured_arm_joints": {
                "left": measured.get("left_arm"),
                "right": measured.get("right_arm"),
            },
            "opening": {
                "left": measured.get("left_opening"),
                "right": measured.get("right_opening"),
            },
            # [x, y, z, qw, qx, qy, qz] in the world frame.
            "left_ee_pose": left_pose,
            "right_ee_pose": right_pose,
            "step_count": int(self._step_count),
            "control_steps_used": int(getattr(env, "take_action_cnt", 0) or 0) if env is not None else 0,
            "policy_steps_max": int(episode.get("max_steps") or 0) or None,
            "policy_steps_used": int(episode.get("committed_timestep") or 0),
            "policy_steps_left": (
                max(0, int(episode["max_steps"]) - int(episode["committed_timestep"]))
                if episode.get("max_steps")
                else None
            ),
            "recent_chunks": list(self._act_log[-6:]),
        }

    def _note_success_source(self) -> None:
        """Attribute a latched ``eval_success`` to whoever was stepping.

        ``_run_chunk`` (the policy) marks "policy" the moment ``take_action``
        latches success. The only other code that steps the simulator is the
        intervention mixin's ``_sim_action`` (repair steps), so a latched
        success first seen anywhere else -- here, via ``_oracle`` (which the
        mixin calls after every plan) or at the start of ``harness_advance``
        -- latched during a repair. A rewind that un-latches success clears it.
        """
        episode = self._episode
        env = self._env
        if episode is None or env is None or not self._env_live:
            return
        if bool(getattr(env, "eval_success", False)):
            if episode.get("success_latched_by") is None:
                episode["success_latched_by"] = "repair"
        else:
            episode["success_latched_by"] = None

    def _success_during_repair(self) -> bool:
        return bool((self._episode or {}).get("success_latched_by") == "repair")

    def _oracle(self) -> dict[str, Any]:
        """Privileged outcome, in the official sense.

        Official success is ``TASK_ENV.eval_success``, latched inside
        ``take_action`` the first physics step ``check_success()`` holds
        (``_base_task.py``); the episode is a success iff it latched before
        ``take_action_cnt`` reached ``step_lim``. We do NOT call
        ``check_success()`` here: it is a task method, and reading it outside
        the official cadence is not how the official loop scores.
        """
        env = self._require_env()
        self._note_success_source()
        success = bool(env.eval_success)
        used = int(env.take_action_cnt)
        limit = int(env.step_lim) if env.step_lim is not None else 0
        return {
            "task_success": success,
            "episode_score": 1.0 if success else 0.0,
            "process_score": None,
            "has_process_score": False,
            "end_flag": bool(success or (limit and used >= limit)),
            "control_steps_used": used,
            "step_limit": limit,
            "success_during_repair": self._success_during_repair(),
        }

    def _frames_from_obs(self, obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        cams = obs.get("observation", {}) or {}
        out: dict[str, np.ndarray] = {}
        for cam_key, alias in CAMERA_ALIASES.items():
            data = cams.get(cam_key)
            raw = data.get("rgb") if isinstance(data, Mapping) else None
            if raw is None:
                continue
            out[alias] = _resize_nearest(_as_rgb_uint8(raw), FRAME_SIZE)
        return {alias: out[alias] for alias in FRAME_CAMERAS if alias in out}

    def _episode_dir(self) -> Path:
        episode = self._require_episode()
        return self._output_dir / (
            f"{episode['task']}_s{int(episode['seed'])}_ep{int(episode['episode_index'])}"
        )

    # ---- harness surface --------------------------------------------------

    def harness_env_meta(self) -> dict[str, Any]:
        episode = self._episode or {}
        return {
            "benchmark": "RoboTwin",
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
            "policy_protocol": "openpi_ws",
            "pi0_step": self._pi0_step,
            "step_budget_scale": self._step_budget_scale,
            "action_dim": 14,
            "task_config": self._task_config,
            "instruction_type": self._instruction_type,
            "seed_source": self._seed_source,
            "seed_cache_dir": str(self._seed_cache_dir),
            "robotwin_root": str(self._robotwin_root),
            "cuda_device": self._cuda_device,
            "gpu_env": dict(self._gpu_env),
            "rt_denoiser": os.environ.get("ROBOTWIN_RT_DENOISER", "oidn"),
            "episodes_run": int(self._episodes_run),
            "pid": os.getpid(),
            "rss_mb": _rss_mb(),
            "telemetry_schema": self._telemetry_schema,
            "supports_intervention": bool(self._intervention_rpc()),
            "round_index": int(getattr(self, "_round_index", 0)),
            "round_budget": getattr(self, "_round_budget", None),
            "round_used": int(getattr(self, "_round_used", 0)),
            "resets_used": int(getattr(self, "_resets_used", 0)),
            "intervene_control_steps": int(getattr(self, "_intervene_steps", 0)),
        }

    def harness_episode_begin(
        self,
        task: str,
        seed: int = 0,
        episode_index: int = 0,
        obj_instance_split: str | None = None,
        max_steps_extra: int = 0,
    ) -> dict[str, Any]:
        """HARNESS ONLY: build one episode and stop before the first action.

        ``obj_instance_split`` is a RoboCasa concept; accepted and echoed, no
        effect (RoboTwin draws its objects from the seed). Any task may follow
        any other in one process (see ``_ensure_task_env``).
        """
        started = time.monotonic()
        env = self._ensure_task_env(task)
        self._close_scene()
        self._episode = None
        resolved = self._resolve_episode(task, int(seed), int(episode_index))
        robotwin_seed = int(resolved["seed"])

        from envs.utils.create_actor import UnStableError  # lazy

        args = self._task_args(task)
        self._task_args_cache = args
        try:
            env.setup_demo(now_ep_num=int(episode_index), seed=robotwin_seed, is_test=True,
                           **copy.deepcopy(args))
        except UnStableError as exc:
            self._env_live = True
            self._close_scene()
            raise RuntimeError(
                f"seed {robotwin_seed} ({task}, cache index {episode_index}) was unstable on "
                f"setup although it passed the expert check: {exc}"
            ) from exc
        self._env_live = True
        instruction = self._choose_instruction(task, robotwin_seed, resolved["episode_info"])
        env.set_instruction(instruction=instruction)

        native_step_lim = int(env.step_lim)
        max_steps = int(round(native_step_lim * self._step_budget_scale)) + max(0, int(max_steps_extra))
        env.step_lim = max_steps
        horizon = int(self._policy_chunk_hint)

        self._step_count = 0
        self._act_log = []
        self._empty_closures = []
        self._empty_now = {"left": False, "right": False}
        self._travel_cm = {"left": 0.0, "right": 0.0}
        # Fresh treatment-arm bookkeeping per episode (mixin host contract).
        self._init_intervention_state()
        self._episode = {
            "task": str(task),
            "seed": int(seed),
            "episode_index": int(episode_index),
            "obj_instance_split": obj_instance_split,
            "robotwin_seed": robotwin_seed,
            "cache_index": int(resolved["cache_index"]),
            "seed_source": resolved["seed_source"],
            "seed_cache": resolved["seed_cache"],
            "seed_cache_sha256": resolved["seed_cache_sha256"],
            "seed_base": resolved["seed_base"],
            "episode_info": dict(resolved["episode_info"]),
            "task_config": self._task_config,
            "instruction": instruction,
            "committed_timestep": 0,
            "next_query_index": 0,
            "policy_calls": 0,
            # "policy" | "repair" | None: who was stepping the sim when
            # eval_success latched (see _note_success_source).
            "success_latched_by": None,
            "max_steps": int(max_steps),
            "native_step_lim": native_step_lim,
            "horizon": horizon,
            "horizon_confirmed": False,
            "total_chunks": int(math.ceil(max_steps / max(1, horizon))),
            "frames_written": 0,
        }
        self._episodes_run += 1
        self._episode_dir().mkdir(parents=True, exist_ok=True)
        return {
            "ok": True,
            "task": str(task),
            "seed": int(seed),
            "episode_index": int(episode_index),
            "obj_instance_split": obj_instance_split,
            "robotwin_seed": robotwin_seed,
            "task_config": self._task_config,
            "instruction": instruction,
            "horizon": horizon,
            "horizon_confirmed": False,
            "max_steps": int(max_steps),
            "native_step_lim": native_step_lim,
            "total_chunks": int(self._episode["total_chunks"]),
            "policy_family": self._policy_family,
            "build_s": round(time.monotonic() - started, 2),
        }

    def _run_chunk(self, env: Any, obs: Mapping[str, Any], actions: np.ndarray, budget: int) -> tuple[int, Any]:
        """Execute one chunk the way ``deploy_policy.eval`` does. Returns
        (actions executed, last observation or None)."""
        executed = 0
        last_obs = None
        for action in actions:
            if env.eval_success or int(env.take_action_cnt) >= budget:
                break
            env.take_action(action)
            executed += 1
            if env.eval_success:
                if self._episode is not None and self._episode.get("success_latched_by") is None:
                    self._episode["success_latched_by"] = "policy"
                # take_action already refreshed now_obs on success; the official
                # loop's remaining get_obs calls are after the episode ended.
                break
            last_obs = env.get_obs()
        return executed, last_obs

    def harness_advance(self, num_chunks: int = 1) -> dict[str, Any]:
        """HARNESS ONLY: let the policy run ``num_chunks`` chunks (each one
        official ``eval`` call). Same return field set as RoboDojo."""
        episode = self._require_episode()
        env = self._require_env()
        self._note_success_source()  # a repair may have latched success since the last call
        client = self._policy()

        budget = int(episode["max_steps"])
        committed = int(env.take_action_cnt)
        query_index = int(episode["next_query_index"])
        executed_total = 0
        frames: list[dict[str, np.ndarray]] = []
        frame_steps: list[int] = []
        stopped = "chunks_done"
        episode_dir = self._episode_dir()

        for _ in range(int(num_chunks)):
            if env.eval_success:
                stopped = "task_success"
                break
            if committed >= budget:
                stopped = "step_budget_exhausted"
                break

            chunk_t0 = committed
            obs = self._observation()  # the official outer loop's get_obs
            request = build_policy_request(obs, episode["instruction"], reset=episode["policy_calls"] == 0)
            infer_t0 = time.monotonic()
            chunk = client.infer(request)
            infer_s = time.monotonic() - infer_t0
            episode["policy_calls"] = int(episode["policy_calls"]) + 1
            if chunk.ndim != 2 or chunk.shape[1] != 14:
                raise RuntimeError(f"policy returned an action chunk of shape {chunk.shape}; expected (H, 14)")
            actions = chunk[: self._pi0_step]
            if not episode["horizon_confirmed"]:
                episode["horizon"] = int(actions.shape[0])
                episode["horizon_confirmed"] = True
                episode["total_chunks"] = int(math.ceil(budget / max(1, actions.shape[0])))

            start = self._proprio_state(obs)
            sim_t0 = time.monotonic()
            executed, last_obs = self._run_chunk(env, obs, actions, budget)
            sim_s = time.monotonic() - sim_t0
            committed = int(env.take_action_cnt)
            executed_total += executed
            self._step_count += executed

            query_index += 1
            end_obs = last_obs if last_obs is not None else self._observation()
            shot = self._frames_from_obs(end_obs)
            frames.append(shot)
            frame_steps.append(int(committed))
            for alias, image in shot.items():
                _write_png(episode_dir / f"c{query_index - 1:03d}_t{committed:05d}_{alias}.png", image)
            episode["frames_written"] = int(episode["frames_written"]) + len(shot)

            end = self._proprio_state(end_obs)
            moved_cm: list[float | None] = []
            for side in ("left", "right"):
                a, b = start.get(f"{side}_ee_pose"), end.get(f"{side}_ee_pose")
                if a and b:
                    d = float(np.linalg.norm(np.asarray(b[:3]) - np.asarray(a[:3]))) * 100.0
                    self._travel_cm[side] += d
                    moved_cm.append(round(d, 2))
                else:
                    moved_cm.append(None)
            # Stall signal: last EXECUTED joint target vs simulated joints.
            track_err = None
            track_err_arm = None
            grip_err = None
            if executed > 0:
                want = np.asarray(actions[executed - 1], dtype=np.float64)
                la, ra = (end["measured_arm_joints"] or {}).get("left"), (end["measured_arm_joints"] or {}).get("right")
                if la and ra and len(la) == 6 and len(ra) == 6:
                    got = np.asarray(la + ra, dtype=np.float64)
                    track_err = round(float(np.max(np.abs(np.concatenate([want[0:6], want[7:13]]) - got))), 3)
                    # per arm [left, right]: with two arms the single max cannot say WHICH arm is held
                    # back (rtj12: screener blamed the left arm, judge the right, same number).
                    track_err_arm = [round(float(np.max(np.abs(want[0:6] - got[0:6]))), 3),
                                     round(float(np.max(np.abs(want[7:13] - got[6:12]))), 3)]
                op = end["opening"]
                if op.get("left") is not None and op.get("right") is not None:
                    grip_err = [round(float(np.clip(want[6], 0, 1)) - float(op["left"]), 3),
                                round(float(np.clip(want[13], 0, 1)) - float(op["right"]), 3)]
            opening = end["opening"]
            grip_cmd = end["gripper"]
            oracle = self._oracle()
            entry = {
                "chunk": int(query_index - 1),
                "t": [int(chunk_t0), int(committed)],
                "steps": int(executed),
                "horizon": int(actions.shape[0]),
                "joints_end": [round(float(v), 4) for v in end["joint_positions"]],
                "gripper": opening,
                "grip": [
                    round(float(opening["left"]), 3) if opening.get("left") is not None else None,
                    round(float(opening["right"]), 3) if opening.get("right") is not None else None,
                ],
                "grip_cmd": [
                    round(float(grip_cmd["left"]), 3) if grip_cmd.get("left") is not None else None,
                    round(float(grip_cmd["right"]), 3) if grip_cmd.get("right") is not None else None,
                ],
                "grip_err": grip_err,
                # What the POLICY commanded in this chunk (judge needs the action history, not only
                # its effect): last 14-dim joint target, per-arm range of the gripper command
                # (0 close .. 1 open), and the measured arm joints at chunk end.
                "cmd_end": ([round(float(v), 3) for v in np.asarray(actions[executed - 1]).reshape(-1)]
                            if executed > 0 else None),
                "cmd_grip_range": ([[round(float(np.min(actions[:executed, 6])), 2), round(float(np.max(actions[:executed, 6])), 2)],
                                    [round(float(np.min(actions[:executed, 13])), 2), round(float(np.max(actions[:executed, 13])), 2)]]
                                   if executed > 0 else None),
                "measured_joints_end": {
                    side: ([round(float(v), 3) for v in (end["measured_arm_joints"] or {}).get(side) or []] or None)
                    for side in ("left", "right")
                },
                "moved_cm": moved_cm,
                "track_err_rad": track_err,
                "track_err_arm_rad": track_err_arm,
                "left_ee_pose": end["left_ee_pose"],
                "right_ee_pose": end["right_ee_pose"],
                "left_ee_start": start.get("left_ee_pose"),     # per-chunk displacement in the judge's table
                "right_ee_start": start.get("right_ee_pose"),
                "task_success": bool(oracle["task_success"]),
                "infer_s": round(infer_s, 3),
                "sim_s": round(sim_s, 3),
            }
            self._act_log.append(entry)
            for side in ("left", "right"):
                op_side = opening.get(side)
                pose = end.get(f"{side}_ee_pose")
                cmd = grip_cmd.get(side)
                if op_side is None or pose is None:
                    continue
                was_empty = self._empty_now.get(side, False)
                now_empty = float(op_side) <= EMPTY_CLOSURE_NORM and (cmd is None or float(cmd) <= GRIP_CMD_SHUT)
                self._empty_now[side] = now_empty
                if now_empty and not was_empty:
                    self._empty_closures.append({
                        "t": int(committed),
                        "arm": side,
                        "eef": [round(float(v), 4) for v in pose[:3]],
                        "opening": round(float(op_side), 3),
                    })
                    del self._empty_closures[:-20]
            if len(self._act_log) > 400:
                del self._act_log[:-400]
            if env.eval_success:
                stopped = "task_success"
                break
            if committed >= budget:
                stopped = "step_budget_exhausted"
                break

        episode["committed_timestep"] = int(committed)
        episode["next_query_index"] = int(query_index)
        oracle = self._oracle()
        succeeded = bool(oracle["task_success"])
        # No extra render: the newest observation the env holds is now_obs
        # (refreshed by every get_obs, and by take_action on success).
        final_obs = env.now_obs if getattr(env, "now_obs", None) else self._observation()
        final = self._proprio_state(final_obs)

        return {
            "task_success": succeeded,
            "committed_timestep": int(committed),
            "max_steps": int(budget),
            "next_query_index": int(query_index),
            "steps_executed": int(executed_total),
            "chunks_done": len(frames),
            "stopped": stopped,
            "truth": {
                "task_success": succeeded,
                "episode_score": oracle["episode_score"],
                "process_score": None,
                "robotwin_seed": int(episode["robotwin_seed"]),
                "task_config": self._task_config,
                "control_steps_used": int(oracle["control_steps_used"]),
                "success_during_repair": bool(oracle["success_during_repair"]),
                "success_latched_by": episode.get("success_latched_by"),
            },
            "telemetry": {
                "schema": self._telemetry_schema,
                "chunks": self._act_log[-40:],
                "active_arm": self._active_arm(),
                "travel_cm": {k: round(v, 2) for k, v in self._travel_cm.items()},
                "eef_now": {"left": final["left_ee_pose"], "right": final["right_ee_pose"]},
                "opening_now": {
                    side: (round(float(v), 3) if v is not None else None)
                    for side, v in final["opening"].items()
                },
                "grip_cmd_now": final["gripper"],
                "empty_closures": list(self._empty_closures[-8:]),
                "joints_now": final["joint_positions"],
                "measured_joints_now": final.get("measured_arm_joints"),
                "horizon": int(episode["horizon"]),
                "horizon_confirmed": bool(episode["horizon_confirmed"]),
                "steps_used": int(committed),
                "steps_left": max(0, int(budget) - int(committed)),
                "step_limit": int(budget),
            },
            "done": bool(succeeded or committed >= budget or stopped != "chunks_done"),
            "frames": frames,
            "frame_steps": frame_steps,
            "frames_dir": str(episode_dir.resolve()),
        }

    def harness_read_state_privileged(self) -> dict[str, Any]:
        return {"state": self._proprio_state(), "oracle": self._oracle()}

    def harness_oracle_grasp(self) -> dict[str, Any]:
        """HARNESS ONLY: RoboCasa's ``grasp`` verb, RoboTwin flavour.

        ``is_grasped`` per arm = some finger link of that arm is in contact
        with a non-robot, non-static actor (the task objects), read from the
        SAPIEN contact list (the method ``Base_Task.get_gripper_actor_contact
        _position`` uses). Plus the official success latch.
        """
        env = self._require_env()
        robot = env.robot
        finger_links = {
            "left": set(getattr(robot, "left_fix_gripper_name", []) or []),
            "right": set(getattr(robot, "right_fix_gripper_name", []) or []),
        }
        for side in ("left", "right"):
            for joint in getattr(robot, f"{side}_gripper"):
                finger_links[side].add(joint[0].child_link.get_name())
        static = {"ground", "wall", "table", ""}
        robot_links = set()
        for side in ("left", "right"):
            for link in getattr(robot, f"{side}_entity").get_links():
                robot_links.add(link.get_name())
        held: dict[str, set] = {"left": set(), "right": set()}
        for contact in env.scene.get_contacts():
            names = (contact.bodies[0].entity.name, contact.bodies[1].entity.name)
            for side in ("left", "right"):
                for a, b in (names, names[::-1]):
                    if a in finger_links[side] and b not in static and b not in robot_links:
                        held[side].add(b)
        oracle = self._oracle()
        return {
            "is_grasped": bool(held["left"] or held["right"]),
            "is_grasped_by_arm": {side: bool(v) for side, v in held.items()},
            "grasped_actors": {side: sorted(v) for side, v in held.items()},
            "task_success": oracle["task_success"],
            "control_steps_used": oracle["control_steps_used"],
        }

    def harness_world_digest(self) -> dict[str, Any]:
        """HARNESS ONLY: ``scene_digest`` of the live env -- the exact function
        the parity runner applies to the official loop's TASK_ENV, so the two
        sha256 values are comparable. (``harness_state_digest`` may be the
        intervention mixin's richer variant, with a different field set.)"""
        env = self._require_env()
        digest, sha = scene_digest(env)
        return {"sha256": sha, "scene": digest, "take_action_cnt": int(env.take_action_cnt),
                "eval_success": bool(env.eval_success)}

    def reset_step_counter(self) -> dict[str, Any]:
        self._require_env()
        previous = int(self._step_count)
        self._step_count = 0
        return {"step_count": 0, "previous_step_count": previous}

    def harness_release_env(self) -> dict[str, Any]:
        """HARNESS ONLY: close the episode's scene (official ``close_env``
        cadence); keep the task env, its Robot/cuRobo planners and the policy
        connection for the next episode."""
        self._close_scene()
        self._episode = None
        self._act_log = []
        self._step_count = 0
        return {"ok": True, "app_alive": self._env is not None}

    def harness_quit(self) -> dict[str, Any]:
        import threading

        threading.Timer(0.5, lambda: os._exit(0)).start()
        return {"ok": True, "pid": os.getpid()}

    # ---- selftest ---------------------------------------------------------

    def selftest(self, task: str, chunks: int = 3, chunk_len: int = 16) -> dict[str, Any]:
        """One episode driven by a hold-still policy, no server: separates
        "SAPIEN + assets + GPU work" from "the policy server works"."""
        t0 = time.monotonic()
        begin = self.harness_episode_begin(task, 0, 0)
        env = self._require_env()
        step_times: list[float] = []
        for _ in range(int(chunks)):
            obs = self._observation()
            hold = np.asarray(obs["joint_action"]["vector"], dtype=np.float64)
            for _ in range(int(chunk_len)):
                ts = time.monotonic()
                env.take_action(hold)
                env.get_obs()
                step_times.append(time.monotonic() - ts)
        report = {
            "ok": True,
            "begin": begin,
            "control_steps": int(env.take_action_cnt),
            "mean_step_s": round(float(np.mean(step_times)), 4) if step_times else None,
            "total_s": round(time.monotonic() - t0, 2),
            "oracle": self._oracle(),
            "gpu_env": self._gpu_env,
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # -- shared with env_service.py / env_service_robodojo.py --------------
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--port-tries", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="recovery-explore-robotwin")
    parser.add_argument("--log", type=Path, default=None,
                        help="call log; default <output-dir>/env_service_calls.jsonl")
    parser.add_argument("--policy-endpoint", default=None,
                        help="openpi websocket policy server, ws://host:port (http:// and "
                        "bare host:port accepted)")
    parser.add_argument("--policy-family", default="pi05", choices=["pi05"])
    parser.add_argument("--step-budget-scale", type=float, default=1.0)
    parser.add_argument("--cuda-device", type=int, default=None,
                        help="physical GPU; exported as CUDA_VISIBLE_DEVICES (+ Vulkan pin) "
                        "before sapien is imported")
    parser.add_argument("--parent-watch", action="store_true")
    parser.add_argument("--debug-resets", type=int, default=0)       # accepted, ignored
    parser.add_argument("--render-size", type=int, default=FRAME_SIZE)  # accepted, ignored
    parser.add_argument("--anchors-jsonl", type=Path, default=None)  # accepted, ignored
    # -- RoboTwin-only ------------------------------------------------------
    parser.add_argument("--robotwin-root", type=Path,
                        default=Path(os.environ.get("ROBOTWIN_ROOT", DEFAULT_ROBOTWIN_ROOT)))
    parser.add_argument("--task-config", choices=list(TASK_CONFIGS),
                        default=os.environ.get("ROBOTWIN_TASK_CONFIG", DEFAULT_TASK_CONFIG))
    parser.add_argument("--instruction-type", default=DEFAULT_INSTRUCTION_TYPE,
                        choices=["seen", "unseen"])
    parser.add_argument("--pi0-step", type=int, default=DEFAULT_PI0_STEP,
                        help="actions executed per chunk = reply[:pi0_step] (deploy_policy.yml: 50)")
    parser.add_argument("--policy-chunk", type=int, default=DEFAULT_POLICY_CHUNK_HINT,
                        help="assumed chunk length before the first reply (reporting only)")
    parser.add_argument("--seed-source", choices=["cache", "expert"],
                        default=os.environ.get("ROBOTWIN_SEED_SOURCE", "cache"))
    parser.add_argument("--seed-cache", type=Path,
                        default=Path(os.environ["ROBOTWIN_SEED_CACHE"]) if os.environ.get("ROBOTWIN_SEED_CACHE") else None,
                        help="one explicit cache file (overrides --seed-cache-dir lookup)")
    parser.add_argument("--seed-cache-dir", type=Path,
                        default=Path(os.environ.get("ROBOTWIN_SEED_CACHE_DIR", DEFAULT_SEED_CACHE_DIR)))
    parser.add_argument("--st-seed-base", type=int, default=OFFICIAL_ST_SEED_BASE,
                        help="--seed-source expert only: first seed = base * (1 + seed)")
    parser.add_argument("--telemetry-schema", choices=["robotwin", "robodojo"],
                        default=os.environ.get("ROBOTWIN_TELEMETRY_SCHEMA", "robotwin"))
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--selftest-task", default="adjust_bottle")
    parser.add_argument("--selftest-chunks", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gpu_env = configure_gpu(args.cuda_device)
    output_dir = Path(args.output_dir)
    log_path = Path(args.log) if args.log else output_dir / "env_service_calls.jsonl"
    service = RobotwinEnvService(
        output_dir=output_dir,
        run_id=args.run_id,
        log_path=log_path,
        policy_endpoint=args.policy_endpoint,
        policy_family=args.policy_family,
        policy_chunk_hint=args.policy_chunk,
        pi0_step=args.pi0_step,
        robotwin_root=args.robotwin_root,
        task_config=args.task_config,
        instruction_type=args.instruction_type,
        seed_source=args.seed_source,
        seed_cache=args.seed_cache,
        seed_cache_dir=args.seed_cache_dir,
        st_seed_base=args.st_seed_base,
        step_budget_scale=args.step_budget_scale,
        cuda_device=args.cuda_device,
        telemetry_schema=args.telemetry_schema,
    )
    service._gpu_env = gpu_env
    print(f"[robotwin-svc] task_config={args.task_config} root={args.robotwin_root} gpu_env={gpu_env}",
          flush=True)
    if args.selftest:
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
