#!/usr/bin/env python3
"""rex -- thin **model-side** CLI for failure-state recovery exploration (runs on the operator's machine and reaches the service through a port-forward).

One subcommand = one HTTP call = one interface. Compact JSON goes to stdout; images are saved to
local files and their paths printed. Designed for Claude Code to call via Bash and view images with Read.

## This file is the complete definition of "what the model can do" (frozen 2026-09-11)

There is one rule: **the model may use only what a real robot / deployed VLM can obtain**. Everything
else is harness (experiment orchestration + privileged scoring), kept in ``cli/harness.py``, and **no
path** from this file reaches it -- not by defaults, not by "don't pass that flag"; the interface simply does not exist.

Model-side tools (all of them; the parameters listed are all the parameters):

| Tool | Parameters |
|---|---|
| ``state``     | None. End-effector pose / gripper width / step count. **Never includes the oracle** |
| ``render``    | ``--out-dir`` (resolution fixed at 512) |
| ``unproject`` | ``--row --col --camera`` |
| ``move-to``   | ``--xyz --gripper`` |
| ``lift``      | ``--dz --gripper`` |
| ``gripper``   | ``open`` \\| ``close`` |
| ``cosmos``    | ``--num-chunks`` |
| ``meta``      | None. Task instruction + bookkeeping (**no object name**) |

### What was removed, and why (see the docstrings of the endpoints in env_service.py)

* ``state --oracle``: a switch that defaults to off is still a switch -- it lives in --help and in muscle memory,
  and one slip turns a masked run into a privileged run, whose mined experience fails outright on a VLM
  (a VLM has no object_xyz, no contact queries, no success flag). The unmasked read moved to
  ``harness_read_state_privileged``, which this CLI does not expose.
* ``grasped``: it calls ``env._check_grasp``, a MuJoCo contact query; the upstream docstring itself
  says these are "privileged labels for storage only; these never enter inference". Giving it to the model
  amounts to a noise-free success oracle: close, ask "did I get it", branch -- a robot cannot ask that,
  and neither can a VLM. The model now has to judge from gripper width (proprioception) + images before and
  after lifting. The predicate stays in ``harness_oracle_grasp``, for scoring only.
* ``cosmos --seed``: the seed is not information a robot has; exposing it invites "reroll until it hits" -- resampling
  the same state until the policy happens to succeed. That is not a transferable skill, and it destroys attribution
  (you cannot tell "this recovery pose works" from "the fourth sample worked"). Seeds come from the service's own monotonic counter.
* ``cosmos --prompt``: the service already knows this episode's instruction and passes it verbatim. An arbitrary prompt is
  not merely unfair, it is **an immediate 500** -- the policy only accepts instructions present in the precomputed T5 cache.
* ``cosmos --num-candidates``: fixed at 4. K is a deployment-side compute budget, not something the policy changes for itself mid-episode.
* ``move-to --step-clip/--max-steps``, ``render --size``: controller internals and camera resolution;
  a robot does not tune these per command, and making them tunable would make recipes mined under different clips incomparable.
* ``make-failure`` / ``reset`` / ``reset-steps``: experiment orchestration, moved to ``cli/harness.py``.

## Usage

    kubectl port-forward -n <namespace> <pod> 8420:8420 &
    harness.py run-to-failure --task PnPCounterToCab --seed 901   # harness creates the state first
    rex.py meta
    rex.py state
    rex.py render --out-dir ./frames
    rex.py unproject --row 397 --col 248 --camera wrist
    rex.py move-to --xyz 4.82 -0.77 1.03
    rex.py nudge --dxyz 0.03 0 0
    rex.py rotate --drot 0 0.2 0
    rex.py cosmos --num-chunks 2

No numpy / requests dependency; standard library only, so any python3 can run it.
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_ENDPOINT = "http://127.0.0.1:8420"

# numpy dtype -> (struct format, byte size). The server encodes ndarrays with rpc_transport's _NumpyEncoder
# as {"__ndarray__": b64, "dtype": ..., "shape": [...]}; this decodes them back,
# so the CLI does not need numpy just to decode an array.
_FMT = {
    "float64": ("d", 8), "float32": ("f", 4),
    "int64": ("q", 8), "int32": ("i", 4), "int16": ("h", 2), "int8": ("b", 1),
    "uint64": ("Q", 8), "uint32": ("I", 4), "uint16": ("H", 2), "uint8": ("B", 1),
    "bool": ("?", 1),
}


def _decode(obj):
    """Turn the server's ndarray markers back into nested lists."""
    if isinstance(obj, dict):
        if "__ndarray__" in obj and set(obj) <= {"__ndarray__", "dtype", "shape"}:
            raw = base64.b64decode(obj["__ndarray__"])
            dtype = str(obj.get("dtype"))
            shape = list(obj.get("shape") or [])
            if dtype not in _FMT:
                return {"_unsupported_dtype": dtype, "shape": shape}
            fmt, size = _FMT[dtype]
            flat = list(struct.unpack("<%d%s" % (len(raw) // size, fmt), raw))
            for dim in reversed(shape[1:]):
                flat = [flat[i:i + dim] for i in range(0, len(flat), dim)]
            return flat
        if "__npscalar__" in obj and set(obj) <= {"__npscalar__", "dtype"}:
            return obj["__npscalar__"]
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


def call(endpoint: str, method: str, timeout: float = 1800.0, **kwargs):
    body = json.dumps({"method": method, "args": [], "kwargs": kwargs}).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/call", data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        die(f"cannot reach {endpoint}: {exc}. Is the port-forward up?")
    if isinstance(payload, dict) and payload.get("error"):
        die(str(payload["error"]).strip().splitlines()[-1])
    result = payload.get("result") if isinstance(payload, dict) else payload
    return _decode(result)


def die(msg: str) -> None:
    print(json.dumps({"ok": False, "error": msg}), file=sys.stderr)
    raise SystemExit(1)


def emit(obj) -> None:
    print(json.dumps(obj, separators=(",", ":"), default=str))


def write_png(path: Path, rows) -> None:
    """Write a PNG with zero dependencies (rows is an HxWx3 nested list)."""
    import zlib

    h = len(rows); w = len(rows[0])
    raw = bytearray()
    for row in rows:
        raw.append(0)                      # filter type 0
        for px in row:
            raw.extend((int(px[0]) & 255, int(px[1]) & 255, int(px[2]) & 255))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
           + chunk(b"IEND", b""))
    path.write_bytes(png)


# Keep the old private name so imports elsewhere do not break.
_write_png = write_png


def main() -> None:
    ap = argparse.ArgumentParser(prog="rex", description=__doc__.split("\n")[0])
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("meta", help="task instruction and bookkeeping (no object name -- a VLM does not get it either)")

    sub.add_parser("state", help="end-effector pose / gripper width / step count. Never includes the oracle")

    p = sub.add_parser("nudge", help="relative displacement (m, world frame). Small correction: keeps the pose the policy chose and only offsets it")
    p.add_argument("--dxyz", nargs=3, type=float, required=True)
    p.add_argument("--gripper", default="hold", help="hold | open | close | number")

    p = sub.add_parser("render", help="three-camera RGB (512) + primary-camera depth + camera parameters")
    p.add_argument("--out-dir", default="./frames")

    p = sub.add_parser("unproject", help="pixel -> world coordinates")
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--col", type=int, required=True)
    p.add_argument("--camera", default="primary", choices=["primary", "secondary", "wrist"])

    p = sub.add_parser(
        "move-to",
        help="OSC servo to world coordinates. At most 5 cm per call; relaxed to 60 cm (retracing the path) when the target is "
             "within 8 cm of any point in state's visited_points / grasp_attempts",
    )
    p.add_argument("--xyz", nargs=3, type=float, required=True)
    p.add_argument("--gripper", default="hold", help="hold | open | close | number")

    sub.add_parser(
        "reset",
        help="return to the state at the start of this intervention and reissue an action budget. At most 5 times per episode",
    )


    p = sub.add_parser("lift", help="raise/lower along world z")
    p.add_argument("--dz", type=float, required=True)
    p.add_argument("--gripper", default="hold")

    p = sub.add_parser("rotate", help="wrist rotation (axis-angle, radians). At most 0.35 rad (about 20 deg) per call, clipped by the server")
    p.add_argument("--drot", nargs=3, type=float, required=True, metavar=("RX","RY","RZ"))
    p.add_argument("--gripper", default="hold", help="hold | open | close | number")

    p = sub.add_parser("gripper", help="open/close")
    p.add_argument("action", choices=["open", "close"])

    p = sub.add_parser(
        "exec-plan",
        help="hand a plan of <=8 primitives to the server for sequential execution. Stops and returns at a checkpoint or an anomaly",
    )
    p.add_argument("--plan-file", required=True, help="JSON file: array of steps")
    p.add_argument("--frames-dir", default=None, help="save the three-camera images from the stopping moment here")

    p = sub.add_parser(
        "cosmos",
        help="run Cosmos closed-loop from the current state. prompt/seed/candidate count are set by the service, not by your parameters",
    )
    p.add_argument("--num-chunks", type=int, default=2)
    p.add_argument("--out-dir", default="./frames")

    a = ap.parse_args()
    E = a.endpoint

    if a.cmd == "meta":
        emit(call(E, "env_meta"))
    elif a.cmd == "state":
        # No arguments. Masking happens on the server, not by dropping fields here -- no flag can lift it
        emit(call(E, "read_state"))
    elif a.cmd == "nudge":
        g = {"open": -1.0, "close": 1.0}.get(a.gripper, a.gripper)
        emit(call(E, "nudge", dxyz=list(a.dxyz), gripper=g))
    elif a.cmd == "render":
        out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
        r = call(E, "render")
        paths = {}
        for cam, img in (r.get("rgb") or {}).items():
            fp = out / f"{cam}.png"
            write_png(fp, img)
            paths[cam] = str(fp.resolve())
        summary = {"images": paths, "cameras": list((r.get("cameras") or {}).keys()),
                   "size": r.get("size")}
        depth = r.get("depth")
        if depth:
            flat = [v for row in depth for v in row if isinstance(v, (int, float))]
            if flat:
                summary["depth_range_m"] = [round(min(flat), 4), round(max(flat), 4)]
            # The full depth map is neither saved nor echoed: a 512x512 float array is 5 MB, a disaster to read
            # into context, and unproject is enough for localisation (it looks up depth on the server).
            summary["depth"] = "summarized; use `unproject` for 3D points"
        emit(summary)
    elif a.cmd == "unproject":
        emit(call(E, "unproject", row=a.row, col=a.col, camera=a.camera))
    elif a.cmd == "move-to":
        g = {"open": -1.0, "close": 1.0}.get(a.gripper, a.gripper)
        emit(call(E, "move_to", xyz=list(a.xyz), gripper=g))
    elif a.cmd == "reset":
        emit(call(E, "reset_window"))
    elif a.cmd == "lift":
        g = {"open": -1.0, "close": 1.0}.get(a.gripper, a.gripper)
        emit(call(E, "lift", dz=a.dz, gripper=g))
    elif a.cmd == "rotate":
        g = {"open": -1.0, "close": 1.0}.get(a.gripper, a.gripper)
        emit(call(E, "rotate", drot=list(a.drot), gripper=g))
    elif a.cmd == "gripper":
        emit(call(E, "gripper", action=a.action))
    elif a.cmd == "exec-plan":
        plan = json.loads(Path(a.plan_file).read_text())
        if isinstance(plan, dict):           # allow passing a judge turn's whole JSON directly
            plan = plan.get("plan") or []
        r = call(E, "execute_plan", plan=plan)
        # Frames are not echoed (a few MB of arrays in context is a disaster); they are saved and only paths are reported.
        frames = r.pop("frames", None) or {}
        r.pop("cameras", None)
        if a.frames_dir and frames:
            out = Path(a.frames_dir); out.mkdir(parents=True, exist_ok=True)
            # Tile the three cameras side by side into one image instead of three files: every extra image costs the model
            # another tool round trip; measured, going from three images to one at checkpoints cut the rounds per intervention turn from 3 to 1.
            cams = [c for c in ("primary", "secondary", "wrist") if c in frames]
            rows = []
            if cams:
                h = len(frames[cams[0]])
                for y in range(h):
                    row = []
                    for c in cams:
                        row.extend(frames[c][y])
                    rows.append(row)
                fp = out / "checkpoint.png"
                write_png(fp, rows)
                r["image"] = str(fp.resolve())
                r["image_layout"] = " | ".join(cams)
        emit(r)
    elif a.cmd == "cosmos":
        out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
        r = call(E, "call_cosmos", num_chunks=a.num_chunks)
        # The server returns list[{camera: HxWx3}], one entry per chunk -- not a flat list of images.
        # imagined = what Cosmos expected to happen, real = what actually happened; comparing them is the basis for
        # spotting "grasped in imagination, missed in reality" -- with the privileged predicate gone, this is one of the model's
        # main ways to judge success (the other is state's gripper_width plus render before and after lift).
        paths = {}
        for key, tag in (("imagined_frames", "imagined"), ("real_frames", "real")):
            frames = r.get(key)
            if not isinstance(frames, list):
                continue
            for i, per_cam in enumerate(frames):
                if not isinstance(per_cam, dict):
                    continue
                for cam, img in per_cam.items():
                    try:
                        fp = out / f"{tag}_chunk{i}_{cam}.png"
                        write_png(fp, img)
                        paths[f"{tag}_chunk{i}_{cam}"] = str(fp.resolve())
                    except Exception as exc:
                        paths[f"{tag}_chunk{i}_{cam}"] = f"<unwritable: {exc}>"
        emit({k: v for k, v in r.items()
              if k not in ("imagined_frames", "real_frames")} | {"frames": paths})


if __name__ == "__main__":
    main()
