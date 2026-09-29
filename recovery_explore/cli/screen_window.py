#!/usr/bin/env python3
"""Window screener: run the local Qwen over a window before calling the judge. The driver calls this once per window.

Behaviour:
* The judge is called only if ``flag`` is true; if false, log it and move on to the next window.
* **fail-closed**: if the screener raises, returns nothing, or cannot be reached at all, treat it as flag=true.
  An unavailable screener must never turn into a pass -- that would leave the whole trajectory unwatched without any sign of it.

The GPU lease is also managed here. An external GPU holder, if one is used, keeps the card **by default**; only at the
moment the screener is actually called is a file placed in the shared lease directory asking it to yield, and the file is
deleted right afterwards so the holder comes back. In other words a lease exists only in SCREEN=1 windows; at all other
times (including the whole SCREEN=0 control arm) the holder never lets go. Lease files carry an mtime, so if this
process crashes the holder can clear them after STALE_S.

Usage:
    screen_window.py --montage w3/wam_montage.png --telemetry w3/telemetry.txt \
        --task "pick the ..." --prev-flags "false,false,true" --out screen_w3.json
Exit codes: 0 = normal (whether the window passes depends on flag); 3 = **the screener module cannot be imported at all**.
3 is not the same as fail-closed: unreachable service, service error, invalid reply -- those are runtime faults,
and returning flag=true per fail-closed and carrying on is correct. An import failure means **the configuration is wrong**:
the whole run would proceed with every window flagged, silently turning --arm b into --arm a, with ~30% more calls and
results that do not belong to that arm, and no sign of it. So it must be loud: a banner on stderr + exit code 3,
and the driver decides to stop (see the handling of screener_missing in judge_driver_v7.sh).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

# The screener module (screener.py) sits next to this file. SCREENER_DIR can point elsewhere,
# for older deployments that keep it outside the repository.
SCREENER_DIR = os.environ.get("SCREENER_DIR") or os.path.dirname(os.path.abspath(__file__))
# Lease directory (VLLM_LEASE_DIR). An external GPU holder, if one is used, can watch this directory to yield the card;
# if it runs on another machine this must be a path on a shared volume. Unset = no lease coordination (the holder never yields).
_LEASE_DIR = os.environ.get("VLLM_LEASE_DIR", "")
LEASE_DIR = Path(_LEASE_DIR) if _LEASE_DIR else None


def _emit(result: dict, out: str | None) -> None:
    line = json.dumps(result, ensure_ascii=False)
    if out:
        try:
            Path(out).write_text(line + "\n", encoding="utf-8")
        except Exception:
            pass
    print(line)


class _Lease:
    """Lease held while the screener is being called; an external GPU holder, if one is used, can watch for it and yield the card.

    Failing to take the lease does not block the call -- at worst the holder does not yield and the screener is a bit
    slower. Never skip the screener because the lease could not be written (that would be a second failure path besides fail-closed).
    """

    def __init__(self, tag: str) -> None:
        self.path: Path | None = (
            LEASE_DIR / f"{socket.gethostname()}-{os.getpid()}-{tag}" if LEASE_DIR else None)

    def __enter__(self) -> "_Lease":
        if LEASE_DIR is None:
            return self
        try:
            LEASE_DIR.mkdir(parents=True, exist_ok=True)
            assert self.path is not None
            self.path.write_text(str(time.time()))
        except Exception:
            self.path = None
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            if self.path is not None:
                self.path.unlink(missing_ok=True)
        except Exception:
            pass



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--montage", required=True, help="overview image for this window (PNG/JPEG)")
    ap.add_argument("--telemetry", required=True, help="telemetry table text file for this window")
    ap.add_argument("--task", required=True)
    ap.add_argument("--prev-flags", default="", help="comma-separated true/false: the flags of the earlier windows in this episode")
    ap.add_argument("--host", default=os.environ.get("SCREENER_HOST", "127.0.0.1"),
                    help="host running vLLM. On another machine give its IP (the driver passes $SCREENER_HOST)")
    ap.add_argument("--out", default=None, help="write the full result here (screen_w{N}.json)")
    ap.add_argument("--tag", default="w", help="window tag in the lease filename, to tell who is holding it")
    a = ap.parse_args()

    result = {"flag": True, "risk": None, "rule_hit": None, "why": "", "port": None,
              "dt": None, "source": "screener"}

    # Import failure is handled separately: it is a configuration problem, not a runtime fault, and must not fall into the fail-closed path below.
    sys.path.insert(0, SCREENER_DIR)
    try:
        import screener  # type: ignore
    except Exception as exc:
        result["why"] = f"screener not importable from {SCREENER_DIR}: {type(exc).__name__}: {exc}"
        result["source"] = "screener_missing"
        _emit(result, a.out)
        sys.stderr.write(
            "\n" + "=" * 72 + "\n"
            "Failed to import the screener module; cannot run --arm b.\n"
            f"  Looked in: {SCREENER_DIR}\n"
            f"  Error:     {type(exc).__name__}: {exc}\n"
            "Continuing would flag every window -- that is --arm a, not --arm b.\n"
            "Point SCREENER_DIR at the directory containing screener.py, or use --arm a to run the no-screener arm explicitly.\n"
            + "=" * 72 + "\n")
        return 3

    try:
        # Cross-machine: the client defaults to 127.0.0.1; when the driver is not on the service host, point it at that host's IP.
        screener.HOST = a.host
        img = Path(a.montage).read_bytes()
        tel = Path(a.telemetry).read_text(encoding="utf-8", errors="replace")
        prev = [t.strip().lower() == "true" for t in a.prev_flags.split(",") if t.strip()]
        with _Lease(a.tag):
            r = screener.screen(img, tel, a.task, prev)
        if not isinstance(r, dict) or "flag" not in r:
            result["why"] = f"screener returned {type(r).__name__}, treating as flag"
        else:
            result.update(r)
            result["flag"] = bool(r.get("flag"))
    except Exception as exc:  # fail-closed
        result["why"] = f"{type(exc).__name__}: {exc}"
        result["source"] = "fail_closed"
        result["flag"] = True

    _emit(result, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
