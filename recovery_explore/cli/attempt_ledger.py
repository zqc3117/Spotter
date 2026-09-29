#!/usr/bin/env python3
"""Failure ledger for one intervention window, pasted into the replan prompt after a rewind.

    python3 attempt_ledger.py <cell_dir> <window> <next_attempt>

A rewind puts the scene back, so the only way the judge can learn from an attempt is to be
told, in plain words, what it sent, what each step did, how the hands ended up and why it
called the attempt a failure. Reads the files judge_driver_v7.sh already writes per segment:
plan_w{W}_a{A}_r{R}.json, exec_w{W}_a{A}_r{R}.json, act_w{W}_a{A}_r{R}.json.
"""
from __future__ import annotations

import glob, json, os, re, sys


def _load(path: str):
    try:
        return json.load(open(path))
    except Exception:
        return None


def _inner(d):
    """act_/replan_ files wrap the model's JSON as a string under "result"."""
    if isinstance(d, dict) and isinstance(d.get("result"), str):
        m = re.search(r"\{.*\}", d["result"], re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {"note": d["result"][-300:]}
    return d if isinstance(d, dict) else {}


def _step_txt(st: dict) -> str:
    op = st.get("op")
    bits = [str(op)]
    if st.get("arm"):
        bits.append(str(st["arm"]))
    for k in ("dxyz", "xyz", "drot", "dz", "state", "chunks", "to"):
        if st.get(k) is not None:
            bits.append(f"{k}={st[k]}")
    if isinstance(st.get("target"), dict):
        bits.append(f"target={st['target']}")
    for side in ("left", "right"):
        if isinstance(st.get(side), dict):
            bits.append(f"{side}={st[side]}")
    return " ".join(bits)


def _outcome(e: dict) -> str:
    if e.get("error"):
        return "REJECTED/ERROR: " + str(e["error"])[:160]
    op = e.get("op")
    if op == "policy":
        return f"policy ran {e.get('chunks')} chunk(s)"
    if op == "gripper":
        return f"fingers now {e.get('opening_norm')}" + (" -- CLOSED ON NOTHING" if e.get("closed_empty") else "")
    stop = e.get("servo_stop")
    s = f"{stop}, {e.get('final_dist_m')} m from target"
    if e.get("clipped"):
        s += ", CLIPPED"
    return s


def _hands(state: dict) -> str:
    op, cmd = (state or {}).get("opening") or {}, (state or {}).get("gripper") or {}
    out = []
    for side in ("left", "right"):
        o, c = op.get(side), cmd.get(side)
        if o is None:
            continue
        o = float(o)
        word = ("SHUT ON NOTHING" if o <= 0.05 and (c is None or float(c) <= 0.5)
                else "open" if o >= 0.85 else "partly closed")
        out.append(f"{side} {o:.2f} ({word})")
    return ", ".join(out)


def ledger(cd: str, w: int, next_attempt: int) -> str:
    lines = []
    for a in range(1, int(next_attempt)):
        segs = sorted(glob.glob(os.path.join(cd, f"plan_w{w}_a{a}_r*.json")),
                      key=lambda p: int(re.search(r"_r(\d+)\.json$", p).group(1)))
        if not segs:
            continue
        lines.append(f"ATTEMPT {a} -- failed, the scene was rewound afterwards:")
        for p in segs:
            r = re.search(r"_r(\d+)\.json$", p).group(1)
            plan = _load(p) or []
            ex = _load(os.path.join(cd, f"exec_w{w}_a{a}_r{r}.json")) or {}
            act = _inner(_load(os.path.join(cd, f"act_w{w}_a{a}_r{r}.json")))
            lines.append(f"  segment {int(r) + 1} you sent: " + " | ".join(_step_txt(s) for s in plan if isinstance(s, dict)))
            for e in ex.get("executed") or []:
                lines.append(f"    step {e.get('step')} {e.get('op')}: {_outcome(e)}")
            lines.append(f"    stopped because: {ex.get('stop_reason')}; hands after: {_hands(ex.get('state'))}")
            if act.get("note"):
                lines.append(f"    you judged it '{act.get('result')}': {str(act['note'])[:260]}")
    if not lines:
        return ""
    return ("WHAT YOU ALREADY TRIED IN THIS INTERVENTION (all of it failed and was undone):\n"
            + "\n".join(lines)
            + "\n\nLearn from this before you plan. Name, in your diagnosis, what made the last attempt fail "
              "and what you change because of it. Your next plan must differ from every plan above in at "
              "least one real way: a different position (by 5 mm or more), a different approach direction "
              "or height, the other arm, a different gripper state, or grasping yourself instead of handing "
              "back. The simulator rejects a plan that repeats one of these exactly.")


if __name__ == "__main__":
    print(ledger(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
