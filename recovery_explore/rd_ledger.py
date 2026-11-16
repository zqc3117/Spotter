#!/usr/bin/env python3
"""rdv9: one short block listing the ops that already failed in this episode (from this cell's exec_w*.json outcomes),
so the staged policy (switch method after two failures of the same op, then hand back for good) has its evidence.
Usage: rd_ledger.py CELL_DIR"""
import collections, glob, json, os, re, sys
FAIL = {"unreachable", "no_grasp", "descent_blocked", "closed_empty", "push_blocked", "no_motion", "slipped",
        "held_by_width_only", "target_on_robot", "no_object", "press_short", "grasped_not_placed", "released_unverified"}
d = sys.argv[1]; c = collections.OrderedDict()
for f in sorted(glob.glob(os.path.join(d, "exec_w*.json")), key=lambda x: [int(n) for n in re.findall(r"\d+", os.path.basename(x))]):
    w = re.search(r"exec_w(\d+)", f).group(1)
    try: ex = json.load(open(f)).get("executed") or []
    except Exception: continue
    for e in ex:
        o = str(e.get("outcome") or "")
        if o in FAIL or o.startswith("stopped_short"):
            c.setdefault((e.get("op"), o), []).append(w)
if c:
    print("\nOPS THAT ALREADY FAILED IN THIS EPISODE (op -> outcome, windows):")
    for (op, o), ws in c.items():
        print(f"  {op} -> {o}  x{len(ws)}  (w{', w'.join(ws)})")
    print("Same op failed twice -> switch to a materially different method once; if that fails too, hand that "
          "sub-goal back to the policy for good unless the scene clearly changes.")
