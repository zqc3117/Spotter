#!/usr/bin/env python3
"""Print the few-shot block for one judge brief.

    fewshot_select.py --family cosmos --task CoffeeSetupMug --n 1 [--images-as paths]

--n 0 prints nothing (the driver then behaves exactly as before). Only exemplars whose
notes.json says "approved": true are served. Fallback order when a task has fewer than
n approved exemplars: same family + same task -> other family + same task -> same
family + same task group -> other family + same task group. --images-as paths turns
the [[IMAGE:...]] markers into "open this file" lines for engines that read pictures
with a Read tool (claude / codex); api and qwen interleave the markers themselves.
"""
import argparse, glob, json, os, re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK = os.environ.get("FEWSHOT_BANK", f"{HERE}/fewshot_bank")


ALL24 = """PnPCounterToCab PnPCabToCounter PnPCounterToSink PnPSinkToCounter PnPCounterToMicrowave
PnPMicrowaveToCounter PnPCounterToStove PnPStoveToCounter OpenSingleDoor CloseSingleDoor OpenDoubleDoor
CloseDoubleDoor OpenDrawer CloseDrawer TurnOnStove TurnOffStove TurnOnSinkFaucet TurnOffSinkFaucet
TurnSinkSpout CoffeeSetupMug CoffeeServeMug CoffeePressButton TurnOnMicrowave TurnOffMicrowave""".split()


def group(task):
    if task.startswith("PnP"): return "pnp"
    if task.startswith("Coffee") and "Mug" in task: return "mug"
    if re.match(r"(Open|Close)", task): return "door"
    return "switch"      # TurnOn/Off*, TurnSinkSpout, CoffeePressButton


def approved(family, task=None, grp=None, kind=None):
    out = []
    for d in sorted(glob.glob(f"{BANK}/{family}/*/*/")):
        t = d.rstrip("/").split("/")[-2]
        if task and t != task: continue
        if grp and group(t) != grp: continue
        try:
            n = json.load(open(d + "notes.json"))
            k = json.load(open(d + "raw.json")).get("kind", "repair")
        except Exception:
            continue
        if kind and k != kind: continue
        if n.get("approved") is True and os.path.exists(d + "exemplar.txt"):
            out.append((d, n, t, k))
    return out


def pick(family, task, n):
    """Exactly n exemplars whenever the bank can supply them. Slots alternate repair / pass
    (slot 1 = a repair, slot 2 = an episode the policy finished on its own, ...); each slot
    walks the fallback chain, and a slot whose kind is exhausted takes the other kind."""
    other = "pi05" if family == "cosmos" else "cosmos"
    chain = ((family, dict(task=task)), (other, dict(task=task)),
             (family, dict(grp=group(task))), (other, dict(grp=group(task))),
             (family, {}), (other, {}))
    picks, seen = [], set()
    for slot in range(n):
        want = "repair" if slot % 2 == 0 else "pass"
        # An exemplar of the SAME task beats a repair borrowed from a related task: if no
        # unused same-task repair exists but a same-task pass does, this slot takes the pass.
        same = lambda kind: [d for fam in (family, other) for d, *_ in approved(fam, task=task, kind=kind) if d not in seen]
        if want == "repair" and not same("repair") and same("pass"):
            want = "pass"
        for kind in (want, "pass" if want == "repair" else "repair"):
            got = None
            for fam, kw in chain:
                for d, nn, t, k in approved(fam, kind=kind, **kw):
                    if d not in seen:
                        got = (d, nn, t, fam, k); break
                if got: break
            if got: break
        if not got:
            break
        seen.add(got[0]); picks.append(got)
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="cosmos")
    ap.add_argument("--task", default="")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="verify every task gets n=1,2,3 exemplars")
    ap.add_argument("--max-images", type=int, default=0,
                    help="cap on pictures in the whole block (0 = no cap); context frames go first, then checkpoints")
    ap.add_argument("--images-as", choices=["markers", "paths"], default="markers")
    a = ap.parse_args()
    if a.check:
        tasks = sorted({d.rstrip("/").split("/")[-2] for d in glob.glob(f"{BANK}/*/*/*/")} | set(ALL24))
        bad = 0
        for fam in ("cosmos", "pi05"):
            for t in tasks:
                for n in (1, 2, 3):
                    pk = pick(fam, t, n)
                    if len(pk) < n: bad += 1
                    if n == 2:
                        print(f"{fam:6s} {t:24s} n=2 -> " + ", ".join(
                            f"{k}:{'same' if tt == t else tt}{'' if ff == fam else '(' + ff + ')'}" for _, _, tt, ff, k in pk))
        print("SHORTFALLS:", bad)
        raise SystemExit(1 if bad else 0)
    if a.n <= 0:
        return
    picks = pick(a.family, a.task, a.n)
    if len(picks) < a.n:
        import sys
        print(f"fewshot_select: only {len(picks)} of {a.n} exemplar(s) available for {a.family}/{a.task}", file=sys.stderr)
    if not picks:
        return
    k = len(picks)
    print(f"WORKED EXAMPLE{'S' if k > 1 else ''} ({k}). "
          "Each one is a past episode, from a different kitchen with the objects in other places, "
          "shown in time order with the pictures, the numbers and the decisions that were made. "
          "Copy the way of reasoning and the shape of the repair. Never copy a coordinate, a pixel "
          "or an offset from an example: read those off the live pictures and numbers only. "
          "Most windows of the examples were passed as ok; that is the normal case.\n")
    bodies = [open(d + "exemplar.txt", encoding="utf-8").read() for d, *_ in picks]
    if a.max_images > 0:
        # drop optional frames (caption line + marker line) until the block fits: episode-end and
        # start context first, then mid/before, then repair checkpoints; core stage pictures stay.
        for pat in (r"\[Context\] The last recorded[^\n]*\n\[\[IMAGE:[^\]]+\]\]\n", r"\[Context\] The first window[^\n]*\n\[\[IMAGE:[^\]]+\]\]\n",
                    r"\[Context\][^\n]*\n\[\[IMAGE:[^\]]+\]\]\n", r"\[Repair checkpoint[^\n]*\n\[\[IMAGE:[^\]]+\]\]\n"):
            for j in range(len(bodies) - 1, -1, -1):
                while sum(b.count("[[IMAGE:") for b in bodies) > a.max_images and re.search(pat, bodies[j]):
                    bodies[j] = re.sub(pat, "", bodies[j], count=1)
    for i, (d, n, t, fam, kind) in enumerate(picks, 1):
        body = bodies[i - 1]
        for key in ("DIAG_PASS", "NOTE_PASS", "NOTE_FAULT", "NOTE_AFTER"):
            val = str(n.get(key.lower(), "")).strip()
            if key.startswith("NOTE") and val:
                val = "Commentary: " + val
            body = body.replace("{" + key + "}", val)
        # the released bank stores picture paths relative to the bank root; make them absolute
        body = re.sub(r"\[\[IMAGE:(?!/)([^\]]+)\]\]", lambda m: f"[[IMAGE:{BANK}/{m.group(1)}]]", body)
        if a.images_as == "paths":
            body = re.sub(r"\[\[IMAGE:([^\]]+)\]\]", r"(example picture, open it: \1)", body)
        same = "the same task" if t == a.task else f"a related task ({t})"
        same += ", a repair" if kind == "repair" else ", the policy was rightly left alone"
        print(f"=== Worked example {i} of {k}: {same} ===")
        print(body.rstrip())
        print(f"=== End of worked example {i} ===\n")
    print("Everything below this line is the LIVE episode you are judging.\n")


if __name__ == "__main__":
    main()
