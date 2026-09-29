#!/usr/bin/env python3
"""Pick memory-bank entries for the judge: task-kind matches first, then situation matches.

v1 scored one bag of words against `instruction + situation flags`, with a +1 for
verified entries. In practice the same four aperture/grasp entries won every time
(103 of 122 prompts in l24c) and nothing learned in later rounds was ever shown.
v2 keeps two buckets: entries that mention the kind of object or fixture named in
the instruction, and entries that match the situation words. Light stemming so
"approaching" meets "approach".
"""
from __future__ import annotations
import argparse, glob, os, re

BANK = os.environ.get("MEMORY_BANK") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory_bank")
STOP = set("a an the and or but of to in on at for with from by is are was were be been it its this that these those not no as if then than so you your i we they he she pick place put turn open close press the into onto from onto".split())

def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf): return w[: -len(suf)]
    return w

def words(text):
    return {stem(w) for w in re.findall(r"[a-z][a-z-]{2,}", text.lower()) if w not in STOP}

def entries():
    out = []
    for path in sorted(glob.glob(os.path.join(BANK, "global", "*.md"))):
        raw = open(path).read(); parts = raw.split("---", 2)
        if len(parts) < 3: continue
        fm, body = parts[1], parts[2]
        get = lambda k: (re.search(rf"^{k}:\s*(.+)$", fm, re.M) or [None, ""])[1].strip()
        sym = re.sub(r"[\[\]]", "", get("symptom"))
        # symptom may be a YAML block list after render(): collect "- x" lines under it
        blk = re.search(r"^symptom:\s*\n((?:\s+-\s.*\n?)+)", fm, re.M)
        if blk: sym = " ".join(l.strip("- ").strip() for l in blk.group(1).splitlines())
        how = re.search(r"\*\*How to apply:\*\*\s*\n(.*?)(?:\n\*\*|\Z)", body, re.S)
        howlines = [l.strip("- ").strip() for l in (how.group(1) if how else "").strip().splitlines() if l.strip()]
        head = words(" ".join([get("title"), get("applies_when"), sym, get("kind")]))
        out.append({"id": os.path.basename(path)[:-3], "title": get("title").strip('"'), "conf": get("conf" "idence"), "kind": get("kind"),
                    "how": howlines, "bag": head, "body": words(" ".join(howlines)) | words(body[:1200]) })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default=""); ap.add_argument("--situation", default="")
    ap.add_argument("--limit", type=int, default=5); ap.add_argument("--task-slots", type=int, default=2)
    ap.add_argument("--kind", choices=["pnp", "mech"], default="mech",
                    help="pnp: the instruction's fixture words are locations, not the subject; drop them")
    ap.add_argument("--query", help="v1 compatibility: whole query as situation")
    ap.add_argument("--index", action="store_true",
                    help="print every visible lesson as a one-line title index (first window of an episode)")
    ap.add_argument("--prim-slots", type=int, default=0,
                    help="reserve this many slots for primitive/infra lessons (intervention time)")
    a = ap.parse_args()
    if a.query and not a.instruction: a.instruction, a.situation = "", a.query
    iw, sw = words(a.instruction), words(a.situation)
    if a.kind == "pnp":
        # in pick-and-place tasks cabinet / counter / sink / pan are locations; the subject is the object being moved
        iw -= {stem(w) for w in "cabinet counter sink stove microwave pan plate drawer shelf table top bowl basket".split()}
    es = entries(); picked = []
    # Entry types are exclusive: fixture words in the title/applies_when mark a mechanism lesson, object-transport words mark a PnP lesson.
    # PnP tasks do not see pure mechanism entries (drawer handles, faucet levers), and mechanism tasks do not see pure transport entries
    # (whether the object follows the hand). Entries touching both (e.g. "empty closure is not a failure on mechanisms")
    # are visible to both.
    FIX = {stem(w) for w in "door drawer knob faucet button lever burner fixture mechanism panel spout tap handle".split()}
    PNP = {stem(w) for w in "object transport placement place deliver carry destination source drop lift held holding".split()}
    def visible(e):
        m, pn = bool(e["bag"] & FIX), bool(e["bag"] & PNP)
        if a.kind == "pnp": return not (m and not pn)
        return not (pn and not m)
    es = [e for e in es if visible(e)]
    if a.index:
        # Show the full title index once at the start of an episode: each title is itself a rule; 47 lines is about 1.5k tokens.
        # Entries matching the situation get their How-to-apply expanded in later windows; this guarantees every lesson is seen at least once.
        order = ["perception", "strategy", "primitive", "infra", "failure"]
        print("Index of lessons from earlier runs (titles only; those matching the moment are expanded under each window; evidence, not orders):")
        for k in order:
            grp = [e for e in es if e["kind"] == k]
            if not grp: continue
            print(f"  [{k}]")
            for e in sorted(grp, key=lambda e: e["title"]): print(f"  - {e['title']}")
        return
    def bonus(e): return {"verified": 0.5, "probable": 0.25}.get(e["conf"], 0.0)
    # bucket 1: task-kind — must share a word with the instruction (object / fixture / verb)
    def score(e, wi, ws):
        return wi*len(iw & e["bag"]) + 0.5*wi*len(iw & e["body"]) + ws*len(sw & e["bag"]) + 0.3*ws*len(sw & e["body"]) + bonus(e)
    task = sorted([e for e in es if (iw & e["bag"]) or (iw & e["body"])], key=lambda e: -score(e, 3, 1))
    for e in task[: a.task_slots]: picked.append(e)
    # bucket 1b: primitive / infra lessons — how the tools behave — only asked for at
    # intervention time, when a plan is about to be written
    if a.prim_slots:
        prim = sorted([e for e in es if e not in picked and e["kind"] in ("primitive", "infra") and (sw & e["bag"] or sw & e["body"])],
                      key=lambda e: -score(e, 0.5, 1))
        for e in prim[: a.prim_slots]: picked.append(e)
    # bucket 2: situation
    sit = sorted([e for e in es if e not in picked and (len(sw & e["bag"]) + len(sw & e["body"])) >= 2], key=lambda e: -score(e, 0.5, 1))
    for e in sit: 
        if len(picked) >= a.limit: break
        picked.append(e)
    if not picked: return
    print("Lessons from earlier runs that match this situation (they are evidence, not orders; the images and telemetry win if they disagree):")
    for e in picked:
        print(f"\n- {e['title']}")
        for line in e["how"][:2]: print(f"    {line}")

if __name__ == "__main__": main()
