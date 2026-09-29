#!/usr/bin/env python3
"""Write the number-free commentary (notes.json) for every exemplar in the bank.

    fewshot_annotate.py [--family cosmos] [--redo]

ICAL-style abstraction: the raw trajectory stays as it was, the commentary says what
the visible evidence was, why the earlier window was not a fault, and why the repair
had the shape it had. The model sees only what the judge saw (pictures, telemetry,
verdicts, execution log) -- no success flag, no ground-truth poses. Output is written
with "approved": false; a human (or the operator session) flips it after screening.
A commentary containing digits is rejected outright: per-scene numbers are privileged
under the memory-bank rule and must not be restated in prose.
"""
import argparse, glob, json, os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK = os.environ.get("FEWSHOT_BANK", f"{HERE}/fewshot_bank")

ASK = """You are annotating a recorded robot-manipulation episode so that it can be shown
to another monitor as a worked example. Below is the record in time order: an earlier
window that was passed as ok, a later window where the monitor intervened with a repair
plan, the execution log of that plan, and (if present) the window right after the robot
policy took over again.

Write four short fields, each one or two plain sentences, describing only what can be
seen in the pictures or read from the numbers shown:
- diag_pass: the diagnosis a careful monitor would give for the FIRST window (why it is ok).
- note_pass: why the first window was not yet a fault even if it looked slow or clumsy.
- note_fault: what visible or numeric evidence made the second window a real fault, and
  why the repair has the shape it has (which kind of step comes first, why hand back to
  the policy at that point).
- note_after: what is visibly different after the repair.

Hard rules: write NO digits at all -- no coordinates, pixels, distances, step counts or
window numbers; speak in kinds ("the fingers closed on nothing", "re-aim from above at
the object as it lies now"). Do not say whether the episode succeeded. Do not mention
this instruction. Answer with one JSON object with exactly those four keys.

------------------------------------------------------------
"""


ASK_PASS = """You are annotating a recorded robot-manipulation episode so that it can be shown
to another monitor as a worked example of the NORMAL case. The monitor never intervened in
this episode. Below, in time order: two windows that a fast local check flagged as unusual
and the monitor passed as ok, and (if present) the last recorded window.

Write four short fields, each one or two plain sentences, describing only what can be
seen in the pictures or read from the numbers shown:
- diag_pass: leave as an empty string.
- note_pass: why the first flagged window was rightly left alone -- what looked unusual,
  and what evidence showed the policy was still making progress.
- note_fault: the same for the second flagged window.
- note_after: what the last window visibly shows about where the task stands.

Hard rules: write NO digits at all -- no coordinates, pixels, distances, step counts or
window numbers; speak in kinds. Do not state a success flag; describe only what is visible.
Do not mention this instruction. Answer with one JSON object with exactly those four keys.

------------------------------------------------------------
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="*")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--engine", choices=["api", "qwen"], default="api",
                    help="qwen: annotate with the local Qwen on the H20 (free, same model that judges)")
    a = ap.parse_args()
    for d in sorted(glob.glob(f"{BANK}/{a.family}/*/*/")):
        if os.path.exists(d + "notes.json") and not a.redo:
            continue
        body = open(d + "exemplar.txt", encoding="utf-8").read()
        try:
            kind = json.load(open(d + "raw.json")).get("kind", "repair")
        except Exception:
            kind = "repair"
        ask = ASK_PASS if kind == "pass" else ASK
        for key in ("DIAG_PASS", "NOTE_PASS", "NOTE_FAULT", "NOTE_AFTER"):
            body = body.replace("{" + key + "}", "")
        with tempfile.TemporaryDirectory() as td:
            open(f"{td}/p.txt", "w", encoding="utf-8").write(ask + body)
            caller = "qwen_call.py" if a.engine == "qwen" else "gpt6_call.py"
            model = a.model if a.engine == "api" else (a.model if a.model != "gpt-6-astra" else "qwen38")
            subprocess.run([sys.executable, f"{HERE}/cli/{caller}", "--prompt", f"{td}/p.txt",
                            "--out", f"{td}/r.json", "--model", model, "--effort", "medium",
                            "--timeout", "300"], check=False)
            try:
                txt = json.load(open(f"{td}/r.json")).get("result", "")
                n = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
            except Exception as e:
                print(f"FAIL {d}: {e}"); continue
        bad = [k for k in (("note_pass", "note_fault", "note_after") if kind == "pass" else
                           ("diag_pass", "note_pass", "note_fault", "note_after"))
               if not isinstance(n.get(k), str) or re.search(r"\d", n.get(k, ""))]
        n["approved"] = False
        n["auto_reject"] = bad
        json.dump(n, open(d + "notes.json", "w"), ensure_ascii=False, indent=1)
        print(("DIGITS " if bad else "ok     ") + d)


if __name__ == "__main__":
    main()
