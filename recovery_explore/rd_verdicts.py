#!/usr/bin/env python3
"""rdv7 (RD_PERSIST): rewrite the shared plan spec for RoboDojo -- stdin -> stdout.
1. Verdict definitions: "ok" means the policy is visibly progressing; stuck / failed expectation / no progress is
   "intervene" with a different approach each time; "giveup" only when no steps are left or the objects are gone.
2. Drop the "Three rules the simulator enforces" paragraph: those plan gates are off on RoboDojo (ROBODOJO_PLAN_GATES=0),
   so telling the judge a manual close is refused, or that a missed step locks it out, was false and discouraging.
3. With RD_STEPWISE=1, the "executed IN ORDER ... without asking you again" sentence is replaced."""
import os, re, sys

s = sys.stdin.read()
OLD = ('- verdict "ok": the policy is coping. plan MUST be empty. Most windows are ok.\n'
       '- verdict "intervene": something is wrong that you can fix. plan is 1 to 8 steps,\n'
       '  executed IN ORDER by the simulator without asking you again, so write the whole\n'
       '  repair, not just its first move.\n'
       '- verdict "giveup": beyond rescue. plan MUST be empty. This is final for the episode.')
step = os.environ.get("RD_STEPWISE", "0") == "1"
staged = os.environ.get("RD_STAGED", "0") == "1"
NEW = ('- verdict "ok": the policy is visibly making progress toward the goal in these pictures -- the object it\n'
       '  works on moved toward where the task wants it, a step got done, or a grasp or placement is under way and\n'
       '  is not repeating an earlier failure. plan MUST be empty.\n'
       '- verdict "intervene": anything else while steps remain -- the task is stuck, the same step failed again,\n'
       '  your last expectation did not hold, or nothing moved toward the goal in the last two windows. Each time,\n'
       '  try an approach that differs from what already failed. "No reliable short fix", a repair that needs\n'
       '  several steps, or few steps left are NOT reasons to answer ok: ok hands the steps back to a policy that\n'
       '  is not progressing. plan is 1 to 8 steps, ' +
       ('of which only the first is executed before you see its result\n  and choose the next (stepwise mode).\n'
        if step else 'executed IN ORDER by the simulator without asking you again,\n  so write the whole repair, not just its first move.\n') +
       '- verdict "giveup": only when no steps are left to act with, or the objects the task needs have left every\n'
       '  camera and the reach of both arms. plan MUST be empty. This is final for the episode.')
if staged:
    NEW = ('- verdict "ok": the policy is coping -- it is progressing, or it has just missed and is plausibly about to\n'
           '  try again on its own. plan MUST be empty.\n'
           '- verdict "intervene": only on clear failure evidence in these pictures -- an object placed in the wrong\n'
           '  place, a dropped object, the same failed attempt repeated, or no change at all over two consecutive\n'
           '  windows. What you do then depends on the stage (see STAGED REPAIR below). plan is 1 to 8 steps, ' +
           ('of which only the first is executed before you see its result\n  and choose the next (stepwise mode).\n'
            if step else 'executed IN ORDER by the simulator without asking you again.\n') +
           '- verdict "giveup": only when no steps are left to act with, or the objects the task needs have left every\n'
           '  camera and the reach of both arms. plan MUST be empty. This is final for the episode.')
if OLD in s:
    s = s.replace(OLD, NEW)
else:
    sys.stderr.write("rd_verdicts: verdict block not found, spec left unchanged\n")
s = re.sub(r"Three rules the simulator enforces on every plan.*?\n\n", "", s, count=1, flags=re.S)
sys.stdout.write(s)
