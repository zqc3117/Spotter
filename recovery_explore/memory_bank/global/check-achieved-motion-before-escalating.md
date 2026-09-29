---
id: check-achieved-motion-before-escalating
scope: global
kind: strategy
title: When a step reports servo_missed, the plan did not happen; do not escalate the offset
applies_when: a plan segment aborted with servo_missed, or the executed log shows far less motion than commanded
symptom: [servo_missed, did not reach target, moved less, residual error, descend, offset, repeat, same correction, escalate]
evidence:
  cells: [CoffeeSetupMug_s195_ep19, OpenDrawer_s195_ep18, CoffeeServeMug_s195_ep12]
  attempts: 3
  source: mech16-round
confidence: probable
related: [one-change-per-attempt, move-to-can-rotate-and-stall-in-tight-space]
---
A missed servo and a wrong target look identical in the outcome — nothing got
fixed — but they need opposite responses. Escalating the offset after a missed
servo makes the next attempt worse.

**Why:**
In `CoffeeSetupMug_s195_ep19` the repair commanded a 3 cm descent; the
end-effector descended **0.9 cm** and the step reported it did not reach the
target. The judge read the resulting empty closure as "I did not go deep enough"
and commanded a larger descent, which also did not execute. In
`OpenDrawer_s195_ep18` a direct corrected-target move missed by **23.25 cm**;
routing through a visited waypoint reached that waypoint within 0.93 cm, and only
then did small corrections land near the handle. Proximity to a visited point does
not by itself guarantee the servo can get there.

**How to apply:**
- Read `moved_cm` and `servo_ok` in the executed log before interpreting the
  result. If the motion did not happen, your hypothesis was never tested.
- After `servo_missed`, re-establish tracking first: move to a nearby point the
  arm has actually reached, confirm the endpoint, then make small corrections.
- Do not repeat the same descend-and-close with a bigger number, and do not call
  the repair fixed without evidence the object or mechanism moved.
- Zero contact force does not mean the motion succeeded; it usually means you
  never reached anything.

**Falsify:**
If a task shows servo misses that resolve themselves on retry without any change
of approach, the re-establish step is wasted budget; measure retry success before
spending actions on it.

**Related:** [[one-change-per-attempt]] [[move-to-can-rotate-and-stall-in-tight-space]]
