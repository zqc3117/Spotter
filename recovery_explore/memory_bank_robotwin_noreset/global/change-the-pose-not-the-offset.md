---
id: change-the-pose-not-the-offset
scope: global
kind: strategy
title: "After several failed grasps of the same object, change the pose: wider fingers, other side, different wrist angle"
applies_when: "The same object has resisted more than two grasp attempts and each retry looked like the last one with a small offset"
symptom: [repeated attempts, same pose, small block, never lifted, wider opening, rotate wrist, other side, change approach]
evidence:
  cells: [stack_blocks_three_s0_ep0]
  attempts: 1
  source: operator-review
confidence: single-shot
related: [escalate-to-own-grasp-after-repeated-empty-closures]
---

**Why:** A reviewer watching many attempts on one small block asked why nothing about the attempt
ever changed: each retry approached from the same side, with the same finger opening and the same
wrist angle, and missed the same way.

**How to apply:**
- Count attempts on the object across the episode, not within one intervention.
- From the third attempt on, change a pose variable, not a distance: open the fingers wider before
  descending, come at the object from another side, or turn the wrist so the fingers straddle its
  narrow axis.
- Say in the diagnosis which pose variable you changed and why the previous one could not work.

**Falsify:** If changed poses keep missing while the object is plainly within reach, the fault is
in the aim rather than the pose, and re-aiming is the repair.
