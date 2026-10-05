---
id: check-wrist-orientation-before-pixel-aim
scope: global
kind: primitive
title: "Turn the wrist fingertips-down before a pixel grasp when it points sideways"
applies_when: "About to pixel-aim a re-grasp after a miss, and the wrist view or pose shows the gripper pointing sideways instead of down at the table"
symptom: [pixel aim, wrist orientation, sideways, rotate wrist, servo missed, empty close, re-approach]
evidence:
  cells: [scan_object_s0_ep0]
  attempts: 1
  source: judge-learn-rtL1
confidence: single-shot
related: []
---

**Why:** A pixel move sends the fingertip centre above the surface point and assumes the fingers will straddle the object from above. With the gripper axis near horizontal every pixel target left the fingers off the object, and nudges down in the gripper frame moved away from it. Several rewinds failed for the same reason: the approach line was wrong, not the offset.

**How to apply:**
- Before a pixel re-grasp, check that the gripper points down (wrist picture shows the table below the fingers).
- If it points sideways, first rotate the wrist toward fingertips-down, then pixel-aim.
- Or hand back from a corrected pose so the policy re-approaches on its own line.

**Falsify:** If a pixel grasp from a sideways wrist lands and holds, pixel aiming is orientation-robust and this step is unnecessary.
