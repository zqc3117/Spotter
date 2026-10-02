---
id: verify-pixel-landing-before-close
scope: global
kind: primitive
title: "Check where a pixel move landed before closing or handing back"
applies_when: "A pixel move_to was used to place the hand for a grasp, especially next to a larger object or on a narrow part such as a handle"
symptom: [pixel aim, landed beside, wrong surface, handle, small object, verify, fingertips now, empty close]
evidence:
  cells: [put_bottles_dustbin_s0_ep0, lift_pot_s0_ep0]
  attempts: 2
  source: judge-learn-rtL1
confidence: probable
related: []
---

**Why:** The pixel lookup can pick the surface of a neighbouring larger object or the table next to a narrow part. A close from there came up empty, repeatedly. On a two-hand lift by narrow handles, pixel aims landed beside the handle while the policy's own approach had been lined up correctly.

**How to apply:**
- After a pixel move, compare the reported 3D point and the wrist picture with where the target really is before closing.
- If it landed on a neighbour or the table, correct with a small relative nudge from there or re-aim at another pixel of the target.
- For narrow parts (handles, rims) where the policy's own approach already looked right, prefer restoring that approach (small nudge + hand-back) over a pixel aim.

**Falsify:** If pixel moves next to larger objects land on target and close successfully, the extra check is unnecessary.
