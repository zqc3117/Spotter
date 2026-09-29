---
id: corrected-approach-then-verified-grasp
scope: global
kind: strategy
title: Combine a corrected approach with an explicit grasp check
applies_when: A free-object grasp missed and the policy withdrew with empty fingers
symptom:
- object
- remains
- source
- closed
- hand
- leaves
evidence:
  cells:
  - PnPStoveToCounter_s195_ep10
  attempts: 1
  source: treatment trajectory and repair checkpoints
confidence: single-shot
related: []
---
**Why:**
A short policy hand-back from a corrected approach can restore alignment without completing the grasp. That intermediate state is useful, but it is not evidence of recovery. In this episode, reopening and returning slightly below the recorded miss let the policy approach again; a small additional descent, close, and lift then established transport. The object remained attached during a subsequent policy chunk, and placement succeeded.

**How to apply:**
After confirming an empty grasp at the source, reopen and retrace to a modestly corrected approach pose. If a short policy hand-back leaves the fingers aligned but the object still supported, finish the small local correction rather than repeatedly handing back or abandoning the repair. Close, lift several centimetres, and compare third-person views. Once the object rises with the fingers, test retention during one policy chunk before declaring the repair complete.

**Falsify:**
If the object stays on its support during the lift or slips during resumed motion, the grasp is not repaired regardless of aperture. If the corrected approach produces contact, drift, or misalignment, do not descend blindly; back off or re-stage and choose another approach.
