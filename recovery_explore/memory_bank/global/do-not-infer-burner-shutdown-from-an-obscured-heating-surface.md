---
id: do-not-infer-burner-shutdown-from-an-obscured-heating-surface
scope: global
kind: perception
title: Do not infer burner shutdown from an obscured heating surface
applies_when: The target burner is covered by cookware and the arm has withdrawn from
  the stove controls.
symptom:
- glow
- visible
- around
- target
- pan
- supervisor
- assumes
- burner
- false
- without
- confirming
- control
evidence:
  cells:
  - TurnOffStove_s195_ep18
  attempts:
  - Windows 14 and 15 passed based on apparent absence of target glow and arm withdrawal.
  source: Trajectory images, action history, and final outcome reporting treatment
    success False.
confidence: single-shot
related: []
---
**Why:**
In windows 14 and 15, a pan covered the rear-left burner while the rear-right burner visibly glowed red. The supervisor correctly distinguished the neighboring burner but treated the absence of visible target glow as evidence of shutdown. The treatment ultimately failed. This outcome does not establish the exact mechanical failure, but it shows that the completion inference was not reliable.

**How to apply:**
Treat a cookware-obscured burner as having uncertain state unless its own control or another target-specific indicator confirms shutdown. Inspect the wrist view for the target knob's index and compare it across frames while the hand is at the control. Withdrawal and empty fingers do not establish completion. Preserve uncertainty when the control cannot be read; intervene only when trajectory evidence establishes a correctable failure and supports a small, targeted repair.

**Falsify:**
A clearly readable target knob in its off position, a target-specific off indicator, or a directly observed shutdown transition would support completion despite cookware obscuring the burner. If such evidence was present earlier in this episode, the proposed perception error would need reassessment.
