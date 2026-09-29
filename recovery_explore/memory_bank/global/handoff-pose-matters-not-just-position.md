---
id: handoff-pose-matters-not-just-position
scope: global
kind: strategy
title: A wrist left rotated at handoff can derail the policy's transport, not only drop the object
applies_when: your repair rotated the end-effector and you are about to hand control back
symptom: [policy went sideways after handoff, transport derailed, odd trajectory after intervention]
evidence:
  cells: [x25-PnPCounterToSink_s195_ep1]
  attempts: 2
  source: judge-rounds
confidence: single-shot
related: [grasp-can-survive-verification-and-still-drop-at-handoff, place-orientation-decides-whether-it-stays-put]
---
Handing back with a visibly rotated wrist produces transport that wanders, even when the grasp itself holds.

**Why:**
Orientation is part of the observation the policy conditions on. A pose outside its training distribution produces actions outside its competence.

**How to apply:**
- Spend one `rotate` to bring the wrist back toward the pose the policy had before you started, unless the task needs the new orientation.

**Falsify:**
If handing back rotated makes no measurable difference, drop this.

**Related:** [[grasp-can-survive-verification-and-still-drop-at-handoff]] [[place-orientation-decides-whether-it-stays-put]]
