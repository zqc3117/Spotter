---
id: target-may-already-be-delivered
scope: global
kind: strategy
title: Check whether the object is already on the destination before treating a still scene as a stall
applies_when: the object has not moved for several windows and the arm is wandering
symptom: [object not moving, looks stalled, is it done, already delivered]
evidence:
  cells: [x25-PnPSinkToCounter_s195_ep4, x25-PnPStoveToCounter_s195_ep9]
  attempts: 6
  source: judge-rounds
confidence: verified
related: [empty-source-corroborates-delivery]
---
A static object plus a moving arm reads the same whether the task is finished or the grasp never happened. Which one it is depends on where the object is, not on what the arm is doing.

**Why:**
The instruction names a source and a destination. The object sitting at the destination is success; the object sitting at the source is failure. Treating the first as a stall throws away a finished episode.

**How to apply:**
- Locate the object against the instruction's two named places before judging.
- Corroborate with the source: an emptied source supports delivery, see [[empty-source-corroborates-delivery]].

**Falsify:**
An instruction with an ambiguous destination would break this test.

**Related:** [[empty-source-corroborates-delivery]]
