---
id: dropped-out-of-reach-is-terminal
scope: global
kind: failure
title: An object that has fallen off the work surface cannot be recovered from proprioception and images
applies_when: the object has disappeared from all three cameras after a drop
symptom: [object gone, cannot find it, fell on the floor, disappeared from view]
evidence:
  cells: [x25-PnPSinkToCounter_s195_ep10, x25-PnPCounterToSink_s195_ep7, jd-PnPCabToCounter_s195_ep0]
  attempts: 4
  source: judge-rounds
confidence: probable
related: [three-strikes-then-hand-back]
---
Sweeping the arm back along its own path to hunt for a dropped object consumes the budget and does not find it. Before concluding it is gone, take one wide view: it may have fallen clear of the counter entirely.

**Why:**
Search needs a position estimate, and a dropped object provides none. Without an oracle there is nothing to aim at.

**How to apply:**
- One `render` at the current pose to confirm it is not simply occluded.
- If it is gone, `RESULT: giveup` immediately rather than spending the budget.

**Falsify:**
A wider camera or a search primitive would make recovery feasible and retire this entry.

**Related:** [[three-strikes-then-hand-back]]
