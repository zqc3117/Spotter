---
id: empty-source-corroborates-delivery
scope: global
kind: perception
title: Confirm delivery by checking the source is now empty, not only by looking at the destination
applies_when: deciding whether the object reached the destination container or surface
symptom: [is it in the pan, delivered or not, destination occluded, cannot see the object]
evidence:
  cells: [x25-PnPSinkToCounter_s195_ep0, x25-PnPCounterToSink_s195_ep5]
  attempts: 4
  source: judge-rounds
confidence: verified
related: [target-may-already-be-delivered]
---
The destination is often partly occluded by the arm. The source is usually not, and an object that has left it has to be somewhere.

**Why:**
Two weak views that agree beat one view you cannot read. The source view also catches the case where the object was knocked out of the source without ever being carried.

**How to apply:**
- Compare the source region against the first window of the episode, not against intuition.
- If the source is empty and the destination is unreadable, prefer ok and look again next window.

**Falsify:**
Scenes where the source region leaves the camera's view invalidate the comparison.

**Related:** [[target-may-already-be-delivered]]
