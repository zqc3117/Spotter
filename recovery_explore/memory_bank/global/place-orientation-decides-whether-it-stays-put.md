---
id: place-orientation-decides-whether-it-stays-put
scope: global
kind: primitive
title: Straighten the wrist before opening the fingers, or the object rolls off where you put it
applies_when: the object is held and you are about to release it on the destination
symptom: [rolled off, fell over after release, placed flat, not upright]
evidence:
  cells: [x25-PnPStoveToCounter_s195_ep11]
  attempts: 2
  source: judge-rounds
confidence: single-shot
related: [handoff-pose-matters-not-just-position, move-to-and-lift-do-not-restore-orientation]
---
A cylindrical container released on its side rolls, and the episode fails after a correct transport.

**Why:**
The grasp pose sets the release pose. Nothing between them corrects orientation, and neither `move-to` nor `lift` changes it.

**How to apply:**
- `rotate` to bring the object upright before the release, not after.
- The cap is 0.35 rad per call, so plan two calls if the tilt is larger.

**Falsify:**
Objects that are stable in any orientation make this irrelevant.

**Related:** [[handoff-pose-matters-not-just-position]] [[move-to-and-lift-do-not-restore-orientation]]
