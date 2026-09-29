---
id: topple-check-before-horizontal-recentering
scope: global
kind: primitive
title: Raise before recentering horizontally over an upright object
applies_when: the fingers are open and hovering at object height and you want to shift the arm sideways
symptom: [recentering knocked it over, upright object fell, cup tipped while aligning]
evidence:
  cells: [x25-PnPCabToCounter_s195_ep0]
  attempts: 1
  source: judge-rounds
confidence: single-shot
related: [do-not-sweep-in-sideways]
---
A horizontal `move-to` at object height drags an open gripper through the object.

**Why:**
Open fingers are wider than the object and lower than its top; the horizontal path intersects it.

**How to apply:**
- One `lift` of three to five centimetres first, then the horizontal correction, then descend.
- This costs one extra action and prevents the failure that ends the episode.

**Falsify:**
Objects much shorter than the finger opening are not at risk.

**Related:** [[do-not-sweep-in-sideways]]
