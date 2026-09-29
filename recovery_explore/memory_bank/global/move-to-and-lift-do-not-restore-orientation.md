---
id: move-to-and-lift-do-not-restore-orientation
scope: global
kind: infra
title: move-to and lift control position only; whatever rotation you accumulate stays
applies_when: planning a repair that involves several position commands
symptom: [wrist drifted, orientation changed, pose looks wrong after moves]
evidence:
  cells: [x25-PnPStoveToCounter_s195_ep6]
  attempts: 1
  source: judge-rounds
confidence: single-shot
related: [move-to-can-rotate-and-stall-in-tight-space, place-orientation-decides-whether-it-stays-put]
---
Neither primitive pulls the wrist back to any reference pose. Rotation accumulated during a stall stays until `rotate` removes it.

**Why:**
They command a position target and leave orientation to the controller's null-space.

**How to apply:**
- Read `eef_quat` in `state` if a sequence of moves behaved oddly.
- Correct rotation explicitly with `rotate`; do not expect a later `move-to` to tidy it.

**Falsify:**
An orientation-locked controller would make this moot.

**Related:** [[move-to-can-rotate-and-stall-in-tight-space]] [[place-orientation-decides-whether-it-stays-put]]
