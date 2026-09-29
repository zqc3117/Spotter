---
id: closing-after-a-descent-that-stopped-short-closes-above-the-object
scope: global
kind: primitive
title: A close issued right after a descent that stopped short lands above the object
applies_when: a plan has a lift or nudge downward followed by a gripper close, and the descent step reported servo_missed or travelled less than asked
symptom: [descent stopped short, closed_empty, servo_missed, close above object, lift -0.05, nudge down]
evidence:
  cells: [g120r-PnPCounterToMicrowave_s195_ep16, g120rpi-PnPCounterToMicrowave_s195_ep14, g120rpi-PnPMicrowaveToCounter_s195_ep18, g120r-PnPMicrowaveToCounter_s195_ep19]
  attempts: 4
  source: campaign-2026-09-16
confidence: verified
related: []
---
A close issued right after a descent that stopped short lands above the object.

**Why:**
The close does not know the descent failed; it shuts at whatever height the servo gave up, which is above the graspable body. The stop reason is reported before the close runs only when the plan is split there.

**How to apply:**
- Put the close in a later segment than the descent, so a short descent stops the plan before the fingers shut.
- When a descent reports short travel, lower with one absolute waypoint rather than accumulating nudges, or let the policy do the descent.

**Falsify:**
Closes after short descents that nevertheless caught the object.
