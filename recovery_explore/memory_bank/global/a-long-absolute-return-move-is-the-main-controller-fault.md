---
id: a-long-absolute-return-move-is-the-main-controller-fault
scope: global
kind: primitive
title: A long absolute move back to where the grasp was missed is the main source of servo misses and overshoot
applies_when: planning the first step of a repair after the policy has withdrawn from a missed grasp or a control
symptom: [servo_missed, overshoot, move_clipped, long move_to, return to missed grasp, xyz target]
evidence:
  cells: [g120n-CoffeeServeMug_s195_ep15, g120n-PnPCabToCounter_s195_ep10, g120r-PnPMicrowaveToCounter_s195_ep13, g120rpi-OpenDrawer_s195_ep18, g120rpi-PnPCounterToCab_s195_ep12, g120r-TurnOffStove_s195_ep10]
  attempts: 6
  source: campaign-2026-09-16
confidence: verified
related: []
---
A long absolute move back to where the grasp was missed is the main source of servo misses and overshoot.

**Why:**
The policy pulls the hand twenty to forty centimetres back after a miss. One absolute move_to across that distance passes through arm configurations the Cartesian controller handles badly; across two runs without rewind, two thirds of interventions contained at least one such fault, and most of those faults were on the return move.

**How to apply:**
- Prefer `retreat`, which returns to the takeover pose along the path the arm already reached, over an xyz target you compose.
- If you must travel, do it as two or three short moves and look between them.
- After one fault on a return move, do not send a second one from the new place: it fails the same way.

**Falsify:**
A return move that lands within tolerance as often as short nudges do would remove the reason to split it.
