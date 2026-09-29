---
id: keypad-and-dispenser-poses-blow-up-cartesian-corrections
scope: global
kind: primitive
title: In front of a microwave keypad, under a coffee dispenser and at a stove knob in contact, a two-centimetre correction can travel a metre
applies_when: planning any move_to, nudge or lift while the hand is at a keypad, under a dispenser, or seated on a stove knob
symptom: [overshoot, blow-up, keypad, microwave, dispenser, coffee machine, stove knob, contact pose, travelled 100 cm, controller misbehaving]
evidence:
  cells: [g120r-TurnOnMicrowave_s195_ep17, g120r-TurnOnMicrowave_s195_ep15, g120r-CoffeeServeMug_s195_ep15, g120rpi-CoffeeServeMug_s195_ep15, g120rpi-TurnOffMicrowave_s195_ep16, g120r-TurnOnMicrowave_s195_ep19, l30b-TurnOffStove_s195_ep14, l30a-PnPCounterToMicrowave_s195_ep13]
  attempts: 8
  source: campaign-2026-09-16
confidence: verified
related: []
---
In front of a microwave keypad and under a coffee dispenser, a two-centimetre correction can travel a metre.

**Why:**
The arm reaches these controls near its extension limit and the Cartesian controller is unstable there; overshoots of 77 to 116 centimetres came from three-centimetre requests across both policy families; a three-centimetre clearance lift from a stove knob in contact travelled 99 centimetres, and a retreat from beside the stove overshot the same way.

**How to apply:**
- Prefer passing the window; these tasks mostly complete on their own.
- If you act, change only the fingers or hand back from the current pose; do not restage with a move.

**Falsify:**
Repeated small corrections at these poses landing within tolerance.
