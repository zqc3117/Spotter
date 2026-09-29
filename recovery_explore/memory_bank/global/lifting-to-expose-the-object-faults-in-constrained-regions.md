---
id: lifting-to-expose-the-object-faults-in-constrained-regions
scope: global
kind: primitive
title: A small lift to expose an occluded object faults in constrained regions instead of revealing anything
applies_when: the object is hidden by the hand or a fixture and you are tempted to raise or back off a few centimetres to see it
symptom: [occluded, expose, lift 3 cm, back off, servo_missed, overshoot, cabinet, sink, keypad, dispenser]
evidence:
  cells: [g120r-TurnOffSinkFaucet_s195_ep11, g120rpi-TurnOffSinkFaucet_s195_ep16, g120rpi-TurnSinkSpout_s195_ep16, g120r-TurnOnSinkFaucet_s195_ep16, g120rpi-TurnOffStove_s195_ep10, g120r-PnPCabToCounter_s195_ep19]
  attempts: 6
  source: campaign-2026-09-16
confidence: verified
related: []
---
A small lift to expose an occluded object faults in constrained regions instead of revealing anything.

**Why:**
Inside a cabinet, at the rear edge of a sink, against a keypad or over a dispenser tray the arm is near joint limits or structure; a three-centimetre lift from there often travels sideways, misses, or blows up, and the view afterwards is no better.

**How to apply:**
- Do not spend a move to see. Decide from the rows you have; when no row shows the state, say so and pass.
- Reopening the fingers costs no travel and is the only harmless way to change the picture from a constrained pose.

**Falsify:**
A run where exposure lifts from these regions reached their target and the next frame showed the object.
