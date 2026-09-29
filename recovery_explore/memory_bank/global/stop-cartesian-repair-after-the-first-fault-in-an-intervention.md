---
id: stop-cartesian-repair-after-the-first-fault-in-an-intervention
scope: global
kind: failure
title: After the first servo miss or overshoot in an intervention, further Cartesian steps from that region fail the same way
applies_when: a step has just reported servo_missed, overshoot or move_clipped and you are deciding the next segment
symptom: [servo_missed, overshoot, second attempt, repeated fault, drift]
evidence:
  cells: [g120r-CoffeeServeMug_s195_ep15, g120r-TurnOffStove_s195_ep12, g120rpi-CoffeeSetupMug_s195_ep15, g120rpi-PnPCounterToCab_s195_ep12, g120r-OpenDrawer_s195_ep19]
  attempts: 5
  source: campaign-2026-09-16
confidence: verified
related: []
---
After the first servo miss or overshoot in an intervention, further Cartesian steps from that region fail the same way.

**Why:**
In the rewound runs the second and third attempts after a fault repeated the fault with two- and three-centimetre variations; none of them converged, and the intervention ended by handing back the untouched state. Without a rewind the same attempts leave the hand a further step away each time.

**How to apply:**
- Treat the first fault as the end of Cartesian repair from that place: retreat, then hand back, or hand back with the fingers open if retreat itself faults.
- Do not spend the remaining action budget on offsets of the same move.

**Falsify:**
An intervention where a third small correction after two faults produced a verified grasp.
