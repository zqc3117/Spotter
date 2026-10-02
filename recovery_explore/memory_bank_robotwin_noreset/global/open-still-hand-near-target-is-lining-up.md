---
id: open-still-hand-near-target-is-lining-up
scope: global
kind: strategy
title: "A hand hovering open and still near its target is lining up, not stalled"
applies_when: "The working hand is open, has stopped or nearly stopped moving near or above its target, the object has not been disturbed and there has been no close on this target yet"
symptom: [open gripper, parked hand, stationary, hovering, no travel, approach phase, stall, lining up, two-hand lift]
evidence:
  cells: [move_stapler_pad_s0_ep0, place_can_basket_s0_ep0, place_shoe_s0_ep0]
  attempts: 3
  source: judge-learn-rtL1
confidence: verified
related: []
---

**Why:** Several interventions treated an open, still hand as a stalled or missed grasp and pushed it with Cartesian moves; those moves stalled or fell short from the pose, and the episodes that the policy was on course to finish ended in failure. An open hand that has not closed yet has not missed anything. On a two-hand lift, the second hand often waits open while the first settles its grip.

**How to apply:**
- An open, still hand with no close yet and nothing knocked over is not a fault: answer ok.
- Intervene only on a named fault: an empty closure on the grasping arm, an object knocked or dropped, an arm jammed with large joint error.
- If a two-hand lift has one hand holding and the other open and still for many windows late in the episode, the smallest repair is a 1-2 cm nudge of the open hand toward its grip and a hand-back of 2 chunks. Never a large move or a manual close as the first repair.

**Falsify:** If open, still hands left alone repeatedly run out the step budget without ever closing, waiting is wrong and a small early nudge is the better default.
