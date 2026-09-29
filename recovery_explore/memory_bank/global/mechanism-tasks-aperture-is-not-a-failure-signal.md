---
id: mechanism-tasks-aperture-is-not-a-failure-signal
scope: global
kind: perception
title: On doors, drawers, knobs, faucets and buttons, an empty closure is not a failure
applies_when: the instruction is to open or close a door or drawer, turn a knob, spout or faucet, or press a button, rather than to move a free object
symptom: [door, drawer, knob, faucet, spout, button, handle, mechanism, open, close, turn, press, empty close, closed on nothing, aperture, gripper open]
evidence:
  cells: [OpenDoubleDoor_s195_ep14, CloseDrawer_s195_ep11, CloseDrawer_s195_ep15, CloseSingleDoor_s195_ep19, TurnOnMicrowave_s195_ep11, TurnOffMicrowave_s195_ep11, TurnSinkSpout_s195_ep14, TurnOnSinkFaucet_s195_ep19, CoffeePressButton_s195_ep15]
  attempts: 9
  source: mech16-round
confidence: verified
related: [closed-gripper-is-not-a-grasp, verify-the-mechanism-not-the-arm]
---
These tasks never require the hand to keep hold of anything. The door stays open
after you let go; the knob stays turned; the button stays pressed. So "the fingers
closed on nothing" and "the gripper is open" carry no bad news at all here.

**Why:**
Nine episodes across six mechanism tasks logged an empty closure or a wide-open
gripper and **still succeeded on the control arm**. In `OpenDoubleDoor_s195_ep14`
telemetry recorded an empty closure at step 183 with a 10.4 mm aperture while the
door was in fact swinging farther open, and both arms finished the task. The
signal that made this a reliable failure indicator for pick-and-place — the hand
must retain the object all the way to the destination — simply does not apply.

**How to apply:**
- Use the aperture for one thing only: to know that the fingers are not holding
  anything *right now*. Never to conclude the task is failing.
- Decide from the mechanism's own state across adjacent third-person frames: has
  the door swung, has the drawer come out, has the knob rotated, is water running.
- Releasing and repositioning is normal technique on these tasks, not a dropped
  object. An arm that lets go and moves to the next phase is making progress.
- `closed_empty` as a plan abort condition still means your own repair step
  grabbed nothing; that is about your action, not about the policy's trajectory.

**Falsify:**
A mechanism task where the handle must be held continuously to the end — a spring
door that closes the moment you release it — would put this outside its scope.
Check whether the mechanism keeps its state after release before relying on it.

**Related:** [[closed-gripper-is-not-a-grasp]] [[verify-the-mechanism-not-the-arm]]
