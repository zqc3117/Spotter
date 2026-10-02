---
id: closed-gripper-is-not-a-grasp
scope: global
kind: perception
title: A closed gripper is not a grasp; only the object leaving its support proves one
applies_when: any time you decide whether a grasp attempt succeeded
symptom: [false grasp, gripper closed, looks grasped, object did not move, drops after lift, empty close]
evidence:
  cells: [b10-PnPStoveToCounter_ep0, b10-PnPStoveToCounter_ep6, jd-PnPCabToCounter_s195_ep0]
  attempts: 8
  source: judge-rounds
confidence: verified
related: [width-plateau-cuts-both-ways, grasp-can-survive-verification-and-still-drop-at-handoff]
---
A close command always executes, whether or not anything is between the fingers. Closing on nothing, pinching an edge, and toppling the object then closing all look identical in the gripper signal alone.

**Why:**
The only information that separates them is relative motion: does the object travel with the gripper. That requires the arm to move after closing, so a verdict taken at the moment of closure is always premature. Calling a closure a success leaves every later stage operating on an empty hand.

**How to apply:**
- Judge from the two third-person cameras, never the wrist camera alone. A grasp is shown by the object leaving its support with the fingers; a miss by a new closed-all-the-way entry, or by the object still at its source, or fallen, while the hand is somewhere else. Until you can point to one of these, the window is undecided, not a miss.
- Aperture below 12 mm with a new closed-all-the-way entry means the fingers are shut on nothing; trust that as a negative. A close command whose fingers stopped between about 12 and 50 mm and stayed there, with no new entry, is how a held object usually reads, see [[width-plateau-cuts-both-ways]].
- The object must keep its position relative to the fingers across several frames; a lift of a few centimetres is hard to see in the overview rows.
- Under an open command a mid-range aperture is not evidence of a hold.

**Falsify:**
A task where the object is fixed in place and is never meant to leave its support (pushing, opening a door) would put this rule outside its scope.

**Related:** [[width-plateau-cuts-both-ways]] [[grasp-can-survive-verification-and-still-drop-at-handoff]]
