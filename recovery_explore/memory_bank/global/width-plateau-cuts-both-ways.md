---
id: width-plateau-cuts-both-ways
scope: global
kind: perception
title: A steady mid-range aperture under a close command usually means a grasp, but it survives a mid-transport drop
applies_when: using gripper aperture as evidence for or against a grasp
symptom: [width plateau, aperture unchanged, thought it was holding, dropped but width stayed, false positive]
evidence:
  cells: [x25-PnPSinkToCounter_s195_ep0, x25-PnPStoveToCounter_s195_ep1, x25-PnPCabToCounter_s195_ep0]
  attempts: 9
  source: judge-rounds
confidence: verified
related: [closed-gripper-is-not-a-grasp]
---
Aperture resting well away from both fully open and fully shut, and staying there across large end-effector motion, is supporting evidence for a grasp. It is not proof: the same plateau persists after the object has already slipped out mid-transport, sometimes for tens of centimetres of travel.

**Why:**
The fingers hold whatever position the controller drove them to. Nothing pushes them closed again when the object leaves, so the plateau is a property of the controller, not of the contents. Those drops are the exception: over earlier pick-and-place runs, a close command with the width holding between 25 and 50 mm while the arm moved was a held object 406 times in 445.

**How to apply:**
- Under a close command, read the plateau as a grasp unless the images show the object left behind or fallen; under an open command it is not evidence of a hold.
- Re-check the images after any large transport move; do not assume the plateau carried the grasp with it.
- An aperture near zero is not automatically an empty close either: a thin object can read close to zero.

**Falsify:**
A gripper with force feedback or a compliant closure loop would make the plateau informative on its own, and this entry would no longer apply.

**Related:** [[closed-gripper-is-not-a-grasp]]
