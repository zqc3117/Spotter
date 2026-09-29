---
id: wrist-camera-cannot-settle-a-grasp-question
scope: global
kind: perception
title: The wrist view is too close and too mobile to answer whether the object is held
applies_when: the third-person views are occluded and you are tempted to judge from the wrist camera
symptom: [wrist view looks right, object fills the wrist frame, cannot see from outside, parallax]
evidence:
  cells: [x25-PnPStoveToCounter_s195_ep12, x25-PnPSinkToCounter_s195_ep12, x25-PnPCounterToSink_s195_ep1]
  attempts: 6
  source: judge-rounds
confidence: probable
related: [closed-gripper-is-not-a-grasp]
---
An object resting on the surface and an object in the fingers look alike from the wrist. The camera rotates with the wrist, so apparent motion there carries almost no information about world motion, and a gap under the object after a lift is as likely to be parallax as clearance.

**Why:**
Near-field views with a moving frame confound object motion, camera motion and perspective.

**How to apply:**
- Decide grasp questions from the two third-person cameras.
- Use the wrist view for alignment at close range, which is what it is good for.
- The green line with the red dot in these frames is a marker fixed to the gripper, not an object.

**Falsify:**
A wrist camera with a wider field of view could support the judgement.

**Related:** [[closed-gripper-is-not-a-grasp]]
