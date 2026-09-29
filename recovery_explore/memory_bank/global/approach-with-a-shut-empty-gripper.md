---
id: approach-with-a-shut-empty-gripper
scope: global
kind: failure
title: Some policies drive the gripper shut from the first step and approach the object with a closed empty hand
applies_when: reading the action history at the start of an episode
symptom: [gripper closed from step 0, approaches with fingers shut, never opens, cannot grasp]
evidence:
  cells: [jd-PnPCounterToStove_s195_ep0]
  attempts: 1
  source: judge-rounds
confidence: single-shot
related: [policy-can-place-with-an-empty-hand]
---
In the action history the gripper command sits at close for every chunk from the first, and the aperture drops below 12 mm within a few steps and stays there while the arm travels toward the object.

**Why:**
A hand that is already shut cannot take the object however well it is aimed, so the whole approach is wasted. This is visible in the telemetry long before it is visible in the images.

**How to apply:**
- Read the gripper command column in the action history at the first window.
- If it is at close while the arm is still far from the object, one `gripper open` is the whole fix.
- Do not rebuild the approach; only open the hand and let the policy continue.

**Falsify:**
If opening the gripper does not change the outcome across several such trajectories, the closed hand is a symptom rather than the cause.

**Related:** [[policy-can-place-with-an-empty-hand]]
