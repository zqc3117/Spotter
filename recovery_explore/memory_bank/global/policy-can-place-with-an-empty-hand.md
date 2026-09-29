---
id: policy-can-place-with-an-empty-hand
scope: global
kind: failure
title: From a mid-grasp state the policy may close on nothing and head for the target anyway
applies_when: the policy is resumed from a state where it has already begun a grasp
symptom: [empty hand, goes to place without the object, does not retry the grasp, skips to place]
evidence:
  cells: [x25-PnPCounterToSink_s195_ep0, x25-PnPStoveToCounter_s195_ep2, jd-PnPCounterToStove_s195_ep0]
  attempts: 6
  source: judge-rounds
confidence: verified
related: [closed-gripper-is-not-a-grasp, approach-with-a-shut-empty-gripper]
---
The policy does not always return to the object after a miss. It closes the empty gripper and translates toward the task's destination as if carrying something, and it will not go back on its own.

**Why:**
The state after a missed grasp looks, to the policy, like the state after a successful one: gripper shut, arm above the object. Nothing in its input distinguishes them.

**How to apply:**
- An empty-closed gripper travelling toward the destination is a failure in progress, not transport. Intervene.
- Open the gripper first, then bring the end-effector back over the object.
- Do not wait for the policy to notice; it will not.

**Falsify:**
A policy given proprioceptive grasp feedback should recover on its own, which would retire this entry for that policy.

**Related:** [[closed-gripper-is-not-a-grasp]] [[approach-with-a-shut-empty-gripper]]
