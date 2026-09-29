---
id: do-not-sweep-in-sideways
scope: global
kind: primitive
title: Descend onto the object from above; sliding in laterally topples whatever is next to it
applies_when: closing the last few centimetres toward an object that has neighbours or sits in a container
symptom: [knocked it over, toppled the cup, hit the neighbour, object tipped, grasped the side]
evidence:
  cells: [jd-PnPCabToCounter_s195_ep0, x25-PnPCabToCounter_s195_ep4]
  attempts: 3
  source: judge-rounds
confidence: probable
related: [fix-at-the-miss-not-where-the-arm-ended-up, topple-check-before-horizontal-recentering]
---
A lateral translation into the grasp pose pushes the object or its neighbour over before the fingers ever close, and what follows is a pinch on a tilted wall that fails at the first lift.

**Why:**
At grasp range the fingers occupy most of the free space around the object. Any horizontal motion at that height is a collision.

**How to apply:**
- Stop above and slightly short of the object, then descend.
- Sit back a little rather than reaching past the object's centre; let the fingers straddle it roughly parallel instead of poking it with the tips.
- Check `collateral` in every action's result; `toppled: true` means undo the situation with `reset` rather than continuing.

**Falsify:**
An isolated object on an empty surface has room for a lateral approach, so the rule matters only where clearance is tight.

**Related:** [[fix-at-the-miss-not-where-the-arm-ended-up]] [[topple-check-before-horizontal-recentering]]
