---
id: grasped-and-hovering-means-do-not-intervene
scope: global
kind: strategy
title: "Object held and end-effector already over the destination: let the policy finish"
applies_when: the object is visibly in the fingers and the arm is above the place named in the instruction
symptom: [already holding, above target, should I intervene, almost done]
evidence:
  cells: [x25-PnPStoveToCounter_s195_ep3, x25-PnPSinkToCounter_s195_ep2, rd-PnPCounterToMicrowave_s195_ep6]
  attempts: 3
  source: judge-rounds
confidence: verified
related: [place-orientation-decides-whether-it-stays-put]
---
When the object is squarely in the fingers and the end-effector is already over the destination surface, the remaining motion is the part the policy does well. Intervening here costs budget and risks the grasp.

**Why:**
Release and descent over a clear surface is short, and every action you spend moves the arm into a pose the policy did not choose.

**How to apply:**
- Verdict ok, and say why, so the next window can tell progress from stalling.
- The one exception is orientation: if the object is held in a pose that will not stand once released, straighten the wrist first, see [[place-orientation-decides-whether-it-stays-put]].

**Falsify:**
If trajectories in this state routinely fail at release, the release itself needs help and this entry is wrong.

**Related:** [[place-orientation-decides-whether-it-stays-put]]
