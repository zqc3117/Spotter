---
id: fix-at-the-miss-not-where-the-arm-ended-up
scope: global
kind: strategy
title: Repair the moment of the missed grasp, not the retreated pose the policy left you in
applies_when: the telemetry shows an empty close and the arm has since moved away
symptom: [arm retreated, too far to fix, budget runs out, cannot reach the object]
evidence:
  cells: [jd-PnPCounterToStove_s195_ep0, jd-PnPCounterToStove_s195_ep1, jd-PnPCabToCounter_s195_ep0]
  attempts: 12
  source: judge-rounds
confidence: probable
related: [do-not-sweep-in-sideways, one-change-per-attempt]
---
After a missed grasp the policy withdraws, often twenty to fifty centimetres. Correcting from there spends the whole budget on travel and arrives short.

**Why:**
The error to correct is two or three centimetres at the moment the fingers shut. Every centimetre of travel afterwards is unrelated to that error.

**How to apply:**
- Read the recorded empty-close position from `grasp_attempts` in `state`.
- Write the correction into the target: aim three centimetres above that point, or two to the side, rather than aiming at the point itself.
- A `move-to` whose target is within eight centimetres of a visited point may travel up to sixty centimetres in one action; elsewhere it is clipped to five.
- Approach from above the object, never sliding along the surface, see [[do-not-sweep-in-sideways]].

**Falsify:**
If corrections launched from the retreated pose succeed as often, the anchor buys nothing.

**Related:** [[do-not-sweep-in-sideways]] [[one-change-per-attempt]]
