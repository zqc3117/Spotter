---
id: destination-proximity-is-not-placement-proof
scope: global
kind: perception
title: Destination proximity is not proof of completed placement
applies_when: An object is visible near its destination while the arm withdraws or
  stalls elsewhere
symptom:
- repeatedly
- declaring
- completion
- unchanged
- object
- destination
- without
- verifying
- orientation
- alignment
evidence:
  cells:
  - CoffeeSetupMug_s195_ep10
  attempts: 1
  source: Trajectory observations and final unsuccessful episode outcome
confidence: single-shot
related: []
---
**Why:**
I repeatedly treated an object resting in the destination region as evidence that placement was complete and used that interpretation to dismiss a sustained arm stall. The episode ultimately failed. This does not establish the exact placement defect, but it shows that the visual completion judgment was insufficiently supported. Repeating the same ambiguous view does not strengthen it.

**How to apply:**
Separate destination proximity from the requested final relationship. Check the object's orientation, support, and alignment with the relevant destination feature using both third-person views. An object can be stable and near the correct destination without being correctly placed.

When the arm stalls persistently after apparent placement, revisit the completion judgment rather than automatically carrying it forward. Intervene only if a specific visible defect supports a concrete repair; uncertainty alone does not justify disturbing the object.

**Falsify:**
If clear views establish the required orientation, support, and alignment, and independent evidence attributes failure to an unrelated requirement, placement ambiguity was not the cause. A verified completed placement should still be left undisturbed despite an unrelated arm stall.
