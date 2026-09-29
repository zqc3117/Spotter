---
id: placement-contact-can-resemble-a-stall
scope: global
kind: strategy
title: Placement contact can resemble a stall
applies_when: A carried object has reached the destination and the gripper is beginning
  to open.
symptom:
- downward
- hand
- travel
- nearly
- stops
- despite
- substantial
- command
evidence:
  cells:
  - PnPCounterToSink_s195_ep10
  attempts: 1
  source: Observed trajectory and final episode outcome.
confidence: single-shot
related: []
---
**Why:**
A commanded-versus-travelled mismatch can reflect contact at placement depth rather than a failure requiring repair. In this episode, visible transport and descent were followed by greatly reduced downward travel while the fingers began opening. Leaving the policy uninterrupted resulted in success.

**How to apply:**
Interpret reduced travel alongside the task phase. If the object has visibly travelled with the hand to the destination and the aperture is increasing, allow the release sequence to continue rather than immediately diagnosing a stall. Check subsequent frames for stable support and hand withdrawal; opening alone does not establish completion.

**Falsify:**
This interpretation does not apply if the object remains at the source, falls outside the destination, or visibly tips during release. Persistent immobility without release progress, or an open hand parked over the placed object across windows, warrants reassessment and possibly intervention.
