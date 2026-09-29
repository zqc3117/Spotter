---
id: preserve-progress-after-recovery
scope: global
kind: strategy
title: Preserve policy progress after a recovery
applies_when: A previous grasp failure has been repaired and the policy is transporting
  or placing the object.
symptom:
- historical
- empty
- closures
- smaller
- travelled-than-commanded
- distances
- can
- suggest
- failure
- despite
- current
- visual
evidence:
  cells:
  - PnPSinkToCounter_s195_ep10
  attempts: 1
  source: Post-repair observation windows and episode outcome.
confidence: single-shot
related: []
---
**Why:**
A past missed grasp does not describe the current grasp. In this episode, later third-person frames showed the object travelling with the hand toward the destination, followed by release and initial withdrawal. Leaving those windows uninterrupted preserved a successful outcome despite historical empty closures and substantial differences between commanded and travelled displacement.

**How to apply:**
After recovery, reassess the current object-hand relationship. If the object translates with the fingers toward the destination, allow transport to continue. During placement, look for the object settling on the destination while the fingers open and the hand begins leaving. Do not diagnose a stall solely from a command-versus-travel discrepancy when both the hand and object are making visible progress.

**Falsify:**
This rule no longer supports waiting if the object stays at the source while the hand departs, slips during transport, or stops advancing toward the destination across windows. After release, renewed intervention may be warranted if the object tips or the hand remains parked over it across repeated windows.
