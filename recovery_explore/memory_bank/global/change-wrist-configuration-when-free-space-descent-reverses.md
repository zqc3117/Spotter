---
id: change-wrist-configuration-when-free-space-descent-reverses
scope: global
kind: strategy
title: Change wrist configuration when a contact-free descent reverses
applies_when: Picking a small object from a recessed surface when downward corrections
  stop short or move upward without measured contact
symptom:
- sink pickup
- descent reversal
- zero force
- servo miss
- empty close
- wrist configuration
evidence:
  cells:
  - PnPSinkToCounter_s195_ep10
  attempts:
  - A downward correction moved upward with zero contact force, followed by an empty
    close; re-staging with a lateral approach change and wrist rotation reached a
    lower waypoint, but pickup remained unverified.
  source: treatment intervention logs and checkpoint images; both control and treatment
    failed
confidence: single-shot
related: []
---
**Why:**  
A descent that reverses without measured contact may reflect poor Cartesian reachability in the current arm configuration rather than insufficient commanded depth. Here, further descent followed by closing produced an empty grasp. Re-staging with a changed wrist configuration and approach line reached a lower waypoint, but the intervention ended before testing the grasp. This supports an alternative to try, not a proven recovery.

**How to apply:**  
For a small object inside a sink or other recess, compare the actual vertical motion with the command before closing. If descent reverses while contact force remains near zero, retreat to a known approach pose and change the wrist configuration before attempting the same grasp height again. Initially preserve the target location so the effect of the configuration change is interpretable. Reserve enough repair steps to inspect descent completion, close, and verify pickup by lifting; reaching a lower hover alone is not the repair.

**Falsify:**  
This hypothesis is weakened if changing wrist configuration does not improve downward tracking, or if repeated trials show that the original configuration reaches the grasp height reliably. A successfully tracked descent followed by another empty close would instead point toward target alignment or finger geometry as the remaining problem.
