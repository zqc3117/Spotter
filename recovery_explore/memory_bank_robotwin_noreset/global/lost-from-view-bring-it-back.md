---
id: lost-from-view-bring-it-back
scope: global
kind: failure
title: "An object or hand that leaves the head camera view is a fault: find it and bring it back into view"
applies_when: "The target object or the working hand is no longer in the head camera picture, or a pointing task is aimed the wrong way"
symptom: [out of view, left the frame, lifted too high, knocked away, lost object, scan direction, wrist view, find it]
evidence:
  cells: [handover_mic_s0_ep0, place_dual_shoes_s0_ep0, scan_object_s0_ep0]
  attempts: 3
  source: operator-review-rtG1
confidence: verified
related: [dropped-or-toppled-object-is-a-fault]
---

**Why:** A hand lifted a microphone out of the head camera picture, a shoe was knocked out of view,
a scanner pointed the wrong way; in each case the judge treated the missing view as missing
information and waited, and the episode ran out of steps.

**How to apply:**
- Treat "not in the head camera picture" as a fault, not as a reason to wait.
- Find the object in the other cameras first; the other wrist view often still shows it.
- Move the arm so that the head view and the wrist view of the working hand see the target again.
- For pointing tasks, turn so the target moves toward the centre of the wrist picture.

**Falsify:** If bringing the hand back into view repeatedly costs the policy the steps it needed to
finish, leaving a briefly hidden hand alone is the better default.
