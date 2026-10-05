---
id: identify-target-and-destination-first
scope: global
kind: perception
title: "Name the object and its destination before judging; a clean run to the wrong thing is a fault"
applies_when: "Any place / put / move task where the scene holds more than one candidate destination (a mat, a plate, a stand, a small bin next to a big bin)"
symptom: [wrong destination, wrong object, placed beside, mat, bin, container, looks fine but wrong, identification]
evidence:
  cells: [place_mouse_pad_s0_ep0, dump_bin_bigbin_s0_ep0]
  attempts: 2
  source: operator-review
confidence: probable
related: []
---

**Why:** Reviewers watching the recordings found episodes where the arm ran a smooth, unstalled
trajectory and set the object down next to the mat instead of on it, or emptied a bin into the
wrong receptacle. Nothing stalled, no closure came back empty, so every "is it stuck" signal
stayed quiet and the window was passed. The fault was in what the policy was aiming at, not in
how it moved.

**How to apply:**
- In the diagnosis of every window, name the target object and the destination, as things you can
  point at in the head view; use the working hand's wrist view when two candidates look alike.
- Judge placement against the destination you named, not against smoothness of motion.
- If the object is heading for, or has landed on, the wrong thing, that is a fault: pick it up
  again and place it on the destination.
- Say which is which in words ("the wide dark mat, not the paper beside it") so a later window can
  check the same thing.

**Falsify:** If the destination cannot be told apart in any available view, naming it is guesswork
and the rule does not apply; report the ambiguity instead of intervening on it.
