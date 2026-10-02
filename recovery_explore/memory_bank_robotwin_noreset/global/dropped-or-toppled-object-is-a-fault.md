---
id: dropped-or-toppled-object-is-a-fault
scope: global
kind: failure
title: "An object dropped in a handover, or knocked over on approach, is a fault to repair where it now lies"
applies_when: "The object falls out of a hand during a transfer, or the arm topples a neighbouring object on its way to the target"
symptom: [dropped, fell, toppled, knocked over, handover, on its side, moved object, re-grasp, new orientation]
evidence:
  cells: [handover_mic_s0_ep0, pick_dual_bottles_s0_ep0]
  attempts: 2
  source: operator-review
confidence: probable
related: [identify-target-and-destination-first]
---

**Why:** Reviewers saw a microphone slip out of the receiving hand and a bottle knocked flat, and
in both cases nothing was repaired: the policy kept reaching for where the object used to be, and
the episode ran out of steps. A moved object does not raise a stall or an empty-closure flag, so
it has to be seen in the pictures.

**How to apply:**
- Compare the object's place and orientation with the earlier windows; a change nobody commanded
  is a fault.
- Re-aim at where it lies now with a pixel move_to, and grasp it in its new orientation: an object
  on its side is taken across its body, not around the end that now faces up.
- A dropped object is worth one repair even late in the episode, because the policy will not
  re-target it on its own.
- Do not read "the object moved" as a reason to give up.

**Falsify:** If re-grasping a toppled object in its new orientation repeatedly fails while the same
object grasped upright succeeds, the orientation-aware re-grasp is not the right repair.
