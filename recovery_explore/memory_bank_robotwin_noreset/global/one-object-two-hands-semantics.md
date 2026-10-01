---
id: one-object-two-hands-semantics
scope: global
kind: strategy
title: "When one object needs both hands at once, a one-hand-at-a-time policy is already wrong"
applies_when: "The instruction implies both hands act on the same object together: a pot by two handles, tipping one bin into another, a wide or heavy object, or a handover between the hands"
symptom: [both hands, one at a time, dual lift, pot, handles, tip, handover, semantics, move_both, lift_both]
evidence:
  cells: [lift_pot_s0_ep0, handover_block_s0_ep0]
  attempts: 2
  source: operator-review
confidence: probable
related: [escalate-to-own-grasp-after-repeated-empty-closures]
---

**Why:** In these episodes the policy kept trying one hand, then the other, on an object that only
moves when both hands hold it. Each single-hand attempt looked like an ordinary approach, so the
windows read as "still working"; the task could never succeed that way.

**How to apply:**
- Decide from the instruction whether the object needs one hand or two before the first repair.
- If it needs two and only one hand is engaging it, repair with the both-hand steps so the hands
  move in the same control steps, then hand back.
- On a handover, the giving hand must be still and holding before the receiving hand closes.
- Do not spend repairs perfecting a single-hand grasp on a two-hand object.

**Falsify:** If a one-hand grasp on such an object ever lifts and carries it to the destination,
the object does not need two hands and this rule does not apply to it.
