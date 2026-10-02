---
id: escalate-to-own-grasp-after-repeated-empty-closures
scope: global
kind: strategy
title: "After repeated empty closures on the same target, grasp yourself instead of handing back"
applies_when: "The arm that owns the grasp has shut on nothing two or more times in this episode, or a hand-back in this intervention ended shut on nothing again"
symptom: [empty close, closed on nothing, repeated miss, hand-back failed, escalate, grasp yourself, pixel aim, one arm done]
evidence:
  cells: [blocks_ranking_size_s0_ep0, place_bread_basket_s0_ep0]
  attempts: 2
  source: judge-learn-rtL1
confidence: probable
related: []
---

**Why:** Re-opening the fingers and handing back from almost the same pose let the policy repeat the same approach, and it missed the same way. A nudge only changes the offset; it does not change the approach line or who makes the close. In two-object tasks the arm whose partner has already placed its object kept missing no matter how close the hand was put.

**How to apply:**
- Count empty closures on the arm that owns the grasp across the whole episode.
- At two, stop handing back for the close. In ONE plan: PIXEL move_to with the fingers open a few centimetres above the object centre, nudge down, close, lift a little to check the object comes up.
- Hand back 1-2 chunks only after the lift shows a hold, so the policy transports and places.
- Change the category of the action (who closes, approach line, grasp height), not just the offset.

**Falsify:** If a self-made close from a pixel-aimed pose also comes up empty on the lift, the fault is elsewhere (wrong target, orientation, reach) and this escalation is not the fix.
