---
id: one-change-per-attempt
scope: global
kind: primitive
title: Change one variable per attempt so the next attempt knows what moved the needle
applies_when: a repair attempt failed and you are about to try again
symptom: [tried again and failed, many changes at once, cannot tell what helped]
evidence:
  cells: [prior-no-grid-data_s0, jd-PnPCounterToStove_s195_ep0]
  attempts: 4
  source: judge-rounds
confidence: probable
related: [fix-at-the-miss-not-where-the-arm-ended-up, three-strikes-then-hand-back]
---
Reopen, lift clear, recentre over the object, then descend, and alter exactly one quantity from the previous attempt: the horizontal offset, or the depth, or the wrist angle.

**Why:**
A budget of eight actions allows very few attempts. Changing three things at once spends an attempt and produces no information for the next one.

**How to apply:**
- Say in your summary which quantity you changed and by how much.
- On a retry after `reset`, change a different quantity rather than repeating the same move harder.

**Falsify:**
If failures come from a single dominant cause, changing everything at once would be faster.

**Related:** [[fix-at-the-miss-not-where-the-arm-ended-up]] [[three-strikes-then-hand-back]]
