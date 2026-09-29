---
id: three-strikes-then-hand-back
scope: global
kind: strategy
title: Three failed repairs on one trajectory means stop and let the policy run out
applies_when: you have repaired, watched it fail, and repaired again on the same episode
symptom: [keeps failing, should I try again, give up, wasting budget]
evidence:
  cells: [x25-PnPStoveToCounter_s195_ep10, jd-PnPCabToCounter_s195_ep0, prior-no-grid-data_s0]
  attempts: 4
  source: judge-rounds
confidence: probable
related: [dropped-out-of-reach-is-terminal, one-change-per-attempt]
---
Repeated repairs on the same trajectory stop paying after about three. Beyond that the arm accumulates poses the policy never sees and the episode gets worse, not better.

**Why:**
Each intervention leaves the arm somewhere the policy did not choose. The cost compounds while the chance of rescue does not.

**How to apply:**
- Write `RESULT: giveup` and stop. Giving up is terminal for the episode by design.
- Give up early when the object has left the reachable workspace, see [[dropped-out-of-reach-is-terminal]].

**Falsify:**
If a fourth or fifth repair rescues episodes at a measurable rate, raise the number.

**Related:** [[dropped-out-of-reach-is-terminal]] [[one-change-per-attempt]]
