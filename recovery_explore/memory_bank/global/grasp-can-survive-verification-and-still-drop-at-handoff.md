---
id: grasp-can-survive-verification-and-still-drop-at-handoff
scope: global
kind: failure
title: A grasp verified by lift and lateral tests can still drop within the first chunks after handoff
applies_when: you verified a grasp with your own actions and are about to hand control back to the policy
symptom: [drops after handoff, verified then lost, policy loses the object, drops in first chunks]
evidence:
  cells: [x25-PnPStoveToCounter_s195_ep0, x25-PnPCabToCounter_s195_ep2, x25-PnPSinkToCounter_s195_ep1]
  attempts: 7
  source: judge-rounds
confidence: probable
related: [handoff-pose-matters-not-just-position, three-strikes-then-hand-back]
---
Repeated verification (lift, lateral shift, multi-pose third-person check) can all pass and the object still falls out within one to three chunks of the policy taking over.

**Why:**
Verification proves the grasp exists now, under your motions. The policy's own motions are larger and differently oriented, and the wrist pose you left behind may be one the policy never sees in training.

**How to apply:**
- Do not spend extra budget on more verification; it does not predict survival.
- Do spend one action straightening the wrist before handing back, see [[handoff-pose-matters-not-just-position]].
- If the same trajectory drops right after handoff twice, stop intervening on it.

**Falsify:**
If straightening the wrist before handoff removes the post-handoff drops across several trajectories, the cause is orientation and this entry should be narrowed to that.

**Related:** [[handoff-pose-matters-not-just-position]] [[three-strikes-then-hand-back]]
