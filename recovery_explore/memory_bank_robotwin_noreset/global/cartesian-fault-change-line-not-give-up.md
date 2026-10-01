---
id: cartesian-fault-change-line-not-give-up
scope: global
kind: strategy
title: "After Cartesian moves fail from a pose, change the line or the arm; do not give up while steps remain"
applies_when: "Planned moves from the current arm pose keep ending short of the target (move_residual) or with no path (plan_failed), and the policy still has a good share of its steps left"
symptom: [move residual, plan failed, stuck configuration, ik, change approach, give up, other arm]
evidence:
  cells: [open_microwave_s0_ep0, handover_mic_s0_ep0, open_laptop_s0_ep0]
  attempts: 3
  source: judge-learn-rtL1
confidence: verified
related: []
---

**Why:** Repeating Cartesian moves of the same kind from the same pose failed the same way each time. Giving up early also lost an episode: the policy was left alone although a different repair was still possible and steps remained.

**How to apply:**
- After one failed move, change a real thing from where the arm is now: a different approach height or direction, the other arm, or hand back to the policy.
- Do not send a third Cartesian plan from the same pose.
- Give up only when the object is out of reach for good (fallen off the table, destroyed) or the steps left cannot fit a hand-back that finishes the task.

**Falsify:** If changing the line from such poses never produces a successful hand-back, the pose itself is the fault and an early hand-back to the policy should be the first move.
