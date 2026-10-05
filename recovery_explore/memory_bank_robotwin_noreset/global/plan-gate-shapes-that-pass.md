---
id: plan-gate-shapes-that-pass
scope: global
kind: infra
title: "Send plan shapes the gate accepts on the first try"
applies_when: "Writing an intervention plan"
symptom: [plan gate, rejected, policy first, manual close, plan shape, wasted intervention]
evidence:
  cells: [place_a2b_left_s0_ep0]
  attempts: 1
  source: judge-learn-rtL1
confidence: single-shot
related: []
---

**Why:** Two rejected plans used up an intervention and left the policy with far fewer steps than a valid first plan would have.

**How to apply:**
- Start with a repair step, never with policy (unless a step already faulted or a repair already ran since the last hand-back).
- First plan: place the OPEN fingers, then policy 1-2 chunks.
- A manual close is accepted only after a failed hand-back in this intervention or after two empty closures in the episode, and it must be followed by a lift in the same plan.

**Falsify:** If a plan that breaks one of these shapes is accepted, the gate changed and this entry must be updated.
