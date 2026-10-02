---
id: idle-arm-empty-closure-is-not-the-fault
scope: global
kind: failure
title: "An empty closure on the idle arm is not the fault to repair"
applies_when: "Two-arm task where the telemetry reports an empty closure on the arm that is parked or waiting rather than the arm over the target"
symptom: [empty close, idle arm, parked arm, wrong arm, bimanual, which arm, closed on nothing]
evidence:
  cells: [place_object_basket_s0_ep0]
  attempts: 1
  source: judge-learn-rtL1
confidence: single-shot
related: []
---

**Why:** An intervention spent its actions re-opening and repositioning the arm that was not doing the task. An idle hand closing on air is harmless; the task only advances when the arm that owns the grasp holds the object.

**How to apply:**
- Before planning, match the empty-closure flag to the arm that is over the named target in the head and wrist pictures.
- If the flag is on the parked or waiting arm, treat it as no fault and answer ok, or direct any repair at the arm that owns the grasp.
- Never spend an action on the idle arm's aperture.

**Falsify:** If a task needs the idle arm later and its shut gripper blocks that step (it must open to receive or grasp), the idle closure does matter; then an open command on that arm is the right repair.
