---
id: check-descent-completion-before-closing-on-a-mug
scope: global
kind: failure
title: Check descent completion before closing on a mug
applies_when: Retrying a mug grasp near a dispenser after an empty grasp
symptom:
- deeper
- descent
- misses
- target
- action
- sequence
- continues
- close
- lift
evidence:
  cells:
  - CoffeeServeMug_s195_ep10
  attempts:
  - w7_a1_r0
  - w7_a2_r0
  - w7_a3_r0
  source:
  - exec_w7_a1_r0.json
  - exec_w7_a2_r0.json
  - replan_w7_a3.json
  - exec_w7_a3_r0.json
confidence: single-shot
related: []
---
**Why:**
In attempt w7_a1_r0, a commanded 3 cm descent moved down about 1.9 cm. The next retry increased the descent to 5 cm, but the end effector moved down only 1.3 cm and sideways about 2 cm, leaving 4.22 cm target error with servo_ok false. The sequence nevertheless closed and lifted; the lift also missed, and the subsequent diagnosis reported a tipped mug. Wrist contact force remained zero throughout these actions, so that signal did not establish a clean approach. The later orientation adjustment also ended in an empty grasp. Both episode arms failed; no successful recovery was demonstrated.

**How to apply:**
Put an observation checkpoint after the approach descent and before closing. Compare achieved position with the requested target and inspect the mug and fingers. If the servo reports failure or substantial lateral drift, stop the close-and-lift sequence, reassess alignment and clearance, and choose a revised approach. Do not interpret a failed grasp as evidence that the fingers merely need to descend farther. After closing, verify that the mug follows a small test lift before resuming transport.

**Falsify:**
This proposed checkpoint would be unnecessary for this failure mode if repeated comparable trials showed that descents with similar target error and drift still reliably aligned the fingers and produced stable lifts without disturbing the mug. A successful deeper retry after an accurately achieved, visually aligned approach would also limit the lesson to failed descents rather than deeper grasps generally.
