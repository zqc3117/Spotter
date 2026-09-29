---
id: reopening-an-empty-gripper-can-break-a-policy-stall
scope: global
kind: strategy
title: Reopening an empty gripper can restart a stalled approach without Cartesian
  correction
applies_when: A free-object policy remains closed on air near the source, especially
  when Cartesian repair has proved unreliable
symptom:
- empty close
- closed gripper
- source unchanged
- stalled approach
- oscillation
- controller overshoot
evidence:
  cells:
  - PnPStoveToCounter_s195_ep10
  attempts: A positional correction overshot and was rewound; a later gripper-only
    reopening followed by two policy chunks restarted descent and enabled a verified
    grasp.
  source: 'Intervention logs, checkpoint images, and paired outcomes: control failed,
    treatment succeeded.'
confidence: single-shot
related: []
---
**Why:**  
A stalled empty closure can reflect a policy that has stopped approaching because its fingers are already closed. Here, reopening alone changed the policy’s behavior: it left the stalled pose and descended with open fingers around the target. This succeeded where a Cartesian correction had produced severe overshoot. The useful alternative was changing gripper state, not finding a stronger positional correction.

**How to apply:**  
When the target remains at the source and the hand persistently stays closed on nothing, consider reopening the fingers and letting the policy perform a short fresh approach. This is particularly useful after an unsafe Cartesian attempt has been rewound. Check whether reopening actually restarts descent or alignment; do not assume it repairs the grasp. Once the fingers straddle the target, verify pickup by closing and lifting, then check retention through policy motion.

**Falsify:**  
If reopening makes the policy immediately close empty again, preserves the same stalled motion, or causes withdrawal without a renewed approach, gripper state did not unlock the approach in that case. Repeated such outcomes would undermine this as a useful alternative to positional correction.
