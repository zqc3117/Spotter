---
id: brief-restricted-motion-near-a-faucet-can-precede-success
scope: global
kind: strategy
title: Brief restricted motion near a faucet can precede success
applies_when: A faucet policy has approached the lever and first shows reduced end-effector
  travel near contact.
symptom:
- water
- remains
- running
- substantial
- commands
- produce
- little
- travel
- hand
- obscures
- lever
evidence:
  cells:
  - TurnOffSinkFaucet_s195_ep14
  attempts:
  - w4
  - w5
  - w6
  source: verdict_w4.json, verdict_w5.json, w5/telemetry.txt, w6/telemetry.txt, and
    user-reported terminal outcomes
confidence: single-shot
related: []
---
**Why:**
In this episode, travel fell from 8.4 and 8.0 cm per chunk to 2.3 and 1.2 cm while water was still running. The fifth verdict left the policy running without intervention. The final telemetry recorded two more chunks with 2.5 and 2.4 cm travel, and the user reported success for both treatment and control. Low travel near the lever therefore did not establish a failure requiring recovery. This does not establish the precise contact mechanism or an advantage over control.

**How to apply:**
When restricted motion first appears after a progressing approach, allow another observation window if there is no independent evidence of failure. Compare the faucet state and contact geometry across views and successive frames before deciding to retract or reposition. Treat reduced travel and lever occlusion as reasons to inspect, not sufficient evidence of a stall. The distances here are observations, not thresholds.

**Falsify:**
If subsequent windows show unchanged faucet state with repeated ineffective contact, or clear geometry preventing lever actuation, the transient-contact interpretation is unsupported and recovery may be warranted. Repeated comparable episodes that fail after waiting would weaken this strategy.
