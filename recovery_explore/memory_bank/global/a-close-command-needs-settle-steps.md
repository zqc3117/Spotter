---
id: a-close-command-needs-settle-steps
scope: global
kind: infra
title: Aperture read immediately after a close command has not settled yet
applies_when: reading gripper width right after commanding a close
symptom: [width looks wrong, read too early, aperture still moving]
evidence:
  cells: [x25-PnPStoveToCounter_s195_ep13]
  attempts: 1
  source: judge-rounds
confidence: single-shot
related: [width-plateau-cuts-both-ways]
---
The fingers take several simulation steps to reach their final position. A reading taken immediately after the command reflects the motion, not the result.

**Why:**
The command is a position target for a controller that needs time to converge.

**How to apply:**
- Let roughly eight steps pass, or issue a small hold move, before trusting the number.
- The telemetry's per-chunk width column is already settled; prefer it.

**Falsify:**
A gripper that closes within a step would remove the delay.

**Related:** [[width-plateau-cuts-both-ways]]
