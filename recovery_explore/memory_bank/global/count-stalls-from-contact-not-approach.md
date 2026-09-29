---
id: count-stalls-from-contact-not-approach
scope: global
kind: strategy
title: Count ineffective-contact windows separately from approach
applies_when: A policy approaches a fixture and then slows near its control.
symptom:
- fixture
- appears
- unchanged
- during
- approach
- followed
- reduced
- hand
- travel
- gripper
- repositioning
evidence:
  cells:
  - CoffeePressButton_s195_ep10
  - OpenDoubleDoor_s195_ep10
  - TurnOffMicrowave_s195_ep11
  - TurnOnMicrowave_s195_ep11
  - TurnOnSinkFaucet_s195_ep13
  attempts: 1
  source: Observed trajectory with no intervention; both control and treatment succeeded.
confidence: verified
related: []
---
**Why:**
An unchanged fixture during approach is expected. Counting those windows toward an ineffective-contact threshold can trigger intervention just as the policy reaches the control. Here, continued approach followed by one reduced-travel window and finger repositioning was compatible with eventual success.

**How to apply:**
Separate approach from attempted operation using adjacent images, with travel measurements as confirmation. Start counting ineffective-contact windows when the hand is positioned to operate the control, not merely when the control first appears unchanged. Inspect the wrist view for button or indicator changes; empty fingers and reopening alone do not establish failure.

**Falsify:**
If the hand was already pressing without effect during the earlier windows, those windows do count. A clearly wrong control or three consecutive windows of motion commands with no hand or fixture movement warrants intervention rather than extending the approach interpretation.
