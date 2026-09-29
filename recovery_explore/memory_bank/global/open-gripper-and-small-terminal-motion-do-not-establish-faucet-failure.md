---
id: open-gripper-and-small-terminal-motion-do-not-establish-faucet-failure
scope: global
kind: perception
title: Open gripper and small terminal motion do not establish faucet failure
applies_when: Monitoring a faucet activation attempt with an open gripper and reduced
  end-effector travel near the end of the approach.
symptom:
- gripper
- aperture
- stays
- terminal
- chunks
- travel
- only
- about
evidence:
  cells:
  - TurnOnSinkFaucet_s195_ep10
  attempts:
  - ctrl
  - w7
  source: ctrl/telemetry.txt and w7/telemetry.txt; user-reported control and treatment
    success.
confidence: single-shot
related: []
---
**Why:**
Both arms succeeded despite an open gripper throughout the recorded trajectory. In the final three chunks, control travel was 0.6, 0.8, and 0.9 cm; treatment travel was 0.5, 0.8, and 1.0 cm. Thus, neither an open gripper nor several small terminal movements alone establishes failure. These records do not establish the contact mechanism or a benefit from intervention.

**How to apply:**
When monitoring another faucet activation attempt, treat these signals as ambiguous. Check faucet state and visible interaction before prescribing gripper closure or a recovery solely because translation has slowed. Use task progress and success evidence to decide whether intervention is needed.

**Falsify:**
If the faucet remains off and observation shows the gripper missing the handle or making no useful contact, this episode does not justify continuing unchanged. A handle that requires a secure grasp may also need closure; open-gripper success here does not establish that closure is generally unnecessary.
