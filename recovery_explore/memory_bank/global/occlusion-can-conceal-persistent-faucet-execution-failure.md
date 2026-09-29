---
id: occlusion-can-conceal-persistent-faucet-execution-failure
scope: global
kind: failure
title: Occlusion can conceal persistent faucet execution failure
applies_when: A fixture control is obscured while repeated substantial motion commands
  produce no measurable hand movement.
symptom:
- hand
- stays
- same
- pose
- many
- chunks
- despite
- repeated
- translation
- rotation
- commands
evidence:
  cells:
  - TurnOnSinkFaucet_s195_ep15
  attempts: 1
  source: Windows 13 through 15 telemetry and third-person images; final control and
    treatment outcomes both failed.
confidence: single-shot
related: []
---
**Why:**
Chunks 18 through 29 reported 0.0 cm travelled despite commands of roughly 9 cm in x and -14 cm in z, with substantial rotation commands. The faucet state was obscured, and the supervisor repeatedly passed because it could not verify fixture failure. Both arms ultimately failed. This shows that uncertainty about fixture state can persist alongside an unresolved execution problem; it does not prove that intervention would have helped.

**How to apply:**
Distinguish small achieved movement during normal fixture manipulation from sustained failure to execute substantial commands. Track that discrepancy across windows even when the control is obscured. Reassess visible contact and obstruction evidence instead of treating each occluded window as a fresh routine pass. Keep any repair contingent on a supported contact diagnosis and a known safe correction; this episode supplies neither a validated direction nor a successful repair.

**Falsify:**
Visible water flow or other direct evidence that the requested fixture state has been reached defeats the failure interpretation. Successful episodes with the same prolonged command-motion discrepancy would weaken its usefulness as a warning sign.
