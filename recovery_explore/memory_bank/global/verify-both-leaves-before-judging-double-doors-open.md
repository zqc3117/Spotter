---
id: verify-both-leaves-before-judging-double-doors-open
scope: global
kind: perception
title: Verify both leaves before judging double doors open
applies_when: A task asks to open cabinet doors and one or both leaves are partly
  occluded.
symptom:
- open
- panel
- visible
- sustained
- arm
- commands
- produce
- almost
- motion
evidence:
  cells:
  - OpenDoubleDoor_s195_ep12
  attempts: 1
  source: Window 31 telemetry and montage, compacted intervention history, and final
    paired outcome report.
confidence: single-shot
related: []
---
**Why:**
At window 31, repeated large translation commands produced only about 0.1 cm of travel per chunk. The judgment passed because the cabinet doors appeared open, without separately establishing the state of each leaf. The final report recorded control success and treatment failure. An earlier intervention was recorded as fixed, but its details are unavailable; this outcome does not establish which action caused failure.

**How to apply:**
For a plural door instruction, track each leaf separately against the cabinet frame using the third-person views. A visible open panel supports only that panel's state. If the other leaf is obscured, report uncertainty rather than treating the whole fixture as complete. Sustained large commands with negligible travel warrant checking for obstruction and unfinished fixture motion, but justify a repair only when the relevant leaf and a useful correction can be identified.

**Falsify:**
If both leaves can be independently seen sufficiently open, negligible arm travel can be harmless post-completion behavior. If replay shows both leaves were already open at this window, this judgment was not the source of the failure; investigate earlier actions instead.
