---
id: measure-knob-rotation-relative-to-the-fixed-panel
scope: global
kind: perception
title: Measure knob rotation relative to the fixed panel
applies_when: Assessing rotary control progress through a moving wrist camera.
symptom:
- knob
- appears
- turn
- wrist
- view
- index
- remains
- fixed
- relative
- panel
- scale
evidence:
  cells:
  - TurnOnStove_s195_ep15
  attempts:
  - w5
  - w6
  - w7
  - w8
  - w9
  source: verdict_w5.json through verdict_w9.json; w9/telemetry.txt; user-reported
    control and treatment success
confidence: single-shot
related: []
---
**Why:**
The episode judgments twice reported knob progress and then corrected that assessment because the surrounding panel rolled with the wrist camera. At w9, the judgment reported an unchanged index across successive windows, and telemetry recorded only 0.1 cm of hand travel in each of the last two chunks despite substantial commands. Both arms ultimately succeeded, so this episode does not establish an intervention benefit.

**How to apply:**
Compare the knob index against fixed scale markings in successive views before declaring rotation. Use the panel as the visual reference, accounting for camera roll and perspective. When markings are obscured, leave progress uncertain. Combine persistent absence of relative index movement with command and motion history before diagnosing ineffective engagement; small hand translation alone is compatible with successful rotation.

**Falsify:**
If the index moves relative to the fixed scale after accounting for camera motion, the apparent turn is real and a stall diagnosis based on small hand translation should be rejected. Independent knob-angle measurements or a stable camera showing rotation would also contradict a camera-motion explanation.
