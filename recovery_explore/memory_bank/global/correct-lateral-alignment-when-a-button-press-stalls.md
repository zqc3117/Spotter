---
id: correct-lateral-alignment-when-a-button-press-stalls
scope: global
kind: strategy
title: Correct lateral alignment when a button press stalls
applies_when: A hand repeatedly commands forward motion at a button while the control
  stays unchanged and the wrist view shows lateral misalignment.
symptom:
- near-zero
- end-effector
- travel
- despite
- continued
- pressing
- commands
- tip
- beside
- intended
- button
evidence:
  cells:
  - TurnOnMicrowave_s195_ep15
  attempts:
  - 'Window 8: retract 2.5 cm, shift laterally 2.5 cm toward START, then command a
    3.5 cm forward press.'
  source: 'Trajectory telemetry, wrist images, and terminal report: control arm failed;
    treatment arm succeeded. An earlier intervention also occurred, so this correction''s
    independent contribution is unknown.'
confidence: single-shot
related: []
---
**Why:**
Repeated forward commands can fail when the pressing tip is beside the button. The wrist camera can show this alignment even when third-person views mainly show the hand. In this episode, a small lateral correction preceded eventual treatment success, although the immediate image did not confirm activation.

**How to apply:**
Check commanded motion and actual travel, then inspect the exact named button in the wrist view. If the control stays unchanged and lateral misalignment is visible, retract slightly, shift a few centimetres toward the button, and re-approach. Infer directions from the current scene; do not reuse this episode's world-coordinate signs. Leave a reachable pose for the policy to continue and distinguish that from a confirmed press.

**Falsify:**
Do not apply this correction when the button has already activated or alignment is unclear. The final forward move here missed its servo target and reported zero force, so neither the move nor its immediate image proves contact. If alignment improves but the button remains unchanged, investigate depth or wrist orientation instead of repeating lateral shifts.
