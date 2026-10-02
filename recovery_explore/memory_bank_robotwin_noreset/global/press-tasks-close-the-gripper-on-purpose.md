---
id: press-tasks-close-the-gripper-on-purpose
scope: global
kind: perception
title: "On press, stamp and click tasks the policy closes the gripper on purpose"
applies_when: "The instruction is to press, stamp, click, push or activate something (stapler, bell, seal, switch, button) rather than pick an object up"
symptom: [press, stamp, click, button, stapler, bell, seal, switch, closed gripper, empty close, not a miss]
evidence:
  cells: [press_stapler_s0_ep0]
  attempts: 1
  source: judge-learn-rtL1
confidence: single-shot
related: []
---

**Why:** The policy presses with closed fingers. The telemetry then shows the hand shut on nothing, which looks exactly like a missed grasp; opening the gripper and moving the hand to repair it interrupted a press that was on its way.

**How to apply:**
- On these tasks, a shut empty hand descending onto or resting on the fixture is the press itself: answer ok.
- Look for progress in the pictures (the button, lid or seal moved) rather than in the gripper opening.
- Intervene only if the hand is clearly pressing beside the fixture or has stopped far from it for several windows.

**Falsify:** If a fixture needs a real grip (a handle or lever to pull), the closed hand is a grasp and a shut-on-nothing reading is a real miss.
