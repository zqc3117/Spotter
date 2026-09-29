---
id: a-tipped-mug-on-the-target-is-not-a-placement
scope: global
kind: failure
title: A tipped mug on the target is not a placement, and a far-off stalled arm is
  the window to act
applies_when: A mug or cup sits roughly on the destination (drip tray, shelf, plate)
  but on its side or tilted, while the arm has withdrawn and keeps commanding motion
  it cannot make
symptom:
- mug tipped
- lying on side
- drip tray
- placed but tilted
- arm stalled
- withdrawn
- near step limit
- commanded not travelled
evidence:
  cells:
  - CoffeeSetupMug_s195_ep10
  attempts: 1
  source: The episode ended with the mug on its side on the coffee machine drip tray
    and the arm stuck far away at the edge of the workspace, moving 0.2 cm per chunk
    against 4 cm commands. In both of the last two windows I saw, I answered ok, reasoning
    that "the mug is under the dispenser" and "too few policy steps left." Both the
    control and treatment arms failed. My single earlier intervention used zero actions,
    so it changed nothing.
confidence: single-shot
related:
- check-descent-completion-before-closing-on-a-mug
---
Seeing the mug near the dispenser made the task look done, but a mug on its side under the dispenser does not count as a success. Separately, I passed on acting because only a few policy steps were left, but that budget limits only hand-backs, not my own moves.

**Why:** For place tasks, success depends on the object's pose, not just its location. A tilted or toppled mug fails even when it is directly under the target. When the arm has also withdrawn and stalled, the policy has stopped working on the task. It will not come back and set the mug upright, so each window that stalls is a lost chance to fix it. Also, the remaining policy-step budget does not charge for your own move, nudge, rotate and gripper steps. Seeing few policy steps left is a reason not to hand back, not a reason to skip a repair.

**How to apply:** Once the object reaches the destination, check its orientation in both third-person rows. If it is tilted or on its side and the arm has pulled away or stalled (large commanded moves, near-zero travel, over two or more windows), treat that as a failed placement and consider intervening right away. Don't wait for the final windows. Plan the repair entirely with your own steps. Use retraced move_to steps to get back near the object, then stand it upright from above and across its body. Don't end the plan with a policy hand-back if the policy has no steps left to use.

**Falsify:** Suppose a later episode shows a mug on its side on the destination and the task is judged successful anyway. Or suppose a self-driven righting attempt, started early enough, keeps knocking the mug off the destination. In either case this lesson is wrong, and passing is the better call.
