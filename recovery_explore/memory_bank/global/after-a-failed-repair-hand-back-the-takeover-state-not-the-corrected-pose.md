---
id: after-a-failed-repair-hand-back-the-takeover-state-not-the-corrected-pose
scope: global
kind: strategy
title: A failed repair ended by returning to the takeover pose and handing back beats a second correction from where the hand is
applies_when: the first attempt of an intervention did not produce the object in hand or the control moved, and you are choosing what to do next
symptom: [repair failed, not fixed, hand back, takeover pose, retreat, second correction, untouched state]
evidence:
  cells: [g120r-CoffeeSetupMug_s195_ep14, g120r-TurnOffStove_s195_ep17, g120rpi-PnPCounterToCab_s195_ep12, g120r-TurnOnStove_s195_ep18, g120r-CoffeeSetupMug_s195_ep13]
  attempts: 5
  source: campaign-2026-09-16
confidence: verified
related: []
---
A failed repair ended by returning to the takeover pose and handing back beats a second correction from where the hand is.

**Why:**
With the scene handed back as it was, the policy finished from its own pose in five of these episodes. With the correction left in place the same policies re-approached from a displaced pose and ran out of steps.

**How to apply:**
- End a failed first attempt with retreat and an empty-plan fixed.
- Reserve a second attempt for a materially different plan; never a third.

**Falsify:**
Episodes where handing back the takeover state failed and the committed correction succeeded.
