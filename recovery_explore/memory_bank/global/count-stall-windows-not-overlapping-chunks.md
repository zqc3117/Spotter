---
id: count-stall-windows-not-overlapping-chunks
scope: global
kind: strategy
title: Count stalled observation windows, not overlapping history chunks
applies_when: Supervising fixture manipulation with low hand travel and uncertain
  control-state changes
symptom:
- several
- consecutive
- history
- rows
- show
- large
- motion
- commands
- little
- translation
evidence:
  cells:
  - TurnOnSinkFaucet_s195_ep13
  - TurnOffSinkFaucet_s195_ep11
  attempts: 1
  source: Two observed windows were passed without intervention; both control and
    treatment runs ultimately succeeded
confidence: probable
related: []
---
**Why:**  
Consecutive low-travel chunks can span fewer observation windows than they appear to, especially when action histories overlap. In this episode, three consecutive low-travel chunks covered only two observed windows of reduced travel. Waiting rather than treating those chunks as three stalled windows preserved a run that ultimately succeeded. This supports patience at initial contact, not a conclusion that the fixture was already activated.

**How to apply:**  
Track persistent lack of fixture change across distinct observation windows. Do not count repeated history rows as fresh evidence or substitute chunk count for a window-based stall threshold. Check the control itself in every available view; discount apparent rotation when the surrounding panel rotates with the wrist camera. If the required persistence threshold has not been reached and no independent failure is visible, pass.

**Falsify:**  
This lesson would be weakened by repeated episodes where waiting through the prescribed observation windows loses recoverable runs that earlier intervention reliably saves. A visible wrong-control interaction or another independent failure justifies action without waiting for the stall threshold.
