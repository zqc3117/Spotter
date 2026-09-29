---
id: slow-open-gripper-approach-is-not-a-missed-grasp
scope: global
kind: strategy
title: Slow open-gripper approach is not a missed grasp
applies_when: The hand is approaching a free object with open fingers and reduced
  but nonzero travel.
symptom:
- several
- chunks
- show
- much
- less
- hand
- travel
- commanded
- making
- continuing
- approach
- resemble
evidence:
  cells:
  - PnPCounterToMicrowave_s195_ep12
  - PnPCounterToCab_s195_ep10
  attempts: 1
  source: Observed approach and release windows followed by successful episode outcome
    without intervention.
confidence: probable
related: []
---
**Why:**  
Commanded-versus-travelled discrepancies alone do not establish a failed approach. In this episode, repeated downward commands produced only small movements while the fingers remained open. Passing allowed the policy to complete the grasp, placement, and withdrawal successfully.

**How to apply:**  
Check whether the hand is still advancing and whether a grasp has actually been attempted before diagnosing a miss. If travel remains nonzero, the fingers are open, and the images show no definite collision or other failure, keep watching rather than interrupting solely because motion is slow. After closure, judge transport from object motion in third-person views—not aperture alone.

**Falsify:**  
Do not extend this rule to a confirmed empty closure followed by withdrawal, an object left at the source while the hand departs, or sustained near-zero hand travel despite substantial commands. Those observations provide stronger evidence of failure than slow approach does.
