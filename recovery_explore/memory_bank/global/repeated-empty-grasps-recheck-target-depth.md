---
id: repeated-empty-grasps-recheck-target-depth
scope: global
kind: strategy
title: Recheck target depth when local grasp corrections repeat the miss
applies_when: A free object remains on its support after empty grasps and small corrections
  around the recorded miss do not resolve them.
symptom:
- empty close
- repeated miss
- cabinet
- target depth
- servo missed
- occlusion
- approach reset
evidence:
  cells:
  - PnPCabToCounter_s195_ep10
  attempts: Several local corrections and one approach reset failed; visually locating
    the target and approaching its actual depth produced a verified grasp.
  source: Intervention logs and camera checkpoints; treatment succeeded while control
    failed.
confidence: single-shot
related: []
---
**Why:**

A recorded empty-close pose identifies where the hand missed, not necessarily where the object is. Repeatedly descending near that pose can preserve a substantial depth error. In this episode, local height and lateral corrections failed, and the policy repeated the empty grasp after an approach reset. Raising the hand exposed a usable view of the target; visual targeting then revealed that it lay farther into the cabinet.

**How to apply:**

- Use the missed-grasp pose to retrace the approach efficiently, but do not treat it as an object-location measurement.
- If local corrections repeatedly fail, stop accumulating nudges. Re-stage and clear the view enough to locate the actual target.
- Use a visible target surface to establish the approach, allowing for surface-versus-centre bias. Keep the hand above the support during lateral alignment.
- If that target is outside the retracing allowance, use bounded waypoints rather than assuming one long move will reach it. Account for actual landing error before the next step.
- Verify pickup by the object rising with the hand in a third-person view, then verify retention after a short policy transport segment.

**Falsify:**

If clear views already show the fingers centred around the target at the correct depth, further visual retargeting may not help. Investigate orientation, insufficient descent, contact, or grasp geometry instead. Repeated servo misses alone do not establish a depth error; the target's visible position must support that diagnosis.
