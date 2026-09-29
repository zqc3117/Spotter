---
id: pixel-targets-need-native-camera-coordinates
scope: global
kind: infra
title: Convert montage pixels to native camera coordinates before targeting
applies_when: A repair uses pixel targets selected from a resized multi-camera montage
symptom:
- pixel target
- montage
- image scaling
- wrong waypoint
- move clipped
- unprojection
evidence:
  cells:
  - PnPStoveToCounter_s195_ep10
  attempts: 1
  source: treatment trajectory and repair checkpoints
confidence: single-shot
related: []
---
**Why:** The first repair selected the correct lime visually but supplied coordinates measured in a downscaled montage tile. Unprojection produced a waypoint well above and away from the pan, and the move was clipped. After re-staging, targeting the lime using the full-resolution camera image reached the intended approach region. The subsequent grasp repair succeeded.

**How to apply:** Distinguish montage coordinates, tile-local coordinates, and native camera coordinates. Remove the tile’s montage offset and account for resizing before submitting a pixel target. If the mapping is unclear, use an available full-resolution checkpoint rather than guessing. When the returned waypoint is inconsistent with the visible target, check coordinate scaling before interpreting the result as an obstacle or escalating the motion.

**Falsify:** This explanation would be wrong if the targeting interface accepts montage-scaled coordinates directly, or if verified native-camera coordinates produce the same misplaced waypoint. In that case, investigate camera selection, depth, or calibration instead.
