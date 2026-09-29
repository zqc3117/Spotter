---
id: allow-a-brief-slowdown-during-drawer-handle-grasp-closure
scope: global
kind: strategy
title: Allow a brief slowdown during drawer handle grasp closure
applies_when: A drawer-opening policy slows briefly as the gripper closes after approaching
  the handle.
symptom:
- end-effector
- travel
- drops
- one
- action
- chunk
- gripper
- aperture
- decreases
- nonzero
- width
evidence:
  cells:
  - OpenDrawer_s195_ep17
  attempts:
  - treatment
  - control
  source: w7/telemetry.txt and ctrl/telemetry.txt; user-reported success for both
    arms
confidence: single-shot
related: []
---
**Why:**
In the treatment trajectory, travel fell to 1.6 cm during steps 144-160 as the gripper closed from 48.1 to 32.8 mm. The next two chunks travelled 13.2 and 13.5 cm with aperture near 32 mm, followed by reported success. The control showed the same pattern and also succeeded. A single low-motion chunk at grasp closure can therefore be a normal transition into pulling; this episode does not establish a treatment benefit.

**How to apply:**
Interpret motion together with gripper state and action phase. If a slowdown coincides with closure after handle approach, allow the next policy chunk and check for resumed pulling before declaring a stall or attempting a regrasp. Treat nonzero aperture as supporting evidence of contact, not proof of a secure handle grasp. Use observed displacement rather than commanded translation magnitude to assess progress.

**Falsify:**
This interpretation fails if subsequent chunks remain nearly stationary, the gripper closes on empty space, or visual evidence shows a missed handle or no drawer movement despite hand motion. Those observations warrant reassessing contact rather than continuing to wait on the basis of this episode.
