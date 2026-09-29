---
id: distinguish-clipped-waypoints-from-servo-drift
scope: global
kind: strategy
title: Distinguish a clipped waypoint from servo drift before re-staging
applies_when: A long move toward a previously visited approach point stops early
symptom:
- hand
- stops
- short
- requested
- target
- repair
- risks
- repeatedly
- returning
- starting
- pose
evidence:
  cells:
  - PnPCabToCounter_s195_ep10
  attempts: 1
  source: Intervention execution logs and checkpoint images
confidence: single-shot
related: []
---
**Why:**
A capped move and a servo miss require different responses. In this episode, a long return stopped at the 60 cm cap without contact. Completing the remaining distance brought the hand near the intended approach. Subsequent re-staging required additional capped moves and consumed repair actions before any grasp attempt. The episode ultimately failed; successful repositioning alone did not establish a successful repair.

**How to apply:**
Read the stop reason alongside requested distance, actual travel, endpoint error, and contact force. If the hand faithfully reached a capped waypoint, compute the remaining distance and inspect the route before continuing toward the original target. Do not treat that expected truncation as accumulated drift requiring a return to the takeover pose.

If the servo instead misses its executable waypoint, reassess the approach and re-stage as appropriate. If actual travel is several times the executable displacement, follow the controller blow-up stop rule rather than sending another Cartesian move.

Budget the return route before starting: permission to retrace a visited region does not remove the per-call distance cap, and clipping aborts later plan steps. Preserve enough repair actions to attempt and verify the grasp.

**Falsify:**
This distinction would be unhelpful if cap-only stops consistently left the arm in unsafe configurations or completing the remaining route repeatedly caused collisions or controller failures. Test across additional episodes whether continuing after clean clipping saves actions without increasing those failures.
