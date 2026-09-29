---
id: move-to-can-rotate-and-stall-in-tight-space
scope: global
kind: infra
title: A long move-to near structure quietly rotates the wrist and burns steps without arriving
applies_when: commanding a large move-to inside a cabinet, under a hood, or down into a container
symptom: [move-to ok false, did not arrive, wrist rotated, steps exploded, stalled]
evidence:
  cells: [x25-PnPCabToCounter_s195_ep0, x25-PnPCounterToMicrowave_s195_ep6, x25-PnPCounterToSink_s195_ep0]
  attempts: 9
  source: judge-rounds
confidence: probable
related: [move-to-and-lift-do-not-restore-orientation]
---
The servo pushes against contact, the step count climbs, and the end-effector arrives rotated or not at all. `ok: false` here means the servo did not converge, not that the interface failed.

**Why:**
Position control with no contact model keeps pushing toward an unreachable target, and the null-space absorbs the rest as rotation.

**How to apply:**
- Approach constrained space in short steps, checking after each.
- Descending into a container with a raised rim is the worst case; stop at rim height and check.
- If a call returns `ok: false`, read `state` before issuing another; do not repeat it harder.

**Falsify:**
A controller with contact detection would remove the stall and the drift together.

**Related:** [[move-to-and-lift-do-not-restore-orientation]]
