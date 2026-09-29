---
id: avoid-wrist-reorientation-after-unexplained-cartesian-misses
scope: global
kind: strategy
title: Do not escalate an unexplained Cartesian miss into wrist reorientation
applies_when: A fixture remains unchanged, the hand is stalled away from its control,
  and a Cartesian correction misses without contact.
symptom:
- faucet running
- remote stall
- servo miss
- zero force
- wrist reorientation
- overshoot
- failed recovery
evidence:
  cells:
  - TurnOffSinkFaucet_s195_ep11
  attempts:
  - combined retreat
  - vertical-only correction
  - wrist-pitch change followed by lift
  source: paired control and treatment outcomes plus intervention execution logs
confidence: single-shot
related: []
---
**Why:**  
The untreated policy succeeded; the supervised trajectory failed. The visible running faucet and stationary hand established lack of progress, but did not establish an obstruction or a safe retreat direction. A combined retreat moved downward instead of upward with zero contact force. After a reset, vertical-only moves executed, but the policy returned toward its previous pose. Escalating to wrist reorientation followed by another short lift then produced travel exceeding twenty times the requested distance. The faucet remained on, and the policy did not recover before its budget expired.

**How to apply:**  
Separate evidence that progress has stopped from evidence supporting a particular repair. When the hand is away from the fixture, reversing the policy's commands is not a validated clearance maneuver. If that maneuver misses in an unexpected direction without contact, reconsider the controller configuration rather than assuming a collision.

Do not treat a policy's return after a height correction as evidence that wrist pitch needs changing. Without visible alignment evidence at the intended control, speculative reorientation can make the next Cartesian move much less reliable. Prefer preserving the policy's pose and remaining steps over escalating an unsupported repair. A successfully executed reposition alone does not establish that the fixture operation has recovered.

**Falsify:**  
In comparable paired runs, test whether preserving the stalled pose allows eventual fixture operation. This lesson would be weakened if untreated runs consistently remain stalled while a reproducible, visually justified wrist correction restores bounded Cartesian motion and successful operation without overshoot.
