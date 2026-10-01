---
id: read-the-task-before-judging
scope: global
kind: strategy
title: "Parse the task before judging: objects, destination, order, one hand or both"
applies_when: "The start of every episode, and any window where the policy seems busy but the task is not getting closer"
symptom: [task parse, wrong object, wrong order, big and small, both hands, one hand, semantics, ranking, lift together]
evidence:
  cells: [lift_pot_s0_ep0, blocks_ranking_size_s0_ep0]
  attempts: 2
  source: operator-review-rtG1
confidence: probable
related: [one-object-two-hands-semantics, identify-target-and-destination-first]
---

**Why:** In one episode the policy worked a two-handled pot with a single hand from the first
window; in another it ordered blocks by the wrong size from the start. Neither ever looked stuck,
so the judge passed every window. The mistake was in what the policy was doing, and it was visible
from the first pictures to anyone who had read the instruction carefully.

**How to apply:**
- Before judging the first window, write down which objects the instruction names, where each must
  end up, in what order, whether one hand or both act on the same object, and what the head camera
  shows when the task is done.
- For ranking or sorting tasks, decide the order from the pictures (which is biggest, which colour
  comes first) before the policy moves anything.
- Every later window, compare the policy against that parse; a contradiction is a fault even when
  nothing is stuck, and it is worth intervening early and laying out the right route.

**Falsify:** If a policy that visibly contradicts a careful parse still completes the task, the parse
was wrong, not the policy; re-read the instruction.
