---
id: verify-mug-body-straddling-before-closing
scope: global
kind: strategy
title: Verify mug body straddling before closing
applies_when: A mug remains on a counter after empty grasp closures and the commanded
  approach is not reliably reached.
symptom:
- mug
- empty close
- counter
- grasp alignment
- approach drift
- servo miss
- handoff
evidence:
  cells:
  - l30bx-CoffeeSetupMug_s195_ep15
  attempts: 2
  source: learn-l30bx
confidence: probable
related: []
---
**Why:** Repeated closures and a lateral target adjustment failed to acquire the mug. The subsequent approach for a policy grasp also missed. This supports testing an approach that verifies actual finger placement before closure; it does not establish that a particular grasp direction will succeed.

**How to apply:** After an empty mug grasp, reopen and reacquire a view of the mug body and both fingers. Choose an unobstructed approach toward the body, with the handle clear of the finger path. Move in stages and verify that the fingers actually straddle opposite sides of the body below the rim before closing. If the approach misses, resolve that mismatch before another closure or policy handoff. After closing, use a short lift to verify that the mug follows before handing transport back to the policy.

**Falsify:** In a future comparable failure, if visibly confirmed body straddling and a reached approach still produce an empty closure or failed test lift, approach alignment alone is insufficient; investigate grasp height, aperture, or contact geometry instead.
