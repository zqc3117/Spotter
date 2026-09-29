---
id: a-narrow-aperture-can-still-be-carrying
scope: global
kind: perception
title: A closed-empty event during transport can be a real grasp; check the object
  before reopening
applies_when: telemetry reports an empty closure while the arm is already carrying
  something toward the destination
symptom:
- mug
- cup
- coffee
- empty close
- closed on nothing
- aperture
- narrow
- transport
- carrying
- reopen
- replacement grasp
- slipped
evidence:
  cells:
  - CoffeeServeMug_s195_ep13
  - CoffeeServeMug_s195_ep12
  - CoffeeSetupMug_s195_ep10
  attempts: 2
  source: mech16-round
confidence: verified
related:
- closed-gripper-is-not-a-grasp
- width-plateau-cuts-both-ways
- one-change-per-attempt
---
The aperture threshold that detects an empty hand also fires on a thin-walled
object gripped near its rim. Reopening on that evidence throws away a grasp that
was working.

**Why:**
`CoffeeServeMug_s195_ep13` is the clean counter-example: the **successful control
arm** showed the same signature the treatment arm treated as proof of failure —
about 9.2 mm aperture at steps 144 to 160 plus a telemetry event labelled
closed-empty — while third-person images show the red mug travelling from under
the dispenser to the serving counter. The judge reopened, attempted replacement
grasps, and the episode failed. Control succeeded; treatment did not. The same
episode pair also broke `CoffeeSetupMug_s195_ep19`.

**How to apply:**
- Treat an aperture-derived empty-grasp event as a **hypothesis**, not a verdict,
  whenever the arm is mid-transport rather than still at the source.
- Confirm with the object: does it move with the hand between adjacent
  third-person frames, and is it away from where it started.
- If the object is travelling with the hand, leave it alone however narrow the
  aperture reads. Reopen only when you can see the object is not with the hand.
- When the images are inconclusive and continuing is safe, pass the window. The
  policy is mid-transport; one more window costs little and tells you a lot.

**Falsify:**
A rigid object that cannot be gripped below 12 mm would make the threshold
trustworthy again. The rule is about thin-walled or rim-gripped objects; check the
object's wall thickness before extending it.

**Related:** [[closed-gripper-is-not-a-grasp]] [[width-plateau-cuts-both-ways]]
