---
id: verify-the-mechanism-not-the-arm
scope: global
kind: perception
title: Judge mechanism progress from the mechanism, not from arm motion or a nearby indicator
applies_when: deciding whether a knob, faucet, spout, drawer, door or button task is making progress
symptom: [knob, stove, burner, faucet, spout, drawer, door, button, no progress, stalled, rotation, small movement, uncertain, cannot tell]
evidence:
  cells: [TurnOnStove_s195_ep10, TurnOffSinkFaucet_s195_ep19, TurnSinkSpout_s195_ep18, TurnOnSinkFaucet_s195_ep16]
  attempts: 4
  source: mech16-round
confidence: probable
related: [mechanism-tasks-aperture-is-not-a-failure-signal, three-strikes-then-hand-back]
---
On these tasks the end-effector barely moves even when the repair is working, so
displacement tells you almost nothing, and the thing you can actually see —
a flame, running water, a lit panel — may belong to a different control.

**Why:**
In `TurnOnStove_s195_ep10` the judge returned ok ten windows running, each time
citing a nonzero aperture as evidence of engagement and a visible flame as
evidence of progress. The flame was the **wrong burner**: window 10 described a
flame farther right with the nearer burner dark, window 14 described the front
right burner as unlit. Both arms failed. The judge was not short of information;
it kept reading the two weakest signals available.

**How to apply:**
- Name the specific control the instruction asks for, and track that one. A
  neighbouring burner, a second door leaf or another tap is not evidence.
- Look for the control's own displacement between adjacent third-person frames:
  knob index mark rotated, drawer face further out, door edge swung.
- When you cannot see the control's state, say so and pass. Repeatedly citing the
  same aperture number as reassurance is not verification; it is the absence of it.
- Alternating rotation directions across windows with tiny end-effector movement
  is a sign the arm is not engaged at all, not that it is nearly there.

**Falsify:**
A task whose only observable is a downstream indicator (a machine that only shows
a status light) would force indicator-based judgement; then say explicitly that
the indicator is the evidence, and confirm it belongs to the requested control.

**Related:** [[mechanism-tasks-aperture-is-not-a-failure-signal]] [[three-strikes-then-hand-back]]
