---
id: knob-reapproach-after-empty-closure
scope: global
kind: primitive
title: Reseat on a knob after a confirmed empty closure by clearing then descending with fingers open
applies_when: a stove or faucet knob task where the fingers demonstrably closed on nothing and the knob has not turned
symptom: [knob, stove, burner, faucet, empty close, closed on nothing, lost engagement, reseat, reapproach, slipped off]
evidence:
  cells: [TurnOffStove_s195_ep18, TurnOffStove_s195_ep19]
  attempts: 2
  source: mech16-round
confidence: probable
related: [mechanism-tasks-aperture-is-not-a-failure-signal, verify-the-mechanism-not-the-arm, do-not-sweep-in-sideways]
---
This is the one mechanism repair that has actually rescued episodes: both
`TurnOffStove` cells went control-fail to treatment-success with the same
four-step shape.

**Why:**
Unlike a door, a knob does need the fingers on it while it turns, so a confirmed
empty closure there really is lost engagement. The repair that worked was a
reseat, not a bigger rotation: open the fingers, take a clearance waypoint about
3 cm above the pose where contact was last plausible, descend onto it with the
fingers open, close. Four actions, no contact-force spikes, then hand back.

**How to apply:**
- Only after the aperture confirms the fingers are shut on nothing **and** the
  knob has not moved. An empty closure with a turned knob needs no repair.
- Clear first, then descend. Coming in sideways at knob height catches the panel;
  see [[do-not-sweep-in-sideways]].
- Pick the clearance direction and target from the current geometry in the images,
  not from coordinates copied out of an earlier episode.
- After handing back, watch for a new empty-closure event and for visible knob
  rotation. A nonzero aperture alone does not prove you are engaged, and a
  successful reseat does not prove the burner changed state.

**Falsify:**
Two cells, one task. If reseating fails on faucet handles or on a knob that must
be pushed in before turning, the recipe is stove-knob-specific and should be
narrowed.

**Related:** [[mechanism-tasks-aperture-is-not-a-failure-signal]] [[verify-the-mechanism-not-the-arm]]
