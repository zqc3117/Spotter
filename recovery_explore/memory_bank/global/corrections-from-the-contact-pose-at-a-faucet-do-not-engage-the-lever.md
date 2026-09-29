---
id: corrections-from-the-contact-pose-at-a-faucet-do-not-engage-the-lever
scope: global
kind: strategy
title: Corrections sent while the hand is on a faucet do not engage the lever; the outcomes seen were the policy finishing on its own or nothing working
applies_when: a faucet or spout task where the hand has sat at the control for several windows without a visible change
symptom: [faucet, spout, lever, no visible change, hand at control, three windows, water still running, minimal travel]
evidence:
  cells: [g120r-TurnOffSinkFaucet_s195_ep11, g120rpi-TurnOffSinkFaucet_s195_ep12, g120rpi-TurnOffSinkFaucet_s195_ep11, g120r-TurnOnSinkFaucet_s195_ep15, g120rpi-TurnOnSinkFaucet_s195_ep15]
  attempts: 5
  source: campaign-2026-09-16
confidence: verified
related: []
---
Corrections sent while the hand is on a faucet do not engage the lever; the outcomes seen were the policy finishing on its own or nothing working.

**Why:**
Height, lateral and pixel-targeted corrections from the contact pose all ended in empty closures, servo misses or drift with zero contact force; the lever is behind or under the fingers and the rear of the sink constrains the arm. Two of these episodes were succeeding and were broken only in the run that committed the correction.

**How to apply:**
- While the hand is on the faucet, pass, however many windows it has been there.
- Act only after the policy has left the control with the fixture unchanged, and then restage from above the faucet rather than from the contact pose.

**Falsify:**
A faucet correction from the contact pose that produced a visible lever change.
