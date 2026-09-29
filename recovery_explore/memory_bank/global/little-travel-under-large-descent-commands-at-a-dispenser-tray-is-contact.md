---
id: little-travel-under-large-descent-commands-at-a-dispenser-tray-is-contact
scope: global
kind: perception
title: Large descent commands producing under a centimetre of travel at a coffee dispenser tray are contact, not a blocked approach
applies_when: a coffee task where the hand is at the tray under the dispenser and the chunk log shows big commanded dz with tiny travel
symptom: [dispenser, tray, mug under dispenser, little travel, large command, blocked approach, stationary mug]
evidence:
  cells: [g120r-CoffeeServeMug_s195_ep15, g120rpi-CoffeeServeMug_s195_ep15, g120n-CoffeeServeMug_s195_ep15]
  attempts: 3
  source: campaign-2026-09-16
confidence: probable
related: []
---
Large descent commands producing under a centimetre of travel at a coffee dispenser tray are contact, not a blocked approach.

**Why:**
The tray and the machine housing stop the hand; the policy keeps commanding downward while it settles the grasp, and finishes on its own. Every repair attempted from that pose either missed or blew up by a metre, and the runs that committed those repairs failed a task the policy was completing.

**How to apply:**
- Treat this pattern at a dispenser as contact and pass the window.
- Do not send Cartesian moves from under the dispenser; it is a blow-up region.

**Falsify:**
A dispenser-tray stall of this kind that ended with the mug still on the tray and the policy withdrawn.
