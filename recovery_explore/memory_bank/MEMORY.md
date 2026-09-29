# recovery_explore memory bank index

<!-- GENERATED FILE — regenerated wholesale by
     `cli/memory_cli.py build-index`. Do not hand-edit: edit the
     leaves under global/ and task_only/, then rebuild. -->

Entries: 77 global.

## Global

### failure

- [A pinch you place by hand on a small round or slender object closes on air or slips within the first chunks of transport](global/a-hand-placed-pinch-on-a-small-round-or-slender-object-slips.md) `verified` — the repair is about getting an avocado, garlic bulb, lime, carrot, fish, potato or similar into the fingers
  - symptom: closed_empty, manual close, slipped during transport, small object, round object, slender object, pinch
- [A tipped mug on the target is not a placement, and a far-off stalled arm is the window to act](global/a-tipped-mug-on-the-target-is-not-a-placement.md) `single-shot` — A mug or cup sits roughly on the destination (drip tray, shelf, plate) but on its side or tilted, while the arm has withdrawn and keeps commanding motion it cannot make
  - symptom: mug tipped, lying on side, drip tray, placed but tilted, arm stalled, withdrawn, near step limit, commanded not travelled
- [Some policies drive the gripper shut from the first step and approach the object with a closed empty hand](global/approach-with-a-shut-empty-gripper.md) `single-shot` — reading the action history at the start of an episode
  - symptom: gripper closed from step 0, approaches with fingers shut, never opens, cannot grasp
- [Check descent completion before closing on a mug](global/check-descent-completion-before-closing-on-a-mug.md) `single-shot` — Retrying a mug grasp near a dispenser after an empty grasp
  - symptom: deeper, descent, misses, target, action, sequence, continues, close, lift
- [An object that has fallen off the work surface cannot be recovered from proprioception and images](global/dropped-out-of-reach-is-terminal.md) `probable` — the object has disappeared from all three cameras after a drop
  - symptom: object gone, cannot find it, fell on the floor, disappeared from view
- [A grasp verified by lift and lateral tests can still drop within the first chunks after handoff](global/grasp-can-survive-verification-and-still-drop-at-handoff.md) `probable` — you verified a grasp with your own actions and are about to hand control back to the policy
  - symptom: drops after handoff, verified then lost, policy loses the object, drops in first chunks
- [Occlusion can conceal persistent faucet execution failure](global/occlusion-can-conceal-persistent-faucet-execution-failure.md) `single-shot` — A fixture control is obscured while repeated substantial motion commands produce no measurable hand movement.
  - symptom: hand, stays, same, pose, many, chunks, despite, repeated, translation, rotation, commands
- [From a mid-grasp state the policy may close on nothing and head for the target anyway](global/policy-can-place-with-an-empty-hand.md) `verified` — the policy is resumed from a state where it has already begun a grasp
  - symptom: empty hand, goes to place without the object, does not retry the grasp, skips to place
- [After the first servo miss or overshoot in an intervention, further Cartesian steps from that region fail the same way](global/stop-cartesian-repair-after-the-first-fault-in-an-intervention.md) `verified` — a step has just reported servo_missed, overshoot or move_clipped and you are deciding the next segment
  - symptom: servo_missed, overshoot, second attempt, repeated fault, drift

### infra

- [Aperture read immediately after a close command has not settled yet](global/a-close-command-needs-settle-steps.md) `single-shot` — reading gripper width right after commanding a close
  - symptom: width looks wrong, read too early, aperture still moving
- [move-to and lift control position only; whatever rotation you accumulate stays](global/move-to-and-lift-do-not-restore-orientation.md) `single-shot` — planning a repair that involves several position commands
  - symptom: wrist drifted, orientation changed, pose looks wrong after moves
- [A long move-to near structure quietly rotates the wrist and burns steps without arriving](global/move-to-can-rotate-and-stall-in-tight-space.md) `probable` — commanding a large move-to inside a cabinet, under a hood, or down into a container
  - symptom: move-to ok false, did not arrive, wrist rotated, steps exploded, stalled
- [Convert montage pixels to native camera coordinates before targeting](global/pixel-targets-need-native-camera-coordinates.md) `single-shot` — A repair uses pixel targets selected from a resized multi-camera montage
  - symptom: pixel target, montage, image scaling, wrong waypoint, move clipped, unprojection

### perception

- [A closed-empty event during transport can be a real grasp; check the object before reopening](global/a-narrow-aperture-can-still-be-carrying.md) `verified` — telemetry reports an empty closure while the arm is already carrying something toward the destination
  - symptom: mug, cup, coffee, empty close, closed on nothing, aperture, narrow, transport, carrying, reopen, replacement grasp, slipped
- [A mug or cup taken by the handle reads a few millimetres of aperture whether or not the handle is inside](global/a-narrow-aperture-on-a-handle-grasp-is-not-a-miss.md) `probable` — the fingers have just closed on a handled object and the width reads under ten millimetres
  - symptom: aperture 6 mm, aperture 7 mm, mug, cup, handle, closed on nothing, narrow width
- [Check the object between the fingers is the named target, not a larger neighbour](global/check-the-object-between-the-fingers-is-the-named-target.md) `single-shot` — A small target shares a shelf or crowded spot with a bigger distractor, and repeated closes, yours or the policy's, keep shutting on air with something centred between the fingers
  - symptom: check, object, between, fingers, named, target, larger, neighbour
- [A closed gripper is not a grasp; only the object leaving its support proves one](global/closed-gripper-is-not-a-grasp.md) `verified` — any time you decide whether a grasp attempt succeeded
  - symptom: false grasp, gripper closed, looks grasped, object did not move, drops after lift, empty close
- [Destination proximity is not proof of completed placement](global/destination-proximity-is-not-placement-proof.md) `single-shot` — An object is visible near its destination while the arm withdraws or stalls elsewhere
  - symptom: repeatedly, declaring, completion, unchanged, object, destination, without, verifying, orientation, alignment
- [Do not infer burner shutdown from an obscured heating surface](global/do-not-infer-burner-shutdown-from-an-obscured-heating-surface.md) `single-shot` — The target burner is covered by cookware and the arm has withdrawn from the stove controls.
  - symptom: glow, visible, around, target, pan, supervisor, assumes, burner, False, without, confirming, control
- [Confirm delivery by checking the source is now empty, not only by looking at the destination](global/empty-source-corroborates-delivery.md) `verified` — deciding whether the object reached the destination container or surface
  - symptom: is it in the pan, delivered or not, destination occluded, cannot see the object
- [Large descent commands producing under a centimetre of travel at a coffee dispenser tray are contact, not a blocked approach](global/little-travel-under-large-descent-commands-at-a-dispenser-tray-is-contact.md) `probable` — a coffee task where the hand is at the tray under the dispenser and the chunk log shows big commanded dz with tiny travel
  - symptom: dispenser, tray, mug under dispenser, little travel, large command, blocked approach, stationary mug
- [Measure knob rotation relative to the fixed panel](global/measure-knob-rotation-relative-to-the-fixed-panel.md) `single-shot` — Assessing rotary control progress through a moving wrist camera.
  - symptom: knob, appears, turn, wrist, view, index, remains, fixed, relative, panel, scale
- [On doors, drawers, knobs, faucets and buttons, an empty closure is not a failure](global/mechanism-tasks-aperture-is-not-a-failure-signal.md) `verified` — the instruction is to open or close a door or drawer, turn a knob, spout or faucet, or press a button, rather than to move a free object
  - symptom: door, drawer, knob, faucet, spout, button, handle, mechanism, open, close, turn, press, empty close, closed on nothing, aperture, gripper open
- [Open gripper and small terminal motion do not establish faucet failure](global/open-gripper-and-small-terminal-motion-do-not-establish-faucet-failure.md) `single-shot` — Monitoring a faucet activation attempt with an open gripper and reduced end-effector travel near the end of the approach.
  - symptom: gripper, aperture, stays, terminal, chunks, travel, only, about
- [The first sight of a neighbouring burner lighting is not yet wrong-control failure; the policy often moves to the named knob within a window](global/the-wrong-burner-lighting-is-often-corrected-by-the-policy-within-a-window.md) `verified` — a stove task where a burner other than the named one has just lit or its knob has turned
  - symptom: wrong burner, neighbouring burner lit, wrong knob, wrong control, rear-right, rear-left, index unchanged
- [Back-projected object points sit on the near surface, about three centimetres short of the graspable centre](global/unprojection-lands-on-the-surface-not-the-centre.md) `probable` — turning a pixel into a world coordinate to aim a grasp
  - symptom: grasp lands short, aimed at the object but missed, unproject offset, alignment off by a few cm
- [Verify both leaves before judging double doors open](global/verify-both-leaves-before-judging-double-doors-open.md) `single-shot` — A task asks to open cabinet doors and one or both leaves are partly occluded.
  - symptom: open, panel, visible, sustained, arm, commands, produce, almost, motion
- [Judge mechanism progress from the mechanism, not from arm motion or a nearby indicator](global/verify-the-mechanism-not-the-arm.md) `probable` — deciding whether a knob, faucet, spout, drawer, door or button task is making progress
  - symptom: knob, stove, burner, faucet, spout, drawer, door, button, no progress, stalled, rotation, small movement, uncertain, cannot tell
- [A steady mid-range aperture supports a grasp but never proves one, and it survives a mid-transport drop](global/width-plateau-cuts-both-ways.md) `verified` — using gripper aperture as evidence for or against a grasp
  - symptom: width plateau, aperture unchanged, thought it was holding, dropped but width stayed, false positive
- [A hand that pulls back from a keypad with the display unchanged is usually re-approaching, not giving up](global/withdrawal-after-a-press-with-the-display-unchanged-is-a-re-approach.md) `verified` — a button task where the hand has just withdrawn ten to forty centimetres after touching the panel and nothing changed
  - symptom: withdrew, display unchanged, START, stop button, keypad, press, not activated, pulled back
- [The wrist view is too close and too mobile to answer whether the object is held](global/wrist-camera-cannot-settle-a-grasp-question.md) `probable` — the third-person views are occluded and you are tempted to judge from the wrist camera
  - symptom: wrist view looks right, object fills the wrist frame, cannot see from outside, parallax

### primitive

- [A long absolute move back to where the grasp was missed is the main source of servo misses and overshoot](global/a-long-absolute-return-move-is-the-main-controller-fault.md) `verified` — planning the first step of a repair after the policy has withdrawn from a missed grasp or a control
  - symptom: servo_missed, overshoot, move_clipped, long move_to, return to missed grasp, xyz target
- [A close issued right after a descent that stopped short lands above the object](global/closing-after-a-descent-that-stopped-short-closes-above-the-object.md) `verified` — a plan has a lift or nudge downward followed by a gripper close, and the descent step reported servo_missed or travelled less than asked
  - symptom: descent stopped short, closed_empty, servo_missed, close above object, lift -0.05, nudge down
- [Descend onto the object from above; sliding in laterally topples whatever is next to it](global/do-not-sweep-in-sideways.md) `probable` — closing the last few centimetres toward an object that has neighbours or sits in a container
  - symptom: knocked it over, toppled the cup, hit the neighbour, object tipped, grasped the side
- [In front of a microwave keypad, under a coffee dispenser and at a stove knob in contact, a two-centimetre correction can travel a metre](global/keypad-and-dispenser-poses-blow-up-cartesian-corrections.md) `verified` — planning any move_to, nudge or lift while the hand is at a keypad, under a dispenser, or seated on a stove knob
  - symptom: overshoot, blow-up, keypad, microwave, dispenser, coffee machine, stove knob, contact pose, travelled 100 cm, controller misbehaving
- [Reseat on a knob after a confirmed empty closure by clearing then descending with fingers open](global/knob-reapproach-after-empty-closure.md) `probable` — a stove or faucet knob task where the fingers demonstrably closed on nothing and the knob has not turned
  - symptom: knob, stove, burner, faucet, empty close, closed on nothing, lost engagement, reseat, reapproach, slipped off
- [A small lift to expose an occluded object faults in constrained regions instead of revealing anything](global/lifting-to-expose-the-object-faults-in-constrained-regions.md) `verified` — the object is hidden by the hand or a fixture and you are tempted to raise or back off a few centimetres to see it
  - symptom: occluded, expose, lift 3 cm, back off, servo_missed, overshoot, cabinet, sink, keypad, dispenser
- [Change one variable per attempt so the next attempt knows what moved the needle](global/one-change-per-attempt.md) `probable` — a repair attempt failed and you are about to try again
  - symptom: tried again and failed, many changes at once, cannot tell what helped
- [Straighten the wrist before opening the fingers, or the object rolls off where you put it](global/place-orientation-decides-whether-it-stays-put.md) `single-shot` — the object is held and you are about to release it on the destination
  - symptom: rolled off, fell over after release, placed flat, not upright
- [Raise before recentering horizontally over an upright object](global/topple-check-before-horizontal-recentering.md) `single-shot` — the fingers are open and hovering at object height and you want to shift the arm sideways
  - symptom: recentering knocked it over, upright object fell, cup tipped while aligning

### strategy

- [A retreat after a policy hand-back throws away the re-approach the policy just made](global/a-retreat-after-a-hand-back-throws-away-the-policys-re-approach.md) `verified` — a hand-back has just ended with the fingers near or on the object and nothing decided, and you are choosing between lift, another chunk, and retreat
  - symptom: retreat, hand-back, policy step, re-approach undone, near the object, undecided, same recipe next window
- [After your own repair failed, a later intervention with the same recipe spends the allowance without changing the outcome](global/a-second-intervention-with-the-same-recipe-spends-the-allowance-for-nothing.md) `verified` — a window shows the same failure you already tried to repair earlier in this episode
  - symptom: repair did not hold, previous recovery failed, same miss, second intervention, third intervention, allowance
- [A failed repair ended by returning to the takeover pose and handing back beats a second correction from where the hand is](global/after-a-failed-repair-hand-back-the-takeover-state-not-the-corrected-pose.md) `verified` — the first attempt of an intervention did not produce the object in hand or the control moved, and you are choosing what to do next
  - symptom: repair failed, not fixed, hand back, takeover pose, retreat, second correction, untouched state
- [Allow a brief slowdown during drawer handle grasp closure](global/allow-a-brief-slowdown-during-drawer-handle-grasp-closure.md) `single-shot` — A drawer-opening policy slows briefly as the gripper closes after approaching the handle.
  - symptom: end-effector, travel, drops, one, action, chunk, gripper, aperture, decreases, nonzero, width
- [Do not escalate an unexplained Cartesian miss into wrist reorientation](global/avoid-wrist-reorientation-after-unexplained-cartesian-misses.md) `single-shot` — A fixture remains unchanged, the hand is stalled away from its control, and a Cartesian correction misses without contact.
  - symptom: faucet running, remote stall, servo miss, zero force, wrist reorientation, overshoot, failed recovery
- [Brief restricted motion near a faucet can precede success](global/brief-restricted-motion-near-a-faucet-can-precede-success.md) `single-shot` — A faucet policy has approached the lever and first shows reduced end-effector travel near contact.
  - symptom: water, remains, running, substantial, commands, produce, little, travel, hand, obscures, lever
- [Center the pinch on shelved oval produce](global/center-the-pinch-on-shelved-oval-produce.md) `single-shot` — Oval produce remains on a cabinet shelf after repeated empty closes or slips during pickup.
  - symptom: oval produce, avocado, cabinet shelf, empty close, missed grasp, slipping, retained grasp
- [Change wrist configuration when a contact-free descent reverses](global/change-wrist-configuration-when-free-space-descent-reverses.md) `single-shot` — Picking a small object from a recessed surface when downward corrections stop short or move upward without measured contact
  - symptom: sink pickup, descent reversal, zero force, servo miss, empty close, wrist configuration
- [When a step reports servo_missed, the plan did not happen; do not escalate the offset](global/check-achieved-motion-before-escalating.md) `probable` — a plan segment aborted with servo_missed, or the executed log shows far less motion than commanded
  - symptom: servo_missed, did not reach target, moved less, residual error, descend, offset, repeat, same correction, escalate
- [Check pinch orientation for elongated objects in bowls](global/check-pinch-orientation-for-elongated-objects-in-bowls.md) `single-shot` — An elongated object remains in a bowl or shallow dish after repeated empty grasps using position corrections.
  - symptom: elongated object, bowl, rim clearance, empty close, repeated miss, pinch orientation, sweet potato
- [Correct lateral alignment when a button press stalls](global/correct-lateral-alignment-when-a-button-press-stalls.md) `single-shot` — A hand repeatedly commands forward motion at a button while the control stays unchanged and the wrist view shows lateral misalignment.
  - symptom: near-zero, end-effector, travel, despite, continued, pressing, commands, tip, beside, intended, button
- [Combine a corrected approach with an explicit grasp check](global/corrected-approach-then-verified-grasp.md) `single-shot` — A free-object grasp missed and the policy withdrew with empty fingers
  - symptom: object, remains, source, closed, hand, leaves
- [Corrections sent while the hand is on a faucet do not engage the lever; the outcomes seen were the policy finishing on its own or nothing working](global/corrections-from-the-contact-pose-at-a-faucet-do-not-engage-the-lever.md) `verified` — a faucet or spout task where the hand has sat at the control for several windows without a visible change
  - symptom: faucet, spout, lever, no visible change, hand at control, three windows, water still running, minimal travel
- [Count stalled observation windows, not overlapping history chunks](global/count-stall-windows-not-overlapping-chunks.md) `probable` — Supervising fixture manipulation with low hand travel and uncertain control-state changes
  - symptom: several, consecutive, history, rows, show, large, motion, commands, little, translation
- [Count ineffective-contact windows separately from approach](global/count-stalls-from-contact-not-approach.md) `verified` — A policy approaches a fixture and then slows near its control.
  - symptom: fixture, appears, unchanged, during, approach, followed, reduced, hand, travel, gripper, repositioning
- [Distinguish a clipped waypoint from servo drift before re-staging](global/distinguish-clipped-waypoints-from-servo-drift.md) `single-shot` — A long move toward a previously visited approach point stops early
  - symptom: hand, stops, short, requested, target, repair, risks, repeatedly, returning, starting, pose
- [Route a carried object around sinks and open containers rather than across them](global/do-not-carry-over-an-open-container.md) `single-shot` — moving a held object horizontally toward the destination
  - symptom: dropped into the sink, lost in the basin, fell into a container on the way
- [Repair the moment of the missed grasp, not the retreated pose the policy left you in](global/fix-at-the-miss-not-where-the-arm-ended-up.md) `probable` — the telemetry shows an empty close and the arm has since moved away
  - symptom: arm retreated, too far to fix, budget runs out, cannot reach the object
- [Restored approach motion does not establish fixture engagement](global/fixture-approach-motion-is-not-engagement.md) `single-shot` — A fixture repair repositions the hand and the policy resumes moving
  - symptom: hand, reaches, control, region, requested, fixture, state, remains, unchanged
- [Grasp dispenser body before policy handoff](global/grasp-dispenser-body-before-policy-handoff.md) `single-shot` — An upright pump dispenser stays on the counter through repeated empty closures and lift attempts.
  - symptom: pump dispenser, empty close, missed grasp, body alignment, lift test, policy handoff
- [Object held and end-effector already over the destination: let the policy finish](global/grasped-and-hovering-means-do-not-intervene.md) `verified` — the object is visibly in the fingers and the arm is above the place named in the instruction
  - symptom: already holding, above target, should I intervene, almost done
- [A wrist left rotated at handoff can derail the policy's transport, not only drop the object](global/handoff-pose-matters-not-just-position.md) `single-shot` — your repair rotated the end-effector and you are about to hand control back
  - symptom: policy went sideways after handoff, transport derailed, odd trajectory after intervention
- [Keep a caught round object until it is clear of the source](global/keep-a-caught-round-object-until-it-is-clear-of-the-source.md) `single-shot` — A repair grasp has finally caught a small round or rolling object (fruit, ball, egg) inside a container such as a pan, bowl or pot, after the policy repeatedly closed on nothing there, and you are deciding when to hand control back.
  - symptom: round object, lime, fruit in pan, repeated empty close, caught then lost, hand back, squeezed out, regrasp loop, slipped, transport
- [Lift-test a stalled closed hand before opening it; near the object the stall can be a grasp the policy cannot carry out](global/lift-test-a-stalled-closed-hand-before-opening-it.md) `probable` — the hand has stopped travelling with the fingers shut close to the object, inside a cabinet or a sink, and the aperture is narrow but not zero
  - symptom: stalled, fingers shut, narrow aperture, cabinet, sink, closed empty unclear, reopen, lift test, mushroom
- [Pinch solid material on ring-shaped objects](global/pinch-solid-material-on-ring-shaped-objects.md) `single-shot` — A donut or another ring-shaped object remains on its support after repeated empty grasp attempts.
  - symptom: donut, ring-shaped object, empty close, grasp height, rim, cabinet, repeated miss
- [Placement contact can resemble a stall](global/placement-contact-can-resemble-a-stall.md) `single-shot` — A carried object has reached the destination and the gripper is beginning to open.
  - symptom: downward, hand, travel, nearly, stops, despite, substantial, command
- [Preserve policy progress after a recovery](global/preserve-progress-after-recovery.md) `single-shot` — A previous grasp failure has been repaired and the policy is transporting or placing the object.
  - symptom: historical, empty, closures, smaller, travelled-than-commanded, distances, can, suggest, failure, despite, current, visual
- [Reacquire visible support before regrasping slender objects](global/reacquire-visible-support-before-regrasping-slender-objects.md) `single-shot` — A slender object slips during extraction from an enclosed fixture and its landing position becomes obscured.
  - symptom: slender object, carrot, slipped, empty grasp, obscured landing, microwave, regrasp
- [Reopening an empty gripper can restart a stalled approach without Cartesian correction](global/reopening-an-empty-gripper-can-break-a-policy-stall.md) `single-shot` — A free-object policy remains closed on air near the source, especially when Cartesian repair has proved unreliable
  - symptom: empty close, closed gripper, source unchanged, stalled approach, oscillation, controller overshoot
- [Recheck target depth when local grasp corrections repeat the miss](global/repeated-empty-grasps-recheck-target-depth.md) `single-shot` — A free object remains on its support after empty grasps and small corrections around the recorded miss do not resolve them.
  - symptom: empty close, repeated miss, cabinet, target depth, servo missed, occlusion, approach reset
- [When the fingers keep catching a neighbouring object, slide the hover point along the target's long axis](global/shift-along-the-object-axis-to-avoid-a-neighbour.md) `single-shot` — repeated attempts pick up or collide with an object adjacent to the target
  - symptom: grabs the wrong object, neighbour in the way, distractor, fingers hit the next item
- [Slow open-gripper approach is not a missed grasp](global/slow-open-gripper-approach-is-not-a-missed-grasp.md) `probable` — The hand is approaching a free object with open fingers and reduced but nonzero travel.
  - symptom: several, chunks, show, much, less, hand, travel, commanded, making, continuing, approach, resemble
- [Check whether the object is already on the destination before treating a still scene as a stall](global/target-may-already-be-delivered.md) `verified` — the object has not moved for several windows and the arm is wandering
  - symptom: object not moving, looks stalled, is it done, already delivered
- [Three failed repairs on one trajectory means stop and let the policy run out](global/three-strikes-then-hand-back.md) `probable` — you have repaired, watched it fail, and repaired again on the same episode
  - symptom: keeps failing, should I try again, give up, wasting budget
- [Unstick a blocked knob press by restaging on the knob, not beside it](global/unstick-a-blocked-knob-press-by-restaging-on-the-knob-not-beside-it.md) `single-shot` — A stove or appliance knob task where the policy has pushed for several windows with commanded displacement growing chunk over chunk, travel near zero, fingers shut on nothing, and the named burner still lit
  - symptom: unstick, blocked, knob, press, restaging, knob, beside
- [Verify mug body straddling before closing](global/verify-mug-body-straddling-before-closing.md) `probable` — A mug remains on a counter after empty grasp closures and the commanded approach is not reliably reached.
  - symptom: mug, empty close, counter, grasp alignment, approach drift, servo miss, handoff
- [Verify target isolation before closing in clutter](global/verify-target-isolation-before-closing-in-clutter.md) `single-shot` — A small elongated target shares a shelf with a larger distractor and grasp attempts close empty or pick up the distractor.
  - symptom: fish, clutter, cabinet shelf, empty close, wrong object, distractor, grasp alignment, handoff
