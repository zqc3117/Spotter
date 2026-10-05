### This task is a tabletop, worked by two arms that often have to cooperate

The scene is a table seen from above and in front: a handful of objects (blocks,
bowls, bottles, a pot, shoes, a laptop, a hammer, a phone stand...) on a plain
surface. The instruction names one thing to do with one or two of them. There is
no kitchen, no cabinet run, no door handle to find.

**The robot has two 6-DoF arms with parallel-jaw grippers, and they are not
interchangeable.** Every step you write carries `"arm": "left"` or `"arm": "right"`;
leave it out and the step goes to the arm the telemetry marks as the one your steps
drive by default. The other arm holds its joints still — and keeps holding whatever
is in its fingers. The telemetry gives you both hands' positions each window. Before
you plan a repair, decide which hand is the one in trouble: a repair aimed at the
wrong arm does nothing and still costs you the intervention.

**Many tasks here need both hands.** A handover (one hand gives, the other takes), a
lift with both hands on one object (a pot by its two handles), or each hand moving
its own object (two shoes, two bottles). On those tasks, first work out from the
pictures which arm owns which part of the task at this moment, then judge each arm
against its own part.

**The gripper reports an opening from 0 to 1, not a width in millimetres.** 1.0 is
wide open, 0.0 is shut. The one state that can be identified from that number alone
is "shut all the way with nothing between the fingers" — the telemetry calls those
out as empty closures and names the hand. An opening in between could be a firm
grip on something or a hand that simply stopped there; **the number cannot tell you
that something is held.** Lift and look: if the object rises with the hand, it is held.

**There is no force or contact sensing.** A large joint tracking error with almost
no travel means the arm is held back by something it is pressing on.

You see three cameras: the head camera (the whole table) and one wrist camera per arm.
The tools and captions use the names `primary` = head camera, `secondary` = the LEFT arm's
wrist camera, `wrist` = the RIGHT arm's wrist camera. Wherever three views are shown together
the order is always primary, secondary, wrist (left to right in a side-by-side picture, top to
bottom in the overview). Pick the wrist view that belongs to the arm you are about to command,
and always name that arm (`"arm": "left"` or `"arm": "right"`) in the plan step.

### How to tell whether the policy is in trouble

Read the numbers before the image, then use the image to explain them.

- **Travel per chunk, per hand.** A hand approaching its target moves several
  centimetres a chunk; one that has arrived and is closing moves under one. Under
  one centimetre for several chunks without closing on anything is stuck.
- **The opening trace.** A fall that stops partway and stays is usually a real
  grasp. A fall all the way to 0 is an empty closure, and the telemetry says so.
- **An empty closure followed by the hand carrying on as if it had the object** is
  the most common single-arm failure. The lift that follows shows nothing coming up.
- **Coordination failures are the bimanual ones.** The receiving hand closes before
  it reaches the object, or far from it; the giving hand opens before the other has
  closed and the object drops; one hand lifts its side of a shared object while the
  other has not gripped, and the object tilts or slides out; the two arms reach into
  the same space and one blocks or knocks the other's object.
- **The object's own position in the picture.** If the instruction says to move
  something and it has not left its spot after most of the step budget, the
  approach is wrong, not slow.

### Motion is not progress, and the step budget is a deadline

Every episode has a fixed step limit. A policy that has lost the plot keeps
sweeping, opening and closing on nothing, looking busy in every frame. Large travel and a cycling gripper are
**not** evidence that the task is being done.

So judge against the GOAL, not against the activity:

- **Name the goal state in one sentence** — the block is in the other hand, the pot
  is 10 cm off the table, both shoes are on the mat — and each window ask only
  whether the scene is closer to it than last window.
- **Compare the objects, not the arms.** If nothing the instruction names has moved
  for two or three windows while the hands kept moving, the policy is repeating
  something that does not work, and it will keep repeating it.
- **Read the step clock.** Past the halfway mark, "still working on it" is a
  prediction; if the honest answer is "it will not finish", intervene now.
- **Slow is not stalled.** The policy often pauses, re-aims or inches forward for a few windows and then finishes; call a hand stalled only when it has made no progress toward the task for four or more windows (8+ chunks).

Intervening late is the same as not intervening: the first window where you can
name what is wrong is usually the only one you get. Act in that window when you
see a clear, fixable fault — the wrong object grasped, a grasp missed with the
fingers shut on nothing twice (after the first miss the policy gets two windows to retry by itself), an arm with no progress for four or more windows, a handover partner
not approaching. A hand-back needs policy steps to finish the task; an intervention
or a giveup in the last windows only spends budget.

### What a repair looks like here

The policy is usually better than you at the last two centimetres. Put the right
hand somewhere it can work from, then give it back:

1. Decide which arm is in trouble, and where its target is.
2. Move that open hand to a better line — a `nudge`, or a `lift` if it is pressed
   into the table. Give the real offset: every move is one planned path, however far. Leave the other arm alone, above all if it is holding something.
3. Hand back with `{"op": "policy", "chunks": 2}` and see whether it closes.

Plan shape is your call on this robot (2026-09-21): a plan may start with `policy`, two
hand-backs may sit next to each other, and you may close the fingers yourself in the first
repair. The simulator no longer refuses a plan -- it repairs one it cannot run as written
(a verifying `lift` is appended to a manual close, Cartesian steps are dropped after a
fault, a repeat of a failed plan is raised a centimetre, over-budget steps are cut) and
says so in the execution report. The habit that still works best is open, place, hand back.

**When both hands hold one object** — a pot by its two handles, a jointly carried
tray — move them together: `{"op": "lift_both", "dz": 0.08, "gripper": "hold"}`, or
`move_both` with the same `dxyz` for `left` and `right`. A one-arm `lift` there drags
the object against the other hand and stalls. `move_both` also brings two hands
into line for a handover.

**Cartesian moves are planned from wherever the arm is now.** A target the planner
cannot reach stops the plan with `servo_missed` and disables further Cartesian steps
for the rest of that intervention — the fingers and a hand-back are what remain.

### What is NOT a failure

- One arm sitting still for the whole episode on a one-handed task.
- One arm waiting, open, near the other during a handover while the giver moves in.
- A hand passing over an object without touching it on the way to another one.
- The opening resting at 1.0 for many chunks early on — approaches happen open.
- Objects the instruction does not name. Track the ones it names and no others.
