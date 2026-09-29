### This task operates a fixture, not a free object

The instruction asks you to move part of the kitchen itself — a door, a drawer, a
knob, a spout, a button. **Nothing has to end up in the hand.** The door stays
open after you let go, the knob stays turned, the button stays pressed.

**So the gripper tells you almost nothing here.** An empty closure, a wide-open
gripper, the fingers letting go and the arm backing off — on a pick-and-place task
each of those is evidence of failure, and on this task each of them is ordinary
technique. The fingers do not have to hold a handle for a door to keep swinging;
an empty closure and a fixture that is moving can happen at the same moment.

**The one reliable test is whether the fixture itself changed.** Compare
adjacent frames from whichever row shows the control — for a door, a drawer or a
spout that is a third-person row; for a knob, a button or a keypad it is usually
the wrist row, which looks straight at the control while the third-person cameras
see the hand covering it — and look at the thing the instruction names:

- the door's edge has swung relative to the cabinet frame, or it has not
- the drawer face is further out from the unit, or it is not
- the knob's index mark has rotated, the lever has tilted, the button is depressed
- water is running from the spout, or it is not

**The wrist camera rolls with the wrist.** A knob that seems to turn while the
whole panel turns with it has not turned. Read the index mark against the
panel's own fixed scale, never against the edge of the frame; when the
markings are hidden, the rotation is unknown, not confirmed.

**Track the control the instruction names, and no other.** A flame on a
neighbouring burner, the second leaf of a double door, another tap — none of those
are evidence. A flame on the wrong burner looks like progress and is not.

**End-effector displacement is a weak signal on these tasks — in both
directions.** Turning a knob or pressing a button moves the hand by millimetres,
so `moved_cm` near zero *while the control is changing* is normal and is not a
stall. But near-zero travel with the control unchanged, window after window,
while the policy keeps issuing commands, is the hand not engaged with it — that
is the failure mode on these tasks, and it does not announce itself with a big
number. An arm making large confident movements while the fixture never changes
is the other one.

### What justifies intervening

- **Not engaged**: the fingers are shut on nothing *and* the fixture has not
  changed across several windows. On a knob or a lever this is real: unlike a
  door, these need the hand on them while they move
- **Gave up on the control**: the hand was at the control, the fixture did not
  change, and the policy has since backed away from it or wandered off to work
  somewhere else. That is the policy having tried and failed. While the hand is
  still on the control and pressing, it has not failed yet, however many windows
  it has been there: pressing a lever, turning a knob or pushing a button produces
  almost no end-effector travel while it is happening, and some policies take
  many windows to complete a press
- **Wrong control**: the arm is working on a different knob, tap or door leaf than
  the instruction names, and is still on it a window later. The first sight of a
  neighbouring burner lighting or the wrong leaf moving is not yet the signal: the
  policy often moves to the named control on its own within the next window, and
  a repair sent at that moment takes over an arm that was about to correct itself
- **Parked short and left**: the policy pushed on nothing a few millimetres from
  the control, the control never changed, and it has now withdrawn or moved to
  re-approach elsewhere. On a knob or a button this is the common way to fail.
  One withdrawal is not it: a hand that pulls back from a button or a lever with
  the fixture unchanged and then turns back toward it is re-approaching, which is
  how these policies press. Act when the withdrawal is followed by work somewhere
  else, or when the same withdraw-and-return has now happened twice with nothing
  changed
- **Alternating without progress**: rotation commands flipping direction window
  after window while the fixture stays put — the hand is not engaged at all

### What does not justify intervening

- an empty closure, on its own, on any of these tasks
- the gripper being open, or the hand releasing and repositioning
- `moved_cm` near zero while the hand is on a knob or button *and the control
  is still changing*
- the fixture being partly open when the task is not finished yet
- not being able to see the fixture's state in the third-person rows — check
  the wrist row first; only when no row shows the control, say so and pass
- reduced travel with the hand at the control, for any number of windows: that
  is contact, and contact is not failure. The signal to act on is the policy
  leaving the control with the fixture unchanged, or working the wrong control

The two lists are ordered: a justification that is met is not cancelled by an
item on this list. "I cannot confirm the control changed" is a reason to look
harder at every row, not a reason to pass a hand that has not moved for three
windows.

**When the fixture has already reached the state the instruction asks for, do not
touch it.** Several of these tasks succeed early and then the arm simply withdraws;
an intervention there can undo a finished task. But only once you have seen it
reach that state: an indicator you cannot see — a burner under a pan, a lamp
behind the hand — proves nothing either way, and withdrawal with empty fingers is
not completion. Read the control itself.

### Repairing a fixture

The repair section of this brief is written for grasps. On a fixture the usual
faults, and the fixes that have worked, are:

- **Button not going down**: the tip is beside the button. The wrist row shows
  this; the third-person rows hide it behind the hand. Back off two or three
  centimetres, shift toward the button, press again along the same axis. Zero
  contact force after the press means it never touched
- **Knob not turning**: the fingers are not on it. Come in from above, close on
  the knob, then `rotate`; a rotation with the fingers open turns nothing

Stop when the hand is somewhere the policy can continue from, and report whether
the fixture actually changed — not whether your plan completed.
