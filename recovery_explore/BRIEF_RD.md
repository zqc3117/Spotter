### This task is a tabletop, worked by two arms

The scene is a desk: objects sit on a wooden surface, sometimes with a monitor, a
keyboard, a fan or a box in the frame. The instruction names one thing to do with
one or two of those objects. There is no kitchen, no fixture, no door or knob.

**The robot has two arms and they are not interchangeable.** Every step you write
carries `"arm": "left"` or `"arm": "right"`; leave it out and the step goes to the
right arm. The other arm holds still. The telemetry gives you both hands' positions
each window and marks which one your steps drive by default. Before you plan a
repair, decide which hand is the one in trouble — a repair aimed at the idle arm
does nothing and still costs you the intervention.

**The gripper reports the measured finger opening, not a width in millimetres.** 1.0 is
wide open; the fingers meet at about 0.19. An opening at or below 0.20 after a close
is "shut all the way with nothing between the fingers" — the telemetry calls those
out as empty closures and names the hand. A close that stops above that is resting
on something: usually a grasp, sometimes an edge it cannot hold; **the number alone
cannot tell you that something is held.** As on any robot, the way to find out whether a
grasp took is to lift and look: if the object rises with the hand, it is held.

**There is no force sensor on this robot.** Nothing will stop a plan because the
wrist pushed into something. What you get instead is the joint tracking error in
the telemetry: the gap between the joint target the policy commanded and the joints
that came back. A large tracking error with almost no travel means the arm is being
held back by something it is pressing on — the table, the object, its own limits.
That is the closest thing here to "it hit something", and it is worth reading every
window.

### How to tell whether the policy is in trouble

Read the numbers before the image, then use the image to explain them.

- **Travel per chunk, per hand.** A hand approaching its target moves several
  centimetres a chunk. A hand that has arrived and is closing moves under one. A
  hand that has moved under one centimetre for several chunks in a row and has not
  closed on anything is stuck, not working.
- **The opening trace.** Watch it fall after a close. A fall that stops above 0.20 and
  stays there is usually a real grasp. A fall to about 0.19 is the fingers meeting: an
  empty closure, and the telemetry will say so.
- **An empty closure followed by the hand carrying on as if it had the object** is
  the single most common failure on this benchmark, exactly as it is on any other.
  The lift that follows will show nothing coming up.
- **The object's own position in the picture.** Compare the newest frame against
  the previous window's: if the instruction says to pick something up and that
  thing has not left the table after most of the step budget is gone, the approach
  is wrong, not slow.

### Motion is not progress, and the budget is a deadline

This is the mistake to avoid on this benchmark. A policy that has lost the
plot on a tabletop does not freeze — it keeps sweeping the arm back and forth
at 30 cm a chunk, opening and closing on nothing, looking busy in every frame.
Large travel and a cycling gripper are **not** evidence that the task is being
done. Judged on motion alone, every failing episode looks fine until the step
budget runs out and it is too late to do anything.

So judge against the GOAL, not against the activity:

- **Name the goal state in one sentence** from the instruction — the shell is
  10 cm off the table, the three blocks are in a line, the numbers read
  largest-first — and each window ask only whether the scene is closer to it
  than it was last window.
- **Compare the object, not the arm.** Has the thing the instruction names
  moved since the last window? On a multi-step task, has the count of pieces
  correctly placed gone up? If the answer has been no for two or three windows
  while the hand kept moving, the policy is repeating something that does not
  work, and it will keep repeating it.
- **Read the step clock as a deadline.** The telemetry tells you how many
  policy steps are left. Past the halfway mark, "it is still working on it" is
  a prediction, and you should be able to say what it will have achieved by the
  end. If the honest answer is "it will not finish", that is the window to
  intervene — not three windows later when there is no budget left to repair
  anything.
- **A grasp that has been "in progress" for two windows is not in progress.**
  Closing takes a chunk. An opening that hovers between 0.2 and 0.6 across
  several windows while the object stays on the table means the fingers are on
  something they cannot hold, or on nothing.

Intervening late is the same as not intervening. On a short task — many of
these have only four chunks in the whole episode — that means the first window
where you can name what is wrong is usually the only one you get.

### What a repair looks like here

The policy is usually better than you are at the last two centimetres. Your job is
to put the hand somewhere the policy can work from, then give it back:

1. Look at where the hand is and where the object is.
2. Move the open hand to a better line — a `nudge` of a few centimetres, or a small
   `lift` if the hand is pressed into the table.
3. Hand back with `{"op": "policy", "chunks": 2}` and see whether it closes.

Closing the hand yourself is allowed at any time. It usually pays off after a hand-back
has already failed; following your own close with a small `lift` shows whether the
grasp actually holds.

**Cartesian moves are solved by inverse kinematics from wherever the arm is now.**
A target the solver cannot reach stops the plan with `servo_missed`; the next plan
can try a different, reachable target, or open the fingers and hand back. A move that asks the hand to travel into the table
surface will report the same thing, because it is the same situation: the arm
cannot get there.

### What is NOT a failure

- The idle arm sitting motionless at the edge of the frame for the whole episode.
  Most of these tasks are one-handed; the second arm is simply not used.
- A hand passing over an object without touching it on the way to another one.
- The opening resting at 1.0 for many chunks early on — the approach happens with
  the hand open.
- Objects in the scene that the instruction does not name. Several of these desks
  are cluttered on purpose. Track the thing the instruction names and no other.

### The policy on this benchmark -- read this before you decide

The policy here is a video world-action model. Two things follow from that.

**It does not follow the wording of the instruction.** It imitates what these tasks
usually look like; it does not read which object, colour, number or order the
instruction names. When it fails, it is very often doing a clean, plausible version
of the wrong thing: blocks stacked in the wrong order, the wrong digit or button,
the wrong object, the wrong slot. Handing such an episode back after a small fix
does not help -- it redoes the same thing. So when you do fix a *what* or *which
order* error, do that part yourself, all the way to its end state, and hand back
only for steps the policy was already doing correctly.

**Steps are shared, so spend them where they change the outcome.** Every step your ops take comes out
of the same budget the policy needs to finish. But a task left unfinished scores nothing for the parts
left undone: difficulty, a step that needs several ops, or partial progress so far are not reasons to stop
helping, but neither is every pause: a first miss is usually best handed back for the policy to retry once.
When that retry fails too, take the step over and finish it completely -- pick_place to the target, place,
push, press -- one op at a time, checking each result, then hand back. If the same op fails twice, switch
method once; if that also fails, leave the step to the policy. Pass when the policy is progressing.

How the tools have actually done on this benchmark:

| op | result |
| --- | --- |
| `press` | reported "pressed" every time in earlier runs, but that report never checked that the fingertips reached the surface |
| `push` | reported "pushed" every time in earlier runs, even when the push motion failed; the object was never checked |
| `pick_place` | placed 60 of 115; failed as unreachable 24, grasped but not placed 15, closed on nothing 8, slipped 6 |
| `grasp` | holding 17 of 30 |

And "placed" is not the same as "in the goal": several placements landed beside
the basket or slot. Pick a place target well inside the receptacle, not on its rim.

Intervene when:

1. You can name the specific problem in the picture (wrong item, wrong order,
   wrong slot, part left unfinished, the same miss repeated) and say what the goal
   state looks like.
2. Your tools can make progress on it from here -- a grasp, a place, a push, a press,
   a re-staged hand-back. You do not need a plan that finishes the whole task in one
   go: you need the next step that moves the scene toward the goal, and you will see
   each result before choosing the one after.
3. Steps remain. A repair that takes several ops is worth it when the alternative is
   the policy repeating the same failure until the budget runs out.

Inside one episode, do not repeat what already failed: if a target came back
`unreachable`, `closed_empty` or `grasped_not_placed`, do not send the same op at
the same target again -- change the approach (another point, the other arm,
pi0_pick, a push) instead.

When those three hold, act at the first window you can see the problem; the earlier
window leaves more steps for both of you. Misses from earlier runs that met them:

- cloth still unfolded, 20 steps left, the hand withdrawing;
- charger released beside the power strip but never seated;
- the hand moving on after pressing the wrong button.

"It may be withdrawing after a success" is not a reason to pass when the picture
does not show the goal state.
