### This task moves a free object

**The one reliable test is whether the object moves with the gripper.** Compare
adjacent third-person frames:

- the arm clearly moved and the object stayed put relative to the counter → not held
- object and gripper kept their relative position while both translated → held
- the object left its support and travels through the air with the arm → held

**"Object still, arm moving" is ambiguous until you say where the object is.**
Still at the **source** (in the cabinet, in the sink, on the shelf) → the grasp
missed or the object was dropped, a failure. Still at the **destination** → the
task may be done; leave it alone. Do not assume a motionless object was placed.

**A placement is not finished while the hand is still over the object.** Once
the fingers have opened and the object is resting where the instruction asked,
the hand has to leave. If the policy releases and then parks over the object,
window after window, the task is not done: back the hand off — a `lift` of
several centimetres, then a move away from the object — and hand back. Do not
touch the object itself.

**A mug or cup taken by the handle reads a narrow aperture either way.** On CoffeeSetupMug and CoffeeServeMug a close that stops at 6–12 mm is the handle between the fingers, and only a close to under 4 mm counts as empty (the telemetry uses that threshold for these two tasks). Do not retreat from a 6–12 mm close on a mug; let the policy lift, and judge from whether the mug leaves the counter. The
fingers close to a few millimetres around a handle whether or not the handle is
inside them, so a small width right after the close on a handled object is not the
miss it is on a bulk object. Wait for the lift: a mug that rises with the hand was
held. Calling a miss on the number alone, and reaching in to regrasp, is how a
run that was carrying its mug loses it.

**Your own close on a small round or slender object is the weak link.** Avocados,
garlic, limes, carrots, fish: a pinch you place by hand on these closes on air or
holds for a moment and slips within the first chunks of transport, far more often
than the policy's own grasp does. When the repair is about getting such an object
into the hand, put the open fingers above it on a good line and let the policy make
the close; if you close yourself, lift and look before you believe it.

**Judge the state now, not the history.** An empty closure earlier in the episode
is history; if the object is now sitting at the destination while the hand
withdraws, the placement stands and the earlier event is no reason to regrasp.
After a release, watch that the object stays upright as the hand leaves — a
container that tips as the fingers open is the failure to catch.

### What justifies intervening

- **Dropped**: it travelled with the gripper, then came to rest while the arm kept moving
- **Closed on nothing**: the gripper closed and the arm lifted, and the object never left its support
- **Stalled**: several consecutive chunks of near-zero commanded displacement and an end-effector that does not move
- **No progress**: across two or more windows the object has not left the source
- **Holding but not delivering**: it left the source with the gripper, then the arm mills around instead of heading for the destination. Push the end-effector toward the destination; do not touch the gripper
- **Wrong region**: the arm shuttles somewhere the instruction never mentions
- **Placement pose**: held in an orientation that will not stay put once released, e.g. a cylinder on its side. Fix it with `rotate` **before** opening the fingers

**One missed grasp is not yet a failure.** After closing on nothing the policy
usually reopens and tries again on its own, and most of those second tries land.
The intervention that breaks a run is the one sent at the first empty closure:
it takes over an arm that was about to retry. Wait through the first miss. Act
when the policy has missed twice, or when after a miss it withdraws and heads
away from the source with the fingers shut and does not turn back. And when you
do act and the repair does not take, the next intervention has to differ in kind;
the same open-place-hand-back sent three times in a row is what emptied the
allowance on the runs that lost these cells.

**The first two have a precondition: the policy must have attempted a grasp.**
An approach legitimately takes several windows. Read the **grasp attempt list**
in the action history: if it is empty, the approach has not finished and there
is nothing to diagnose. An entry reading "closed all the way, nothing between
the fingers" means the grasp missed. Calling "no progress" before the policy
has reached the object is the classic misjudgement here.

**A missed grasp is the moment to act, and waiting costs you the chance.** The
typical miss has the gripper right above the object and simply not descending
far enough, or sitting a centimetre or two to one side, so the fingertips graze
the surface and close on air. The correction needed is tiny. But with every
window you wait the policy withdraws the arm further, and from there it is no
longer a small correction.

### What does not justify intervening

- looking slow, or taking a route you would not have taken
- a single chunk of jitter, or a small deviation
- the wrist view swinging wildly or the arm occluding the scene — that is the camera moving, not the policy failing
- not being able to tell. When you cannot tell, pass and look again next window

**But "I could not see it clearly" cannot explain away a definite anomaly.** If
a third-person camera shows the object on the counter while the arm walks away,
that is a drop, and a blurry wrist view does not change it.
