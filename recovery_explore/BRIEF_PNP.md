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

**After one empty closure, decide on the first window you see; do not wait for a
second miss.** Having closed on nothing, the policy on these tasks seldom reopens
and comes back for another try; far more often it keeps the fingers shut and
carries the empty hand away, and each window of that takes the arm further from
the line it missed on. So decide on this window. If a later close stopped between
about 12 and 50 mm with no new entry, or a third-person row shows the object
moving with the hand, the policy has already regrasped: the miss is history,
answer ok. If the fingers are still shut to a few millimetres and the hand is
moving away from the source, nothing came with it: repair now. One kind of
entry needs a look before it is called a miss: when the fingers had stayed
between about 12 and 50 mm under a close for several chunks while the hand
travelled, and only then ran shut, what they held has left them — set down where
the policy was taking it, or dropped on the way — and fingers opening soon after
are the policy letting go. Find the object near where the entry was logged:
resting at the destination, it was placed, answer ok; lying anywhere else, it was
dropped. Compare the pre-grasp anchor's step with the entry's: when it is many
chunks earlier, the anchor lies back where the object was first picked up, and a
retreat there drags the arm back across the scene. And when you do
act and the repair does not take, the next intervention has to differ in kind;
the same open-place-hand-back sent again and again spends the allowance without
changing the outcome.

**The first two have a precondition: the policy must have attempted a grasp.**
An approach legitimately takes several windows. Read the **grasp attempt list**
in the action history: if it is empty, the approach has not finished and there
is nothing to diagnose. An entry reading "closed all the way, nothing between
the fingers" means the grasp missed. Calling "no progress" before the policy
has reached the object is the classic misjudgement here.

**A missed grasp is the moment to act, and waiting costs you the chance.** The
typical miss has the gripper right above the object and simply not descending
far enough, or sitting a centimetre or two to one side, so the fingertips graze
the surface and close on air. The correction needed is tiny, but every window
you wait the policy withdraws and turns the wrist further, until a retreat can no
longer bring the hand back. On these tasks the first repair after such a miss
usually does better without the `policy 2` ending of the default repair:
`retreat` to `pre_grasp`, then with the fingers open lower the hand a centimetre
or two (or shift it that much toward the object's centre if the frames show it
was off to one side), then answer `fixed` with an empty plan. The policy makes
its own close from the corrected pose and has the next windows to itself; a
hand-back inside the repair gives it a chunk or two and asks you to judge a close
that has not settled. If the retreat reports it stopped short of the pre-grasp
point, the hand is not on the old line: do not lower or hand back from there;
answer `fixed` with an empty plan. All of this is about fingers that shut on
nothing.

**A close that stopped short is a grasp until the pictures show otherwise.** Read
the width together with the grip cmd column. While the fingers are commanded open
(grip cmd near -1) the width says nothing about holding. After a close (grip cmd
near +1), fingers with nothing between them run down to a few millimetres and log
a "closed all the way" entry; fingers that stopped between about 12 and 50 mm and
stay there, with no new entry for that close, are resting on something -- on
these tasks usually the object. That supports a grasp without proving it (the
width can outlast a slip), but it moves the burden: in that state an arm
travelling away from the source is more likely carrying than withdrawing, a pause
of a window is not yet a stall, and a smaller width you predicted but did not see
is not a miss. Call a miss or a drop when you can point to it: a new empty
closure in the event list, or, in a third-person row, the object still at its
source while the hand is elsewhere, or at rest away from the hand. If you cannot,
the evidence favours ok and the next window will show more. This matters most
in the windows after you hand back, when that grasp is what your repair was for.
Read it afresh each window instead of carrying an earlier reading forward, and
when a repair is meant to put the object in the hand, write its `expect` as the
object leaving its support with the fingers stopped at its width, not as a width
of a few millimetres. (On the two mug tasks the handle reading above applies
instead.)

### What does not justify intervening

- looking slow, or taking a route you would not have taken
- a single chunk of jitter, or a small deviation
- the wrist view swinging wildly or the arm occluding the scene — that is the camera moving, not the policy failing
- not being able to tell. When you cannot tell, pass and look again next window

**But "I could not see it clearly" cannot explain away a definite anomaly.** If
a third-person camera shows the object on the counter while the arm walks away,
that is a drop, and a blurry wrist view does not change it.
