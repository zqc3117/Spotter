#!/usr/bin/env python3
"""harness -- CLI for experiment orchestration and privileged scoring. **Not for the model.**

``cli/rex.py`` is the model's entire tool surface; this file is the other half: creating
failure states, rewinding to them, running a full episode, and reading the oracles the model
must never see. The two files are kept apart so that "what the model can touch" is a
**file-level** fact -- not a default argument, not a convention.
No code path in rex.py reaches any subcommand here.

    Everything harness.py has, rex.py does not:
      make-failure     script an "empty grasp" state (no policy service needed)
      run-to-failure   run the real policy until the oracle says the grasp really failed (higher fidelity)
      resume           run the episode to the end from the current state and see whether the task succeeds
      reset            rewind to the moment of failure (checked by hash + observation)
      reset-steps      zero the step counter only
      meta             full bookkeeping, including object_name
      state            oracle read (object_xyz / is_grasped / task_success)
      grasp            predicates only: is_grasped + task_success

## Paired trials (why this interface exists)

    harness.py run-to-failure --task PnPCounterToCab --seed 901
      -> only meaningful if triggered=true; if false, try another seed (the service has already released the env)
    harness.py resume                    # control: no intervention; can the policy recover on its own?
    harness.py reset                     # rewind to the same moment
    <explorer intervenes via rex.py>     # treatment
    harness.py resume                    # does it succeed after the intervention?
    harness.py grasp                     # when a finer criterion is needed

Both arms continue from the **same episode clock**, so the comparison is "did the
intervention help", not "how many steps did the explorer waste".

Reuses the HTTP/ndarray/PNG helpers from ``rex.py`` -- a one-way dependency: harness knows
rex, rex does not know harness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:  # normal use: run this script directly
    from rex import DEFAULT_ENDPOINT, call, emit, write_png  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - when imported as a package
    from recovery_explore.cli.rex import DEFAULT_ENDPOINT, call, emit, write_png


def write_montage(frames, frame_steps, out_dir: Path, max_cols: int = 10) -> dict:
    """Tile the per-chunk frames recorded by resume into one image: one row per camera, at most max_cols columns (first and last always kept)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(frames)
    if n == 0:
        return {"montage": None, "n_frames": 0}
    if n <= max_cols:
        idx = list(range(n))
    else:
        idx = sorted({round(i * (n - 1) / (max_cols - 1)) for i in range(max_cols)})
    cams = [c for c in ("primary", "secondary", "wrist") if c in frames[0]]
    rows = []
    for cam in cams:
        strip = [frames[i][cam] for i in idx]
        h = len(strip[0])
        for y in range(h):
            row = []
            for img in strip:
                row.extend(img[y])
            rows.append(row)
    fp = out_dir / "wam_montage.png"
    write_png(fp, rows)
    return {"montage": str(fp.resolve()), "n_frames": n, "shown_frame_indices": idx,
            "shown_steps": [frame_steps[i] for i in idx], "cameras": cams,
            "layout": f"每行一个相机({'/'.join(cams)})，从左到右按时间顺序，列对应 shown_steps"}


def robodojo_telemetry_lines(tel: dict, r: dict) -> list:
    """The RoboDojo flavour of the per-window telemetry the judge reads.

    Deliberately the same SHAPE as the RoboCasa one -- a short table of the
    last few chunks, then the current pose, then the step clock -- so a judge
    that has learned to read one reads the other. The columns differ because
    the robot differs:

    * two arms, so travel and opening are reported per arm;
    * the policy commands JOINT positions, so the stall signal is the tracking
      error (commanded joint target vs the joint state that came back), not
      "commanded 5 cm, moved 0 cm";
    * the gripper reports a normalised opening in 0..1, NOT a width in mm.
      There is no finger-gap measurement on this robot and no force sensor,
      so "closed on nothing" is the only identifiable closed state, exactly as
      on RoboCasa but derived from the opening instead of a width.

    The "End-effector is now at [...]" line is load-bearing: judge_driver_v7.sh
    greps it to build the intervention re-stage point. The ACTIVE arm is
    printed last because the driver takes the last match.
    """
    allc = tel.get("chunks", []) or []
    # RoboTwin: the judge gets the last 20 chunks of action history (user, 2026-09-19), one line per
    # chunk written as CHANGES: per-hand displacement in cm and the policy's gripper command only
    # when it changed. Absolute poses are given once, below. RT_TEL_CHUNKS overrides the 20.
    rt = str(tel.get("schema") or "") == "robotwin" or any("cmd_end" in c for c in allc)
    nkeep = int(os.environ.get("RT_TEL_CHUNKS", "20")) if rt else int(os.environ.get("TEL_CHUNKS", "3"))
    shown = allc[-nkeep:] if nkeep > 0 else allc
    active = str(tel.get("active_arm") or "right")
    other = "left" if active == "right" else "right"

    lines = [f"What the arms did in the last {len(shown)} chunk(s)"
             + (f" (of {len(allc)} so far this episode; earlier chunks not shown)"
                if len(allc) > len(shown) else "")
             + ". Travel is in cm, opening is 0 (shut) to 1 (open):",
             # Explicit per-arm column names. The first version wrote one
             # "L/R" column per quantity; the judge read the pair correctly
             # but then had to keep re-deriving which side was which, and the
             # phrase "L/R" gives no clue that the LEFT number comes first.
             "chunk  steps        left cm  right cm   left open  right open   joint err L/R (rad)"]
    if rt:
        lines[-1] = ("chunk steps     left hand moved (dx,dy,dz cm)  right hand moved (dx,dy,dz cm)  "
                     "open L / R    policy grip cmd L / R    joint err L / R (rad)")
        lines.insert(-1, "Each line is what CHANGED in that chunk. Grip cmd is what the policy told the fingers "
                         "(0 close .. 1 open, a-b = range within the chunk); '=' means same as the chunk before.")
        prev_cmd = None
        for c in shown:
            def _d(side):
                a = c.get(f"{side}_ee_start"); b = c.get(f"{side}_ee_pose")
                if a and b:
                    return "({:+.1f},{:+.1f},{:+.1f})".format(*[(float(b[i]) - float(a[i])) * 100 for i in range(3)])
                return "(?)"
            g = c.get("grip") or [None, None]
            cg = c.get("cmd_grip_range")
            cmd = ["{}-{}".format(*r) if r[0] != r[1] else f"{r[0]}" for r in cg] if cg else ["?", "?"]
            cmd_s = " / ".join("=" if prev_cmd and prev_cmd[k] == cmd[k] else cmd[k] for k in range(2))
            prev_cmd = cmd
            err = c.get("track_err_arm_rad")
            lines.append("{:<5} {:<9} {:<30} {:<31} {:<13} {:<24} {}".format(
                c.get("chunk"), f'{c["t"][0]}-{c["t"][1]}', _d("left"), _d("right"),
                f"{g[0]} / {g[1]}", cmd_s, ("{} / {}".format(*err) if err else c.get("track_err_rad"))))
    for c in ([] if rt else shown):
        moved = c.get("moved_cm") or [None, None]
        grip = c.get("grip") or [None, None]
        lines.append("{:<6} {:<12} {:<8} {:<10} {:<10} {:<12} {}".format(
            c.get("chunk"), f'{c["t"][0]}-{c["t"][1]}',
            str(moved[0]), str(moved[1]), str(grip[0]), str(grip[1]),
            ("{} / {}".format(*c["track_err_arm_rad"]) if c.get("track_err_arm_rad") else c.get("track_err_rad"))))

    # Commanded vs. actually tracked. When a position-controlled arm is blocked, the tracking error stays
    # up instead of falling -- the only decidable "stuck" signal on this robot (it has no force sensor).
    if shown:
        last = shown[-1]
        err = last.get("track_err_rad")
        moved = last.get("moved_cm") or [0, 0]
        per = last.get("track_err_arm_rad")
        if per and max(float(v) for v in per) > 0.15:
            who = [f"the {sd} arm ({v} rad)" for sd, v in zip(("left", "right"), per) if float(v) > 0.15]
            lines.append(f"  The last chunk ended with {' and '.join(who)} away from the joint target it was "
                         f"commanded to reach: something is holding that arm back.")
        elif err is not None and float(err) > 0.15:
            lines.append(f"  The last chunk ended {err} rad away from the joint target it was "
                         f"commanded to reach: something is holding the arm back.")
        elif max(float(v or 0) for v in moved) < 0.5:
            lines.append("  Neither hand moved more than 0.5 cm in the last chunk.")

    lines.append("")
    empties = tel.get("empty_closures") or []
    if empties:
        lines.append(f"Gripper closed all the way with nothing between the fingers "
                     f"({len(empties)} time(s) this episode, newest {min(len(empties), 4)} shown). "
                     "An opening cannot tell 'holding something' from 'open', so the absence of an")
        lines.append("entry is not evidence that anything is held.")
        for e in empties[-4:]:
            lines.append(f'  step {e.get("t")}: {e.get("arm")} hand, end-effector was at '
                         f'{e.get("eef")}, opening {e.get("opening")}')
    else:
        lines.append("So far this episode neither gripper has closed all the way on nothing.")

    # Name the working hand outright. Which arm the policy uses varies by task
    # AND by layout, so the judge cannot assume and must not have to infer it
    # from the table every window.
    total = {"left": 0.0, "right": 0.0}
    for c in allc:
        m = c.get("moved_cm") or []
        if len(m) == 2:
            total["left"] += float(m[0] or 0.0)
            total["right"] += float(m[1] or 0.0)
    if min(total.values()) > 15.0:
        # Bimanual episode: both arms have done real work. Calling one of them "the one doing
        # this task" made the judge ignore the other (rtj12 lift_pot / pick_dual_bottles).
        lines.append(f"BOTH hands are working in this task (left travelled {total['left']:.1f} cm, "
                     f"right {total['right']:.1f} cm). Check each hand separately against the task, "
                     f'and write "arm" in every step; a step without it goes to the {active} hand.')
    elif max(total.values()) > 0:
        lines.append(f"The {active} hand is the one doing this task "
                     f"(travelled {total[active]:.1f} cm so far against "
                     f"{total[other]:.1f} cm for the {other}). Your steps drive it "
                     f'unless you write "arm": "{other}" explicitly.')
    else:
        lines.append(f'Neither hand has moved yet; steps go to the {active} hand '
                     f'unless you write "arm": "{other}".')
    lines.append("")

    poses = tel.get("eef_now") or {}
    for side in (other, active):
        pose = poses.get(side)
        if not pose:
            continue
        tag = "(the arm your steps drive by default)" if side == active else "(the other arm)"
        lines.append(f'{side} hand {tag}: End-effector is now at {[round(float(v), 4) for v in pose[:3]]}, '
                     f'opening {(tel.get("opening_now") or {}).get(side)}, '
                     f'wrist quaternion (w,x,y,z) {[round(float(v), 3) for v in pose[3:]]}.')

    if rt:
        mj = tel.get("measured_joints_now") or {}
        cmd = (allc[-1].get("cmd_end") if allc else None) or tel.get("joints_now")
        def _j(v):
            return [round(float(x), 3) for x in v] if v else None
        lines.append("Joint state now (rad; 6 arm joints per arm, shoulder to wrist):")
        lines.append(f"  measured   left {_j(mj.get('left'))}   right {_j(mj.get('right'))}")
        if cmd and len(cmd) >= 14:
            lines.append(f"  the policy's last command: left {_j(cmd[0:6])} grip {round(float(cmd[6]), 2)}   "
                         f"right {_j(cmd[7:13])} grip {round(float(cmd[13]), 2)}")
        lines.append("  A large gap between measured and commanded on one arm = that arm is blocked or pressing on something.")
    _ct, _mx = r.get("committed_timestep"), r.get("max_steps")
    hz = tel.get("horizon")
    if _ct is not None and _mx:
        lines.append(f"Policy steps used so far: {int(_ct)} of {int(_mx)} "
                     f"({max(0, int(_mx) - int(_ct))} left"
                     + (f"; one chunk is {int(hz)} steps" if hz else "") + "). "
                     "Chunks you hand back with a policy step come out of these; your own moves do not.")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="harness", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- failure states -------------------------------------------------
    p = sub.add_parser(
        "make-failure",
        help="script a clean empty-grasp state (no policy service needed; once per service process)",
    )
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, default=901)
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--miss-offset-m", type=float, default=0.035)
    p.add_argument("--obj-instance-split", default="B")

    p = sub.add_parser(
        "run-to-failure",
        help="run the real policy from t=0 until the oracle declares an eligible failure (PnP: attempted a grasp and missed)",
    )
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, default=901)
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument(
        "--max-chunks", type=int, default=None,
        help="max number of 16-step chunks to run; default is the chunk count for TASK_MAX_STEPS",
    )
    p.add_argument("--obj-instance-split", default="B")
    p.add_argument("--record-dir", default=None,
                   help="consistency-probe output dir: per chunk, saves imagined t32 / real t16 / real t32A / real t32E")
    p.add_argument("--no-stop-on-trigger", action="store_true",
                   help="keep running the full episode after the failure triggers (for building a discrimination dataset with episode-level labels)")

    p = sub.add_parser(
        "resume",
        help="run the episode to the end from the current state with the policy -- answers 'did the intervention actually rescue this episode'",
    )
    p.add_argument(
        "--max-steps", type=int, default=None,
        help="override the episode step budget (default TASK_MAX_STEPS[task]); for short probes only",
    )
    p.add_argument("--frames-dir", default=None,
                   help="record one frame per chunk and tile them into wam_montage.png in this dir (feedback for the rounds experiment)")

    # ---- orchestration ----------------------------------------------------
    sub.add_parser("reset", help="rewind to the failure state (unlimited, ~8 ms, checked by hash + observation)")
    sub.add_parser("reset-steps", help="zero the step counter only; simulation state is untouched")
    p = sub.add_parser("open-budget", help="open an intervention window (no rewind): set the action budget, optionally disable cosmos")
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--allow-cosmos", action="store_true")
    # Token: if given, close-budget must pass back the same one. The driver draws a fresh random string per
    # window and the model never sees it, so the "close budget" path cannot be used by the model to drop the server's privilege gate.
    p.add_argument("--control-token", default=None)
    p = sub.add_parser("close-budget", help="close the intervention window; returns how many actions it used")
    p.add_argument("--control-token", default=None)
    p = sub.add_parser("judge-lock", help="judge turn starts: lock privileged methods (there is no budget while judging, so the gate would be open)")
    p.add_argument("--control-token", default=None)
    p = sub.add_parser("judge-unlock", help="judge turn ends: unlock; requires the token")
    p.add_argument("--control-token", default=None)

    p = sub.add_parser("episode-begin", help="create an episode and stop at t=0 (judge experiment)")
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, default=901)
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--obj-instance-split", default="B")
    p.add_argument("--max-steps-extra", type=int, default=0,
                   help="extra steps on top of the task's default step limit; used for control-arm reruns, see rerun_control.sh")

    p = sub.add_parser("advance", help="advance the policy by N chunks, saving one frame per chunk (judge experiment)")
    p.add_argument("--num-chunks", type=int, default=1)
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--tag", default="w")
    # Where privileged ground truth (object_xyz / is_grasped / task_success) is written.
    # **It must never be inside --frames-dir**: that directory's path is written straight into the judge prompt
    # ("image directory $CD"); a judge with Bash would read the oracle with a single ls.
    # Without this argument nothing is written or echoed -- forgetting it only loses post-hoc analysis, it never leaks.
    p.add_argument("--truth-dir", default=None,
                   help="harness-only: privileged ground truth is written here; discarded if omitted. Do not point it at frames-dir")

    p = sub.add_parser("round-begin", help="rewind to the failure state and open a round with an action budget (rounds experiment)")
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--index", type=int, default=None)
    sub.add_parser("round-status", help="actions used this round + privileged success predicate")
    # Two subcommands for validation scripts only: state-digest takes a whole-scene fingerprint (rewind fidelity /
    # rerun consistency are judged by it); checkpoint takes a snapshot without opening an intervention window.
    sub.add_parser("state-digest", help="comparable fingerprint of the whole scene + both arm poses (for validation scripts)")
    p = sub.add_parser("checkpoint", help="take a snapshot without opening a window (for validation scripts)")
    p.add_argument("--tag", default="manual")
    sub.add_parser("restore-window", help="restore the most recent snapshot (for validation scripts)")
    sub.add_parser("reset-window", help="return to the state at the start of this intervention and reissue the budget (called by the driver when the model says 'not fixed, start over')")
    sub.add_parser("release-env", help="drop the current episode so the next one can be created in the same process (control-arm-only runs)")

    p = sub.add_parser("restore-snapshot", help="restore a captured anchor (cross-process; see the warning in failure_factory)")
    p.add_argument("--anchor-id", required=True)
    p.add_argument("--anchors-jsonl", default=None)

    # ---- privileged reads ---------------------------------------------------
    sub.add_parser("quit", help="make env_service exit after replying (in three-tier mode sim_lane.sh restarts a fresh process)")
    sub.add_parser("meta", help="full bookkeeping, including object_name (the model-side meta omits it)")
    sub.add_parser(
        "state",
        help="state read with oracle: object_xyz / is_grasped / task_success. For scoring",
    )
    sub.add_parser(
        "grasp",
        help="predicates only: is_grasped + task_success (env._check_grasp, privileged)",
    )

    p = sub.add_parser("unproject-batch", help="unproject several pixels at once (harness convenience)")
    p.add_argument("--rows", nargs="+", type=int, required=True)
    p.add_argument("--cols", nargs="+", type=int, required=True)
    p.add_argument("--camera", default="primary", choices=["primary", "secondary", "wrist"])

    a = ap.parse_args()
    E = a.endpoint

    if a.cmd == "make-failure":
        emit(call(E, "make_failure", task=a.task, seed=a.seed,
                  episode_index=a.episode_index, miss_offset_m=a.miss_offset_m,
                  obj_instance_split=a.obj_instance_split))
    elif a.cmd == "run-to-failure":
        emit(call(E, "run_episode_to_failure", task=a.task, seed=a.seed,
                  episode_index=a.episode_index, max_chunks=a.max_chunks,
                  obj_instance_split=a.obj_instance_split,
                  record_dir=a.record_dir,
                  stop_on_trigger=not a.no_stop_on_trigger))
    elif a.cmd == "resume":
        if a.frames_dir:
            r = call(E, "resume_episode", max_steps=a.max_steps, record_frames=True)
            frames = r.pop("frames", []) or []
            steps = r.pop("frame_steps", []) or []
            track = r.pop("eef_track", None)
            r.update(write_montage(frames, steps, Path(a.frames_dir)))
            if track:
                Path(a.frames_dir, "eef_track.json").write_text(
                    json.dumps(track), encoding="utf-8")
            emit(r)
        else:
            emit(call(E, "resume_episode", max_steps=a.max_steps))
    elif a.cmd == "reset":
        emit(call(E, "reset_to_failure"))
    elif a.cmd == "reset-steps":
        emit(call(E, "reset_step_counter"))
    elif a.cmd == "open-budget":
        emit(call(E, "harness_open_action_budget", budget=a.budget,
                  allow_cosmos=bool(a.allow_cosmos),
                  control_token=a.control_token))
    elif a.cmd == "close-budget":
        emit(call(E, "harness_close_action_budget", control_token=a.control_token))
    elif a.cmd == "judge-lock":
        emit(call(E, "harness_judge_lock", control_token=a.control_token))
    elif a.cmd == "judge-unlock":
        emit(call(E, "harness_judge_unlock", control_token=a.control_token))
    elif a.cmd == "episode-begin":
        emit(call(E, "harness_episode_begin", task=a.task, seed=a.seed,
                  episode_index=a.episode_index, obj_instance_split=a.obj_instance_split,
                  max_steps_extra=getattr(a, "max_steps_extra", 0) or 0))
    elif a.cmd == "quit":
        emit(call(E, "harness_quit"))
    elif a.cmd == "advance":
        r = call(E, "harness_advance", num_chunks=a.num_chunks)
        frames = r.pop("frames", []) or []
        steps = r.pop("frame_steps", []) or []
        # pop, not get: truth must not remain in the emitted JSON; the driver's log
        # and stdout are files the judge can read too.
        truth = r.pop("truth", None)
        out = Path(a.frames_dir); out.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, (per_cam, st) in enumerate(zip(frames, steps)):
            for cam, img in per_cam.items():
                fp = out / f"{a.tag}_c{i:02d}_t{st:04d}_{cam}.png"
                write_png(fp, img)
                paths.append(str(fp.resolve()))
        # Also tile one overview image: one image costs the judge fewer tokens than dozens, and trends are easier to see
        r.update(write_montage(frames, steps, out))
        r["frame_files"] = paths
        # The action history is written as a short text block that the driver pastes into the prompt.
        # In round two only one image was passed; the model did not know what the policy had commanded or when the
        # gripper had closed, so it re-located the object from pixels and every intervention started over from far away.
        tel = r.pop("telemetry", None) or {}
        # RoboDojo's per-row columns differ from RoboCasa's (two arms, joint position control, opening is a
        # normalised 0..1 value rather than mm), so the table below would KeyError. The server reports its schema and
        # we branch here; the rest of the rendering (montage, truth dump, emit) is shared by both families.
        # RoboTwin (aloha-agilex) is the same shape -- two arms, normalised opening, no force
        # sensing -- so its env service emits schema "robotwin" and shares the formatter.
        # RoboCasa emits no schema key and keeps the table below.
        if tel and str(tel.get("schema") or "") in ("robodojo", "robotwin"):
            lines = robodojo_telemetry_lines(tel, r)
            tp = out / "telemetry.txt"
            tp.write_text("\n".join(lines) + "\n")
            r["telemetry_file"] = str(tp.resolve())
            r["grasp_attempts"] = len(tel.get("empty_closures") or [])
            tel = {}
        if tel:
            # Only the last few chunks, and only columns that map to visible changes. A whole-episode table reaches
            # 30 rows x 12 numbers by window 15, slowing inference and soaking up attention on numbers; what matters is
            # what changed in the image. The per-row absolute eef_to is of no use to the model (the current position is
            # given once at the end), and the base column is always the same for arm-only policies, so it is printed only when the base actually moved.
            # 2026-09-20 (user): the judge was missing proprioception, the actions taken, and history
            # beyond 3 chunks. Now: the last TEL_CHUNKS (default 20) chunks as a table that also gives
            # where the hand ended and the wrist orientation; every earlier chunk of the episode is
            # sampled into a trajectory summary below, re-sent every window, so it survives compaction.
            nkeep = int(os.environ.get("TEL_CHUNKS", "20"))
            allc = tel.get("chunks", [])
            shown = allc[-nkeep:] if nkeep > 0 else allc
            earlier = allc[:len(allc) - len(shown)]
            lines = [f"What the arm did in the last {len(shown)} chunk(s)"
                     + (f" (of {len(allc)} so far this episode; the earlier ones are summarised below)"
                        if earlier else "")
                     + ". Commands are in cm and radians; hand = end-effector position at the end of"
                     " the chunk (world m); wrist = roll/pitch/yaw in degrees:",
                     "chunk  steps     commanded dxyz(cm)     commanded drot(rad)  grip cmd  travelled(cm) width(mm)  hand xyz (m)              wrist rpy (deg)"]
            for c in shown:
                lines.append("{:<6} {:<9} {:<22} {:<20} {:<9} {:<13} {:<10} {:<25} {}".format(
                    c["chunk"], f'{c["t"][0]}-{c["t"][1]}',
                    str(c["cmd_dpos_cm"]), str(c["cmd_drot_rad"]), c["cmd_gripper"],
                    c["moved_cm"], c["width_mm"], str(c.get("eef_to", "?")), str(c.get("eef_rpy_deg", "?"))))
            if earlier:
                k = max(1, -(-len(earlier) // 12))  # at most ~12 sampled rows
                samp = earlier[::k]
                if samp[-1] is not earlier[-1]:
                    samp.append(earlier[-1])
                lines.append("")
                lines.append(f"Earlier in this episode (every {k} chunk(s) of the first {len(earlier)}): "
                             "step, hand xyz (m), wrist rpy (deg), gripper width (mm), grip cmd")
                for c in samp:
                    lines.append("  step {:<5} hand {:<25} wrist {:<22} width {:<6} grip cmd {}".format(
                        c["t"][1], str(c.get("eef_to", "?")), str(c.get("eef_rpy_deg", "?")),
                        c.get("width_mm"), c.get("cmd_gripper")))
                base = c.get("cmd_base")
                if base is not None and (c.get("base_mode_frac") or 0) > 0:
                    lines.append(f'       base command {base} frac {c.get("base_mode_frac")}')
            # Commanded vs. actually travelled -- a stall is exactly these two disagreeing; point it out directly
            # so the model does not have to compare a pile of numbers itself.
            if shown:
                last = shown[-1]
                want = max((abs(v) for v in (last.get("cmd_dpos_cm") or [0])), default=0.0)
                got = last.get("moved_cm") or 0.0
                if want >= 3.0 and got < 1.0:
                    lines.append(f"  The last chunk commanded up to {want:.1f} cm of motion "
                                 f"but the hand moved {got} cm.")
            ga = tel.get("grasp_attempts") or []
            lines.append("")
            if ga:
                lines.append(f"Gripper open/close events this episode ({len(ga)} total, newest {min(len(ga), 4)} shown). The only state that can be")
                lines.append("identified from aperture alone is 'shut all the way with nothing between the")
                lines.append("fingers'; aperture cannot tell 'holding something' from 'open', so the absence")
                lines.append("of an entry is not evidence that anything is held.")
                for e in ga[-4:]:
                    # The server writes English "closed-empty" / "reopened" (_grasp_events in
                    # env_service.py), but this once checked for the Chinese word for "closed", so every empty
                    # closure was rendered as "opened again" -- while the brief told the judge to look for the
                    # string "closed all the way", which never appeared in the table. Accept both; do not rely on language alignment.
                    ev = ("closed all the way, nothing between the fingers"
                          if str(e.get("event", "")).startswith(("closed", "闭合"))
                          else "opened again")
                    lines.append(f'  step {e["t"]}: {ev}; end-effector was at {e["eef"]}, aperture {e.get("width_mm")} mm')
                lines.append("  A move-to whose target is within 8 cm of one of these points, or of any point in")
                lines.append("  visited_points, may travel up to 60 cm in that one call (retracing the arm's path).")
            else:
                lines.append("So far this episode the gripper has never closed all the way on nothing.")
            pg = tel.get("pre_grasp_pose") or (r.get("pre_grasp_pose") if isinstance(r, dict) else None)
            if pg:
                lines.append(f'Pre-grasp anchor: at step {pg.get("t")}, just before the empty closure at step {pg.get("miss_t")}, '
                             f'the end-effector was at {pg.get("eef")} with the fingers open ({pg.get("width_mm")} mm). '
                             'The plan step {"op": "retreat", "to": "pre_grasp"} returns the hand there along its own path and '
                             'opens the fingers; a policy hand-back from there lets the policy redo its own approach and grasp. '
                             'Look at the object first: the miss may have moved it.')
            if tel.get("eef_rpy_now_deg") is not None:
                lines.append(f'Wrist orientation now: roll/pitch/yaw {tel.get("eef_rpy_now_deg")} deg '
                             f'(quaternion x,y,z,w {tel.get("eef_quat_now")}).')
            if tel.get("joint_pos_now"):
                lines.append(f'Arm joint positions now (rad): {tel.get("joint_pos_now")}.')
            # keep this line LAST among the pose lines and its wording unchanged: judge_driver_v7.sh
            # greps the last "End-effector is now at [...]" for the re-stage point.
            lines.append(f'End-effector is now at {tel.get("eef_now")}, aperture {tel.get("width_mm")} mm.')
            # Policy step clock. Chunks handed back to the policy are deducted from it (16 steps per chunk); the model's own
            # move_to / nudge are not. Without the number, the model treated hand-backs as free probes and four cells ended at the step limit.
            _ct, _mx = r.get("committed_timestep"), r.get("max_steps")
            if _ct is not None and _mx:
                lines.append(f"Policy steps used so far: {int(_ct)} of {int(_mx)} ({max(0, int(_mx) - int(_ct))} left; "
                             f"one chunk is 16 steps). Chunks you hand back with a policy step come out of these; "
                             f"your own moves do not.")
            tp = out / "telemetry.txt"
            tp.write_text("\n".join(lines) + "\n")
            r["telemetry_file"] = str(tp.resolve())
            r["grasp_attempts"] = len(ga)
        if truth and a.truth_dir:
            import json as _json
            tdir = Path(a.truth_dir)
            if tdir.resolve() == out.resolve() or out.resolve() in tdir.resolve().parents:
                raise SystemExit(
                    "--truth-dir must not be inside --frames-dir: the judge can read that directory")
            tdir.mkdir(parents=True, exist_ok=True)
            with open(tdir / f"truth_{a.tag}.json", "w") as fh:
                _json.dump({"committed": r.get("committed_timestep"), **truth}, fh)
        emit(r)
    elif a.cmd == "round-begin":
        emit(call(E, "harness_round_begin", budget=a.budget, round_index=a.index))
    elif a.cmd == "round-status":
        emit(call(E, "harness_round_status"))
    elif a.cmd == "state-digest":
        emit(call(E, "harness_state_digest"))
    elif a.cmd == "checkpoint":
        emit(call(E, "harness_checkpoint", tag=a.tag))
    elif a.cmd == "restore-window":
        emit(call(E, "restore_snapshot", which="window"))
    elif a.cmd == "reset-window":
        emit(call(E, "reset_window"))
    elif a.cmd == "release-env":
        emit(call(E, "harness_release_env"))
    elif a.cmd == "restore-snapshot":
        emit(call(E, "restore_snapshot", anchor_id=a.anchor_id,
                  anchors_jsonl=a.anchors_jsonl))
    elif a.cmd == "meta":
        emit(call(E, "harness_env_meta"))
    elif a.cmd == "state":
        emit(call(E, "harness_read_state_privileged"))
    elif a.cmd == "grasp":
        emit(call(E, "harness_oracle_grasp"))
    elif a.cmd == "unproject-batch":
        emit(call(E, "unproject_batch", rows=a.rows, cols=a.cols, camera=a.camera))


if __name__ == "__main__":
    main()
