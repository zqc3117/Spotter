"""
Local screener client. Pure standard library; import it directly with python3 on any machine.

    from screener import screen
    r = screen(image_bytes, telemetry_text, task="PnPCounterToCab")
    r -> {"flag": bool, "risk": int, "rule_hit": bool, "why": str, "port": int, "dt": float}

flag = rule_hit or risk >= THRESHOLD. rule_hit means "some chunk in this window has an aperture < 12 mm",
on PnP* tasks only, read from the width(mm) column of the newest SCREEN_RULE_ROWS telemetry rows (default 3).
The task name comes from SCREEN_TASK_NAME (judge_driver_v7.sh sets it when it calls screen_window.py);
without it, the task argument itself is checked for a PnP* name.
If flag is true the window goes to Opus; if false the VLA keeps going.
"""
import base64, json, re, time, urllib.request, itertools, threading

# ---- endpoints -------------------------------------------------------------
# Four replicas by default, one per GPU; QWEN_PORTS overrides the list. Use 127.0.0.1 on the same host, the cluster IP from another host.
HOST = "127.0.0.1"          # overridden by the caller across machines (screen_window.py --host / $SCREENER_HOST)
PORTS = [int(p) for p in __import__("os").environ.get("QWEN_PORTS", "8301,8302,8303,8304").split(",")]  # one port per replica
MODEL = "qwen38"
THRESHOLD = 3               # aggressive setting; the conservative setting uses the v5 prompt + threshold 3
EMPTY_MM = 12.0             # aperture below this = empty grasp

# Every screen_window.py call is a new process, so the cycle always started at 8301 -> all 48 lanes hit the first replica
# (2026-09-17 q500: 8301 got 713 calls, the others 200 each; GPU0 100%, GPU2 0%). Stagger the starting port by pid.
import os as _os
_k = _os.getpid() % len(PORTS)
_rr = itertools.cycle(PORTS[_k:] + PORTS[:_k])
_lock = threading.Lock()
def _next_port():
    with _lock:
        return next(_rr)

# ---- prompt (v7; every block is required) ------------------------------
CALIB = """HOW TO READ THE GRIPPER APERTURE (Franka Panda, calibrated):
  > 50 mm   fingers wide open, holding NOTHING
  12-50 mm  partly closed; MAY hold something (a small object reads about 30 mm),
            or may just be at the 28.8 mm rest position holding nothing
  < 12 mm   shut all the way, fingers touching. NOTHING is between them.
            This is positive proof of an EMPTY grasp.
A closed gripper is NEVER evidence of a grasp. If the aperture is under 12 mm and
the arm is travelling toward the destination, it is carrying NOTHING and the task
is already failing. Never call that "transport"."""

RUBRIC = """SCORE ANCHORS -- use the whole range:
  9-10  proof of failure in the numbers: aperture under 12 mm after a close
        command; the gripper reopened right after closing (object slipped);
        the arm carrying nothing toward the destination.
  6-8   strong suspicion, no proof: the end-effector has barely moved over
        several chunks (STUCK -- this is a 7 or 8, never a 3); the gripper is
        opening and closing repeatedly; the fingers are not aligned with the
        object about to be grasped; the arm is heading to the wrong object or
        the wrong destination; something got knocked over.
  3-5   mild doubt you cannot pin on anything specific.
  0-2   unambiguously on track: the arm is making steady progress toward the
        right place and nothing in the numbers contradicts the picture.
Being STUCK is a failure mode, not normal operation. Score it 7 or higher."""

SYS = f"""You are the cheap first-pass screener for a robot arm doing a kitchen
pick-and-place task. An expert judge re-examines every window you flag, so a
false alarm costs one extra review while a miss loses the whole episode.

{CALIB}

{RUBRIC}

Answer in EXACTLY this format, risk first:
RISK: <integer 0-10>
WHY: <one short sentence>"""

# RoboDojo (two ARX X5 arms, tabletop): the telemetry reports the MEASURED finger opening,
# normalised 1 (open) .. about 0.19 where the fingers meet (probe 2026-10-20). The Franka
# millimetre calibration above does not apply, so this family gets its own block.
if _os.environ.get("FAMILY_OVERRIDE") == "robodojo":
    CALIB = """HOW TO READ THE GRIPPER OPENING (two ARX X5 arms, measured finger position):
  0.9-1.0    open, holding NOTHING
  0.21-0.9   closed onto something: usually a grasp (thin objects read just above 0.2),
             or fingers resting on an edge they cannot hold
  <= 0.20    the fingers met: shut all the way with NOTHING between them.
             This is positive proof of an EMPTY grasp.
Each row gives the left and right opening; one hand usually does the work while the
other rests open. A hand at <= 0.20 travelling toward the destination carries nothing."""
    RUBRIC = RUBRIC.replace("aperture under 12 mm after a close\n        command", "an opening at or below 0.20 after a close\n        command")
    SYS = f"""You are the cheap first-pass screener for a two-arm robot doing a tabletop
manipulation task. An expert judge re-examines every window you flag, so a
false alarm costs one extra review while a miss loses the whole episode.

{CALIB}

{RUBRIC}

Answer in EXACTLY this format, risk first:
RISK: <integer 0-10>
WHY: <one short sentence>"""


# rdv8 (SCREEN_PROMPT=v1): rdv7's verdict meaning -- the screener must flag whenever the task is not visibly
# getting closer to done. Offline replay on dev windows: recall on GPT-intervene windows 60% -> 90%.
if _os.environ.get("FAMILY_OVERRIDE") == "robodojo" and _os.environ.get("SCREEN_PROMPT") == "v1":
    SYS = f"""You are the first-pass screener for a two-arm robot doing a tabletop manipulation task. An expert judge
looks at every window you flag and can repair the task; a window you pass is never looked at. A false alarm costs
one extra review; a miss can lose the episode. Flag whenever the task is not visibly getting closer to done.

{CALIB}

Movement is NOT progress. Large arm travel with open grippers often means the policy is wandering, withdrawing
early, or repeating an attempt. Judge progress from the IMAGE against the TASK INSTRUCTION, step by step:
  - Which objects has the task already put in the right place, in the right order/orientation? Which not yet?
  - Is an object on the wrong destination, in the wrong order, the wrong object picked, or the wrong basket/pad?
  - Was something dropped, knocked over, left outside its destination, or released before it arrived?
  - Are the hands withdrawing or resting while part of the task is still undone?
  - Has the same attempt been repeated (grasps closing on nothing, hand returning to the same spot)?
  - If the judge acted recently, did what it expected actually happen?
  - Is the remaining budget small for what is left?

SCORE (risk) -- use the whole range:
  9-10  proof of failure: a wrong placement/order/destination visible in the image; an object dropped or left
        outside its target; fingers shut on nothing (opening <= 0.20) after a close; the judge's expectation
        visibly did not hold.
  6-8   likely trouble: no visible task progress over this window; hands withdrawing or idle with the task
        unfinished; repeated attempts; heading to the wrong object or place; unsure whether a step succeeded.
  3-5   mild doubt.
  0-2   clearly progressing: the image shows the next unfinished step being done correctly right now.

Answer in EXACTLY this format, risk first:
RISK: <integer 0-10>
WHY: <one short sentence naming the task step you checked>"""
try:
    THRESHOLD = int(_os.environ.get("SCREEN_THRESHOLD") or THRESHOLD)
except ValueError:
    pass

# ---- rule ----------------------------------------------------------------
_ROW = re.compile(r"^\d+\s+\d+-\d+\s+.*?([\d.]+)\s*$", re.M)
# cli/harness.py prints bracketed columns after width(mm) (hand xyz, wrist rpy), so the row's
# last token is not the width. With the bracketed fields removed a row reads
# "chunk steps grip_cmd travelled width", so the width is field 4. The rule reads the newest rows
# only, this window's chunks plus the last one before it, so a closure many chunks back does not
# fire it. judge_driver_v7.sh passes SCREEN_RULE_ROWS=WINDOW+1; without it the rule reads RULE_ROWS.
RULE_ROWS = 3
_BRACKETED = re.compile(r"\[[^\]]*\]")
_WIDTH_FIELD = 4


def _is_pnp(task: str = "") -> bool:
    """Only pick-and-place (PnP*) tasks use the rule; mechanism and coffee tasks do not."""
    name = _os.environ.get("SCREEN_TASK_NAME") or str(task or "")
    return name.startswith("PnP")


def _rule_rows() -> int:
    """How many of the newest table rows the rule reads (SCREEN_RULE_ROWS, else RULE_ROWS)."""
    try:
        return max(1, int(_os.environ.get("SCREEN_RULE_ROWS") or RULE_ROWS))
    except ValueError:
        return RULE_ROWS


def rule_hit(telemetry_text: str, task: str = "") -> bool:
    """PnP* tasks: some width(mm) among the newest _rule_rows() chunks of the table is < 12. Other tasks: always False."""
    if not _is_pnp(task):
        return False
    rows = [line for line in telemetry_text.splitlines() if re.match(r"^\d+\s+\d+-\d+", line)]
    for line in rows[-_rule_rows():]:
        try:
            if float(_BRACKETED.sub(" ", line).split()[_WIDTH_FIELD]) < (4.0 if "Mug" in (task or "") else EMPTY_MM):   # mug-handle grasps read 6-12 mm, so for mug tasks only below 4 mm counts as empty
                return True
        except (ValueError, IndexError):
            pass
    return False

# ---- call ----------------------------------------------------------------
def _mime(b: bytes) -> str:
    return "image/png" if b[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"

def score(image_bytes: bytes, telemetry_text: str, task: str,
          prev_verdicts=None, timeout=None, retries=2, mime=None):
    timeout = timeout or (240 if _os.environ.get("SCREEN_THINK") == "1" else 60)
    """Query the model only; returns (risk, why, port, dt). risk=-1 means the reply could not be parsed."""
    hist = ""
    if prev_verdicts:
        hist = "\nPrevious windows (this episode): " + ", ".join(
            "CHECK" if v else "FINE" for v in prev_verdicts[-4:]) + "\n"
    exp = _os.environ.get("SCREEN_EXPECT", "").strip()
    if exp:
        hist += f"\nThe judge acted {_os.environ.get('SCREEN_EXPECT_AGE', '?')} window(s) ago and expected: {exp}\n"
    think = _os.environ.get("SCREEN_THINK") == "1"
    mime = mime or _mime(image_bytes)
    img = base64.b64encode(image_bytes).decode()
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": 2500 if think else 120,
            "chat_template_kwargs": ({"enable_thinking": True, "thinking_budget": int(_os.environ.get("SCREEN_THINK_BUDGET", "1500"))}
                                     if think else {"enable_thinking": False}),
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img}"}},
                    {"type": "text", "text": f"Task: {task}\n{hist}\n{telemetry_text[:6000]}\n\nRISK and WHY:"}]}]}
    data = json.dumps(body).encode()
    last = None
    for _ in range(retries + 1):
        port = _next_port()
        try:
            req = urllib.request.Request(f"http://{HOST}:{port}/v1/chat/completions",
                                         data=data, headers={"Content-Type": "application/json"})
            t0 = time.time()
            d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            txt = d["choices"][0]["message"]["content"] or ""
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
            m = re.search(r"RISK:\s*(\d+)", txt)
            w = re.search(r"WHY:\s*(.+)", txt)
            return (int(m.group(1)) if m else -1, (w.group(1).strip() if w else txt.strip()[:160]),
                    port, time.time() - t0)
        except Exception as e:
            last = e
    raise RuntimeError(f"screener unreachable on {PORTS}: {last}")

def screen(image_bytes: bytes, telemetry_text: str, task: str, prev_verdicts=None, **kw):
    """Rule OR model. This is the function the driver should call."""
    hit = rule_hit(telemetry_text, task)
    risk, why, port, dt = score(image_bytes, telemetry_text, task, prev_verdicts, **kw)
    return {"flag": hit or risk >= THRESHOLD, "risk": risk, "rule_hit": hit,
            "why": why, "port": port, "dt": round(dt, 3)}

def health():
    """Ping each of the four ports once."""
    out = {}
    for p in PORTS:
        try:
            urllib.request.urlopen(f"http://{HOST}:{p}/v1/models", timeout=5).read()
            out[p] = "ok"
        except Exception as e:
            out[p] = f"down ({type(e).__name__})"
    return out

if __name__ == "__main__":
    import sys
    print(health())
    if len(sys.argv) >= 3:
        r = screen(open(sys.argv[1], "rb").read(), open(sys.argv[2]).read(),
                   sys.argv[3] if len(sys.argv) > 3 else "PnPCounterToCab")
        print(json.dumps(r, ensure_ascii=False, indent=1))
