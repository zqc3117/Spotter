"""
Local screener client. Pure standard library; import it directly with python3 on any machine.

    from screener import screen
    r = screen(image_bytes, telemetry_text, task="PnPCounterToCab")
    r -> {"flag": bool, "risk": int, "rule_hit": bool, "why": str, "port": int, "dt": float}

flag = rule_hit or risk >= THRESHOLD. rule_hit means "some chunk in this window has an aperture < 12 mm".
If flag is true the window goes to Opus; if false the VLA keeps going.
"""
import base64, json, re, time, urllib.request, itertools, threading

# ---- endpoints -------------------------------------------------------------
# Four replicas, one per GPU. Use 127.0.0.1 on the same host, the cluster IP from another host.
HOST = "127.0.0.1"          # overridden by the caller across machines (screen_window.py --host / $SCREENER_HOST)
PORTS = [8301, 8302, 8303, 8304]
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

# ---- rule ----------------------------------------------------------------
_ROW = re.compile(r"^\d+\s+\d+-\d+\s+.*?([\d.]+)\s*$", re.M)
def rule_hit(telemetry_text: str, task: str = "") -> bool:
    """Any chunk in the telemetry table has width(mm) < 12."""
    for line in telemetry_text.splitlines():
        if re.match(r"^\d+\s+\d+-\d+", line):
            try:
                if float(line.split()[-1]) < (4.0 if "Mug" in (task or "") else EMPTY_MM):   # mug-handle grasps read 6-12 mm, so for mug tasks only below 4 mm counts as empty
                    return True
            except ValueError:
                pass
    return False

# ---- call ----------------------------------------------------------------
def _mime(b: bytes) -> str:
    return "image/png" if b[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"

def score(image_bytes: bytes, telemetry_text: str, task: str,
          prev_verdicts=None, timeout=60, retries=2, mime=None):
    """Query the model only; returns (risk, why, port, dt). risk=-1 means the reply could not be parsed."""
    hist = ""
    if prev_verdicts:
        hist = "\nPrevious windows (this episode): " + ", ".join(
            "CHECK" if v else "FINE" for v in prev_verdicts[-4:]) + "\n"
    mime = mime or _mime(image_bytes)
    img = base64.b64encode(image_bytes).decode()
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": 120,
            "chat_template_kwargs": {"enable_thinking": False},
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
            txt = d["choices"][0]["message"]["content"]
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
