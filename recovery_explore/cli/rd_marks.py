"""rd7: current three camera frames from the service, object marks numbered on the primary one, and the prompt
list. Writes <out> (a copy of the window montage) plus <out>.newest (the three current frames) which qwen_call
sends as the pictures to point in -- RoboDojo montages have as many columns as the window had chunks, so the
old fixed 4-column crop of the montage picked the wrong tile. Prints <out>; nothing on failure."""
import argparse, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from PIL import Image, ImageDraw
import rex

ap = argparse.ArgumentParser()
ap.add_argument("--endpoint", required=True); ap.add_argument("--montage", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--text", required=True)
a = ap.parse_args()
CAM_DEPTH, CAM_PARAMS, CAM_EST = (os.environ.get(k, "0") == "1" for k in ("RD_CAM_DEPTH", "RD_CAM_PARAMS", "RD_CAM_EST"))
try:
    rnd = rex.call(a.endpoint, "render", timeout=60)
    r = rex.call(a.endpoint, "object_marks", timeout=60)
except SystemExit:
    sys.exit(0)
d = Path(a.out).parent
paths = []
for cam in ("primary", "secondary", "wrist"):
    img = (rnd.get("rgb") or {}).get(cam)
    if img is None:
        sys.exit(0)
    fp = d / f"now_{cam}.png"; rex.write_png(fp, img); paths.append(str(fp))
marks = (r or {}).get("marks") or []
im = Image.open(paths[0]).convert("RGB").resize((512, 512))
dr = ImageDraw.Draw(im)
for mk in marks:
    y = mk["pixel"][0] * 512 / 1000.0; x = mk["pixel"][1] * 512 / 1000.0
    dr.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(255, 235, 0), width=2)
    lbl = str(mk["id"])
    for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        dr.text((x + 11 + ox, y - 12 + oy), lbl, fill=(0, 0, 0))
    dr.text((x + 11, y - 12), lbl, fill=(255, 235, 0))
mp = d / "now_primary_marks.png"; im.save(mp); paths[0] = str(mp)
shutil.copyfile(a.montage, a.out)
open(a.out + ".newest", "w").write(json.dumps(paths))
lines = ["OBJECTS FOUND IN THE HEAD CAMERA'S DEPTH IMAGE (numbered with yellow circles on the current primary picture;",
         "[row, col] is that object's top centre in the primary camera, 0-1000; sizes measured from depth, objects in a",
         "hand or under the arms are not listed):"]
BASE = {"left": (-0.3, -0.45), "right": (0.3, -0.45)}; REACH = float(os.environ.get("RD_REACH", "0.58"))
for mk in marks:
    ok = [a for a, b in BASE.items() if ((mk["xyz"][0] - b[0]) ** 2 + (mk["xyz"][1] - b[1]) ** 2) ** 0.5 <= REACH]
    rch = {2: "either arm", 1: (ok[0] + " arm only") if ok else "", 0: "OUT OF REACH of both arms"}[len(ok)]
    geo = (f", depth {mk['depth_m']:.2f} m, xyz [{mk['xyz'][0]:.3f}, {mk['xyz'][1]:.3f}, {mk['xyz'][2]:.3f}] m"
           if CAM_DEPTH and mk.get("depth_m") is not None else "")
    lines.append(f"  #{mk['id']}  pixel [{mk['pixel'][0]}, {mk['pixel'][1]}]  footprint {mk['narrow_cm']:.0f} x {mk['long_cm']:.0f} cm, {mk['height_cm']:.0f} cm tall{geo}, reach: {rch}")
if not marks:
    lines.append("  (nothing found standing on the table)")
if CAM_DEPTH or CAM_PARAMS:
    cp = rex.call(a.endpoint, "camera_params", timeout=60) if (CAM_DEPTH or CAM_PARAMS) else {}
if CAM_DEPTH:
    hd = cp.get("hands") or {}
    lines.append("  Hands now (world xyz, m): " + ", ".join(f"{sd} {hd[sd]}" for sd in ("left", "right") if sd in hd))
    lines.append("  depth = metres from the head camera along its viewing axis to the object's top centre (measured by the")
    lines.append("  depth camera); xyz = that point in the world frame: x to the robot's right, y away from the robot, z up.")
    lines.append("  Use the difference between a hand's xyz and an object's xyz to size nudges and lifts in metres.")
if CAM_PARAMS:
    lines += ["CAMERAS. A pixel target is [row, col] in 0-1000 within one camera image (row down, col right). Camera axes:",
              "z forward along the view, x toward larger col, y toward larger row. A pixel seen at depth d metres (along z)",
              "is the camera point X=(col-cx)*d/fx, Y=(row-cy)*d/fy, Z=d, and the world point P = R*[X,Y,Z] + t (metres,",
              "world frame as above)."]
    names = {"primary": "primary (head, fixed)", "secondary": "secondary (left wrist, moves with the arm; values for now)",
             "wrist": "wrist (right wrist, moves with the arm; values for now)"}
    for cam, c in (cp.get("cameras") or {}).items():
        if "K" not in c:
            continue
        K = c["K"]
        lines.append(f"  {names.get(cam, cam)}: fx={K[0][0]:.1f} fy={K[1][1]:.1f} cx={K[0][2]:.1f} cy={K[1][2]:.1f}  "
                     f"R={json.dumps(c['R'])}  t={json.dumps(c['t'])}")
if CAM_EST:
    lines += ["ESTIMATES REQUIRED. Whenever your JSON carries a non-empty plan, add a field \"estimates\" BEFORE \"plan\": one",
              "entry per object that plan acts on, {\"name\": \"...\", \"camera\": \"primary\", \"pixel\": [row, col],",
              "\"depth_m\": d, \"xyz\": [x, y, z]} -- the object's centre pixel, its depth, and the world point you compute",
              "from them with the formula above. Then give the plan exactly as before."]
lines += ["To grasp, place onto, push or press one of these, copy its pixel as the target with \"camera\": \"primary\" --",
          "a point that is not on an object makes the tool fail (no_object). The fingers open at most 8.8 cm: an object",
          "whose narrow side is over 8 cm is grasped at the ~7 cm around the point you give: point at its rim, edge or",
          "handle (not its middle) to pick it up, or push it.",
          "The arms reach about 55 cm from their bases (front corners of the table): a tool aimed at an object marked OUT",
          "OF REACH, or at one the named arm cannot reach, does nothing (unreachable). Leave far objects to the policy."]
open(a.text, "w").write("\n".join(lines) + "\n")
print(a.out)
