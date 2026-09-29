#!/usr/bin/env python3
"""Mine worked examples for the judge's few-shot block from learn-seed runs.

    fewshot_build.py --family cosmos [--seed 195] [--top 3] [--out fewshot_bank]

Source: recovery_explore/runs_*_<family>/results.jsonl rows on the LEARN seed only
(seed 195, idx 10-19 -> never overlaps the seed-500 test cells). A candidate is an
episode where the control failed, the treatment succeeded, and ONE intervention ran
cleanly (no rewind, no replan, plan executed to its end) shortly before the episode
ended. Each exemplar is one episode with two decisions in it, laid out the way
GPT-Policy lays out a demonstration (time-ordered frames + per-stage text + the
numeric records):
    window A  -> verdict ok        (pass.png,  telemetry, verdict)
    window B  -> verdict intervene (fault.png, telemetry, verdict+plan, execution log)
    window B+1 -> what it looked like after the hand-back (after.png)

Writes <out>/<family>/<task>/<rank>_<cell>/{pass,fault,after}.png, raw.json and
exemplar.txt. exemplar.txt uses [[IMAGE:<abs path>]] markers; gpt6_call.py /
qwen_call.py interleave them. notes.txt (the rewritten, number-free commentary) is
produced by fewshot_annotate.py and is REQUIRED before fewshot_select.py serves the
exemplar, so nothing unscreened reaches a prompt.
"""
import argparse, glob, json, os, re, shutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOOD_STOPS = {"checkpoint", "plan_complete", "done", "complete", "completed", "policy_done", "end", "finished", ""}


def jload(p, default=None):
    try:
        return json.load(open(p, encoding="utf-8", errors="replace"))
    except Exception:
        return default


def verdict_of(cd, w):
    v = jload(f"{cd}/verdict_w{w}.json")
    if isinstance(v, dict) and v.get("verdict"):
        return v
    j = jload(f"{cd}/judge_w{w}.json", {})
    m = re.search(r"\{.*\}", str(j.get("result", "")), re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


def exec_log(cd, w):
    """Compact, number-bearing execution log of attempt 1 of window w."""
    rows, last_stop = [], ""
    files = sorted(glob.glob(f"{cd}/exec_w{w}_a1_r*.json"),
                   key=lambda p: int(re.search(r"_r(\d+)\.json", p).group(1)))
    for f in files:
        d = jload(f, {})
        for e in d.get("executed", []):
            bits = [e.get("op", "?")]
            if e.get("state"): bits.append(str(e["state"]))
            if e.get("moved_cm") is not None: bits.append(f"moved {e['moved_cm']} cm")
            if e.get("width_mm") is not None: bits.append(f"aperture {e['width_mm']} mm")
            if e.get("servo_ok") is False: bits.append("servo did not converge")
            rows.append("  - " + ", ".join(bits))
        last_stop = str(d.get("stop_reason", ""))
        rows.append(f"  stop: {last_stop or 'plan finished'}")
    return rows, last_stop, len(files)


def trim_tel(path, keep=8):
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except Exception:
        return ""
    out, tab = [], 0
    for ln in lines:
        if re.match(r"^\d+\s+\d+-\d+", ln):
            tab += 1
        if ln.startswith("  A move-to whose") or ln.startswith("  visited_points"):
            continue
        if ln.startswith("Policy steps used"):
            continue
        out.append(ln)
    # keep only the newest `keep` chunk rows
    rows = [i for i, ln in enumerate(out) if re.match(r"^\d+\s+\d+-\d+", ln)]
    for i in rows[:-keep]:
        out[i] = None
    return "\n".join(l for l in out if l is not None)


def extra_frames(cd, dst, ev, used, w_fault=None):
    """More key frames around the core stages (GPT-Policy uses 6-16 per demonstration):
    episode start, the window just before the fault, each repair checkpoint, episode end.
    Returns {slot: text} with [[IMAGE:]] markers; pictures are copied into dst."""
    os.makedirs(dst, exist_ok=True)
    out = {}
    def put(slot, src, name, caption):
        if src and os.path.exists(src):
            shutil.copyfile(src, f"{dst}/{name}")
            out[slot] = out.get(slot, "") + f"{caption}\n[[IMAGE:{dst}/{name}]]\n"
    wins = sorted(e["window"] for e in ev)
    if wins and wins[0] not in used:
        put("start", f"{cd}/w{wins[0]}/wam_montage.png", "start.png",
            "[Context] The first window of that episode (the policy starting the task):")
    if w_fault:
        if w_fault - 1 not in used and w_fault - 1 >= 1:
            put("before", f"{cd}/w{w_fault - 1}/wam_montage.png", "before.png",
                "[Context] The window just before the fault:")
        cks = sorted(glob.glob(f"{cd}/act{w_fault}/a1r*/checkpoint.png"),
                     key=lambda p: int(re.search(r"a1r(\d+)", p).group(1)))
        for i, ck in enumerate(cks[:3]):
            put("ckpt", ck, f"ckpt{i}.png",
                f"[Repair checkpoint {i + 1}] What the monitor was shown when plan segment {i + 1} stopped "
                "(primary | secondary | wrist, native size):")
    last = wins[-1] if wins else None
    for w in ((last + 1, last) if last else ()):
        if w not in used and os.path.exists(f"{cd}/w{w}/wam_montage.png"):
            put("end", f"{cd}/w{w}/wam_montage.png", "end_ep.png", "[Context] The last recorded window of that episode:")
            break
    return out


def candidates(family, seed):
    for res in glob.glob(f"{HERE}/runs_*_{family}/results.jsonl"):
        run = os.path.basename(os.path.dirname(res))
        for line in open(res, encoding="utf-8", errors="replace"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("seed") != seed or not r.get("treatment_success"):
                continue
            if r.get("control_success") is not False:
                continue
            ev = r.get("events") or []
            ivs = [e for e in ev if e.get("judge") == "intervene"]
            if not 1 <= len(ivs) <= 2:
                continue
            iv = ivs[-1]          # the last repair is the one the episode finished on
            if (iv.get("resets") or 0) > 0 or (iv.get("attempts") or 1) > 1:
                continue
            w = iv["window"]
            cd = f"{HERE}/{run}/lane{r.get('lane')}/{r['cell']}"
            if not os.path.isdir(cd):
                continue
            first_iv = ivs[0]["window"]
            oks = [e["window"] for e in ev if e.get("judge") == "ok" and e["window"] < first_iv]
            scr = [e["window"] for e in ev if e.get("judge") == "screened" and e["window"] < first_iv]
            if not (oks or scr):
                continue
            judged_a = bool(oks)
            wa = max(oks) if oks else max(scr)
            need = [f"{cd}/w{wa}/wam_montage.png", f"{cd}/w{w}/wam_montage.png"]
            if not all(os.path.exists(p) for p in need):
                continue
            aft = f"{cd}/w{w + 1}/wam_montage.png"
            need.append(aft if os.path.exists(aft) else "")
            log, stop, nseg = exec_log(cd, w)
            if stop not in GOOD_STOPS and "checkpoint" not in stop:
                continue
            after = len([e for e in ev if e["window"] > w])
            va = verdict_of(cd, wa) if judged_a else {"verdict": "ok", "diagnosis": "{DIAG_PASS}", "plan": []}
            vb = verdict_of(cd, w)
            if not (va and vb and vb.get("plan")):
                continue
            uses_pixel = "pixel" in json.dumps(vb["plan"])
            engine = "gpt" if re.search(r"gpt|api|astra|^runs_g", run) else "other"
            score = (after <= 3) * 4 + uses_pixel * 2 + (engine == "gpt") * 2 + (nseg <= 2) + judged_a * 2 + (len(ivs) == 1) * 2 + bool(need[2])
            instr = r.get("instruction") or ""
            if not instr:
                for pf in sorted(glob.glob(f"{cd}/p_w*.txt")):
                    m = re.search(r"^Task instruction: (.+)$", open(pf, errors="replace").read(), re.M)
                    if m:
                        instr = m.group(1).strip(); break
            r["instruction"] = instr
            yield dict(score=score, run=run, cell=r["cell"], task=r["task"], cd=cd, wa=wa, wb=w,
                       windows_after=after, instruction=r.get("instruction", ""),
                       va=va, vb=vb, log=log, imgs=need, window_chunks=r.get("window_chunks"), ev=ev)


def write_exemplar(c, dst):
    os.makedirs(dst, exist_ok=True)
    names = ["pass.png", "fault.png", "after.png"]
    for src, n in zip(c["imgs"], names):
        if src:
            shutil.copyfile(src, f"{dst}/{n}")
    has_after = bool(c["imgs"][2])
    used = {c["wa"], c["wb"]} | ({c["wb"] + 1} if has_after else set())
    xf = extra_frames(c["cd"], dst, c["ev"], used, w_fault=c["wb"])
    tel_a = trim_tel(f"{c['cd']}/w{c['wa']}/telemetry.txt")
    tel_b = trim_tel(f"{c['cd']}/w{c['wb']}/telemetry.txt")
    body = f"""Task instruction of the example episode: {c['instruction']}

{xf.get('start', '')}
[Example, stage 1 of 3] An earlier window of that episode. Overview image (one row per camera, left to right in time):
[[IMAGE:{dst}/pass.png]]
{tel_a}
Verdict that was given: {json.dumps({k: c['va'].get(k) for k in ('verdict', 'diagnosis', 'plan')}, ensure_ascii=False)}
{{NOTE_PASS}}

{xf.get('before', '')}
[Example, stage 2 of 3] A later window of the same episode:
[[IMAGE:{dst}/fault.png]]
{tel_b}
Verdict and plan that were given: {json.dumps({k: c['vb'].get(k) for k in ('verdict', 'diagnosis', 'plan')}, ensure_ascii=False)}
Execution log of that plan:
{chr(10).join(c['log'])}
{xf.get('ckpt', '')}{{NOTE_FAULT}}

"""
    if has_after:
        body += f"""[Example, stage 3 of 3] The window right after the hand-back to the policy:
[[IMAGE:{dst}/after.png]]
{{NOTE_AFTER}}
"""
    else:
        body += "[Example, stage 3 of 3] No further window: the episode ended during the hand-back.\n{NOTE_AFTER}\n"
    body += xf.get("end", "")
    body = re.sub(r"\n{3,}", "\n\n", body)
    open(f"{dst}/exemplar.txt", "w", encoding="utf-8").write(body)
    meta = {k: c[k] for k in ("score", "run", "cell", "task", "wa", "wb", "windows_after",
                              "instruction", "window_chunks")}
    json.dump(meta, open(f"{dst}/raw.json", "w"), ensure_ascii=False, indent=1)


def pass_candidates(family, seed):
    """Episodes the policy finished on its own: no intervention, yet the screener flagged
    windows and the judge looked and (rightly) passed them. These teach the normal case."""
    for res in glob.glob(f"{HERE}/runs_*_{family}/results.jsonl"):
        run = os.path.basename(os.path.dirname(res))
        for line in open(res, encoding="utf-8", errors="replace"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("seed") != seed or not r.get("treatment_success"):
                continue
            ev = r.get("events") or []
            if any(e.get("judge") not in ("ok", "screened") for e in ev):
                continue
            oks = [e["window"] for e in ev if e.get("judge") == "ok"]
            if len(oks) < 2:
                continue
            cd = f"{HERE}/{run}/lane{r.get('lane')}/{r['cell']}"
            wa, wb = oks[0], oks[-1]
            last = max(e["window"] for e in ev)
            imgs = [f"{cd}/w{wa}/wam_montage.png", f"{cd}/w{wb}/wam_montage.png"]
            if not all(os.path.exists(p) for p in imgs):
                continue
            end = ""
            for w in (last + 1, last):
                if w > wb and os.path.exists(f"{cd}/w{w}/wam_montage.png"):
                    end = f"{cd}/w{w}/wam_montage.png"; break
            imgs.append(end)
            va, vb = verdict_of(cd, wa), verdict_of(cd, wb)
            if not (va and vb and va.get("verdict") == "ok" and vb.get("verdict") == "ok"):
                continue
            instr = r.get("instruction") or ""
            if not instr:
                for pf in sorted(glob.glob(f"{cd}/p_w*.txt")):
                    m = re.search(r"^Task instruction: (.+)$", open(pf, errors="replace").read(), re.M)
                    if m:
                        instr = m.group(1).strip(); break
            engine = "gpt" if re.search(r"gpt|api|astra|^runs_g", run) else "other"
            score = (engine == "gpt") * 2 + bool(end) * 2 + min(len(oks), 4) + (r.get("control_success") is True)
            yield dict(score=score, run=run, cell=r["cell"], task=r["task"], cd=cd, wa=wa, wb=wb,
                       windows_after=last - wb, instruction=instr, va=va, vb=vb, imgs=imgs,
                       window_chunks=r.get("window_chunks"), ev=ev)


def write_pass(c, dst):
    os.makedirs(dst, exist_ok=True)
    for src, n in zip(c["imgs"], ["pass.png", "pass2.png", "end.png"]):
        if src:
            shutil.copyfile(src, f"{dst}/{n}")
    xf = extra_frames(c["cd"], dst, c["ev"], {c["wa"], c["wb"]} | set(range(c["wb"], 10 ** 6)))
    mid, wm = "", (c["wa"] + c["wb"]) // 2
    if c["wa"] < wm < c["wb"] and os.path.exists(f"{c['cd']}/w{wm}/wam_montage.png"):
        shutil.copyfile(f"{c['cd']}/w{wm}/wam_montage.png", f"{dst}/mid.png")
        mid = f"[Context] A window in between (not flagged, the policy working on):\n[[IMAGE:{dst}/mid.png]]\n\n"
    vj = lambda v: json.dumps({k: v.get(k) for k in ("verdict", "diagnosis", "plan")}, ensure_ascii=False)
    body = f"""Task instruction of the example episode: {c['instruction']}
(In this episode the monitor never intervened. A fast local check flagged these windows as unusual; the monitor looked and passed them.)

{xf.get('start', '')}
[Example, stage 1 of 3] A flagged window. Overview image (one row per camera, left to right in time):
[[IMAGE:{dst}/pass.png]]
{trim_tel(f"{c['cd']}/w{c['wa']}/telemetry.txt")}
Verdict that was given: {vj(c['va'])}
{{NOTE_PASS}}

{mid}[Example, stage 2 of 3] A later flagged window of the same episode:
[[IMAGE:{dst}/pass2.png]]
{trim_tel(f"{c['cd']}/w{c['wb']}/telemetry.txt")}
Verdict that was given: {vj(c['vb'])}
{{NOTE_FAULT}}

"""
    if c["imgs"][2]:
        body += f"[Example, stage 3 of 3] The last window of the episode, the policy still on its own:\n[[IMAGE:{dst}/end.png]]\n{{NOTE_AFTER}}\n"
    else:
        body += "[Example, stage 3 of 3] No later window was recorded.\n{NOTE_AFTER}\n"
    open(f"{dst}/exemplar.txt", "w", encoding="utf-8").write(body)
    meta = {k: c[k] for k in ("score", "run", "cell", "task", "wa", "wb", "windows_after", "instruction", "window_chunks")}
    meta["kind"] = "pass"
    json.dump(meta, open(f"{dst}/raw.json", "w"), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, default=195)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--top-pass", type=int, default=2)
    ap.add_argument("--out", default=f"{HERE}/fewshot_bank")
    a = ap.parse_args()
    assert a.seed < 400, "exemplars must come from the learn seed, never the seed-500 test set"
    by = {}
    for c in candidates(a.family, a.seed):
        by.setdefault(c["task"], []).append(c)
    for task, cs in sorted(by.items()):
        cs.sort(key=lambda c: (-c["score"], c["windows_after"]))
        seen, rank = set(), 0
        for c in cs:
            if c["cell"] in seen:
                continue
            seen.add(c["cell"]); rank += 1
            write_exemplar(c, f"{a.out}/{a.family}/{task}/{rank}_{c['cell']}")
            if rank >= a.top:
                break
        print(f"{task}: {len(cs)} candidate(s), wrote {rank}; best score {cs[0]['score']} "
              f"({cs[0]['run']} {cs[0]['cell']} w{cs[0]['wb']} +{cs[0]['windows_after']}w)")
    print(f"{len(by)} task(s) covered for {a.family} (repair exemplars)")
    byp = {}
    for c in pass_candidates(a.family, a.seed):
        byp.setdefault(c["task"], []).append(c)
    for task, cs in sorted(byp.items()):
        cs.sort(key=lambda c: -c["score"])
        seen, rank = set(), 0
        for c in cs:
            if c["cell"] in seen:
                continue
            seen.add(c["cell"]); rank += 1
            write_pass(c, f"{a.out}/{a.family}/{task}/p{rank}_{c['cell']}")
            if rank >= a.top_pass:
                break
        print(f"{task}: {len(cs)} pass candidate(s), wrote {rank}")
    print(f"{len(byp)} task(s) covered for {a.family} (policy-succeeded exemplars)")


if __name__ == "__main__":
    main()
