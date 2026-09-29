#!/usr/bin/env python3
"""Judge channel that calls the OpenAI Responses API directly, used instead of the codex CLI as a control.

    gpt6_call.py --prompt p.txt --out r.json [--image m.png] [--resume <id>]
                 [--model gpt-6-astra] [--effort medium] [--timeout 600]

The output json has the same shape as the one the codex branch writes, so the driver's parsing is unchanged:
    {"result": str, "session_id": str, "duration_ms": int, "usage": {...}}

Why not use a plain-text CLI wrapper: its input is a single string,
so it can pass neither images nor previous_response_id. The judge needs the overview image in every window, and
the whole trajectory is one conversation -- dropping both would not be "the same model over another channel" but a much weaker
setup that would necessarily score worse and prove nothing. So we build the multimodal request ourselves with the same endpoint/proxy/key.
"""
import argparse, base64, json, os, re, sys, time, urllib.request, urllib.error

TOKEN_FILE = os.environ.get("OPENAI_API_TOKEN_FILE", "/etc/openai_api_token")
PROXY = os.environ.get("OPENAI_PROXY", "http://127.0.0.1:10809")
ENDPOINT = os.environ.get("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses")


def text_of(data: dict) -> str:
    if data.get("output_text"):
        return data["output_text"]
    out = []
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") in ("output_text", "text") and part.get("text"):
                out.append(part["text"])
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image")
    ap.add_argument("--resume")
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()

    prompt = open(a.prompt, encoding="utf-8", errors="replace").read()
    # Few-shot exemplars carry their own pictures as [[IMAGE:<path>]] markers inside the
    # prompt; split on them and interleave, so each picture sits next to its text (the
    # live --image still goes last). No marker -> exactly the old single-text payload.
    def _img(path):
        mime = "image/png" if open(path, "rb").read(8)[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        return {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"}
    content = []
    for i, part in enumerate(re.split(r"\[\[IMAGE:([^\]]+)\]\]", prompt)):
        if i % 2 == 0:
            if part.strip():
                content.append({"type": "input_text", "text": part})
        elif os.path.exists(part):
            content.append(_img(part))
        else:
            content.append({"type": "input_text", "text": "(example picture unavailable)"})
    if not content:
        content = [{"type": "input_text", "text": prompt}]
    if a.image and os.path.exists(a.image):
        mime = "image/png" if open(a.image, "rb").read(8)[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        b64 = base64.b64encode(open(a.image, "rb").read()).decode()
        content.append({"type": "input_image", "image_url": f"data:{mime};base64,{b64}"})

    payload = {"model": a.model, "reasoning": {"effort": a.effort},
               "input": [{"role": "user", "content": content}]}
    # Conversation continuity: previous_response_id lets the server continue from earlier windows, equivalent to codex resume.
    if a.resume:
        payload["previous_response_id"] = a.resume

    key = open(TOKEN_FILE).read().strip()
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))

    rec = {"result": "", "session_id": "", "duration_ms": 0}
    t0 = time.time()
    last = None
    for attempt in range(a.retries + 1):
        try:
            req = urllib.request.Request(
                ENDPOINT, data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                method="POST")
            with opener.open(req, timeout=a.timeout) as resp:
                data = json.loads(resp.read().decode())
            rec["result"] = text_of(data)
            rec["session_id"] = data.get("id") or ""
            u = data.get("usage") or {}
            cached = int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0)
            rec["usage"] = {"input_tokens": max(int(u.get("input_tokens") or 0) - cached, 0),
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": cached,
                            "output_tokens": int(u.get("output_tokens") or 0)}
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:600]
            last = f"HTTP {e.code}: {body}"
            # if previous_response_id has expired or is invalid, drop the session and retry once instead of killing the whole lane
            if e.code in (400, 404) and payload.pop("previous_response_id", None):
                continue
            if e.code in (429, 500, 502, 503, 504) and attempt < a.retries:
                time.sleep(5 * (attempt + 1)); continue
            break
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            if attempt < a.retries:
                time.sleep(5 * (attempt + 1)); continue
            break
    rec["duration_ms"] = int((time.time() - t0) * 1000)
    if not rec["result"] and last:
        # the driver's hard-error guard recognises this shape: an error object aborts the lane instead of silently counting as ok
        rec["result"] = json.dumps({"type": "error", "message": last}, ensure_ascii=False)
    json.dump(rec, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)


main()
