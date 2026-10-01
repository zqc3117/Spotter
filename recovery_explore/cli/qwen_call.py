#!/usr/bin/env python3
"""Judge channel for a local Qwen (vLLM, OpenAI chat completions); same interface as gpt6_call.py.

    qwen_call.py --prompt p.txt --out r.json [--image m.png] [--resume <transcript.json>]

Conversation continuity: vLLM has no previous_response_id, so a local transcript file stands in -- one json per episode,
and later windows carry the earlier turns. The context is only 32k, so when history is included, images from old windows are dropped,
only the most recent turns are kept in full, and the first turn (with the brief) is always kept.
"""
import argparse, base64, json, os, re, sys, time, urllib.request, urllib.error

HOST = os.environ.get("QWEN_HOST", os.environ.get("SCREENER_HOST", "127.0.0.1"))
PORTS = [int(p) for p in os.environ.get("QWEN_PORTS", "8301,8302,8303,8304").split(",")]
KEEP_RECENT_TURNS = int(os.environ.get("QWEN_KEEP_TURNS", "3"))   # number of recent turns kept in full (with images)


def _image_part(path: str) -> dict:
    mime = "image/png" if open(path, "rb").read(8)[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _trim(msgs: list) -> list:
    """Keep the first turn as is; strip images from and truncate middle turns; keep the last KEEP_RECENT_TURNS turns as is."""
    if len(msgs) <= 2 + 2 * KEEP_RECENT_TURNS:
        return msgs
    head, tail = msgs[:2], msgs[-2 * KEEP_RECENT_TURNS:]
    mid = []
    for m in msgs[2:-2 * KEEP_RECENT_TURNS]:
        if m["role"] == "user":
            text = " ".join(p.get("text", "") for p in m["content"] if isinstance(p, dict) and p.get("type") == "text")
            mid.append({"role": "user", "content": [{"type": "text", "text": text[:1500]}]})
        else:
            mid.append({"role": "assistant", "content": str(m["content"])[:1200]})
    return head + mid + tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image")
    ap.add_argument("--resume")
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--effort", default="medium")   # compatibility flag; ignored by the local model
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--think", choices=["0", "1"], default="0",
                    help="1: Qwen3 thinks before it answers (the driver asks this for the repair turns only)")
    a = ap.parse_args()

    prompt = open(a.prompt, encoding="utf-8", errors="replace").read()
    # Few-shot exemplars: [[IMAGE:<path>]] markers in the prompt are interleaved as
    # pictures (same convention as gpt6_call.py). No marker -> the old payload.
    content = []
    for i, part in enumerate(re.split(r"\[\[IMAGE:([^\]]+)\]\]", prompt)):
        if i % 2 == 0:
            if part.strip():
                content.append({"type": "text", "text": part})
        elif os.path.exists(part):
            content.append(_image_part(part))
        else:
            content.append({"type": "text", "text": "(example picture unavailable)"})
    if not content:
        content = [{"type": "text", "text": prompt}]
    if a.image and os.path.exists(a.image):
        content.append(_image_part(a.image))

    if a.resume and os.path.exists(a.resume):
        tpath = a.resume
        try:
            msgs = json.load(open(tpath))
        except Exception:
            msgs = []
    else:
        tpath = os.path.join(os.path.dirname(os.path.abspath(a.out)), f"qwen_session_{int(time.time() * 1000)}.json")
        msgs = []
    msgs.append({"role": "user", "content": content})
    # --think 1: the first request stops at the thinking budget (QWEN_THINK_BUDGET); if the thinking has
    # not closed by then, a closing line and </think> are appended as the assistant's own text and the
    # model continues with the answer (vLLM continue_final_message), which gets its own 3000 tokens.
    thinking = a.think == "1"
    think_budget = int(os.environ.get("QWEN_THINK_BUDGET") or 2000)
    answer_tokens = 3000
    payload = {"model": a.model, "messages": _trim(msgs), "max_tokens": think_budget if thinking else answer_tokens,
               "temperature": 0.2, "chat_template_kwargs": {"enable_thinking": thinking}}
    BUDGET_CLOSE = ("\n\nConsidering the limited time by the user, I have to give the answer based on the "
                    "thinking directly now.\n</think>\n\n")

    rec = {"result": "", "session_id": tpath, "duration_ms": 0}
    t0 = time.time(); last = None
    # Pick a replica by hashing the transcript file name; fall through to the next one if it is unreachable.
    start = sum(map(ord, os.path.basename(tpath))) % len(PORTS)
    for attempt in range(a.retries + 1):
        port = PORTS[(start + attempt) % len(PORTS)]
        try:
            req = urllib.request.Request(f"http://{HOST}:{port}/v1/chat/completions",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=a.timeout) as resp:
                data = json.loads(resp.read().decode())
            ch = (data.get("choices") or [{}])[0].get("message") or {}
            text = ch.get("content") or ""
            finish = (data.get("choices") or [{}])[0].get("finish_reason")
            cut_think = "</think>" not in text
            if thinking and (finish == "length" or (cut_think and "<think>" in text)):
                # out of budget inside the thinking: close it and continue with the answer;
                # out of tokens after the thinking closed (the answer itself was cut): continue the answer
                partial = text + BUDGET_CLOSE if cut_think else text
                cont = dict(payload, messages=list(payload["messages"]) + [{"role": "assistant", "content": partial}],
                            max_tokens=answer_tokens, add_generation_prompt=False, continue_final_message=True)
                req2 = urllib.request.Request(f"http://{HOST}:{port}/v1/chat/completions",
                                              data=json.dumps(cont).encode(),
                                              headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req2, timeout=a.timeout) as resp2:
                    data2 = json.loads(resp2.read().decode())
                text = partial + (((data2.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
                rec["think_budget_hit"] = cut_think
                data.setdefault("usage", {})["completion_tokens"] = (
                    int((data.get("usage") or {}).get("completion_tokens") or 0)
                    + int((data2.get("usage") or {}).get("completion_tokens") or 0))
            # Qwen3's chat template opens <think> in the prompt, so the reply carries the thinking with only
            # the closing tag (or both tags). The thinking goes to the record, never into the answer or the
            # transcript (Qwen3 wants earlier turns without it).
            think = [str(ch["reasoning_content"])] if ch.get("reasoning_content") else []
            m = re.search(r"</think>", text) if thinking else None
            if m:
                think.append(re.sub(r"^\s*<think>", "", text[:m.start()]).strip())
                text = text[m.end():]
            think += re.findall(r"<think>(.*?)</think>", text, flags=re.S)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            if think:
                rec["thinking"] = "\n".join(t for t in think if t)[:8000]
            rec["result"] = text
            u = data.get("usage") or {}
            rec["usage"] = {"input_tokens": int(u.get("prompt_tokens") or 0), "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0, "output_tokens": int(u.get("completion_tokens") or 0)}
            msgs.append({"role": "assistant", "content": text})
            json.dump(msgs, open(tpath, "w"), ensure_ascii=False)
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:600]
            last = f"HTTP {e.code}: {body}"
            if e.code == 400 and "context" in body.lower() and len(msgs) > 2:
                # context overflow: retry with only the first turn and the current one
                msgs = msgs[:2] + msgs[-1:]; payload["messages"] = msgs
                continue
            if attempt < a.retries:
                time.sleep(3); continue
            break
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            if attempt < a.retries:
                time.sleep(3); continue
            break
    rec["duration_ms"] = int((time.time() - t0) * 1000)
    if not rec["result"] and last:
        rec["result"] = json.dumps({"type": "error", "message": last}, ensure_ascii=False)
    json.dump(rec, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)


main()
