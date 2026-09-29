"""VLM-based failure diagnosis for post_failure anchors (2026-09-08).

Given a post_failure anchor (RoboCasa CF-branch collection), ask a local
Qwen VLM (served via vllm's OpenAI-compatible API, see README.md) to look
at:

  - up to 8 sampled frames of the primary camera and 8 of the wrist camera
    covering the 16 steps immediately BEFORE the failure query
    (``prefailure_frames``/``collect_prefailure_frames_via_replay``),
  - the current (post-restore) frame from all three cameras, re-rendered at
    512x512 (``render_camera_512``),
  - the original task instruction and the gripper open/closed state,

and return strict JSON:
    {"failure_type": "missed|slipped|dropped|misaligned|other",
     "object_in_gripper": bool,
     "correction": "<one corrective instruction sentence>"}

This module only talks to the OpenAI-compatible ``/v1/chat/completions``
endpoint -- no custom scorer-style server is needed, vllm's own server IS
the server (matching the user's exact deploy commands).

Parse failures retry ONCE with a stricter reminder appended; a second
failure is recorded as ``{"parse_error": true}`` and the raw text is kept
verbatim in ``raw_response`` for offline inspection -- callers should skip
(not crash) on parse_error, per the user's spec.
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("cf_bench.vlm_failure_diagnosis requires Pillow (pip install pillow)") from exc

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("cf_bench.vlm_failure_diagnosis requires requests (pip install requests)") from exc


VALID_FAILURE_TYPES = ("missed", "slipped", "dropped", "misaligned", "other")

SYSTEM_PROMPT = (
    "You are a robot-manipulation failure analyst. You are shown: (1) up to "
    "8 frames from the robot's main camera and 8 from its wrist camera, "
    "sampled over the 16 simulation steps immediately BEFORE a manipulation "
    "failure was detected (in chronological order), and (2) the current "
    "frame from three cameras (main-left, main-right, wrist) at the exact "
    "moment execution was paused. You also get the original task "
    "instruction and whether the gripper is currently open or closed.\n\n"
    "Diagnose why the task attempt failed and propose ONE short corrective "
    "instruction that, if given to the policy next, would help it recover "
    "and complete the original task from the current state.\n\n"
    "Respond with STRICT JSON only, no markdown fences, no prose outside "
    "the JSON object, matching exactly this schema:\n"
    '{"failure_type": "missed|slipped|dropped|misaligned|other", '
    '"object_in_gripper": true|false, '
    '"correction": "<one imperative sentence, e.g. \'open the gripper, '
    "move slightly left, lower and regrasp the <object>'>\"}"
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _encode_frame(frame: np.ndarray, *, max_side: int = 512) -> str:
    """np.uint8 HWC RGB array -> data: URL (JPEG, quality 90)."""
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    image = Image.fromarray(array, mode="RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _image_content(frame: np.ndarray, *, max_side: int = 512) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": _encode_frame(frame, max_side=max_side)}}


@dataclass(frozen=True)
class VlmDiagnosisInput:
    """Everything the prompt needs for one anchor."""

    task_description: str
    gripper_open: bool
    prefailure_primary_frames: Sequence[np.ndarray]  # up to 8, chronological
    prefailure_wrist_frames: Sequence[np.ndarray]  # up to 8, chronological
    current_frames: Mapping[str, np.ndarray]  # {"primary":..., "secondary":..., "wrist":...} at 512x512
    prefailure_frames_synthetic: bool = False  # True if not a real replay (see collect_prefailure_frames_via_replay)


def build_messages(vlm_input: VlmDiagnosisInput) -> list[dict[str, Any]]:
    gripper_state = "open" if vlm_input.gripper_open else "closed"
    provenance_note = (
        " (NOTE: these pre-failure frames are a synthesized stand-in, not a "
        "literal chronological replay -- reason about them as representative "
        "context, not an exact timeline.)"
        if vlm_input.prefailure_frames_synthetic
        else ""
    )
    text = (
        f"Original task instruction: {vlm_input.task_description!r}\n"
        f"Current gripper state: {gripper_state}\n\n"
        f"Below: up to {len(vlm_input.prefailure_primary_frames)} main-camera "
        f"frames then up to {len(vlm_input.prefailure_wrist_frames)} "
        f"wrist-camera frames from the 16 steps before the failure was "
        f"detected{provenance_note}, followed by the current frame from the "
        "main-left, main-right and wrist cameras. Diagnose the failure and "
        "propose one correction. JSON only."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for frame in vlm_input.prefailure_primary_frames:
        content.append(_image_content(frame))
    for frame in vlm_input.prefailure_wrist_frames:
        content.append(_image_content(frame))
    for camera_name in ("primary", "secondary", "wrist"):
        frame = vlm_input.current_frames.get(camera_name)
        if frame is not None:
            content.append({"type": "text", "text": f"current_{camera_name}:"})
            content.append(_image_content(frame))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    # Qwen3.5-hybrid emits a <think>...</think> reasoning block before the
    # real answer even at temperature=0 (confirmed 2026-09-08 smoke test).
    # Strip it before the greedy `{.*}` regex fallback below, or a stray
    # brace inside the reasoning text could make that regex span from the
    # wrong opening brace all the way to the real JSON's closing one.
    text = _THINK_BLOCK_RE.sub("", text)
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _validate_parsed(parsed: Mapping[str, Any]) -> dict[str, Any] | None:
    failure_type = parsed.get("failure_type")
    object_in_gripper = parsed.get("object_in_gripper")
    correction = parsed.get("correction")
    if failure_type not in VALID_FAILURE_TYPES:
        return None
    if not isinstance(object_in_gripper, bool):
        return None
    if not isinstance(correction, str) or not correction.strip():
        return None
    return {
        "failure_type": failure_type,
        "object_in_gripper": object_in_gripper,
        "correction": correction.strip(),
    }


def call_vlm(
    vlm_input: VlmDiagnosisInput,
    *,
    base_url: str,
    model: str = "qwen38",
    temperature: float = 0.0,
    timeout: float = 120.0,
    max_tokens: int = 800,
) -> dict[str, Any]:
    """Calls the vllm OpenAI-compatible server. Retries once on a parse
    failure (JSON missing/invalid/wrong schema); a second failure returns
    {"parse_error": True, "raw_response": <last raw text>}. Always includes
    "raw_response" (the last attempt's raw text) for offline auditing, even
    on success.

    ``chat_template_kwargs: {"enable_thinking": false}`` is passed on every
    call: Qwen3.5-hybrid emits a verbose <think>...</think> block by default
    even at temperature=0 (confirmed 2026-09-08), which at the old
    max_tokens=300 default reliably ate the whole budget and truncated the
    JSON before it ever closed. Disabling thinking mode is the real fix;
    max_tokens=800 (up from 300) and the <think>-stripping in
    _extract_json are defense in depth in case a served checkpoint ignores
    the flag."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    messages = build_messages(vlm_input)
    last_raw = ""
    for attempt in range(2):
        payload_messages = messages
        if attempt == 1:
            # stricter reminder appended as a fresh user turn, per spec:
            # "retry once on parse failure"
            payload_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON matching the "
                        "required schema. Reply with ONLY the JSON object, "
                        "nothing else -- no markdown fences, no explanation."
                    ),
                }
            ]
        response = requests.post(
            url,
            json={
                "model": model,
                "messages": payload_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        raw_text = body["choices"][0]["message"]["content"]
        last_raw = raw_text
        parsed = _extract_json(raw_text)
        validated = _validate_parsed(parsed) if parsed is not None else None
        if validated is not None:
            validated["raw_response"] = raw_text
            validated["parse_error"] = False
            validated["attempts"] = attempt + 1
            return validated
    return {"parse_error": True, "raw_response": last_raw, "attempts": 2}
