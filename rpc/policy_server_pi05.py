"""HTTP inference server for the pi0.5 (LeRobot PI05Policy) checkpoint,
speaking the SAME wire protocol (rpc/protocol.py InferenceRequest/
InferenceResponse over POST /v1/infer, GET /health) as the official
Cosmos rpc/server.py -- so the existing rpc/client.py CosmosPolicyClient
and all downstream runners work against this server unmodified.

pi0.5-specific alignment (verified against the actual checkpoint on
2026-09-07/08):
  - Loaded via LeRobot's own `get_policy_class("pi05").from_pretrained`
    + `make_pre_post_processors(pretrained_path=...)` -- normalization
    stats (QUANTILES) load from the checkpoint's own safetensors, not
    re-derived here.
  - REQUIRES lerobot>=0.6.x (git main, NOT the PyPI 0.4.4 release, which
    ships a broken pi05 that gates on a transformers_replace patch that
    doesn't exist anywhere) and Python>=3.12 (lerobot main's own
    requirement). Run this file with the dedicated
    `.venvs/pi05_py312/bin/python` interpreter, never $COSMOS_POLICY_ENV.
  - Checkpoint's own config: observation.state shape (16,). CONFIRMED
    2026-09-08 against the actual training data conversion script
    (`robocasa-benchmark/openpi`'s `examples/robocasa/
    convert_robocasa_to_lerobot.py`): state = concat([
    robot0_base_to_eef_pos(3), robot0_base_to_eef_quat(4),
    robot0_base_pos(3), robot0_base_quat(4), robot0_gripper_qpos(2)])
    -- i.e. `proprio_base_rel`(7) + `proprio_ext`(7) + `proprio[0:2]`
    (gripper only) in THIS project's protocol terms. An earlier guess
    (plain `proprio`(9) + `proprio_ext`(7), absolute EEF pose first,
    gripper first) was WRONG -- confirmed by the first 24x5 coarse gate
    (18.3% overall, but every single precision pick-and-place task
    scored exactly 0/5 while push/turn/open/close tasks partially
    succeeded, the exact signature of a garbled EEF-position input).
  - 3 cameras at 128x128 (checkpoint's own input_features), keyed by
    LeRobot's `observation.images.robot0_agentview_left_image` /
    `..._right_image` / `..._eye_in_hand_image` -- mapped 1:1 from this
    project's primary/secondary/wrist camera convention (same mapping
    Cosmos's own prepare_observation uses), resized down from the
    protocol's 224x224 to 128x128 (area-interpolation, since we are
    downsizing).
  - Native execution horizon: n_action_steps=25 out of chunk_size=50
    (confirmed from the loaded config -- supersedes any looser "32"
    assumption). This server returns the FULL 50-step chunk in
    `future_images`-free InferenceResponse.actions (shape (K, 50, 12));
    the CALLER (a new eval runner, not yet written) is responsible for
    executing only the first `n_action_steps` and requesting fresh
    candidates for the next chunk, per the coarse-gate protocol's own
    "predict 50, execute 25" rule -- this server does not know or enforce that
    truncation itself, matching how the Cosmos backend also lets the
    caller decide execution length.
  - values are all 0.0 (no value head); no future_images ever returned
    (return_future_images is accepted but ignored, matching the
    documented plan -- pi0.5 does not imagine future frames).
  - One seed -> one candidate; K = num_queries independent forward
    passes, each with `torch.manual_seed(seed_i)` before sampling (the
    model's own sampling noise is seeded, matching the protocol's
    seeds contract as closely as a diffusion/flow action head allows).

Run:
    CUDA_VISIBLE_DEVICES=0 <pi05-venv>/bin/python \
        -m rpc.policy_server_pi05 \
        --host 0.0.0.0 --port 8900 \
        --checkpoint <checkpoints>/pi05-robocasa-H50

  Run it from the repository root (the module lives in rpc/). The venv is the
  Python 3.12 one with lerobot installed; see README.md section 4.2.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import MEDIA_TYPE

LOGGER = logging.getLogger("pi05_rpc")
MAX_BODY_BYTES = 16 * 1024 * 1024

# Cosmos protocol camera key -> RoboCasa observation key -> pi0.5's LeRobot
# feature key. Mirrors rpc/run_remote_robocasa_collect.py's own
# prepare_observation camera_mapping (primary=agentview_left,
# secondary=agentview_right, wrist=eye_in_hand).
CAMERA_TO_LEROBOT_KEY = {
    "primary_image": "observation.images.robot0_agentview_left_image",
    "secondary_image": "observation.images.robot0_agentview_right_image",
    "wrist_image": "observation.images.robot0_eye_in_hand_image",
}


class PI05Backend:
    """Wraps a loaded PI05Policy behind the InferenceRequest/Response contract."""

    def __init__(self, checkpoint: Path, device: str = "cuda") -> None:
        import torch
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self._torch = torch
        self.device = device
        self.checkpoint = Path(checkpoint)

        LOGGER.info("loading pi05 policy from %s", self.checkpoint)
        policy_cls = get_policy_class("pi05")
        self.policy = policy_cls.from_pretrained(str(self.checkpoint))
        self.policy.to(device)
        self.policy.eval()

        # The checkpoint's preprocessor.json references the tokenizer by its
        # original HF hub name (google/paligemma-3b-pt-224), which is a
        # GATED repo requiring an authenticated, access-granted HF token --
        # not available on this cluster. The checkpoint bundles its own
        # tokenizer files locally (tokenizer/tokenizer.json etc, confirmed
        # 2026-09-08), so override the tokenizer_processor step's
        # tokenizer_name to that local directory instead of re-downloading.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": str(self.checkpoint / "tokenizer")}},
        )

        self.chunk_size = int(self.policy.config.chunk_size)
        self.n_action_steps = int(self.policy.config.n_action_steps)
        self.action_dim = int(self.policy.config.output_features["action"].shape[0])
        self.image_size = 128  # checkpoint's own input_features image shape
        self.requests_served = 0
        self._lock = threading.Lock()  # one GPU, one forward pass at a time
        LOGGER.info(
            "pi05 ready: chunk_size=%d n_action_steps=%d action_dim=%d",
            self.chunk_size, self.n_action_steps, self.action_dim,
        )

    def health(self) -> dict:
        return {
            "status": "ok",
            "kind": "policy_server_pi05",
            "checkpoint": str(self.checkpoint),
            "chunk_size": self.chunk_size,
            "n_action_steps": self.n_action_steps,
            "action_dim": self.action_dim,
            "requests_served": self.requests_served,
        }

    def _resize_to_128(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image

        img = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
        img = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

    def _build_raw_batch(self, observation: dict[str, np.ndarray], task_description: str) -> dict[str, Any]:
        torch = self._torch
        raw: dict[str, Any] = {"task": task_description}
        for camera_key, lerobot_key in CAMERA_TO_LEROBOT_KEY.items():
            image = self._resize_to_128(observation[camera_key])
            # CHW float32 [0, 1], matching LeRobot's own VISUAL feature convention.
            tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
            raw[lerobot_key] = tensor
        # State field order CONFIRMED 2026-09-08 against the actual RoboCasa
        # training data conversion script
        # (robocasa-benchmark/openpi's examples/robocasa/
        # convert_robocasa_to_lerobot.py): state = concat([
        #   robot0_base_to_eef_pos(3), robot0_base_to_eef_quat(4),
        #   robot0_base_pos(3), robot0_base_quat(4), robot0_gripper_qpos(2)])
        # -- i.e. BASE-RELATIVE eef pose first, gripper LAST. This
        # supersedes the earlier (proprio + proprio_ext) guess, which put
        # the ABSOLUTE eef pose first and gripper first too -- confirmed
        # wrong by the first 24x5 coarse gate (18.3% overall, but a
        # striking pattern: ALL 8 precision pick-and-place tasks scored
        # exactly 0/5 while push/turn/open/close tasks partially
        # succeeded -- exactly what a wrong EEF-position input produces,
        # since PnP is far more sensitive to precise reach/grasp than a
        # door/drawer/button/knob interaction is).
        proprio = np.asarray(observation["proprio"], dtype=np.float64)  # [gripper_qpos(2), eef_pos(3), eef_quat(4)]
        proprio_ext = observation.get("proprio_ext")
        proprio_base_rel = observation.get("proprio_base_rel")
        if proprio_ext is None or proprio_base_rel is None:
            raise RuntimeError(
                "pi05 requires both 'proprio_ext' (base_pos+base_quat) and 'proprio_base_rel' "
                "(base_to_eef_pos+base_to_eef_quat) in the observation -- confirm the RoboCasa "
                "env is a MobileRobot (robots=PandaMobile)."
            )
        proprio_ext = np.asarray(proprio_ext, dtype=np.float64)  # [base_pos(3), base_quat(4)]
        proprio_base_rel = np.asarray(proprio_base_rel, dtype=np.float64)  # [base_to_eef_pos(3), base_to_eef_quat(4)]
        gripper_qpos = proprio[0:2]
        state = np.concatenate([proprio_base_rel, proprio_ext, gripper_qpos]).astype(np.float32)
        if state.shape != (16,):
            raise RuntimeError(f"pi05 requires a 16-dim state; got {state.shape}")
        raw["observation.state"] = torch.from_numpy(state)
        return raw

    def infer(self, request: Any) -> Any:
        from .protocol import InferenceResponse

        torch = self._torch
        started = time.monotonic()
        seeds = np.asarray(list(request.seeds), dtype=np.int64)
        raw_batch = self._build_raw_batch(request.observation, request.task_description)

        actions_per_seed = []
        with self._lock, torch.no_grad():
            for seed in seeds:
                torch.manual_seed(int(seed))
                batch = self.preprocessor(dict(raw_batch))
                for key, value in list(batch.items()):
                    if isinstance(value, torch.Tensor) and value.ndim in (1, 3):
                        batch[key] = value.unsqueeze(0)  # add batch dim
                batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                chunk = self.policy.predict_action_chunk(batch)  # (1, chunk_size, action_dim)
                chunk = self.postprocessor(chunk.squeeze(0))  # unnormalize -> (chunk_size, action_dim)
                actions_per_seed.append(chunk.detach().cpu().to(torch.float64).numpy())
            self.requests_served += 1

        actions = np.stack(actions_per_seed, axis=0)  # (K, chunk_size, action_dim)
        values = np.zeros(len(seeds), dtype=np.float64)
        return InferenceResponse(
            request_id=request.request_id,
            seeds=seeds,
            actions=actions,
            values=values,
            selected_index=0,
            elapsed_seconds=time.monotonic() - started,
            future_images={},
        )


class PolicyService:
    def __init__(self, backend: PI05Backend) -> None:
        self.backend = backend

    def health(self) -> dict:
        return self.backend.health()

    def infer(self, payload: bytes) -> bytes:
        from .protocol import InferenceRequest

        request = InferenceRequest.from_bytes(payload)
        response = self.backend.infer(request)
        return response.to_bytes()


def create_http_server(host: str, port: int, service: PolicyService) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._send_json(HTTPStatus.OK, service.health())

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/infer":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid content length"})
                return
            body = self.rfile.read(length)
            try:
                response = service.infer(body)
            except Exception as exc:  # report, never crash the server on one bad request
                LOGGER.exception("inference failed")
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self.send_response(HTTPStatus.OK)
            # rpc/client.py's CosmosPolicyClient strictly checks the
            # response Content-Type equals MEDIA_TYPE (not the generic
            # application/octet-stream cf_bench/scorer_server.py uses --
            # that server has its own client, this one must speak to the
            # existing CosmosPolicyClient unmodified).
            self.send_header("Content-Type", MEDIA_TYPE)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def _send_json(self, status: HTTPStatus, body: dict) -> None:
            data = json.dumps(body, sort_keys=True).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args: object) -> None:
            LOGGER.debug(fmt, *args)

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    backend = PI05Backend(args.checkpoint, device=args.device)
    service = PolicyService(backend)
    server = create_http_server(args.host, args.port, service)
    LOGGER.info("pi05 policy server listening on http://%s:%s", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
