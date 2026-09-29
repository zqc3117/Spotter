"""HTTP client for the portable Cosmos Policy inference service."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
import uuid
from typing import Mapping

import numpy as np

from .protocol import MEDIA_TYPE, InferenceRequest, InferenceResponse, ProtocolError


class RemoteInferenceError(RuntimeError):
    """Raised when the remote inference service rejects or fails a request."""


class CosmosPolicyClient:
    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: float = 900.0,
        retries: int = 1,
    ) -> None:
        endpoint = endpoint.rstrip("/")
        self.infer_url = endpoint if endpoint.endswith("/v1/infer") else endpoint + "/v1/infer"
        self.health_url = self.infer_url.removesuffix("/v1/infer") + "/health"
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if retries < 0:
            raise ValueError("retries must be non-negative")
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def health(self) -> dict:
        request = urllib.request.Request(self.health_url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as exc:
            raise RemoteInferenceError(f"health request failed: {exc}") from exc

    def infer(
        self,
        observation: Mapping[str, np.ndarray],
        task_description: str,
        *,
        seed: int,
        num_queries: int = 1,
        num_denoising_steps_action: int = 5,
        return_future_images: bool = False,
        request_id: str | None = None,
    ) -> InferenceResponse:
        inference_request = InferenceRequest(
            request_id=request_id or str(uuid.uuid4()),
            task_description=task_description,
            seed=seed,
            num_queries=num_queries,
            num_denoising_steps_action=num_denoising_steps_action,
            return_future_images=return_future_images,
            observation={key: np.asarray(value) for key, value in observation.items()},
        )
        return self.infer_request(inference_request)

    def infer_request(self, inference_request: InferenceRequest) -> InferenceResponse:
        payload = inference_request.to_bytes()
        last_error: BaseException | None = None
        deadline = time.monotonic() + self.timeout_seconds
        for attempt in range(self.retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            request = urllib.request.Request(
                self.infer_url,
                data=payload,
                method="POST",
                headers={
                    "Content-Type": MEDIA_TYPE,
                    "Accept": MEDIA_TYPE,
                    "X-Cosmos-Queue-Timeout-Seconds": f"{remaining:.6f}",
                    "X-Cosmos-Wait-Timeout-Seconds": f"{remaining:.6f}",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=remaining) as response:
                    content_type = response.headers.get_content_type()
                    body = response.read()
                if content_type != MEDIA_TYPE:
                    raise RemoteInferenceError(f"unexpected response content type: {content_type}")
                result = InferenceResponse.from_bytes(body)
                if result.request_id != inference_request.request_id:
                    raise ProtocolError("response request_id does not match request")
                return result
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code not in {429, 502, 503, 504} or attempt >= self.retries:
                    raise RemoteInferenceError(f"server returned HTTP {exc.code}: {body}") from exc
                last_error = exc
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
            if attempt < self.retries:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.25 * (2**attempt), 2.0, remaining))
        raise RemoteInferenceError(f"inference request failed: {last_error}") from last_error
