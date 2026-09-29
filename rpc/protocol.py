"""Versioned, pickle-free wire protocol for Cosmos Policy inference."""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

PROTOCOL_VERSION = 2
MEDIA_TYPE = "application/vnd.cosmos-policy.rpc+npz"
CAMERA_KEYS = ("primary_image", "secondary_image", "wrist_image")
MAX_PACKET_BYTES = 16 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024
_METADATA_KEY = "__metadata__"
_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


class ProtocolError(ValueError):
    """Raised when an RPC packet violates the protocol contract."""


@dataclass(frozen=True)
class Packet:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    task_description: str
    seed: int
    num_queries: int
    num_denoising_steps_action: int
    return_future_images: bool
    observation: dict[str, np.ndarray]

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(self.seed + index for index in range(self.num_queries))

    def to_bytes(self) -> bytes:
        metadata = {
            "kind": "inference_request",
            "request_id": self.request_id,
            "task_description": self.task_description,
            "seed": self.seed,
            "num_queries": self.num_queries,
            "num_denoising_steps_action": self.num_denoising_steps_action,
            "return_future_images": self.return_future_images,
        }
        validate_inference_request(self)
        return encode_packet(metadata, self.observation)

    @classmethod
    def from_bytes(cls, payload: bytes) -> "InferenceRequest":
        packet = decode_packet(payload, expected_kind="inference_request")
        metadata = packet.metadata
        request = cls(
            request_id=_required_str(metadata, "request_id"),
            task_description=_required_str(metadata, "task_description"),
            seed=_required_int(metadata, "seed"),
            num_queries=_required_int(metadata, "num_queries"),
            num_denoising_steps_action=_required_int(metadata, "num_denoising_steps_action"),
            return_future_images=_required_bool(metadata, "return_future_images"),
            observation=packet.arrays,
        )
        validate_inference_request(request)
        return request


@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    seeds: np.ndarray
    actions: np.ndarray
    values: np.ndarray
    selected_index: int
    elapsed_seconds: float
    future_images: dict[str, np.ndarray]

    @property
    def selected_actions(self) -> np.ndarray:
        return self.actions[self.selected_index]

    @property
    def selected_value(self) -> float:
        return float(self.values[self.selected_index])

    @property
    def selected_seed(self) -> int:
        return int(self.seeds[self.selected_index])

    def to_bytes(self) -> bytes:
        validate_inference_response(self)
        arrays = {
            "seeds": self.seeds,
            "actions": self.actions,
            "values": self.values,
            **self.future_images,
        }
        metadata = {
            "kind": "inference_response",
            "request_id": self.request_id,
            "selected_index": self.selected_index,
            "elapsed_seconds": self.elapsed_seconds,
        }
        return encode_packet(metadata, arrays)

    @classmethod
    def from_bytes(cls, payload: bytes) -> "InferenceResponse":
        packet = decode_packet(payload, expected_kind="inference_response")
        arrays = dict(packet.arrays)
        try:
            seeds = arrays.pop("seeds")
            actions = arrays.pop("actions")
            values = arrays.pop("values")
        except KeyError as exc:
            raise ProtocolError(f"response is missing array {exc.args[0]!r}") from exc
        response = cls(
            request_id=_required_str(packet.metadata, "request_id"),
            seeds=seeds,
            actions=actions,
            values=values,
            selected_index=_required_int(packet.metadata, "selected_index"),
            elapsed_seconds=_required_number(packet.metadata, "elapsed_seconds"),
            future_images=arrays,
        )
        validate_inference_response(response)
        return response


def encode_packet(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> bytes:
    metadata_dict = dict(metadata)
    metadata_dict["protocol_version"] = PROTOCOL_VERSION
    try:
        metadata_bytes = json.dumps(
            metadata_dict, separators=(",", ":"), sort_keys=True, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"metadata is not valid JSON: {exc}") from exc
    if len(metadata_bytes) > MAX_METADATA_BYTES:
        raise ProtocolError("metadata is too large")

    encoded_arrays: dict[str, np.ndarray] = {
        _METADATA_KEY: np.frombuffer(metadata_bytes, dtype=np.uint8)
    }
    uncompressed_size = len(metadata_bytes)
    for key, value in arrays.items():
        if not _KEY_RE.fullmatch(key) or key == _METADATA_KEY:
            raise ProtocolError(f"invalid array key: {key!r}")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ProtocolError(f"object arrays are forbidden: {key}")
        array = np.ascontiguousarray(array)
        uncompressed_size += array.nbytes
        encoded_arrays[key] = array
    if uncompressed_size > MAX_UNCOMPRESSED_BYTES:
        raise ProtocolError("uncompressed packet is too large")

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **encoded_arrays)
    payload = buffer.getvalue()
    if len(payload) > MAX_PACKET_BYTES:
        raise ProtocolError("compressed packet is too large")
    return payload


def decode_packet(
    payload: bytes,
    *,
    expected_kind: str | None = None,
    max_packet_bytes: int = MAX_PACKET_BYTES,
) -> Packet:
    if not payload:
        raise ProtocolError("packet is empty")
    if len(payload) > max_packet_bytes:
        raise ProtocolError("compressed packet is too large")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if _METADATA_KEY not in archive.files:
                raise ProtocolError("packet has no metadata")
            metadata_array = archive[_METADATA_KEY]
            if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
                raise ProtocolError("metadata has an invalid representation")
            if metadata_array.nbytes > MAX_METADATA_BYTES:
                raise ProtocolError("metadata is too large")
            metadata = json.loads(metadata_array.tobytes().decode("utf-8"))
            arrays: dict[str, np.ndarray] = {}
            uncompressed_size = metadata_array.nbytes
            for key in archive.files:
                if key == _METADATA_KEY:
                    continue
                if not _KEY_RE.fullmatch(key):
                    raise ProtocolError(f"invalid array key: {key!r}")
                array = archive[key]
                if array.dtype.hasobject:
                    raise ProtocolError(f"object arrays are forbidden: {key}")
                uncompressed_size += array.nbytes
                if uncompressed_size > MAX_UNCOMPRESSED_BYTES:
                    raise ProtocolError("uncompressed packet is too large")
                arrays[key] = np.array(array, copy=True)
    except ProtocolError:
        raise
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid NPZ packet: {exc}") from exc

    if not isinstance(metadata, dict):
        raise ProtocolError("metadata must be a JSON object")
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported protocol version: {metadata.get('protocol_version')!r}"
        )
    if expected_kind is not None and metadata.get("kind") != expected_kind:
        raise ProtocolError(f"expected packet kind {expected_kind!r}")
    return Packet(metadata=metadata, arrays=arrays)


def validate_inference_request(request: InferenceRequest) -> None:
    if not request.request_id or len(request.request_id) > 128:
        raise ProtocolError("request_id must contain 1-128 characters")
    if not request.task_description or len(request.task_description) > 4096:
        raise ProtocolError("task_description must contain 1-4096 characters")
    if not 0 <= request.seed <= (2**63 - 1) - request.num_queries:
        raise ProtocolError("seed is outside the supported deterministic range")
    if not 1 <= request.num_queries <= 32:
        raise ProtocolError("num_queries must be between 1 and 32")
    if not 1 <= request.num_denoising_steps_action <= 100:
        raise ProtocolError("num_denoising_steps_action must be between 1 and 100")
    required_keys = set(CAMERA_KEYS) | {"proprio"}
    # proprio_ext (chassis position + quaternion, 7 floats) and
    # proprio_base_rel (EEF pose RELATIVE to the chassis -- position +
    # quaternion, 7 floats, robosuite's own robot0_base_to_eef_pos/quat
    # sensors) are both OPTIONAL, added 2026-09 for non-Cosmos backends:
    # pi0.5's 16-dim state needs proprio_ext (absolute base pose);
    # GR00T-N1's state needs proprio_base_rel (base-relative EEF pose,
    # NOT the same quantity as proprio_ext -- confirmed by reading
    # GR00T's own RobocasaSinglePandaGripperDataConfig, which uses
    # state.end_effector_position/rotation_relative, not absolute). The
    # Cosmos backend ignores both if present. Any other extra key is
    # still rejected.
    allowed_keys = required_keys | {"proprio_ext", "proprio_base_rel"}
    observed_keys = set(request.observation)
    if not required_keys <= observed_keys or not observed_keys <= allowed_keys:
        raise ProtocolError(
            f"observation arrays must be {sorted(required_keys)} "
            f"plus optionally 'proprio_ext'/'proprio_base_rel', got {sorted(observed_keys)}"
        )
    for key in CAMERA_KEYS:
        image = np.asarray(request.observation[key])
        if image.dtype != np.uint8 or image.shape != (224, 224, 3):
            raise ProtocolError(f"{key} must be uint8 with shape (224, 224, 3)")
    proprio = np.asarray(request.observation["proprio"])
    if proprio.shape != (9,) or proprio.dtype != np.float64:
        raise ProtocolError("proprio must be float64 with shape (9,)")
    if not np.isfinite(proprio).all():
        raise ProtocolError("proprio must contain finite values")
    if "proprio_ext" in request.observation:
        proprio_ext = np.asarray(request.observation["proprio_ext"])
        if proprio_ext.shape != (7,) or proprio_ext.dtype != np.float64:
            raise ProtocolError("proprio_ext must be float64 with shape (7,) (base_pos[3] + base_quat[4])")
        if not np.isfinite(proprio_ext).all():
            raise ProtocolError("proprio_ext must contain finite values")
    if "proprio_base_rel" in request.observation:
        proprio_base_rel = np.asarray(request.observation["proprio_base_rel"])
        if proprio_base_rel.shape != (7,) or proprio_base_rel.dtype != np.float64:
            raise ProtocolError(
                "proprio_base_rel must be float64 with shape (7,) (base_to_eef_pos[3] + base_to_eef_quat[4])"
            )
        if not np.isfinite(proprio_base_rel).all():
            raise ProtocolError("proprio_base_rel must contain finite values")


def validate_inference_response(response: InferenceResponse) -> None:
    seeds = np.asarray(response.seeds)
    actions = np.asarray(response.actions)
    values = np.asarray(response.values)
    if seeds.ndim != 1 or seeds.dtype != np.int64:
        raise ProtocolError("response seeds must be a one-dimensional int64 array")
    # Action last-dim is 7 for the Cosmos/RoboCasa Franka+parallel-gripper
    # convention. Widened 2026-09 to also allow 12 for policies whose native
    # action space includes extra dims (e.g. a mobile base) -- passed through
    # untouched by _to_environment_action, never reinterpreted as 7-dim.
    if actions.ndim != 3 or actions.shape[0] != seeds.size or actions.shape[2] not in (7, 12):
        raise ProtocolError("response actions must have shape (N, chunk_size, 7) or (N, chunk_size, 12)")
    if actions.dtype != np.float64 or not np.isfinite(actions).all():
        raise ProtocolError("response actions must contain finite float64 values")
    if values.shape != seeds.shape or values.dtype != np.float64 or not np.isfinite(values).all():
        raise ProtocolError("response values must be finite float64 values with shape (N,)")
    if not 0 <= response.selected_index < seeds.size:
        raise ProtocolError("selected_index is out of range")
    if not np.isfinite(response.elapsed_seconds) or response.elapsed_seconds < 0:
        raise ProtocolError("elapsed_seconds must be finite and non-negative")
    for key, image_batch in response.future_images.items():
        if not key.startswith("future_"):
            raise ProtocolError(f"unexpected response array: {key}")
        batch = np.asarray(image_batch)
        if batch.dtype != np.uint8 or batch.ndim != 4 or batch.shape[0] != seeds.size or batch.shape[-1] != 3:
            raise ProtocolError(f"{key} must be uint8 with shape (N, H, W, 3)")


def _required_str(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str):
        raise ProtocolError(f"metadata field {key!r} must be a string")
    return value


def _required_int(metadata: Mapping[str, Any], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"metadata field {key!r} must be an integer")
    return value


def _required_bool(metadata: Mapping[str, Any], key: str) -> bool:
    value = metadata.get(key)
    if not isinstance(value, bool):
        raise ProtocolError(f"metadata field {key!r} must be a boolean")
    return value


def _required_number(metadata: Mapping[str, Any], key: str) -> float:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"metadata field {key!r} must be numeric")
    return float(value)
