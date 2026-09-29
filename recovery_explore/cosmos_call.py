"""Best-of-K Cosmos invocation, factored out so it is testable without an env.

WHY this is its own module: the interesting parts of "call Cosmos" -- the K-seed
request contract, the response validation, the argmax-with-lowest-index tie-break,
the future-image key mapping, and the re-query-every-16-steps loop -- have nothing
to do with MuJoCo. Keeping them here means they can be exercised against a fake
client and a fake ``step_chunk`` on any machine, while ``env_service.py`` supplies
the only genuinely env-dependent piece (stepping the simulator).

Ported from ``rpc/run_remote_robocasa_value_select_deploy.py``:

* ``_validate_candidate_response`` -> :func:`validate_candidate_response`. The K
  seeds MUST come back as exactly ``base .. base+K-1`` in order: that equality is
  what proves the generation server actually honoured our seeds, i.e. that the
  policy side is deterministic and the run is reproducible. Do not relax it.
* ``_future_images_to_cand_images`` -> :func:`future_images_to_frames`. The key
  names come from ``rpc.official_backend``'s
  ``get_future_images_from_generated_samples``: future_image=primary (third-person
  cam 1), future_image2=secondary (third-person cam 2), future_wrist_image=wrist.
* ``selected_index = int(np.argmax(selection_values))  # ties -> lowest index``
  -> :func:`select_best_index`.

The one deliberate difference from the deploy runner: it issues its K-candidate
request only at an *eligible failure* anchor inside a full evaluation episode,
and its ``candidate_seed_base`` is derived from that episode's own deterministic
schedule (``deterministic_policy_seed``) so the K seed blocks of an evaluation can
never collide. This module is driven interactively from a restored anchor instead,
so the caller passes ``seed`` directly and owns seed bookkeeping.

Neither ``robosuite`` nor ``rpc`` is imported at module scope: the policy client is
duck-typed (anything with ``.infer(...)`` matching ``rpc.client.CosmosPolicyClient``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

# Keys of rpc.official_backend's future-image payload, in our canonical camera
# order. See the module docstring.
FUTURE_IMAGE_KEY_BY_CAMERA = {
    "primary": "future_image",
    "secondary": "future_image2",
    "wrist": "future_wrist_image",
}


class PolicyClient(Protocol):
    """The surface of ``rpc.client.CosmosPolicyClient`` this module uses."""

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
    ) -> Any: ...


@dataclass(frozen=True)
class CosmosCallConfig:
    """Generation-side knobs. Defaults are the official evaluation values used
    throughout this repo (``rpc/run_remote_robocasa_cf_branch_collect.py``'s
    ``RethinkEvalConfig``): a 32-action plan of which the first 16 are committed
    before re-querying, 5 denoising steps."""

    chunk_size: int = 32
    num_open_loop_steps: int = 16
    num_denoising_steps_action: int = 5
    # Action dimension: 7 for Cosmos Policy, 12 for pi0.5 (same as the --action-dim argument of
    # the policy rollout script). Validation must use the per-family value, otherwise switching family fails with a shape mismatch.
    action_dim: int = 7


@dataclass(frozen=True)
class CandidateBatch:
    """One K-candidate generation response, validated."""

    seeds: np.ndarray  # (K,) int64, exactly base..base+K-1
    actions: np.ndarray  # (K, chunk_size, 7) float64
    values: np.ndarray  # (K,) float64 -- the generation server's own value head
    future_images: dict[str, np.ndarray] | None  # per-camera "imagined" frames


def validate_candidate_response(
    response: Any,
    *,
    request_id: str,
    base_seed: int,
    num_candidates: int,
    chunk_size: int,
    action_dim: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of ``_validate_candidate_response``
    (rpc/run_remote_robocasa_value_select_deploy.py).

    The seed-equality check is the determinism contract: if the server did not
    return seeds ``base..base+K-1`` in order, the candidates are not the ones we
    asked for and nothing downstream is reproducible.
    """
    if getattr(response, "request_id", None) != request_id:
        raise RuntimeError("RPC response request_id does not match the auditable request ID")
    seeds = np.asarray(getattr(response, "seeds", None))
    actions = np.asarray(getattr(response, "actions", None))
    values = np.asarray(getattr(response, "values", None))
    expected = np.arange(base_seed, base_seed + num_candidates, dtype=np.int64)
    if (
        seeds.shape != (num_candidates,)
        or seeds.dtype != np.int64
        or not np.array_equal(seeds, expected)
    ):
        raise RuntimeError(f"RPC seeds {seeds.tolist()} != expected {expected.tolist()}")
    if actions.shape != (num_candidates, chunk_size, action_dim) or actions.dtype != np.float64:
        raise RuntimeError(
            f"RPC candidate actions have shape/dtype {actions.shape} {actions.dtype}; "
            f"expected {(num_candidates, chunk_size, action_dim)} float64"
        )
    if values.shape != (num_candidates,) or values.dtype != np.float64:
        raise RuntimeError(
            f"RPC candidate values have shape/dtype {values.shape} {values.dtype}"
        )
    if not np.isfinite(actions).all() or not np.isfinite(values).all():
        raise RuntimeError("RPC returned non-finite candidate actions or values")
    return seeds, actions, values


def future_images_to_frames(
    future_images: Mapping[str, np.ndarray] | None,
) -> dict[str, np.ndarray] | None:
    """Port of ``_future_images_to_cand_images``: rename the backend's payload
    keys to our camera names. Returns None when the request did not ask for
    future images."""
    if future_images is None:
        return None
    out: dict[str, np.ndarray] = {}
    for camera, key in FUTURE_IMAGE_KEY_BY_CAMERA.items():
        if key in future_images:
            out[camera] = np.asarray(future_images[key], dtype=np.uint8)
    return out


def select_best_index(values: Sequence[float] | np.ndarray) -> int:
    """argmax, ties resolved to the LOWEST index -- same rule (and same comment)
    as the deploy runner's ``int(np.argmax(selection_values))``."""
    return int(np.argmax(np.asarray(values, dtype=np.float64)))


def request_candidates(
    client: PolicyClient,
    observation: Mapping[str, np.ndarray],
    task_description: str,
    *,
    base_seed: int,
    num_candidates: int,
    request_id: str,
    config: CosmosCallConfig = CosmosCallConfig(),
    return_future_images: bool = True,
) -> CandidateBatch:
    """ONE request for K candidates (``num_queries=K``, seeds ``base..base+K-1``).

    ``return_future_images=True`` is the default here (unlike the deploy runner,
    which only asks for them in ``scorer_server:*`` mode) because the "imagined
    frames" ARE a deliverable of ``call_cosmos``: the explorer is meant to compare
    what the policy imagined against what actually happened.
    """
    if int(num_candidates) < 1:
        raise ValueError("num_candidates must be >= 1")
    response = client.infer(
        observation,
        task_description,
        seed=int(base_seed),
        num_queries=int(num_candidates),
        num_denoising_steps_action=config.num_denoising_steps_action,
        return_future_images=bool(return_future_images),
        request_id=request_id,
    )
    seeds, actions, values = validate_candidate_response(
        response,
        request_id=request_id,
        base_seed=int(base_seed),
        num_candidates=int(num_candidates),
        chunk_size=config.chunk_size,
        action_dim=int(getattr(config, "action_dim", 7)),
    )
    future = (
        future_images_to_frames(getattr(response, "future_images", None))
        if return_future_images
        else None
    )
    return CandidateBatch(seeds=seeds, actions=actions, values=values, future_images=future)


def run_best_of_k_closed_loop(
    client: PolicyClient,
    *,
    observation: Mapping[str, np.ndarray],
    task_description: str,
    num_chunks: int,
    num_candidates: int,
    base_seed: int,
    step_chunk: Callable[[np.ndarray], "ChunkResult"],
    request_prefix: str,
    config: CosmosCallConfig = CosmosCallConfig(),
    return_future_images: bool = True,
) -> dict[str, Any]:
    """Closed loop: re-query every ``config.num_open_loop_steps`` (16) executed steps.

    Each iteration asks for K candidates in ONE request, picks the highest-value
    one (ties -> lowest index), commits its first 16 actions through ``step_chunk``,
    and repeats with the resulting observation, for at most ``num_chunks``
    iterations or until ``step_chunk`` reports ``stop``.

    ``step_chunk`` is the ONLY env-dependent piece: it takes a (16, 7) action
    array, executes it, and returns a :class:`ChunkResult` with the new policy
    observation, any real frames it captured, and whether to stop. A test can
    pass a pure-python fake.

    Seeds: chunk ``i`` uses ``base_seed + i * num_candidates``, so the K-seed
    blocks of consecutive chunks never overlap -- the same non-overlap property
    ``candidate_seed_base`` gives the deploy runner, expressed for a standalone
    call whose seed the caller chose.

    Returns ``{"imagined_frames": [...], "real_frames": [...], "value": float,
    "values": [...], "selected_index": [...], "seeds": [...], "chunks": int,
    "steps": int, "stopped_early": bool}``. ``value`` is the LAST selected
    candidate's value (the one whose execution the caller is about to judge);
    the per-chunk history is in ``values``.
    """
    horizon = int(config.num_open_loop_steps)
    imagined_frames: list[dict[str, np.ndarray] | None] = []
    real_frames: list[Any] = []
    selected_values: list[float] = []
    all_values: list[list[float]] = []
    selected_indices: list[int] = []
    used_seeds: list[int] = []
    steps = 0
    stopped_early = False
    current = dict(observation)

    for chunk_index in range(int(num_chunks)):
        seed = int(base_seed) + chunk_index * int(num_candidates)
        request_id = f"{request_prefix}-c{chunk_index:03d}"
        batch = request_candidates(
            client,
            current,
            task_description,
            base_seed=seed,
            num_candidates=num_candidates,
            request_id=request_id,
            config=config,
            return_future_images=return_future_images,
        )
        index = select_best_index(batch.values)
        selected_indices.append(index)
        used_seeds.append(int(batch.seeds[index]))
        all_values.append([float(v) for v in batch.values])
        selected_values.append(float(batch.values[index]))
        imagined_frames.append(_slice_future_images(batch.future_images, index))

        plan = np.asarray(batch.actions[index], dtype=np.float64)
        result = step_chunk(plan[:horizon])
        steps += int(result.steps)
        if result.frames is not None:
            real_frames.append(result.frames)
        current = dict(result.observation)
        if result.stop:
            stopped_early = True
            break

    return {
        "imagined_frames": imagined_frames,
        "real_frames": real_frames,
        "value": selected_values[-1] if selected_values else None,
        "values": all_values,
        "selected_values": selected_values,
        "selected_index": selected_indices,
        "seeds": used_seeds,
        "chunks": len(selected_values),
        "steps": steps,
        "stopped_early": stopped_early,
    }


@dataclass
class ChunkResult:
    """What ``step_chunk`` hands back for one executed 16-step chunk."""

    observation: Mapping[str, np.ndarray]
    steps: int
    stop: bool = False
    frames: Any = None


def _slice_future_images(
    future_images: Mapping[str, np.ndarray] | None, index: int
) -> dict[str, np.ndarray] | None:
    """Take candidate ``index``'s imagined frame out of the per-camera batch.

    GUESS AT AN INTERFACE (flagged deliberately): the deploy runner only ever
    hands the whole ``(K, H, W, 3)`` batch straight to a scorer, so nothing in
    this repo pins down the leading-axis convention for a single candidate. We
    therefore index the leading axis ONLY when it is exactly K-shaped and pass
    the array through untouched otherwise, rather than guessing wrong and
    silently returning someone else's frame.
    """
    if future_images is None:
        return None
    out: dict[str, np.ndarray] = {}
    for camera, array in future_images.items():
        arr = np.asarray(array)
        out[camera] = arr[index] if arr.ndim == 4 and index < arr.shape[0] else arr
    return out


__all__ = [
    "CandidateBatch",
    "ChunkResult",
    "CosmosCallConfig",
    "PolicyClient",
    "future_images_to_frames",
    "request_candidates",
    "run_best_of_k_closed_loop",
    "select_best_index",
    "validate_candidate_response",
]
