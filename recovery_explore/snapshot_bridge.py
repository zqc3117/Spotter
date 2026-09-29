"""Glue between the recovery_explore env service and this repo's anchor/snapshot IO.

Everything here is IMPORTED from ``cf_bench`` / ``rpc`` -- nothing is copied. The
restore contract (``cf_bench/snapshot_io.py``) is the load-bearing invariant of the
whole line of work: ``restore_simulator_snapshot`` recomputes the sha256 of the
restored flattened simulator state and raises when it differs from the anchor's
``state_sha256``, and ``cf_bench.cf_branch_replay.restore_anchor`` additionally
raises when any ``policy_input_*`` observation check fails. This module deliberately
adds NO try/except around either -- a weakened hash check would silently turn "we
reproduced the failure state exactly" into "we started somewhere near it", which is
the one thing this experiment cannot tolerate.

Layout:

* :func:`read_anchor_records` / :func:`find_anchor` -- anchors.jsonl reading, same
  line-by-line JSONL shape ``cf_bench/audit_anchors.py::read_anchors`` uses.
* :func:`build_env` -- ``cf_bench.cf_branch_replay.build_env_and_cfg`` plus the
  layout/style consistency assertion (``check_layout_style_match``), which is how
  the replay line proves the reconstructed scene is the anchor's scene.
* :func:`restore` -- the restore itself; returns the anchor's policy observation
  together with the task instruction and the target object's name.

Imports are done lazily inside the functions so this module stays importable (and
``py_compile``-able) off-pod, where robosuite/robocasa do not exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np


@dataclass(frozen=True)
class RestoredAnchor:
    """What the service needs after a successful restore."""

    anchor_id: str
    task: str
    task_description: str
    object_name: str | None
    state_sha256: str
    query_timestep: int
    policy_observation: Mapping[str, np.ndarray]


def read_anchor_records(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield the anchor records of one ``anchors.jsonl``.

    Same shape as ``cf_bench/audit_anchors.py::read_anchors`` (one JSON object per
    non-empty line), minus its multi-file/``_source_file`` bookkeeping.
    """
    path = Path(path)
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def find_anchor(path: Path | str, anchor_id: str) -> dict[str, Any]:
    for record in read_anchor_records(path):
        if str(record.get("anchor_id")) == str(anchor_id):
            return record
    raise KeyError(f"anchor_id {anchor_id!r} not found in {path}")


def anchor_task_description(anchor: Mapping[str, Any]) -> str:
    """The REAL language instruction, not the env class name.

    Copied rule (not code) from ``cf_bench/vlm_correction_rollout.py``: prefer
    ``task_description`` / ``task_text`` (e.g. "pick the sweet potato from the
    counter and place it in the cabinet") over ``task`` (e.g. "PnPCounterToCab").
    Both keys are usually present and non-empty, so a bare ``or`` chain over the
    wrong order silently picks the class name.
    """
    return str(
        anchor.get("task_description") or anchor.get("task_text") or anchor.get("task")
    )


def load_frozen_observation(anchor: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The anchor's pre-branch observation, from its npz ``pre_*`` arrays."""
    from cf_bench.cf_branch_replay import frozen_observation_from_arrays

    with np.load(anchor["arrays_file"]) as npz:
        return frozen_observation_from_arrays(npz)


def build_env(
    anchor: Mapping[str, Any], *, run_id: str, output_dir: Path
) -> tuple[Any, Any]:
    """Reconstruct the exact env the anchor was collected under.

    Delegates to ``cf_bench.cf_branch_replay.build_env_and_cfg`` (PandaMobile, the
    official pickled controller config, our three cameras, the anchor's own
    obj_instance_split / layout strategy) and then asserts the recomputed
    layout/style pair equals the anchor's recorded one -- ``build_env_and_cfg``'s
    own docstring says the caller must do this before trusting anything
    downstream.
    """
    from dataclasses import replace

    from cf_bench.cf_branch_replay import build_env_and_cfg, check_layout_style_match
    from rpc.run_remote_robocasa_cf_branch_collect import create_robocasa_env

    env, cfg = build_env_and_cfg(anchor, run_id=run_id, output_dir=Path(output_dir))
    match = check_layout_style_match(anchor, cfg)

    if not match["matches"]:
        # build_env_and_cfg RE-DERIVES the scene from (split, seed, episode). That
        # derivation was verified on 2026-09-03 against a seed195 split-B anchor,
        # but it does NOT reproduce the 2026-09-02 split-A train pool: episode-0002
        # /anchor-001 records layout 9 / style 0 while the "TRAIN_NONTEST" strategy
        # recomputes 6 / 11 (observed 2026-09-11).
        #
        # The anchor records layout_id/style_id precisely so the scene can be
        # rebuilt, so PIN the recorded pair instead of trusting the re-derivation.
        # official_layout_style_pair() ast.literal_eval's this string and indexes
        # it, so a single-pair list is returned verbatim for any episode_idx, and
        # create_robocasa_env passes exactly that pair to robosuite.make.
        #
        # This cannot silently build the WRONG scene: restore_simulator_snapshot
        # re-hashes the restored flattened state against the anchor's own
        # state_sha256 and raises on any difference, so a mis-built kitchen fails
        # loudly one call later rather than producing quietly bogus physics.
        try:
            env.close()
        except Exception:
            pass
        pinned = f"[({int(anchor['layout_id'])}, {int(anchor['style_id'])})]"
        cfg = replace(cfg, layout_and_style_ids=pinned)
        env = create_robocasa_env(
            cfg, seed=cfg.seed, episode_idx=int(anchor["episode_index"])
        )
        match = check_layout_style_match(anchor, cfg)
        if not match["matches"]:
            try:
                env.close()
            except Exception:
                pass
            raise RuntimeError(
                f"layout/style still mismatched after pinning the anchor's own "
                f"recorded pair for {anchor.get('anchor_id')}: {match}"
            )
    return env, cfg


def target_object_name(env: Any, anchor: Mapping[str, Any]) -> str | None:
    """Best-effort human-readable name of the task's target object ("obj").

    INTERFACE GUESS, flagged: nothing in this repo reads an object's display name
    (the collectors only ever key objects by the internal handle ``"obj"`` --
    ``rpc/run_remote_robocasa_collect.py::_grasped_object_names`` and
    ``_target_object_eef_distance``). So this tries, in order: the anchor record's
    own ``object_name``-ish fields, RoboCasa's episode meta ``object_cfgs`` entry
    whose ``name`` is ``"obj"`` (its ``info.cat`` is the category string RoboCasa
    builds the instruction from), and finally ``env.objects["obj"].name``. Returns
    None rather than inventing a label.
    """
    for key in ("object_name", "target_object", "object_category"):
        value = anchor.get(key)
        if isinstance(value, str) and value:
            return value
    try:
        ep_meta = env.get_ep_meta()
        for cfg in ep_meta.get("object_cfgs", ()) or ():
            if str(cfg.get("name")) != "obj":
                continue
            info = cfg.get("info") or {}
            for candidate in (info.get("cat"), cfg.get("category"), cfg.get("name")):
                if isinstance(candidate, str) and candidate:
                    return candidate
    except Exception:
        pass
    try:
        return str(env.objects["obj"].name)
    except Exception:
        return None


def target_object_xyz(env: Any) -> list[float] | None:
    """World xyz of the target object's body, or None when the task has no "obj".

    Same lookup ``rpc/run_remote_robocasa_collect.py::_target_object_eef_distance``
    uses (``env.obj_body_id["obj"]`` -> ``sim.data.body_xpos``). PRIVILEGED: only
    ever returned under the oracle key, never when oracle masking is on.
    """
    body_ids = getattr(env, "obj_body_id", {})
    if not isinstance(body_ids, Mapping) or "obj" not in body_ids:
        return None
    try:
        position = np.asarray(
            env.sim.data.body_xpos[int(body_ids["obj"])], dtype=np.float64
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    return [float(v) for v in position]


def restore(
    env: Any,
    cfg: Any,
    anchor: Mapping[str, Any],
    *,
    frozen_observation: Mapping[str, np.ndarray] | None = None,
) -> RestoredAnchor:
    """Restore ``anchor``'s failure snapshot onto ``env``, hash-verified.

    Raises (via ``cf_bench.snapshot_io.restore_simulator_snapshot``) when the
    restored flattened state's sha256 differs from the anchor's, and (via
    ``cf_bench.cf_branch_replay.restore_anchor``) when any policy-input array
    differs. Both are the contract, not an inconvenience: do not soften them.

    ``env`` may be restored repeatedly -- every rollout mutates it, so the caller
    re-restores before each attempt (the pattern
    ``cf_bench/vlm_correction_rollout.py`` uses for its paired arms).
    """
    from cf_bench.cf_branch_replay import restore_anchor
    from cf_bench.snapshot_io import load_snapshot

    if frozen_observation is None:
        frozen_observation = load_frozen_observation(anchor)
    snapshot_dir = Path(anchor["snapshot_dir"])
    policy_observation = restore_anchor(
        env, snapshot_dir, frozen_observation, flip_images=cfg.flip_images
    )
    snapshot = load_snapshot(snapshot_dir, env)
    return RestoredAnchor(
        anchor_id=str(anchor.get("anchor_id")),
        task=str(anchor.get("task")),
        task_description=anchor_task_description(anchor),
        object_name=target_object_name(env, anchor),
        state_sha256=str(snapshot.state_sha256),
        query_timestep=int(anchor.get("query_timestep", 0)),
        policy_observation=policy_observation,
    )


__all__ = [
    "RestoredAnchor",
    "anchor_task_description",
    "build_env",
    "find_anchor",
    "load_frozen_observation",
    "read_anchor_records",
    "restore",
    "target_object_name",
    "target_object_xyz",
]
