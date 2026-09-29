"""Cross-process serialization for RoboCasa simulator snapshots.

The in-memory
``SimulatorSnapshot`` used by ``rpc.run_remote_robocasa_oracle_branch3`` and
``rpc.run_remote_robocasa_postfailure_gated_branch3`` cannot be pickled as-is:
each ``ObjectStateSnapshot.target`` is a live reference into the RoboCasa
controller tree, not data. This module factors the pure-data capture/restore
logic out of those two runners (byte-identical in both) and adds a disk format:

- ``.meta.json``    -- shapes/dtypes manifest + state_sha256 + controller
                        target type names (order-sensitive identity check)
- ``.physics.npz``  -- flattened_state, numpy/python RNG state arrays
- ``.aux.npz``      -- physics_auxiliary + environment_values ndarrays
- ``.controllers.pkl`` -- controller_states values (plain dict/scalar/ndarray
                        trees only -- never a live object), pickled because
                        RNG state and nested dict shapes are awkward as npz

``dump_snapshot`` and ``load_snapshot`` are the public entry points. Both
``rpc/run_remote_robocasa_oracle_branch3.py`` and
``rpc/run_remote_robocasa_cf_branch_collect.py`` (T2) should import
``capture_simulator_snapshot`` / ``restore_simulator_snapshot`` from here
instead of keeping their own copies, once T1's smoke passes.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:  # pragma: no cover - exercised only inside the RoboCasa/Cosmos env
    from rpc.run_remote_robocasa_targeted_collect import (
        SimulatorSnapshot as PhysicsSimulatorSnapshot,
        _capture_simulator_snapshot as _capture_physics_snapshot,
    )
    from rpc.run_remote_robocasa_collect import _copy_observation, prepare_observation
except ImportError:  # pragma: no cover - allows importing this module for
    # unit tests that only exercise the pure-Python (de)serialization helpers
    # on a hand-built fake env, without the full cosmos_policy/robosuite stack.
    PhysicsSimulatorSnapshot = None  # type: ignore[assignment,misc]
    _capture_physics_snapshot = None  # type: ignore[assignment]
    _copy_observation = None  # type: ignore[assignment]
    prepare_observation = None  # type: ignore[assignment]


_UNSNAPSHOTTABLE = object()


@dataclass(frozen=True)
class ObjectStateSnapshot:
    target: Any
    values: Mapping[str, Any]


@dataclass(frozen=True)
class SimulatorSnapshot:
    physics: Any  # PhysicsSimulatorSnapshot
    physics_auxiliary: Mapping[str, np.ndarray]
    environment_values: Mapping[str, Any]
    controller_states: tuple[ObjectStateSnapshot, ...]

    @property
    def state_sha256(self) -> str:
        return self.physics.state_sha256


def _clone_state_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple):
        copied = tuple(_clone_state_value(item) for item in value)
        return (
            _UNSNAPSHOTTABLE
            if any(item is _UNSNAPSHOTTABLE for item in copied)
            else copied
        )
    if isinstance(value, list):
        copied = [_clone_state_value(item) for item in value]
        return (
            _UNSNAPSHOTTABLE
            if any(item is _UNSNAPSHOTTABLE for item in copied)
            else copied
        )
    if isinstance(value, Mapping):
        copied = {key: _clone_state_value(item) for key, item in value.items()}
        return (
            _UNSNAPSHOTTABLE
            if any(item is _UNSNAPSHOTTABLE for item in copied.values())
            else copied
        )
    return _UNSNAPSHOTTABLE


def _snapshot_object(target: Any) -> ObjectStateSnapshot:
    values: dict[str, Any] = {}
    for name, value in vars(target).items():
        copied = _clone_state_value(value)
        if copied is not _UNSNAPSHOTTABLE:
            values[name] = copied
    return ObjectStateSnapshot(target=target, values=values)


def _controller_objects(env: Any) -> tuple[Any, ...]:
    objects: list[Any] = []
    seen: set[int] = set()
    for robot in getattr(env, "robots", ()):
        candidates: list[Any] = []
        composite_controller = getattr(robot, "composite_controller", None)
        if composite_controller is not None:
            candidates.append(composite_controller)
        controller = getattr(robot, "controller", None)
        if controller is not None:
            candidates.append(controller)
        part_controllers = getattr(robot, "part_controllers", None)
        if isinstance(part_controllers, Mapping):
            candidates.extend(part_controllers.values())
        for candidate in candidates:
            for target in (
                candidate,
                getattr(candidate, "interpolator_pos", None),
                getattr(candidate, "interpolator_ori", None),
            ):
                if target is None or id(target) in seen:
                    continue
                seen.add(id(target))
                objects.append(target)
        grippers = getattr(robot, "gripper", None)
        if isinstance(grippers, Mapping):
            for gripper in grippers.values():
                if gripper is not None and id(gripper) not in seen:
                    seen.add(id(gripper))
                    objects.append(gripper)
    return tuple(objects)


def capture_simulator_snapshot(env: Any) -> SimulatorSnapshot:
    """Identical in behavior to the private copies in the two branch3 runners."""
    sim_data = getattr(env.sim, "data", None)
    environment_values = {
        name: _clone_state_value(getattr(env, name))
        for name in ("timestep", "cur_time", "done")
        if hasattr(env, name)
    }
    return SimulatorSnapshot(
        physics=_capture_physics_snapshot(env),
        physics_auxiliary={
            name: np.asarray(getattr(sim_data, name)).copy()
            for name in (
                "ctrl",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
                "mocap_pos",
                "mocap_quat",
                "userdata",
            )
            if sim_data is not None and hasattr(sim_data, name)
        },
        environment_values=environment_values,
        controller_states=tuple(
            _snapshot_object(target) for target in _controller_objects(env)
        ),
    )


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes())
    return digest.hexdigest()


def restore_simulator_snapshot(
    env: Any,
    snapshot: SimulatorSnapshot,
    *,
    expected_observation: Mapping[str, np.ndarray],
    flip_images: bool,
) -> tuple[Mapping[str, Any], dict[str, np.ndarray], dict[str, bool]]:
    """Identical in behavior to the private copies in the two branch3 runners."""
    physics = snapshot.physics
    env.sim.set_state_from_flattened(physics.flattened_state)
    env.sim.forward()
    for name, value in snapshot.physics_auxiliary.items():
        destination = getattr(env.sim.data, name)
        destination[...] = value
    for name, value in snapshot.environment_values.items():
        setattr(env, name, _clone_state_value(value))
    for object_state in snapshot.controller_states:
        for name, value in object_state.values.items():
            setattr(object_state.target, name, _clone_state_value(value))
    np.random.set_state(physics.numpy_random_state)
    random.setstate(physics.python_random_state)

    restored_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
    state_hash_exact = _array_sha256(restored_state) == physics.state_sha256
    if not state_hash_exact:
        raise RuntimeError("restored flattened simulator state hash differs from anchor")
    getter = getattr(env, "_get_observations", None)
    if not callable(getter):
        raise RuntimeError("RoboCasa environment lacks _get_observations")
    try:
        raw = getter(force_update=True)
    except TypeError:
        raw = getter()
    resampled = prepare_observation(raw, flip_images)
    policy_observation = _copy_observation(expected_observation)
    checks: dict[str, bool] = {"simulator_state_hash": state_hash_exact}
    for key, expected in expected_observation.items():
        checks[f"policy_input_{key}"] = np.array_equal(
            np.asarray(expected), np.asarray(policy_observation[key])
        )
        checks[f"resampled_sensor_{key}"] = bool(
            key in resampled
            and np.array_equal(np.asarray(expected), np.asarray(resampled[key]))
        )
    if not all(value for key, value in checks.items() if key.startswith("policy_input_")):
        raise RuntimeError("frozen branch policy observation differs from anchor")
    return raw, policy_observation, checks


# --------------------------------------------------------------------------
# Disk serialization (T1's actual new contribution -- everything above this
# line is a byte-identical extraction of existing, already-battle-tested
# logic; do not "improve" it here, that would defeat the point of reusing
# code the two branch3 runners already trust).
# --------------------------------------------------------------------------


def _target_type_names(snapshot: SimulatorSnapshot) -> list[str]:
    return [
        f"{type(state.target).__module__}.{type(state.target).__qualname__}"
        for state in snapshot.controller_states
    ]


def dump_snapshot(snapshot: SimulatorSnapshot, directory: Path | str) -> Path:
    """Write ``snapshot`` to ``directory`` (created if absent). Returns the dir."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    physics = snapshot.physics
    # model_xml is not needed to restore physics (only flattened_state is --
    # see rpc/run_remote_robocasa_targeted_collect.py's own
    # _restore_simulator_snapshot, which never reads it) but the dataclass
    # requires it and it's useful provenance, so persist it as plain text.
    (out / "model.xml").write_text(physics.model_xml)
    np.savez(
        out / "physics.npz",
        flattened_state=np.asarray(physics.flattened_state),
        numpy_random_state_pos=np.asarray(physics.numpy_random_state[2], dtype=np.int64),
        numpy_random_state_has_gauss=np.asarray(
            physics.numpy_random_state[3], dtype=np.int64
        ),
        numpy_random_state_cached_gaussian=np.asarray(
            physics.numpy_random_state[4], dtype=np.float64
        ),
        numpy_random_state_keys=np.asarray(physics.numpy_random_state[1], dtype=np.uint32),
    )
    # np.random.get_state() is ('MT19937', keys(uint32[624]), pos, has_gauss,
    # cached_gaussian); store the algorithm name separately since it's a str.
    aux_arrays = {f"aux__{k}": v for k, v in snapshot.physics_auxiliary.items()}
    np.savez(out / "aux.npz", **aux_arrays) if aux_arrays else None

    with (out / "controllers.pkl").open("wb") as fh:
        pickle.dump(
            [dict(state.values) for state in snapshot.controller_states],
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with (out / "python_random_state.pkl").open("wb") as fh:
        pickle.dump(physics.python_random_state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with (out / "environment_values.pkl").open("wb") as fh:
        pickle.dump(
            dict(snapshot.environment_values), fh, protocol=pickle.HIGHEST_PROTOCOL
        )

    meta = {
        "schema": "cf_branch_snapshot/v1",
        "state_sha256": snapshot.state_sha256,
        "numpy_random_state_algo": physics.numpy_random_state[0],
        "controller_target_types": _target_type_names(snapshot),
        "has_aux": bool(aux_arrays),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return out


def load_snapshot(directory: Path | str, env: Any) -> SimulatorSnapshot:
    """Read a snapshot dumped by ``dump_snapshot`` and rebind it to ``env``.

    ``env`` must be a freshly constructed RoboCasa environment for the same
    task/seed the snapshot was captured from -- this function does not
    recreate the env itself, it only reconstitutes live controller object
    references via ``_controller_objects(env)`` and re-attaches the pickled
    ``values`` onto them in the same order they were captured.
    """
    src = Path(directory)
    meta = json.loads((src / "meta.json").read_text())

    with np.load(src / "physics.npz") as physics_npz:
        flattened_state = physics_npz["flattened_state"]
        numpy_random_state = (
            meta["numpy_random_state_algo"],
            physics_npz["numpy_random_state_keys"],
            int(physics_npz["numpy_random_state_pos"]),
            int(physics_npz["numpy_random_state_has_gauss"]),
            float(physics_npz["numpy_random_state_cached_gaussian"]),
        )

    aux_path = src / "aux.npz"
    physics_auxiliary: dict[str, np.ndarray] = {}
    if meta["has_aux"] and aux_path.exists():
        with np.load(aux_path) as aux_npz:
            physics_auxiliary = {
                key[len("aux__"):]: aux_npz[key]
                for key in aux_npz.files
                if key.startswith("aux__")
            }

    with (src / "python_random_state.pkl").open("rb") as fh:
        python_random_state = pickle.load(fh)
    with (src / "environment_values.pkl").open("rb") as fh:
        environment_values = pickle.load(fh)
    with (src / "controllers.pkl").open("rb") as fh:
        controller_values_list: list[dict[str, Any]] = pickle.load(fh)

    controller_targets = _controller_objects(env)
    expected_types = meta["controller_target_types"]
    if len(controller_targets) != len(expected_types):
        raise RuntimeError(
            "controller object count mismatch on load: "
            f"env has {len(controller_targets)}, snapshot recorded {len(expected_types)}"
        )
    actual_types = [
        f"{type(target).__module__}.{type(target).__qualname__}"
        for target in controller_targets
    ]
    if actual_types != expected_types:
        raise RuntimeError(
            "controller object type order mismatch on load: "
            f"env has {actual_types}, snapshot recorded {expected_types}"
        )
    controller_states = tuple(
        ObjectStateSnapshot(target=target, values=values)
        for target, values in zip(controller_targets, controller_values_list)
    )

    state_sha256 = hashlib.sha256()
    state_sha256.update(str(flattened_state.dtype).encode("ascii"))
    state_sha256.update(b"\0")
    state_sha256.update(json.dumps(flattened_state.shape).encode("ascii"))
    state_sha256.update(b"\0")
    state_sha256.update(np.ascontiguousarray(flattened_state).tobytes())
    if state_sha256.hexdigest() != meta["state_sha256"]:
        raise RuntimeError("loaded flattened_state hash does not match meta.json")

    physics = PhysicsSimulatorSnapshot(
        model_xml=(src / "model.xml").read_text(),
        flattened_state=flattened_state,
        numpy_random_state=numpy_random_state,
        python_random_state=python_random_state,
        state_sha256=meta["state_sha256"],
    )
    return SimulatorSnapshot(
        physics=physics,
        physics_auxiliary=physics_auxiliary,
        environment_values=environment_values,
        controller_states=controller_states,
    )
