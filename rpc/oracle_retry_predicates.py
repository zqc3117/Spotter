"""Privileged, evaluator-only predicates for Oracle Retry-3.

The policy never receives any value accepted by this module.  Callers capture
privileged snapshots at t0, t8, and t16 and use these pure functions only to
decide whether a local interaction clearly failed or a retry recovered it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


JsonDict = dict[str, Any]


def _number(value: Any) -> float | None:
    """Return a finite real number, explicitly excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _boolean(mapping: Mapping[str, Any], *keys: str) -> bool | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            return value
    return None


def _state(snapshot: Mapping[str, Any]) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    nested = snapshot.get("task_state")
    state = nested if isinstance(nested, Mapping) else snapshot
    family = state.get("family", "")
    values = state.get("values", {})
    context = state.get("context", {})
    return (
        str(family).lower() if isinstance(family, str) else "",
        values if isinstance(values, Mapping) else {},
        context if isinstance(context, Mapping) else {},
    )


def _circular_distance(left: float, right: float) -> float:
    return abs((left - right + math.pi) % (2.0 * math.pi) - math.pi)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _selected_joint_values(
    values: Mapping[str, Any], context: Mapping[str, Any]
) -> list[float]:
    keys: list[str] = []
    configured = context.get("progress_keys")
    if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        keys.extend(str(key) for key in configured)
    for name in ("progress_key", "target_joint"):
        key = context.get(name)
        if isinstance(key, str):
            keys.append(key)
    if not keys:
        keys = [str(key) for key in values if "joint" in str(key).lower()]
    if not keys:
        keys = [str(key) for key, value in values.items() if _number(value) is not None]
    selected = []
    for key in dict.fromkeys(keys):
        number = _number(values.get(key))
        if number is not None:
            selected.append(number)
    return selected


def mechanism_progress(
    task_name: str,
    snapshot: Mapping[str, Any],
    *,
    task_description: str = "",
) -> float | None:
    """Return normalized progress toward the task target, or ``None``."""
    family, values, context = _state(snapshot)
    description = task_description.lower()
    if family in {"door", "drawer"}:
        joints = _selected_joint_values(values, context)
        if not joints:
            return None
        openness = _clip01(sum(joints) / len(joints))
        return openness if task_name.startswith("Open") else 1.0 - openness

    if family == "sink":
        if task_name == "TurnSinkSpout":
            angle = _number(values.get("spout_joint"))
            if angle is None:
                return None
            target = (
                1.5 * math.pi
                if "left" in description
                else 0.5 * math.pi
                if "right" in description
                else None
            )
            return None if target is None else _clip01(1.0 - _circular_distance(angle, target) / math.pi)
        angle = _number(values.get("handle_joint"))
        if angle is None:
            return None
        on_progress = _clip01(_circular_distance(angle, 0.0) / 0.40)
        return on_progress if task_name == "TurnOnSinkFaucet" else 1.0 - on_progress

    if family == "stove":
        target_knob = context.get("target_knob")
        if not isinstance(target_knob, str):
            return None
        angle = _number(values.get(target_knob))
        if angle is None:
            return None
        on_progress = _clip01(_circular_distance(angle, 0.0) / 0.35)
        return on_progress if task_name == "TurnOnStove" else 1.0 - on_progress

    if family in {"binary", "button"}:
        turned_on = _boolean(values, "turned_on")
        if turned_on is None:
            return None
        target_on = task_name in {"CoffeePressButton", "TurnOnMicrowave"}
        return 1.0 if turned_on == target_on else 0.0
    return None


def _target_grasped(snapshot: Mapping[str, Any]) -> bool | None:
    direct = _boolean(snapshot, "target_grasped", "target_grasp")
    if direct is not None:
        return direct
    grasped = snapshot.get("grasped_objects")
    if isinstance(grasped, Sequence) and not isinstance(grasped, (str, bytes)):
        return "obj" in {str(item) for item in grasped}
    return None


def _task_success(snapshot: Mapping[str, Any]) -> bool:
    return _boolean(snapshot, "task_success") is True


def _explicit_attempt(evidence: Mapping[str, Any]) -> bool:
    if any(
        _boolean(evidence, key) is True
        for key in ("interaction_attempted", "contact", "near_control_action")
    ):
        return True
    distance = _number(evidence.get("min_control_eef_distance"))
    action_delta = _number(evidence.get("interaction_action_delta"))
    return bool(
        distance is not None
        and action_delta is not None
        and distance < 0.05
        and action_delta >= 0.008
    )


def _pnp_attempt(
    t0: Mapping[str, Any],
    t8: Mapping[str, Any],
    t16: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> tuple[bool, str]:
    if _target_grasped(t8) is True and _target_grasped(t16) is False:
        return True, "target_grasped_at_t8_then_dropped_by_t16"
    distances = [
        _number(snapshot.get("target_object_eef_distance"))
        for snapshot in (t0, t8, t16)
    ]
    apertures = [
        _number(snapshot.get("gripper_aperture")) for snapshot in (t0, t8, t16)
    ]
    valid_distances = [value for value in distances if value is not None]
    valid_apertures = [value for value in apertures if value is not None]
    distance = _number(evidence.get("min_target_object_eef_distance"))
    if distance is None and valid_distances:
        distance = min(valid_distances)
    aperture_delta = _number(evidence.get("gripper_aperture_delta"))
    if aperture_delta is None and valid_apertures:
        aperture_delta = max(valid_apertures) - min(valid_apertures)
    attempted = bool(
        distance is not None
        and aperture_delta is not None
        and distance < 0.05
        and aperture_delta >= 0.008
    )
    return attempted, "near_target_with_gripper_motion" if attempted else "no_explicit_grasp_attempt"


def _progress_payload(progress: Sequence[float | None]) -> JsonDict:
    p0, p8, p16 = progress
    return {
        "t0": p0,
        "t8": p8,
        "t16": p16,
        "gain_t0_t16": None if p0 is None or p16 is None else p16 - p0,
        "delta_t8_t16": None if p8 is None or p16 is None else p16 - p8,
    }


def _result(
    *, phase: str, family: str, decision: str, reason: str,
    eligible: bool, recovered: bool, progress: JsonDict,
) -> JsonDict:
    result = {
        "phase": phase,
        "family": family,
        "decision": decision,
        "reason": reason,
        "eligible_failure": eligible,
        "recovered": recovered,
        "progress": progress,
    }
    json.dumps(result, allow_nan=False)
    return result


def evaluate_initial_failure(
    task_name: str,
    t0: Mapping[str, Any],
    t8: Mapping[str, Any],
    t16: Mapping[str, Any],
    *,
    attempt_evidence: Mapping[str, Any] | None = None,
    task_description: str = "",
) -> JsonDict:
    """Classify an initial t16 anchor without treating terminal failure as evidence."""
    evidence = attempt_evidence or {}
    family = _state(t16)[0] or _state(t0)[0]
    if family == "pnp":
        progress = {"t0": None, "t8": None, "t16": None, "gain_t0_t16": None, "delta_t8_t16": None}
        if _target_grasped(t16) is True:
            return _result(phase="initial", family=family, decision="continue", reason="target_grasped", eligible=False, recovered=True, progress=progress)
        attempted, reason = _pnp_attempt(t0, t8, t16, evidence)
        return _result(phase="initial", family=family, decision="local_retry" if attempted else "wait", reason=reason, eligible=attempted, recovered=False, progress=progress)

    values = [
        mechanism_progress(task_name, snapshot, task_description=task_description)
        for snapshot in (t0, t8, t16)
    ]
    progress = _progress_payload(values)
    p0, p8, p16 = values
    if _task_success(t16) or (p16 is not None and p16 >= 0.99):
        return _result(phase="initial", family=family, decision="continue", reason="target_state_reached", eligible=False, recovered=True, progress=progress)
    if family in {"binary", "button"}:
        attempted = _explicit_attempt(evidence)
        return _result(phase="initial", family=family, decision="local_retry" if attempted else "wait", reason="explicit_binary_interaction_failed" if attempted else "terminal_failure_without_attempt_evidence", eligible=attempted, recovered=False, progress=progress)
    if None in values:
        return _result(phase="initial", family=family, decision="wait", reason="insufficient_privileged_progress_state", eligible=False, recovered=False, progress=progress)
    assert p0 is not None and p8 is not None and p16 is not None
    attempted = max(p8, p16) - p0 >= 0.10
    stalled = abs(p16 - p8) <= 0.02
    regressed = p8 - p16 >= 0.03
    eligible = attempted and (stalled or regressed)
    reason = "progress_stalled_after_attempt" if attempted and stalled else "progress_regressed_after_attempt" if eligible else "no_clear_local_mechanism_failure"
    return _result(phase="initial", family=family, decision="local_retry" if eligible else "wait", reason=reason, eligible=eligible, recovered=False, progress=progress)


def evaluate_retry_recovery(
    task_name: str,
    t0: Mapping[str, Any],
    t8: Mapping[str, Any],
    t16: Mapping[str, Any],
    *,
    attempt_evidence: Mapping[str, Any] | None = None,
    task_description: str = "",
) -> JsonDict:
    """Judge one retry rollout as recovered, clearly failed again, or unresolved."""
    evidence = attempt_evidence or {}
    family = _state(t16)[0] or _state(t0)[0]
    if family == "pnp":
        progress = {"t0": None, "t8": None, "t16": None, "gain_t0_t16": None, "delta_t8_t16": None}
        if _target_grasped(t16) is True:
            return _result(phase="retry", family=family, decision="recovered", reason="target_grasped_after_retry", eligible=False, recovered=True, progress=progress)
        attempted, reason = _pnp_attempt(t0, t8, t16, evidence)
        return _result(phase="retry", family=family, decision="retry_again" if attempted else "wait", reason=reason, eligible=attempted, recovered=False, progress=progress)

    values = [
        mechanism_progress(task_name, snapshot, task_description=task_description)
        for snapshot in (t0, t8, t16)
    ]
    progress = _progress_payload(values)
    p0, p8, p16 = values
    if _task_success(t16) or (p16 is not None and p16 >= 0.99):
        return _result(phase="retry", family=family, decision="recovered", reason="target_state_reached_after_retry", eligible=False, recovered=True, progress=progress)
    if family in {"binary", "button"}:
        attempted = _explicit_attempt(evidence)
        return _result(phase="retry", family=family, decision="retry_again" if attempted else "wait", reason="explicit_binary_interaction_failed_again" if attempted else "no_retry_attempt_evidence", eligible=attempted, recovered=False, progress=progress)
    if None in values:
        return _result(phase="retry", family=family, decision="wait", reason="insufficient_privileged_progress_state", eligible=False, recovered=False, progress=progress)
    assert p0 is not None and p8 is not None and p16 is not None
    if p16 - p0 >= 0.10 and p16 >= p8 - 0.02:
        return _result(phase="retry", family=family, decision="recovered", reason="renewed_progress_after_retry", eligible=False, recovered=True, progress=progress)
    attempted = max(p8, p16) - p0 >= 0.10
    stalled = abs(p16 - p8) <= 0.02
    regressed = p8 - p16 >= 0.03
    failed_again = attempted and (stalled or regressed)
    return _result(phase="retry", family=family, decision="retry_again" if failed_again else "wait", reason="retry_progress_stalled_or_regressed" if failed_again else "retry_not_yet_decisive", eligible=failed_again, recovered=False, progress=progress)
