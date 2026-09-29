# Copyright 2026 The RPent Authors.
# Copyright 2026 recovery_explore contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from RPent:
#   upstream file: rpent/memory/manager.py  (`_validate`, `_split_frontmatter`,
#                  `_canonical_id`, `_render`, and the SCOPES/KINDS/CONFIDENCE
#                  vocabularies)
#   upstream file: rpent/robocasa/libero_explore_prompt.py  (`STEP_DISTIL`,
#                  which specifies the body sections and the frontmatter field
#                  list this schema enforces)
#
# Differences from upstream, deliberate:
#   * scope `suite` is dropped — recovery_explore has global memories plus a
#     per-task JSON tier, no suite tier.
#   * `symptom` and `related` are REQUIRED and VALIDATED here. RPent's prompt
#     asks authors for both but `_validate` never checks them, so they rot.
#   * the three body sections (**Why:** / **How to apply:** / **Falsify:**) are
#     checked mechanically instead of being prompt-only guidance.
#   * `task_only/<task>.json` has its own schema (RPent copies opaque audit and
#     recipe artifacts instead).

"""Frontmatter and task-record contract for the recovery_explore memory bank."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import _yamlish

# --------------------------------------------------------------------------- #
# vocabularies
# --------------------------------------------------------------------------- #

SCOPES = {"global"}
KINDS = {"primitive", "perception", "strategy", "failure", "infra"}
CONFIDENCE = {"single-shot", "probable", "verified"}

#: Frontmatter keys of a global entry, in the order `_render` emits them.
GLOBAL_FIELD_ORDER = (
    "id",
    "scope",
    "kind",
    "title",
    "applies_when",
    "symptom",
    "evidence",
    "confidence",
    "related",
)

#: Body headings every global entry must carry, verbatim.
REQUIRED_SECTIONS = ("**Why:**", "**How to apply:**", "**Falsify:**")
OPTIONAL_SECTIONS = ("**Related:**",)

#: `task_only/<task>.json` keys. Every key is required; `notes` may be "".
TASK_FIELDS: dict[str, tuple[type, ...]] = {
    "task": (str,),
    "best_hover_m": (int, float),
    "best_xy_offset_m": (list,),
    "best_prompt_style": (str,),
    "best_num_chunks": (int,),
    "grasp_rate": (int, float),
    "n_samples": (int,),
    "notes": (str,),
}

#: Draft filename prefixes stripped when deriving the canonical id, so that
#: `new_global_strategy_<slug>.md` publishes as `<slug>.md`.
_PREFIXES = ("new_global_", "new_", "global_")
_KIND_PREFIXES = tuple(f"{kind}_" for kind in sorted(KINDS))

_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_WIKILINK_RE = re.compile(r"^\[\[(?P<id>[^\[\]]+)\]\]$")


class SchemaError(ValueError):
    """A memory leaf or task record violates the contract."""


# --------------------------------------------------------------------------- #
# frontmatter IO  (adapted from RPent `_split_frontmatter` / `_render`)
# --------------------------------------------------------------------------- #


def split_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    """Return `(frontmatter mapping, body)` for one Markdown leaf."""
    text = path.read_text(errors="replace")
    if not text.startswith("---"):
        raise SchemaError("missing YAML frontmatter")
    end = text.find("\n---", 3)
    if end < 0:
        raise SchemaError("unterminated YAML frontmatter")
    try:
        metadata = _yamlish.safe_load(text[3:end])
    except Exception as exc:  # noqa: BLE001 - surface any parser's complaint
        raise SchemaError(f"frontmatter is not valid YAML: {exc}") from exc
    if not isinstance(metadata, dict):
        raise SchemaError("frontmatter must be a mapping")
    return metadata, text[end + 4 :]


def render(metadata: dict[str, Any], body: str) -> str:
    """Serialize one leaf, with frontmatter keys in `GLOBAL_FIELD_ORDER`."""
    ordered = {key: metadata[key] for key in GLOBAL_FIELD_ORDER if key in metadata}
    ordered.update({k: v for k, v in metadata.items() if k not in ordered})
    return "---\n" + _yamlish.safe_dump(ordered) + "---\n" + body.lstrip("\n")


def canonical_id(path: Path, metadata: dict[str, Any]) -> str:
    """Derive the published id: explicit `id`, else the bare filename slug."""
    stem = re.sub(r"_draft$", "", path.stem)
    for prefix in (*_PREFIXES, *_KIND_PREFIXES):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return str(metadata.get("id") or stem).strip()


def normalize_related(value: Any) -> list[str]:
    """Accept `foo` and `"[[foo]]"` alike; return bare ids.

    Frontmatter should carry bare ids — `related: [[a, b]]` is a nested
    sequence to any YAML parser, not a pair of wiki links. `[[id]]` belongs in
    the body's **Related:** line. Quoted `"[[id]]"` is tolerated and unwrapped
    so an author who follows the body convention in frontmatter is corrected
    rather than rejected.
    """
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise SchemaError("related must be a list")
    out: list[str] = []
    for item in value:
        if isinstance(item, list):
            raise SchemaError(
                "related must be a list of bare ids; write `related: [a, b]`, "
                "and keep [[id]] syntax for the body's **Related:** line"
            )
        text = str(item).strip()
        match = _WIKILINK_RE.match(text)
        if match:
            text = match.group("id").strip()
        if not text:
            raise SchemaError("related contains an empty id")
        out.append(text)
    return out


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Validate one global entry's frontmatter. Raises `SchemaError`.

    Extends RPent's `_validate`: same scope/kind/title/applies_when/confidence/
    evidence.cells checks, plus `symptom` and `related`, which our spec
    requires and upstream never enforces.
    """
    scope = metadata.get("scope")
    if scope not in SCOPES:
        raise SchemaError(f"scope must be one of {sorted(SCOPES)}, got {scope!r}")
    if metadata.get("kind") not in KINDS:
        raise SchemaError(f"kind must be one of {sorted(KINDS)}")
    for field in ("id", "title", "applies_when"):
        if not str(metadata.get(field, "") or "").strip():
            raise SchemaError(f"global memory requires a non-empty {field!r}")
    memory_id = str(metadata["id"]).strip()
    if not _ID_RE.match(memory_id):
        raise SchemaError(
            f"id {memory_id!r} must be lowercase kebab-case (a-z, 0-9, '-')"
        )
    if metadata.get("confidence") not in CONFIDENCE:
        raise SchemaError(f"confidence must be one of {sorted(CONFIDENCE)}")

    symptom = metadata.get("symptom")
    if not isinstance(symptom, list) or not symptom:
        raise SchemaError("symptom must be a non-empty list of search keywords")
    if any(not str(word).strip() for word in symptom):
        raise SchemaError("symptom contains an empty keyword")

    evidence = metadata.get("evidence") or {}
    if not isinstance(evidence, dict):
        raise SchemaError("evidence must be a mapping")
    cells = evidence.get("cells")
    if not isinstance(cells, list) or not cells:
        raise SchemaError("evidence.cells must be a non-empty list")
    if any(not str(cell).strip() for cell in cells):
        raise SchemaError("evidence.cells contains an empty cell tag")

    if "related" not in metadata:
        raise SchemaError("global memory requires 'related' (use [] when none)")
    normalize_related(metadata.get("related"))


def validate_body(body: str) -> None:
    """Check the three mandatory body sections. Raises `SchemaError`."""
    missing = [section for section in REQUIRED_SECTIONS if section not in body]
    if missing:
        raise SchemaError(f"body is missing required section(s): {' '.join(missing)}")
    for section in REQUIRED_SECTIONS:
        after = body.split(section, 1)[1]
        head = after.split("**", 1)[0]
        if not head.strip():
            raise SchemaError(f"body section {section} is empty")


def validate_leaf(path: Path) -> dict[str, Any]:
    """Validate one `global/<id>.md` file end to end; return its frontmatter."""
    metadata, body = split_frontmatter(path)
    validate_metadata(metadata)
    validate_body(body)
    if str(metadata["id"]).strip() != path.stem:
        raise SchemaError(
            f"id {metadata['id']!r} does not match filename stem {path.stem!r}"
        )
    return metadata


def validate_task_record(record: Any) -> None:
    """Validate one `task_only/<task>.json` payload. Raises `SchemaError`."""
    if not isinstance(record, dict):
        raise SchemaError("task record must be a JSON object")
    missing = [field for field in TASK_FIELDS if field not in record]
    if missing:
        raise SchemaError(f"task record is missing field(s): {', '.join(missing)}")
    extra = [key for key in record if key not in TASK_FIELDS]
    if extra:
        raise SchemaError(f"task record has unknown field(s): {', '.join(extra)}")
    for field, types in TASK_FIELDS.items():
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, types):
            names = "/".join(t.__name__ for t in types)
            raise SchemaError(f"task record field {field!r} must be {names}")
    if not str(record["task"]).strip():
        raise SchemaError("task record field 'task' must be non-empty")
    offset = record["best_xy_offset_m"]
    if len(offset) != 2 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) for v in offset
    ):
        raise SchemaError("best_xy_offset_m must be [dx, dy] in metres")
    if not 0.0 <= float(record["grasp_rate"]) <= 1.0:
        raise SchemaError("grasp_rate must be a fraction in [0, 1]")
    if int(record["n_samples"]) < 0:
        raise SchemaError("n_samples must be >= 0")
    if int(record["best_num_chunks"]) <= 0:
        raise SchemaError("best_num_chunks must be > 0")


__all__ = [
    "CONFIDENCE",
    "GLOBAL_FIELD_ORDER",
    "KINDS",
    "OPTIONAL_SECTIONS",
    "REQUIRED_SECTIONS",
    "SCOPES",
    "TASK_FIELDS",
    "SchemaError",
    "canonical_id",
    "normalize_related",
    "render",
    "split_frontmatter",
    "validate_body",
    "validate_leaf",
    "validate_metadata",
    "validate_task_record",
]
