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
# Adapted from RPent `rpent/memory/manager.py`, which uses PyYAML directly
# (`yaml.safe_load` / `yaml.safe_dump`). recovery_explore must also run under
# bare `python3` on a laptop with no PyYAML and no permission to pip install,
# so this module is a shim: PyYAML when importable, otherwise a tiny parser /
# emitter for the restricted frontmatter grammar this corpus uses.

"""YAML frontmatter load/dump with a dependency-free fallback.

Supported grammar (the only shapes `schema.py` allows anyway):

    key: scalar                # str / int / float / bool / null
    key: [a, b, c]             # flow sequence of scalars
    key:                       # block sequence of scalars
      - a
      - b
    key:                       # one level of nested mapping, whose values
      sub: scalar              # are scalars or sequences
      sub2: [a, b]

Anything deeper raises ``ValueError`` in the fallback path. That is a feature:
the corpus schema is deliberately flat, and a file that needs more structure is
a file that has drifted off contract.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - depends on the interpreter it runs under
    import yaml as _yaml
except ImportError:  # pragma: no cover
    _yaml = None

HAVE_PYYAML = _yaml is not None


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def _scalar(token: str) -> Any:
    token = token.strip()
    if not token:
        return None
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    low = token.lower()
    if low in ("null", "~"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def _split_flow(token: str) -> list[Any]:
    inner = token.strip()[1:-1].strip()
    if not inner:
        return []
    items: list[str] = []
    depth = 0
    quote = ""
    current = ""
    for char in inner:
        if quote:
            current += char
            if char == quote:
                quote = ""
            continue
        if char in ("'", '"'):
            quote = char
            current += char
            continue
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        if char == "," and depth == 0:
            items.append(current)
            current = ""
            continue
        current += char
    items.append(current)
    return [_scalar(item) for item in items]


def _fallback_load(text: str) -> dict[str, Any]:
    lines = [line.rstrip() for line in text.splitlines()]
    lines = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    root: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if line[0].isspace():
            raise ValueError(f"unexpected indentation at {line.strip()!r}")
        if ":" not in line:
            raise ValueError(f"expected 'key: value' at {line.strip()!r}")
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()
        index += 1
        if rest.startswith("[") and rest.endswith("]"):
            root[key] = _split_flow(rest)
            continue
        if rest:
            root[key] = _scalar(rest)
            continue
        # Block: either a sequence or a nested mapping.
        block: list[str] = []
        while index < len(lines) and lines[index][0].isspace():
            block.append(lines[index])
            index += 1
        if not block:
            root[key] = None
            continue
        if block[0].lstrip().startswith("- "):
            root[key] = [_scalar(item.lstrip()[2:]) for item in block]
            continue
        root[key] = _parse_nested(key, block)
    return root


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _parse_nested(parent: str, block: list[str]) -> dict[str, Any]:
    """Parse one level of nested mapping; its values may be block sequences.

    PyYAML (on the cluster) emits block sequences under `evidence:`; the
    fallback emitter uses flow sequences. Both must read back here.
    """
    nested: dict[str, Any] = {}
    base = _indent(block[0])
    position = 0
    while position < len(block):
        line = block[position]
        stripped = line.strip()
        position += 1
        if stripped.startswith("- "):
            raise ValueError(f"sequence mixed into mapping under {parent!r}")
        if ":" not in stripped:
            raise ValueError(f"expected 'key: value' under {parent!r}")
        sub_key, _, sub_rest = stripped.partition(":")
        sub_key = sub_key.strip()
        sub_rest = sub_rest.strip()
        if sub_rest.startswith("[") and sub_rest.endswith("]"):
            nested[sub_key] = _split_flow(sub_rest)
            continue
        if sub_rest:
            nested[sub_key] = _scalar(sub_rest)
            continue
        items: list[Any] = []
        while position < len(block) and _indent(block[position]) >= base:
            child = block[position].strip()
            if not child.startswith("- "):
                break
            items.append(_scalar(child[2:]))
            position += 1
        if not items:
            raise ValueError(
                f"nesting under {parent}.{sub_key} is deeper than this corpus allows"
            )
        nested[sub_key] = items
    return nested


def safe_load(text: str) -> Any:
    """Parse a YAML frontmatter block into plain Python data."""
    if _yaml is not None:
        return _yaml.safe_load(text)
    return _fallback_load(text)


# --------------------------------------------------------------------------- #
# dumping
# --------------------------------------------------------------------------- #

_NEEDS_QUOTE = (": ", " #", "\n")


def _emit_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "":
        return '""'
    if (
        any(marker in text for marker in _NEEDS_QUOTE)
        or text[0] in "[]{}#&*!|>'\"%@`-?,"
        or text.rstrip() != text
        or text.strip().lower() in ("true", "false", "null", "yes", "no", "on", "off")
    ):
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def _emit_seq(value: list[Any]) -> str:
    return "[" + ", ".join(_emit_scalar(item) for item in value) + "]"


def _fallback_dump(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (list, tuple)):
                    lines.append(f"  {sub_key}: {_emit_seq(list(sub_value))}")
                elif isinstance(sub_value, dict):
                    raise ValueError(f"nesting under {key}.{sub_key} is too deep")
                else:
                    lines.append(f"  {sub_key}: {_emit_scalar(sub_value)}")
        elif isinstance(value, (list, tuple)):
            lines.append(f"{key}: {_emit_seq(list(value))}")
        else:
            lines.append(f"{key}: {_emit_scalar(value)}")
    return "\n".join(lines) + "\n"


def safe_dump(data: dict[str, Any]) -> str:
    """Serialize a frontmatter mapping, preserving key order and unicode."""
    if _yaml is not None:
        return _yaml.safe_dump(
            data, sort_keys=False, allow_unicode=True, default_flow_style=False
        )
    return _fallback_dump(data)


__all__ = ["HAVE_PYYAML", "safe_dump", "safe_load"]
