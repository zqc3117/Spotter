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
#   upstream file: rpent/memory/manager.py  (`MemoryManager.validate`,
#                  `MemoryManager.rebuild_index`, `MemoryManager.merge_memory`,
#                  `_merge_evidence`, and the fcntl.flock merge lock)
#
# Kept from upstream: the exclusive flock around every merge, evidence-set
# union on corroboration, and the confidence ladder (>=3 cells and >=2 distinct
# tasks -> verified; >=2 cells -> probable; else single-shot).
#
# Changed on purpose:
#   * inbox lives at `memory_bank/inbox/<cell_tag>/`, not `_internal/inbox/`,
#     so an exploration session can find it without spelunking `_internal`.
#   * a draft whose id already exists with DIFFERENT prose is archived to
#     `_internal/conflicts/` and the published file is left byte-for-byte
#     untouched. Upstream rewrites the published frontmatter in that case; we
#     do not, because a human resolves conflicts here.
#   * the suite tier is gone; `task_only/` holds JSON best-config records with
#     their own schema instead of opaque audit/recipe pairs.

"""The three memory-bank tools: validate, rebuild_index, merge_inbox."""

from __future__ import annotations

import fcntl
import json
import shutil
from pathlib import Path
from typing import Any

from . import schema
from .schema import SchemaError

BANK_ROOT = Path(__file__).resolve().parent

#: Cell tags shaped `<task>_s<seed>`; the task prefix drives the "distinct
#: tasks" half of the confidence ladder (upstream convention, kept).
_SEED_SEP = "_s"


def _tasks_of(cells: list[str]) -> set[str]:
    return {str(cell).rsplit(_SEED_SEP, 1)[0] for cell in cells}


def grade_confidence(cells: list[str]) -> str:
    """Confidence implied by an evidence cell set (upstream ladder)."""
    tasks = _tasks_of(cells)
    if len(cells) >= 3 and len(tasks) >= 2:
        return "verified"
    if len(cells) >= 2:
        return "probable"
    return "single-shot"


def merge_evidence(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Union two entries' evidence and re-grade confidence.

    Adapted from RPent `_merge_evidence`.
    """
    old_evidence = dict(old.get("evidence") or {})
    new_evidence = dict(new.get("evidence") or {})
    cells = sorted({*old_evidence.get("cells", []), *new_evidence.get("cells", [])})
    evidence: dict[str, Any] = {
        **old_evidence,
        **{k: v for k, v in new_evidence.items() if k not in ("cells", "attempts")},
        "cells": cells,
        "attempts": int(old_evidence.get("attempts") or 0)
        + int(new_evidence.get("attempts") or 0),
    }
    for key in ("solved_seeds", "failed_seeds", "contradicted_by"):
        if key in old_evidence or key in new_evidence:
            evidence[key] = sorted(
                {*old_evidence.get(key, []), *new_evidence.get(key, [])}
            )
    related = sorted(
        {
            *schema.normalize_related(old.get("related")),
            *schema.normalize_related(new.get("related")),
        }
    )
    return {
        **old,
        "evidence": evidence,
        "confidence": grade_confidence(cells),
        "related": related,
    }


class MemoryBank:
    """One recovery_explore memory corpus rooted at `memory_bank/`."""

    def __init__(self, root: str | Path = BANK_ROOT) -> None:
        self._root = Path(root).resolve()

    # -- layout ----------------------------------------------------------- #

    @property
    def root(self) -> Path:
        return self._root

    @property
    def global_dir(self) -> Path:
        return self._root / "global"

    @property
    def task_dir(self) -> Path:
        return self._root / "task_only"

    @property
    def inbox_dir(self) -> Path:
        return self._root / "inbox"

    @property
    def internal_dir(self) -> Path:
        return self._root / "_internal"

    @property
    def conflicts_dir(self) -> Path:
        return self.internal_dir / "conflicts"

    @property
    def merged_dir(self) -> Path:
        return self.internal_dir / "merged"

    @property
    def index_path(self) -> Path:
        return self._root / "MEMORY.md"

    def ensure_layout(self) -> None:
        for directory in (
            self.global_dir,
            self.task_dir,
            self.inbox_dir,
            self.conflicts_dir,
            self.merged_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # -- tool 1: validate -------------------------------------------------- #

    def validate(self) -> list[str]:
        """Return every contract violation found under `global/` and `task_only/`.

        Checks, per spec: schema, `id` == filename stem, duplicate ids across
        files, and the three mandatory body sections. Unlike upstream, a leaf
        with NO frontmatter is reported rather than silently tolerated — every
        file in `global/` is a memory here.
        """
        problems: list[str] = []
        ids: dict[str, Path] = {}
        for path in sorted(self.global_dir.glob("*.md")):
            try:
                metadata = schema.validate_leaf(path)
            except SchemaError as exc:
                problems.append(f"{self._rel(path)}: {exc}")
                continue
            memory_id = str(metadata["id"]).strip()
            if memory_id in ids:
                problems.append(
                    f"{self._rel(path)}: duplicate id {memory_id!r} also in "
                    f"{self._rel(ids[memory_id])}"
                )
            ids[memory_id] = path

        for path in sorted(self.global_dir.glob("*.md")):
            try:
                metadata, _ = schema.split_frontmatter(path)
                related = schema.normalize_related(metadata.get("related"))
            except SchemaError:
                continue
            for target in related:
                if target not in ids:
                    problems.append(
                        f"{self._rel(path)}: related id {target!r} has no entry "
                        "in global/"
                    )

        for path in sorted(self.task_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text())
                schema.validate_task_record(record)
            except (SchemaError, ValueError) as exc:
                problems.append(f"{self._rel(path)}: {exc}")
                continue
            if str(record["task"]).strip() != path.stem:
                problems.append(
                    f"{self._rel(path)}: task {record['task']!r} does not match "
                    f"filename stem {path.stem!r}"
                )
        return problems

    # -- tool 2: rebuild_index --------------------------------------------- #

    def rebuild_index(self) -> Path:
        """Regenerate MEMORY.md wholesale from leaf frontmatter.

        Adapted from RPent `rebuild_index`. Unparseable leaves are listed under
        a "needs repair" heading instead of vanishing — a memory that silently
        drops out of the index is a memory nobody notices is broken.
        """
        self.ensure_layout()
        entries: list[tuple[str, dict[str, Any]]] = []
        broken: list[tuple[str, str]] = []
        for path in sorted(self.global_dir.glob("*.md")):
            try:
                metadata, _ = schema.split_frontmatter(path)
            except SchemaError as exc:
                broken.append((path.name, str(exc)))
                continue
            entries.append((path.name, metadata))

        by_kind: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for name, metadata in entries:
            kind = str(metadata.get("kind") or "unsorted")
            by_kind.setdefault(kind, []).append((name, metadata))

        lines = [
            "# recovery_explore memory bank index",
            "",
            "<!-- GENERATED FILE — regenerated wholesale by",
            "     `cli/memory_cli.py build-index`. Do not hand-edit: edit the",
            "     leaves under global/ and task_only/, then rebuild. -->",
            "",
            f"Entries: {len(entries)} global, "
            f"{len(list(self.task_dir.glob('*.json')))} task_only.",
        ]

        lines.extend(("", "## Global", ""))
        if not entries:
            lines.append("_(none)_")
        for kind in sorted(by_kind):
            lines.extend((f"### {kind}", ""))
            for name, metadata in by_kind[kind]:
                label = metadata.get("title") or metadata.get("id") or name
                applies = str(metadata.get("applies_when") or "").strip()
                confidence = str(metadata.get("confidence") or "?")
                symptom = metadata.get("symptom") or []
                suffix = f" — {applies}" if applies else ""
                lines.append(f"- [{label}](global/{name}) `{confidence}`{suffix}")
                if isinstance(symptom, list) and symptom:
                    keywords = ", ".join(str(word) for word in symptom)
                    lines.append(f"  - symptom: {keywords}")
            lines.append("")

        lines.extend(("## Task-only best configs", ""))
        task_files = sorted(self.task_dir.glob("*.json"))
        if not task_files:
            lines.append("_(none)_")
        for path in task_files:
            try:
                record = json.loads(path.read_text())
            except ValueError as exc:
                broken.append((path.name, str(exc)))
                continue
            lines.append(
                f"- [{record.get('task', path.stem)}](task_only/{path.name}) — "
                f"hover={record.get('best_hover_m')}m "
                f"xy_offset={record.get('best_xy_offset_m')} "
                f"prompt={record.get('best_prompt_style')!r} "
                f"chunks={record.get('best_num_chunks')} "
                f"grasp_rate={record.get('grasp_rate')} "
                f"(n={record.get('n_samples')})"
            )

        if broken:
            lines.extend(("", "## Needs repair", ""))
            for name, reason in broken:
                lines.append(f"- `{name}` — {reason}")

        self.index_path.write_text("\n".join(lines).rstrip() + "\n")
        return self.index_path

    # -- tool 3: merge_inbox ----------------------------------------------- #

    def merge_inbox(self, cell_tag: str, *, dry_run: bool = False) -> dict[str, Any]:
        """Publish one cell's inbox drafts into `global/` and `task_only/`.

        Adapted from RPent `merge_memory`, held under the same exclusive flock.

        Three outcomes per draft:
          * id is new                -> published.
          * id exists, SAME prose    -> corroboration: evidence cells are unioned
                                        and confidence re-graded.
          * id exists, DIFFERENT     -> the draft is archived to
            prose (or the published    `_internal/conflicts/<id>__from_<cell>.md`
            file will not parse)       and the published file is NOT touched.
        """
        self.ensure_layout()
        inbox = self.inbox_dir / cell_tag
        result: dict[str, Any] = {
            "cell": cell_tag,
            "dry_run": dry_run,
            "published": [],
            "corroborated": [],
            "conflicts": [],
            "tasks": [],
            "skipped": [],
        }
        if not inbox.is_dir():
            result["skipped"].append(f"no inbox directory for cell {cell_tag!r}")
            return result

        lock_path = self.internal_dir / "merge.lock"
        lock_path.touch(exist_ok=True)
        with lock_path.open("r+") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                for source in sorted(inbox.glob("*.md")):
                    self._merge_one_draft(source, cell_tag, result, dry_run)
                for source in sorted(inbox.glob("*.json")):
                    self._merge_one_task(source, cell_tag, result, dry_run)
                if not dry_run and not result["skipped"]:
                    archive = self.merged_dir / cell_tag
                    if archive.exists():
                        shutil.rmtree(archive)
                    shutil.move(str(inbox), str(archive))
                elif not dry_run:
                    result["skipped"].append(
                        "inbox kept in place because some drafts did not merge"
                    )
                if not dry_run:
                    self.rebuild_index()
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
        return result

    def _merge_one_draft(
        self,
        source: Path,
        cell_tag: str,
        result: dict[str, Any],
        dry_run: bool,
    ) -> None:
        try:
            metadata, body = schema.split_frontmatter(source)
            memory_id = schema.canonical_id(source, metadata)
            metadata["id"] = memory_id
            schema.validate_metadata(metadata)
            schema.validate_body(body)
        except SchemaError as exc:
            result["skipped"].append(f"{source.name}: {exc}")
            return

        metadata["related"] = schema.normalize_related(metadata.get("related"))
        destination = self.global_dir / f"{memory_id}.md"
        if not destination.exists():
            if not dry_run:
                destination.write_text(schema.render(metadata, body))
            result["published"].append(memory_id)
            return

        try:
            old_metadata, old_body = schema.split_frontmatter(destination)
        except SchemaError as exc:
            self._archive_conflict(
                memory_id, cell_tag, metadata, body, result, dry_run
            )
            result["skipped"].append(
                f"{source.name}: published {destination.name} will not parse "
                f"({exc}); left untouched, draft archived"
            )
            return

        if body.strip() != old_body.strip():
            self._archive_conflict(
                memory_id, cell_tag, metadata, body, result, dry_run
            )
            return

        old_cells = list((old_metadata.get("evidence") or {}).get("cells") or [])
        if cell_tag in old_cells:
            result["skipped"].append(
                f"{source.name}: cell {cell_tag} already credited on {memory_id}"
            )
            return
        merged = merge_evidence(old_metadata, metadata)
        if not dry_run:
            destination.write_text(schema.render(merged, old_body))
        result["corroborated"].append(
            {
                "id": memory_id,
                "cells": merged["evidence"]["cells"],
                "confidence": merged["confidence"],
                "was": old_metadata.get("confidence"),
            }
        )

    def _merge_one_task(
        self,
        source: Path,
        cell_tag: str,
        result: dict[str, Any],
        dry_run: bool,
    ) -> None:
        try:
            record = json.loads(source.read_text())
            schema.validate_task_record(record)
        except (SchemaError, ValueError) as exc:
            result["skipped"].append(f"{source.name}: {exc}")
            return
        task = str(record["task"]).strip()
        destination = self.task_dir / f"{task}.json"
        payload = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
        if destination.exists():
            if destination.read_text() == payload:
                result["skipped"].append(f"{source.name}: identical to published")
                return
            target = self.conflicts_dir / f"{task}__from_{cell_tag}.json"
            target = self._unique(target)
            if not dry_run:
                target.write_text(payload)
            result["conflicts"].append(str(self._rel(target)))
            return
        if not dry_run:
            destination.write_text(payload)
        result["tasks"].append(task)

    def _archive_conflict(
        self,
        memory_id: str,
        cell_tag: str,
        metadata: dict[str, Any],
        body: str,
        result: dict[str, Any],
        dry_run: bool,
    ) -> None:
        target = self._unique(self.conflicts_dir / f"{memory_id}__from_{cell_tag}.md")
        if not dry_run:
            self.conflicts_dir.mkdir(parents=True, exist_ok=True)
            target.write_text(schema.render(metadata, body))
        result["conflicts"].append(str(self._rel(target)))

    # -- helpers ----------------------------------------------------------- #

    @staticmethod
    def _unique(path: Path) -> Path:
        """Never clobber an existing archive — conflicts accumulate."""
        if not path.exists():
            return path
        counter = 2
        while True:
            candidate = path.with_name(f"{path.stem}__{counter}{path.suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    def _rel(self, path: Path) -> Path:
        try:
            return path.relative_to(self._root)
        except ValueError:
            return path


__all__ = ["BANK_ROOT", "MemoryBank", "grade_confidence", "merge_evidence"]
