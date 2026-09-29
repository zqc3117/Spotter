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

"""recovery_explore memory bank: schema, corpus manager, and seed lessons.

Layout under this package:

    MEMORY.md              generated index, rebuilt wholesale
    global/<id>.md         cross-task lessons (frontmatter + 3-section body)
    task_only/<task>.json  per-task best config
    inbox/<cell_tag>/      exploration drafts; NEVER auto-merged
    _internal/conflicts/   archived conflicting drafts; never overwritten
    _internal/merged/      inboxes already published
"""

from .manager import BANK_ROOT, MemoryBank, grade_confidence, merge_evidence
from .schema import SchemaError

__all__ = [
    "BANK_ROOT",
    "MemoryBank",
    "SchemaError",
    "grade_confidence",
    "merge_evidence",
]
