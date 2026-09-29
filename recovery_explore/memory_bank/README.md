# memory_bank — the recovery_explore memory bank

Lessons accumulated across sessions. Exploration sessions **only write drafts into `inbox/<cell_tag>/`**;
the published `global/` and `task_only/` are updated solely by the `merge-inbox` tool, which never
overwrites on conflict and only archives.

The code reuses RPent (Apache-2.0) in copy-and-adapt fashion:

| File | Upstream | What is reused |
|---|---|---|
| `schema.py` | `vendor/rpent/memory/manager.py` (`_validate`/`_split_frontmatter`/`_canonical_id`/`_render`) | frontmatter contract |
| `schema.py` | `vendor/rpent/robocasa/libero_explore_prompt.py` (`STEP_DISTIL`) | three-section body format, field list, dedupe protocol |
| `manager.py` | `vendor/rpent/memory/manager.py` (`validate`/`rebuild_index`/`merge_memory`/`_merge_evidence`) | flock lock, evidence union, automatic confidence promotion |
| `_yamlish.py` | same as above (upstream uses PyYAML directly) | fallback parser when PyYAML is absent |

Deliberate differences from upstream are described in each file's header comment. Key points:
no `suite` layer; `symptom` / `related` **are validated** (upstream only asks for them in the
prompt, the code does not check); the three body sections are **checked mechanically**;
a draft with the same id but a different body **never overwrites a published file** and is
always archived to `_internal/conflicts/`.

---

## Layout

```
MEMORY.md              index, rewritten wholesale by build-index -- do not edit by hand
global/<id>.md         cross-task lessons, one per file
task_only/<task>.json  best configuration per task
inbox/<cell_tag>/      draft area for exploration sessions; never merged automatically
_internal/conflicts/   archived conflicting drafts; nothing is ever silently overwritten
_internal/merged/      archived inboxes that have been merged
```

`cell_tag` has the shape `<task>_s<seed>`, e.g. `PnPCounterToSink_s3`.
Confidence promotion uses it to tell "different tasks" apart; a malformed tag skews the counts.

---

## Running the three tools

The entry point is `recovery_explore/cli/memory_cli.py` (standard library only, nothing to install;
uses PyYAML if available, otherwise the fallback parser in `_yamlish.py`).

```bash
cd <repo>/recovery_explore

# 1) Validate: schema + id must equal file name + duplicate ids + three body sections
python3 cli/memory_cli.py validate
python3 cli/memory_cli.py validate --json     # machine-readable

# 2) Rebuild the index: rewrite MEMORY.md wholesale from the leaves' frontmatter
python3 cli/memory_cli.py build-index --show
#    Refuses to rebuild if validation fails; --force indexes anyway (broken leaves go under "Needs repair")

# 3) Merge: publish inbox drafts into global/ and task_only/
python3 cli/memory_cli.py merge-inbox --dry-run          # preview what would happen, write nothing
python3 cli/memory_cli.py merge-inbox --cell <cell_tag>  # merge a single cell
python3 cli/memory_cli.py merge-inbox                    # merge every cell under inbox/
```

`merge-inbox` has exactly three outcomes per draft:

| Case | Result |
|---|---|
| id not seen before | published to `global/<id>.md` |
| id exists, **same body** | recorded as a corroboration: union of evidence.cells, attempts summed, confidence re-graded |
| id exists, **different body** (or the published file cannot be parsed) | draft archived to `_internal/conflicts/<id>__from_<cell>.md`; **the published file is not touched**, awaiting human review |

Confidence ladder (as upstream): `cells >= 3 and distinct tasks >= 2` -> `verified`;
`cells >= 2` -> `probable`; otherwise `single-shot`. Authors must never write `verified`
by hand; it can only be earned by accumulated evidence.

After a successful merge the whole `inbox/<cell_tag>/` is moved to `_internal/merged/<cell_tag>/`;
if any draft failed to merge, the inbox stays in place so it can be fixed and rerun.
The whole process holds an exclusive flock on `_internal/merge.lock`, so several cells
finishing concurrently is safe.

---

## Writing contract (required reading for exploration sessions)

**Boundary**: you may only create files under `inbox/<your own cell_tag>/`.
Do not create, modify, rename or delete anything in `global/`, `task_only/`, `MEMORY.md` or `_internal/` --
that is the shared, reviewed corpus, and other cells may be reading it.

### 1. Global lessons -> `inbox/<cell_tag>/new_global_<kind>_<slug>.md`

One lesson per file. `<kind>` ∈ `primitive | perception | strategy | failure | infra`.
`<slug>` is 2-5 kebab-case words naming **the lesson**, not the task or the object.

The `id` is the **bare slug** -- the file name without the `new_global_` / `new_` prefix, the kind prefix and the `_draft` suffix:

```
new_global_strategy_short-prompt-after-missed-grasp.md
  -> id: short-prompt-after-missed-grasp        ✅
  -> id: new-global-short-prompt-after-missed-grasp   ❌
```

Frontmatter (fields in this order):

```yaml
---
id: <bare slug, lowercase kebab-case, must equal the final file name>
scope: global
kind: primitive|perception|strategy|failure|infra
title: <one imperative sentence saying what this lesson asks you to do>
applies_when: <trigger: when should a future agent open this entry?>
symptom: [<words someone who is stuck would search for>, ...]   # non-empty list
evidence:
  cells: [<cell_tag>, ...]                  # non-empty list
  attempts: <N>
confidence: single-shot                      # authors always write single-shot
related: [<bare id>, ...]                    # may be []; every id must actually exist
---
```

YAML caveats (upstream pitfalls): quote any value containing `: `;
do not start a value with a quote unless the whole value is quoted; `related` in frontmatter is a
**list of bare ids**, while `[[id]]` is **body** syntax (the validator tolerates quoted `"[[id]]"`
and strips the wrapper).

Body: a one-sentence conclusion, then three required sections and one optional section, with the
headings **copied verbatim**:

```
<one-sentence conclusion>

**Why:**          <mechanism. If unknown, write "observed, cause unknown" --
                  an honest unknown is useful, an invented cause is harmful.>
**How to apply:** <actionable: order, range, thresholds, criteria>
**Falsify:**      <what observation would overturn this entry>
**Related:**      <[[id]] links>
```

**Hard requirement specific to this package**: `How to apply` must be executable by **a local VLM
looking only at images**. Criteria may reference only what is visible in the images (gaps, occlusion,
symmetry, relative motion in agentview / eye-in-hand), and **must not reference oracle coordinates,
`object_xyz`, `is_grasped` or `task_success`** -- step-5 exploration runs with `--mask-oracle`,
and the service does not return those keys at all.
The wording must also **hold across tasks**: write "the object", not "the blue bowl"; write
"the support surface", not "the counter left of the sink". Anything that holds for only one task
belongs in `task_only/`, not here.

### 2. Best per-task configuration -> `inbox/<cell_tag>/<task>.json`

```json
{
  "task": "<task name, must equal the final file name stem>",
  "best_hover_m": 0.12,
  "best_xy_offset_m": [0.01, -0.005],
  "best_prompt_style": "short",
  "best_num_chunks": 8,
  "grasp_rate": 0.6,
  "n_samples": 10,
  "notes": "<free text, may be an empty string>"
}
```

No extra and no missing fields; `grasp_rate ∈ [0,1]`; `best_xy_offset_m` is `[dx, dy]` in metres.

### 3. Dedupe protocol (required before writing a new file)

1. Read `MEMORY.md`, then **open** the entries that might overlap -- compare **bodies**, not index lines.
2. **Already covered and consistent** -> do not create a new file. Write a draft whose id and body are
   **identical** to the published version, changing only `evidence.cells` to your own cell_tag:
   `merge-inbox` recognises this as a corroboration, merges the evidence and promotes confidence
   automatically. This is worth far more than another near-duplicate file.
3. **Covered but contradicted** -> do not edit it. Write your version into the inbox anyway: a different
   body is archived automatically to `_internal/conflicts/`, the published file is left alone, and a human
   decides. In the body, state your observation, attempt numbers and measurements, and the conditions
   under which each version might hold.

### 4. Wrap-up

```bash
python3 cli/memory_cli.py merge-inbox --cell <cell_tag> --dry-run
python3 cli/memory_cli.py merge-inbox --cell <cell_tag>
python3 cli/memory_cli.py validate
```

In the final report, state how many lessons were considered, which layer each went to, how many
corroborations were recorded and how many conflicts were filed.

---

## Seed entries

The 5 entries in `global/` are **priors**, not experimental findings: their `evidence.cells` is the
sentinel cell `prior-no-grid-data_s0`, with `attempts: 0` and `confidence: single-shot`.
This lets later real cells promote them honestly (as soon as a real cell arrives there are 2 cells ->
`probable`; one more from a different task -> `verified`) instead of having them start out carrying
someone else's evidence.
To overturn any of them, use the conflict flow above -- do not edit the file directly.
