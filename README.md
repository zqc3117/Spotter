# 🤖 Spotter: Let the Embodied Model Lead, and the VLM Reflect for It

<p align="center">If our project helps you, please give us a star ⭐ on GitHub to support us.</p>

<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/README-English-111111?style=for-the-badge" alt="English"></a>
  <a href="./README_zh.md"><img src="https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836?style=for-the-badge" alt="中文"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b?style=for-the-badge" alt="arXiv"></a>
  <a href="https://zqc3117.github.io/Spotter/"><img src="https://img.shields.io/badge/Project_Page-Spotter-2ea44f?style=for-the-badge" alt="Project page"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://robocasa.ai"><img src="https://img.shields.io/badge/Simulator-RoboCasa-4c8bf5?style=for-the-badge" alt="RoboCasa"></a>
</p>

| 📄 Paper | 🏠 Project page | 🤖 Model checkpoints | 🌐 Simulator | 🚀 Get started |
|---|---|---|---|---|
| [arXiv](https://arxiv.org/abs/XXXX.XXXXX) | [zqc3117.github.io/Spotter](https://zqc3117.github.io/Spotter/) | [pi0.5](https://huggingface.co/DAVIAN-Robotics/pi05-robocasa-H50) · [Cosmos Policy](https://huggingface.co/nvidia/Cosmos-Policy-RoboCasa-Predict2-2B) | [RoboCasa](https://robocasa.ai) | [Installation](#1-installation) · [Quick start](#3-quick-start-one-pi05-episode) |

**Spotter** adds a lightweight intervention layer between a frozen embodied model (System 1) and a VLM (System 2).
Instead of having the VLM plan every step, the embodied model leads and keeps executing, while a local screener
watches in parallel and the VLM steps in only to confirm and repair an error. The VLM's job becomes much simpler,
so the whole framework works with a **locally served, open-source Qwen and no privileged information**.

## 🔥 Highlights

- **A lightweight intervention layer.** Like a person who keeps a little attention on a task and focuses only when
  something looks wrong, Spotter lets execution flow continuously and brings in the VLM only on a confirmed error.
- **A simpler job for the VLM.** The VLM no longer drives the robot. It only has to notice a failure, repair it with
  a short primitive plan, and check the fix. A local Qwen with no access to the task's success signal is enough.
- **Continuous execution, low overhead.** The embodied model never waits for the VLM on the routine path:
  a successful episode takes only **13–16 s** longer than the policy alone.

## 💡 Why Spotter

Existing VLM-led robot agents are serial: the VLM plans a segment, the embodied model executes it, and the VLM
decides what comes next. This forces a dilemma:

- **Short segments** mean frequent VLM calls and high latency, even on easy tasks.
- **Long segments** mean a deviation during execution cannot be caught in time, and later actions keep going wrong.

Spotter removes this trade-off by supervising in parallel rather than in series.

|  | VLM-led agents | Spotter |
|---|---|---|
| Who drives execution | VLM plans every step | Embodied model leads |
| When the VLM is called | Before every step | Only on a confirmed error |
| Embodied model waits for the VLM | Every step | Only during a repair |
| Needs privileged information | Usually (e.g. the success signal) | No |
| Works with a local Qwen | Only with privileged information | Yes |

### 🎯 Works with a weaker model, without privileged information

With the same local Qwen and no privileged information on RoboCasa (1,200 episodes), a representative VLM-led agent
falls **below the policy alone** (Cosmos Policy 67.9 → 65.2, π0.5 64.3 → 60.5), while Spotter still improves it
(→ 71.7 and 68.4).

### ⚡ Almost no overhead when nothing goes wrong

With Qwen, the judge is called **less than once** per successful episode on average, which adds only **13–16 s**
over the policy alone and takes about **70% less time** than a VLM-led agent using the same Qwen.

## 1. Installation

| Component | Checkpoint | Used by |
|---|---|---|
| pi0.5 | [DAVIAN-Robotics/pi05-robocasa-H50](https://huggingface.co/DAVIAN-Robotics/pi05-robocasa-H50) | `FAMILY_OVERRIDE=pi05` |
| Cosmos Policy | [nvidia/Cosmos-Policy-RoboCasa-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-RoboCasa-Predict2-2B) | `FAMILY_OVERRIDE=cosmos` |
| Qwen judge | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8), served locally with vLLM | judge and screener |

You also need the upstream Cosmos Policy / RoboCasa tree (Python 3.10), pi0.5 / `lerobot`
(Python 3.12), vLLM, and an EGL-capable GPU. This repository does not ship upstream policy
packages or model weights; the Cosmos policy server (`python -m rpc.server`) comes from the
upstream Cosmos Policy repository.

```bash
cp env/env.sh.example env/env.sh
$EDITOR env/env.sh        # point COSMOS_POLICY_ROOT, PI05_PYTHON, checkpoints, QWEN_MODEL, ... at your install
source env/env.sh
```

Reference versions are pinned in [`env/requirements-robocasa.txt`](env/requirements-robocasa.txt)
and [`env/requirements-pi05.txt`](env/requirements-pi05.txt).

### Serve the Qwen judge

The judge and the screener share one locally served Qwen. Start one vLLM replica per GPU
(ports 8301, 8302, ...) and keep the terminal open:

```bash
bash spotter.sh judge --gpus 0,1,2,3
```

With fewer GPUs, pass fewer ids (e.g. `--gpus 0`) and `export QWEN_PORTS=8301` before running.

<details>
<summary>What <code>spotter.sh judge</code> runs</summary>

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve "$QWEN_MODEL" \
  --served-model-name qwen38 --port 8301 \
  --max-model-len 131072 --max-num-seqs 64 \
  --limit-mm-per-prompt '{"image":60,"video":2}' \
  --gpu-memory-utilization 0.70 --seed 0
```
</details>

## 2. Pipeline

```text
policy server ──actions──> env_service (RoboCasa + MuJoCo + EGL)
      ▲                         │ frames + telemetry
      └──────── judge driver <──┘
                    │
             Qwen screener/judge
                    │
             JSON repair plan
```

Cosmos uses 16 steps per policy chunk; pi0.5 uses 25. With `SCREEN=1` a lightweight screener
checks every window and only escalates suspicious windows to the full judge.

## 3. Quick start: one pi0.5 episode

Three commands, each in its own terminal (after `source env/env.sh`):

```bash
bash spotter.sh judge --gpus 0      # 1. Qwen judge + screener (see above)
bash spotter.sh policy pi05 --gpu 1 # 2. the embodied model
bash spotter.sh run pi05            # 3. one episode, control and Spotter on the same scene
```

`run` checks that the policy and Qwen are up, starts the simulation service itself, and prints the
success rates when it finishes. Add `--dry-run` to see the exact commands without running them.

## 4. Full-scale runs

The paper evaluates on `sall500`: 24 RoboCasa tasks × episode index 0–49 at seed 500
(1,200 episodes, the same list for both policy families). The lists are in
[`recovery_explore/episode_sets/`](recovery_explore/episode_sets/); `s500q96` is a 96-episode subset for quick checks.

```bash
bash spotter.sh policy cosmos --gpu 4
bash spotter.sh run cosmos sall500 --lanes 8     # 8 episodes in parallel
bash spotter.sh summary                          # success rates, any time
```

| Option | Meaning |
|---|---|
| `SET` | `smoke` (1 episode, default), `s500q96`, `sall500`, or a file with one `TASK:SEED:EPISODE` per line |
| `--lanes N` | episodes run in parallel; lane `N` uses simulation port `8450+N` |
| `--sim-gpus N` | GPUs the simulation services share, GPU `0..N-1` (default: all) |
| `--arm` | `both` (control + Spotter on each episode, default), `treat` (Spotter only), `ctrl` (policy only) |
| `--run NAME` | run name (default `<family>_<set>`); rerunning the same name resumes it |

Every episode can be run in two arms on the same scene, instruction and step budget:

| Arm | Description |
|---|---|
| **Control** | Policy only; no judge calls. |
| **Treatment** | Policy plus judge and optional interventions. |

All paper settings are built in (Table 3, "full context"); any variable from
[Configuration](#5-configuration) exported in your shell overrides them.

<details>
<summary>Running the driver directly</summary>

`spotter.sh run` launches one driver per lane. The single-episode pi0.5 run above is equivalent to:

```bash
mkdir -p recovery_explore/runs_smoke
printf 'PnPCabToCounter:195:0\n' > recovery_explore/runs_smoke/episodes_pi05.txt
ON_POD=1 RUN=smoke TARGET=1 NGPU=1 FAMILY_OVERRIDE=pi05 \
SIM_HOST=127.0.0.1 SIM_MANAGED=0 POLICY_URL=http://127.0.0.1:8900 \
ENGINE=qwen MODEL=qwen38 SCREEN=1 QWEN_KEEP_TURNS=20 \
STEP_BUDGET_SCALE=1.8 WINDOW=1 TEL_CHUNKS=10 COMPACT_AT=110000 MAX_INTERVENTIONS=8 \
ALLOW_RESET=0 MAX_RESETS=0 FEWSHOT=0 LEARN=0 CTRL_ONLY=0 SKIP_CONTROL=0 \
  bash recovery_explore/judge_driver_v7.sh 0
```

The driver starts and restarts the simulation service on port `8450+lane` by itself. All lanes of a run
claim episodes from the same queue, and each lane stops once the run has `TARGET` results.
</details>

## 5. Configuration

| Setting | Cosmos | pi0.5 |
|---|---:|---:|
| `FAMILY_OVERRIDE` | `cosmos` | `pi05` |
| Steps per chunk | 16 | 25 |
| `WINDOW` (judge window, in chunks) | 2 | 1 |
| `TEL_CHUNKS` | 40 | 10 |
| Policy port base | 8800 | 8900 |
| Policy server | `python -m rpc.server` (upstream Cosmos Policy) | `python -m rpc.policy_server_pi05` |

| Parameter | Paper value | Purpose |
|---|---:|---|
| `STEP_BUDGET_SCALE` | `1.8` | Step budget multiplier; same value for control and treatment. |
| `SCREEN` | `1` | Screener on; `0` calls the judge every window. |
| `MAX_INTERVENTIONS` | `8` | Maximum interventions per episode. |
| `ALLOW_RESET` / `MAX_RESETS` | `0` / `0` | Disable rollbacks so episodes stay comparable. |
| `ENGINE` / `MODEL` | `qwen` / `qwen38` | Local Qwen judge. |
| `COMPACT_AT` | `110000` | Qwen conversation compaction threshold (tokens). |
| `QWEN_KEEP_TURNS` | `20` | Recent judge turns kept in full. |
| `FEWSHOT` | `0` | Set to `1` only with a `FEWSHOT_BANK`. |

How the judge is paced has its own switches, all off or conservative by default:

| Variable | Default | Effect |
|---|---:|---|
| `ASYNC_SCREEN` | `0` | `1` lets the policy keep stepping while the judge works; the loop then resumes at the newest finished window. Faster, but the judge is looking at a window the episode has already moved past — `results.jsonl` records that cost as `stale_steps` and `stale_ivs`. `2` also holds the policy within `ASYNC_K`+1 windows. |
| `MISS_GRACE` | `2` | Windows the policy gets to retry on its own after closing on nothing, before the judge may take over. |
| `FIRST_IV_FRAC` | `35` | The first intervention waits until the policy has closed on nothing at least once, or the episode is this far (%) into its step budget. `0` disables the gate. |
| `MONITOR_AFTER_BUDGET` | `1` | Keep calling the judge once the intervention budget is spent, for the record. `0` stops calling it. |

The paper results are synchronous (`ASYNC_SCREEN=0`). Changing `STEP_BUDGET_SCALE`, `WINDOW`
or the policy family changes the experiment; compare runs by matched task/seed/episode, not
only by aggregate success rate.

## 6. Outputs

Results are written to `recovery_explore/runs_<RUN>_<family>/`:

```text
results.jsonl             one line per episode (control_success, treatment_success, interventions, judge_calls, ...)
lane<N>/driver.log        per-lane progress
lane<N>/jd-.../w<window>/ frames shown to the judge
```

Summarize every run, or one run by name:

```bash
bash spotter.sh summary            # all runs
bash spotter.sh summary cosmos_sall500
```

## 7. Repository layout

```text
spotter.sh            one entry point: serve Qwen / the policy, run episodes, summarize
recovery_explore/     judge driver, RoboCasa service, judge CLIs, briefs, lesson library, episode sets
rpc/                  pi0.5 policy server, RPC protocol, RoboCasa rollout helpers
cf_bench/             snapshot, replay and rendering helpers
env/                  environment template and version pins
```

## 📜 License

This project is released under the [MIT License](LICENSE). Some files in
`recovery_explore/` are adapted from RPent and remain under the Apache License 2.0; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## 🙏 Acknowledgements

We build on [RoboCasa](https://robocasa.ai),
[Cosmos Policy](https://huggingface.co/nvidia/Cosmos-Policy-RoboCasa-Predict2-2B),
[pi0.5](https://huggingface.co/DAVIAN-Robotics/pi05-robocasa-H50), [vLLM](https://github.com/vllm-project/vllm)
and RPent. We thank their authors for releasing their work.

## 📖 Citation

```bibtex
@article{li2026spotter,
  title   = {Spotter: Let the Embodied Model Lead, and the VLM Reflect for It},
  author  = {Li, Long and Zhao, Qichao and Yang, Yue and Xu, Fan and Wang, Zhe and Liew, Alan Wee-Chung and Qu, Chao and Shen, Heng Tao and Pan, Shirui},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```
