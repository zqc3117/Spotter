# 🤖 Spotter：让具身模型主导，让视觉语言模型为它反思

<p align="center">如果我们的项目对你有帮助，欢迎在 GitHub 上点一个 star ⭐ 支持我们</p>

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

| 📄 论文 | 🏠 项目主页 | 🤖 模型权重 | 🌐 仿真器 | 🚀 上手 |
|---|---|---|---|---|
| [arXiv](https://arxiv.org/abs/XXXX.XXXXX) | [zqc3117.github.io/Spotter](https://zqc3117.github.io/Spotter/) | [pi0.5](https://huggingface.co/DAVIAN-Robotics/pi05-robocasa-H50) · [Cosmos Policy](https://huggingface.co/nvidia/Cosmos-Policy-RoboCasa-Predict2-2B) | [RoboCasa](https://robocasa.ai) | [安装](#1-安装) · [快速开始](#3-快速开始跑一集-pi05) |

**Spotter** 在冻结的具身模型（System 1）和 VLM（System 2）之间加入了一个轻量的介入机制。
它不让 VLM 规划每一步，而是让具身模型主导并持续执行；本地筛子并行关注执行进展，只有确认出错时，VLM 才介入确认并纠正。
交给 VLM 的任务因此简单得多，整个框架**用本地部署的开源 Qwen、不借助任何特权信息**就能运行。

## 🔥 核心优势

- **轻量的介入机制。** 就像人做事时分出少量注意力关注进展，发现异常再集中精力处理：执行保持连贯，只在确认出错时才调用 VLM。
- **VLM 的任务更简单。** VLM 不再驱动机器人，只需发现失败、用几个运动原语组成的短计划修复，并检查修复是否奏效。
  本地 Qwen 在看不到任务成功条件的情况下也能胜任。
- **执行不中断，开销低。** 常规执行时具身模型不需要等 VLM，成功的 episode 只比策略单独运行多 **13–16 秒**。

## 💡 为什么需要 Spotter

现有由 VLM 主导的 robot agent 通常是串行的：VLM 先规划一段动作，交给具身模型执行，执行完再由 VLM 判断下一步。这带来一个两难：

- **每次安排的动作太少**：需要频繁调用 VLM，任务再简单时延也高；
- **每次安排的动作太多**：执行中出现偏差时无法及时纠正，后续动作会持续出错。

Spotter 把串行监督改成并行监督，消除了这个两难。

|  | VLM 主导的 agent | Spotter |
|---|---|---|
| 谁主导执行 | VLM 规划每一步 | 具身模型主导 |
| 何时调用 VLM | 每一步之前 | 仅在确认出错时 |
| 具身模型是否等待 VLM | 每一步都等 | 仅在修复期间 |
| 是否需要特权信息 | 通常需要（如任务成功信号） | 不需要 |
| 能否用本地 Qwen | 只有借助特权信息才行 | 可以 |

### 🎯 更弱的模型也能用，不需要特权信息

在 RoboCasa（1200 集）上，用同一个本地 Qwen、不给特权信息时，一种代表性的 VLM 主导 agent 成功率**跌到策略单独运行以下**
（Cosmos Policy 67.9 → 65.2，π0.5 64.3 → 60.5）；Spotter 仍然能带来提升（分别到 71.7 和 68.4）。

### ⚡ 不出错时几乎零开销

用 Qwen 时，成功的 episode 平均调用判别器**不到 1 次**，只比策略单独运行多 **13–16 秒**；和用同一个 Qwen 的 VLM 主导方法相比，耗时少约 **70%**。

## 1. 安装

| 组件 | 权重 | 用于 |
|---|---|---|
| pi0.5 | [DAVIAN-Robotics/pi05-robocasa-H50](https://huggingface.co/DAVIAN-Robotics/pi05-robocasa-H50) | `FAMILY_OVERRIDE=pi05` |
| Cosmos Policy | [nvidia/Cosmos-Policy-RoboCasa-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-RoboCasa-Predict2-2B) | `FAMILY_OVERRIDE=cosmos` |
| Qwen 判别器 | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)（[ModelScope](https://modelscope.cn/models/Qwen/Qwen3.8-27B-FP8)），用 vLLM 本地部署 | 判别器和筛子（screener） |

另外还需要：上游的 Cosmos Policy / RoboCasa 代码（Python 3.10）、pi0.5 / `lerobot`（Python 3.12）、vLLM，以及支持 EGL 的 GPU。本仓库不包含上游策略代码和模型权重；Cosmos 的策略服务（`python -m rpc.server`）来自上游 Cosmos Policy 仓库。

```bash
cp env/env.sh.example env/env.sh
$EDITOR env/env.sh        # 把 COSMOS_POLICY_ROOT、PI05_PYTHON、各权重路径、QWEN_MODEL 等指向你的安装位置
source env/env.sh
```

参考版本见 [`env/requirements-robocasa.txt`](env/requirements-robocasa.txt) 和 [`env/requirements-pi05.txt`](env/requirements-pi05.txt)。

### 部署 Qwen 判别器

判别器和筛子通过 OpenAI 兼容接口访问 vLLM 服务，地址由 `QWEN_HOST`（默认 `127.0.0.1`）和 `QWEN_PORTS`（默认 `8301,8302,8303,8304`）指定。每张卡起一个服务：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve "$QWEN_MODEL" \
  --served-model-name qwen38 --port 8301 \
  --max-model-len 131072 --max-num-seqs 64 \
  --limit-mm-per-prompt '{"image":60,"video":2}' \
  --gpu-memory-utilization 0.70 --seed 0
```

卡够的话可以在 8302–8304 端口再起几个副本，否则把 `QWEN_PORTS` 设成实际使用的端口。

## 2. 流程

```text
策略服务 ──动作──> env_service（RoboCasa + MuJoCo + EGL）
   ▲                      │ 画面 + 遥测
   └────── 判别 driver <──┘
                │
         Qwen 筛子 / 判别器
                │
          JSON 修复计划
```

Cosmos 每个策略块（chunk）执行 16 步，pi0.5 执行 25 步。设置 `SCREEN=1` 时，由轻量的筛子检查每个窗口，只把可疑窗口交给完整的判别器。

## 3. 快速开始：跑一集 pi0.5

先执行 `source env/env.sh`。

### 启动策略服务

```bash
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4 \
  "$PI05_PYTHON" -m rpc.policy_server_pi05 \
    --host 0.0.0.0 --port 8900 --checkpoint "$PI05_CHECKPOINT"
curl -s http://127.0.0.1:8900/health
```

### 启动仿真服务

```bash
MAX_RESETS=0 STEP_BUDGET_SCALE=1.8 SVC_HOST=0.0.0.0 \
  bash recovery_explore/launch_svc.sh 0 8450 http://127.0.0.1:8900 pi05 rounds
python3 recovery_explore/cli/harness.py --endpoint http://127.0.0.1:8450 meta
```

### 建队列并运行 driver

```bash
mkdir -p recovery_explore/runs_smoke
printf 'PnPCabToCounter:195:0\n' > recovery_explore/runs_smoke/episodes_pi05.txt
: > recovery_explore/runs_smoke/episodes_cosmos.txt

ON_POD=1 RUN=smoke TARGET=1 NGPU=1 FAMILY_OVERRIDE=pi05 \
SIM_HOST=127.0.0.1 SIM_MANAGED=0 STEP_BUDGET_SCALE=1.8 \
POLICY_URL=http://127.0.0.1:8900 ENGINE=qwen MODEL=qwen38 \
SCREEN=1 WINDOW=1 TEL_CHUNKS=10 COMPACT_AT=110000 \
MAX_INTERVENTIONS=8 INTERVENE_BUDGET=30 PLAN_ROUNDS=6 \
ALLOW_RESET=0 MAX_RESETS=0 CTRL_ONLY=0 SKIP_CONTROL=1 FEWSHOT=0 LEARN=0 \
  bash recovery_explore/judge_driver_v7.sh 0
```

去掉 `SKIP_CONTROL=1` 就会在同一个 episode 上依次跑对照组和实验组。两组的 `STEP_BUDGET_SCALE` 必须相同。

## 4. 全量实验

同一个 episode 可以在相同的场景、指令和步数预算下分别跑两组：

| 组别 | 说明 |
|---|---|
| **对照组（Control）** | 只有策略，不调用判别器。 |
| **实验组（Treatment）** | 策略加判别器，可按需干预。 |

论文在 `sall500` 上评测：24 个 RoboCasa 任务 × episode 编号 0–49，seed 固定为 500，共 1200 集，两种策略使用同一份列表。列表放在 [`recovery_explore/episode_sets/`](recovery_explore/episode_sets/)，每行一个 `任务:种子:编号`；`s500q96.txt` 是 96 集的子集，适合快速验证。

多开几个 **lane** 共用同一个 run 名即可并行。lane `N` 使用端口 `8450+N` 上的仿真服务；同一个 run 的所有 lane 从共享队列 `recovery_explore/runs_<RUN>/episodes_<策略>.txt` 里领取 episode，每集只会被领取一次。结果数达到 `TARGET` 后 driver 自动结束。

```bash
RUN=paper_cosmos; FAMILY=cosmos; LANES=4
mkdir -p recovery_explore/runs_$RUN
cp recovery_explore/episode_sets/sall500.txt recovery_explore/runs_$RUN/episodes_$FAMILY.txt

for N in $(seq 0 $((LANES-1))); do
  # 每个 lane 一个仿真服务（GPU = N % 显卡数）
  MAX_RESETS=0 STEP_BUDGET_SCALE=1.8 SVC_HOST=0.0.0.0 \
    bash recovery_explore/launch_svc.sh $((N % 4)) $((8450+N)) http://127.0.0.1:8800 $FAMILY rounds &
done

for N in $(seq 0 $((LANES-1))); do
  ON_POD=1 RUN=$RUN TARGET=1200 NGPU=4 FAMILY_OVERRIDE=$FAMILY \
  SIM_HOST=127.0.0.1 SIM_MANAGED=0 STEP_BUDGET_SCALE=1.8 \
  POLICY_URL=http://127.0.0.1:8800 ENGINE=qwen MODEL=qwen38 \
  SCREEN=1 WINDOW=2 TEL_CHUNKS=40 COMPACT_AT=110000 QWEN_KEEP_TURNS=20 \
  MAX_INTERVENTIONS=8 ALLOW_RESET=0 MAX_RESETS=0 FEWSHOT=0 \
    bash recovery_explore/judge_driver_v7.sh $N > recovery_explore/runs_$RUN/lane$N.out 2>&1 &
done
wait
```

跑 pi0.5 时改用 `FAMILY=pi05`、pi0.5 的策略服务（端口从 8900 起）、`WINDOW=1` 和 `TEL_CHUNKS=10`（见下表）。只跑对照组时设 `CTRL_ONLY=1`。

## 5. 参数配置

| 设置 | Cosmos | pi0.5 |
|---|---:|---:|
| `FAMILY_OVERRIDE` | `cosmos` | `pi05` |
| 每块步数 | 16 | 25 |
| `WINDOW`（判别窗口，单位：块） | 2 | 1 |
| `TEL_CHUNKS` | 40 | 10 |
| 策略服务起始端口 | 8800 | 8900 |
| 策略服务 | `python -m rpc.server`（上游 Cosmos Policy） | `python -m rpc.policy_server_pi05` |

| 参数 | 论文取值 | 作用 |
|---|---:|---|
| `STEP_BUDGET_SCALE` | `1.8` | 步数预算倍率，对照组和实验组必须相同。 |
| `SCREEN` | `1` | 开启筛子；`0` 表示每个窗口都调用判别器。 |
| `MAX_INTERVENTIONS` | `8` | 每集最多干预次数。 |
| `ALLOW_RESET` / `MAX_RESETS` | `0` / `0` | 关闭回滚，保证各集可比。 |
| `ENGINE` / `MODEL` | `qwen` / `qwen38` | 使用本地 Qwen 判别器。 |
| `COMPACT_AT` | `110000` | Qwen 对话压缩阈值（token）。 |
| `QWEN_KEEP_TURNS` | `20` | 完整保留的最近判别轮数。 |
| `FEWSHOT` | `0` | 只有提供了 `FEWSHOT_BANK` 才设为 `1`。 |

判别器的节奏另有几个开关，默认都是关闭或保守取值：

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `ASYNC_SCREEN` | `0` | `1` 表示判别器思考时策略继续执行，之后从最新完成的窗口接着判。更快，但判别器看到的是已经过去的窗口，这部分代价在 `results.jsonl` 里记为 `stale_steps` 和 `stale_ivs`。`2` 还会把策略限制在 `ASYNC_K`+1 个窗口之内。 |
| `MISS_GRACE` | `2` | 策略抓空之后，先给它自己重试的窗口数，之后判别器才可以接管。 |
| `FIRST_IV_FRAC` | `35` | 第一次干预要等到策略至少抓空过一次，或者已经用掉这个百分比的步数预算。`0` 表示不设这个门槛。 |
| `MONITOR_AFTER_BUDGET` | `1` | 干预次数用完后继续调用判别器，只做记录。`0` 表示不再调用。 |

论文结果都是同步模式（`ASYNC_SCREEN=0`）。改动 `STEP_BUDGET_SCALE`、`WINDOW` 或策略种类都相当于换了实验；比较不同 run 时应按任务/种子/编号逐集配对，而不只看总成功率。

## 6. 输出

结果写在 `recovery_explore/runs_<RUN>_<策略>/` 下：

```text
results.jsonl             每集一行（control_success、treatment_success、interventions、judge_calls 等）
lane<N>/driver.log        各 lane 的进度日志
lane<N>/jd-.../w<窗口>/    给判别器看的画面
```

快速汇总：

```bash
python3 - recovery_explore/runs_<RUN>_<策略>/results.jsonl <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
for arm in ("control_success", "treatment_success"):
    vals = [bool(r[arm]) for r in rows if r.get(arm) is not None]
    if vals: print(f"{arm}: {sum(vals)}/{len(vals)} = {100*sum(vals)/len(vals):.1f}%")
EOF
```

## 7. 目录结构

```text
recovery_explore/     判别 driver、RoboCasa 服务、判别用命令行工具、简报、经验库、评测集
rpc/                  pi0.5 策略服务、RPC 协议、RoboCasa rollout 工具
cf_bench/             快照、重放和渲染工具
env/                  环境模板和版本锁定
```

## 📜 许可证

本项目以 [MIT 许可证](LICENSE) 发布。`recovery_explore/` 中有部分文件改编自 RPent，仍遵循 Apache License 2.0，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 🙏 致谢

本工作基于 [RoboCasa](https://robocasa.ai)、[Cosmos Policy](https://huggingface.co/nvidia/Cosmos-Policy-RoboCasa-Predict2-2B)、[pi0.5](https://huggingface.co/DAVIAN-Robotics/pi05-robocasa-H50)、[vLLM](https://github.com/vllm-project/vllm) 和 RPent，感谢这些工作的作者开源。

## 📖 引用

```bibtex
@article{li2026spotter,
  title   = {Spotter: Let the Embodied Model Lead, and the VLM Reflect for It},
  author  = {Li, Long and Zhao, Qichao and Yang, Yue and Xu, Fan and Wang, Zhe and Liew, Alan Wee-Chung and Qu, Chao and Shen, Heng Tao and Pan, Shirui},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```
