<p align="center">
  <img src="assets/spotter_logo.png" width="160" alt="Spotter logo">
</p>

<h1 align="center">Spotter：让具身模型主导，让视觉语言模型为它反思</h1>

<p align="center">如果我们的项目对你有帮助，欢迎在 GitHub 上点一个 star ⭐ 支持我们</p>

<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/README-English-111111?style=for-the-badge" alt="English"></a>
  <a href="./README_zh.md"><img src="https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836?style=for-the-badge" alt="中文"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.36808"><img src="https://img.shields.io/badge/arXiv-2609.36808-b31b1b?style=for-the-badge" alt="arXiv"></a>
  <a href="https://zqc3117.github.io/Spotter/"><img src="https://img.shields.io/badge/Project_Page-Spotter-2ea44f?style=for-the-badge" alt="Project page"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://robocasa.ai"><img src="https://img.shields.io/badge/Simulator-RoboCasa-4c8bf5?style=for-the-badge" alt="RoboCasa"></a>
</p>

| 📄 论文 | 🏠 项目主页 | 🤖 模型权重 | 🌐 仿真器 | 🚀 上手 |
|---|---|---|---|---|
| [arXiv](https://arxiv.org/abs/2609.36808) | [zqc3117.github.io/Spotter](https://zqc3117.github.io/Spotter/) | [pi0.5](https://huggingface.co/DAVIAN-Robotics/pi05-robocasa-H50) · [Cosmos Policy](https://huggingface.co/nvidia/Cosmos-Policy-RoboCasa-Predict2-2B) | [RoboCasa](https://robocasa.ai) | [安装](#-环境配置) · [快速开始](#-用发布的权重评测) |

**Spotter** 在冻结的具身模型（System 1）和 VLM（System 2）之间加入了一个轻量的介入机制。
它不让 VLM 规划每一步，而是让具身模型主导并持续执行；本地筛子并行关注执行进展，只有确认出错时，VLM 才介入确认并纠正。
交给 VLM 的任务因此简单得多，整个框架**用本地部署的开源 Qwen、不借助任何特权信息**就能运行。

<p align="center">
  <img src="assets/teaser.png" width="100%" alt="Spotter 总览：具身模型逐段执行，快速 VLM 并行筛查每一段，前沿 VLM 只在确认出错时判断并修复；右侧为成功率和单集耗时。">
</p>

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

## 📑 目录

- [文件结构](#-文件结构)
- [环境配置](#-环境配置)
- [模型准备](#-模型准备)
- [仿真资源与评测集](#-仿真资源与评测集)
- [用发布的权重评测](#-用发布的权重评测)
- [复现论文结果](#-复现论文结果)
- [重建经验库与示例库](#-重建经验库与示例库)
- [接入你自己的具身模型](#-接入你自己的具身模型)
- [输出](#-输出) · [配置参考](#-配置参考)
- [致谢](#-致谢) · [许可证](#-许可证) · [引用](#-引用)

## 📂 文件结构

```text
Spotter/
├── spotter.sh                        # 统一入口：部署 Qwen / 策略服务、运行、汇总结果
├── recovery_explore/                 # Spotter 本体
│   ├── judge_driver_v7.sh            # 单个 lane 的监督循环：筛查 -> 判别 -> 修复 -> 验证
│   ├── env_service.py                # RoboCasa 仿真服务（RPC），由 launch_svc.sh 启动
│   ├── primitives.py                 # 判别器组合修复计划所用的运动原语
│   ├── JUDGE_BRIEF_V7.md             # 判别器提示词；BRIEF_PNP.md / BRIEF_MECH.md 是分任务类型的补充
│   ├── cli/                          # 筛子、判别通道（Qwen / GPT）、harness、few-shot 工具
│   ├── memory_bank/                  # 冻结的 77 条通用经验
│   ├── episode_sets/                 # sall500.txt（论文，1200 集）· s500q96.txt（96 集，调试用）
│   ├── fewshot_bank/                 # （需下载）1-shot 设置用的示例
│   └── runs_<RUN>_<策略>/            # （运行时生成）results.jsonl、各 lane 日志、给判别器看的画面
├── rpc/                              # π0.5 策略服务、RPC 协议、RoboCasa rollout 与采集脚本
├── cf_bench/                         # 纠错能力实验用的快照与重放工具
├── env/                              # env.sh.example、requirements-robocasa.txt、requirements-pi05.txt
├── assets/                           # logo
└── LICENSE · THIRD_PARTY_NOTICES.md · LICENSES/
```

## 🔧 环境配置

Spotter 用到三个 Python 环境：

| 环境 | Python | 运行什么 | 安装 |
|---|---|---|---|
| `robocasa` | 3.10 | 仿真服务、判别 driver、Cosmos 策略服务 | 按 [Cosmos Policy](https://github.com/NVlabs/cosmos-policy) 的 RoboCasa 说明安装，再装 `env/requirements-robocasa.txt` |
| `pi05` | 3.12 | π0.5 策略服务（LeRobot） | `env/requirements-pi05.txt` |
| `vllm` | 3.10+ | Qwen 筛子和判别器 | `pip install vllm` |

```bash
# 1) RoboCasa + Cosmos Policy：按 https://github.com/NVlabs/cosmos-policy 的 RoboCasa 说明安装，然后
pip install -r env/requirements-robocasa.txt

# 2) π0.5
python3.12 -m venv .venvs/pi05 && .venvs/pi05/bin/pip install -r env/requirements-pi05.txt

# 3) Qwen 用的 vLLM
python3 -m venv .venvs/vllm && .venvs/vllm/bin/pip install vllm

# 4) 让 Spotter 找到上面的安装位置
cp env/env.sh.example env/env.sh && $EDITOR env/env.sh
```

> **注意**：无显示器的 MuJoCo 渲染需要支持 EGL 的 NVIDIA GPU（`MUJOCO_GL=egl`，已在 `env/env.sh` 中设置）。

## 📦 模型准备

**1）下载权重。**

```bash
huggingface-cli download DAVIAN-Robotics/pi05-robocasa-H50         --local-dir checkpoints/pi05-robocasa-H50
huggingface-cli download nvidia/Cosmos-Policy-RoboCasa-Predict2-2B --local-dir checkpoints/Cosmos-Policy-RoboCasa-Predict2-2B
huggingface-cli download Qwen/Qwen3.8-27B-FP8                      --local-dir checkpoints/Qwen3.8-27B-FP8
```

国内用户也可以从 [ModelScope](https://modelscope.cn/models/Qwen/Qwen3.8-27B-FP8) 下载 Qwen。

**2）下载后的目录。**

```text
checkpoints/
├── pi05-robocasa-H50/                        # model.safetensors、config.json、policy_*processor*、tokenizer/
├── Cosmos-Policy-RoboCasa-Predict2-2B/       # Cosmos-Policy-RoboCasa-Predict2-2B.pt、robocasa_t5_embeddings.pkl 等
└── Qwen3.8-27B-FP8/
```

**3）在 `env/env.sh` 中设置：**

```bash
export PI05_CHECKPOINT=$PWD/checkpoints/pi05-robocasa-H50
export COSMOS_CHECKPOINT=$PWD/checkpoints/Cosmos-Policy-RoboCasa-Predict2-2B/Cosmos-Policy-RoboCasa-Predict2-2B.pt
export QWEN_MODEL=$PWD/checkpoints/Qwen3.8-27B-FP8
export VLLM_VENV=$PWD/.venvs/vllm
```

**4）1-shot 设置用的示例库（可选，151 MB）。**

```bash
curl -L https://github.com/zqc3117/Spotter/releases/download/v1.0/fewshot_bank.tar.gz | tar -xz -C recovery_explore
```

> **GPT 判别器（可选）**：设置 `ENGINE=api`，并配好 `OPENAI_RESPONSES_URL` 和 `OPENAI_API_TOKEN_FILE`；
> 判别器换成 GPT-6 Astra，筛子仍然是本地 Qwen。

## 🧭 仿真资源与评测集

**RoboCasa 资源**随上面 Cosmos Policy 的 RoboCasa 安装步骤一起装好。

**评测集**随仓库提供，每行一个 `任务:种子:编号`：

| 文件 | 集数 | 用途 |
|---|---:|---|
| `episode_sets/sall500.txt` | 1200 | 论文评测：24 个任务 × 编号 0–49，seed 500 |
| `episode_sets/s500q96.txt` | 96 | 调试用子集，每个任务 4 集 |

经验和示例都是在 seed 195 上学到的，和测试用的 seed 不重叠。
**RoboTwin 2.0 和真机实验**不包含在本次发布中。

## 🚀 用发布的权重评测

每个服务各开一个终端，先执行 `source env/env.sh`。

**1）部署 Qwen 筛子和判别器**（每张卡一个副本，端口 8301、8302……）。

```bash
bash spotter.sh judge --gpus 0,1,2,3
curl -s http://127.0.0.1:8301/v1/models        # 检查
```

卡少时少写几张（如 `--gpus 0`），并在运行前 `export QWEN_PORTS=8301`。

**2）部署具身模型**：π0.5 用 8900 端口，Cosmos Policy 用 8800 端口。

```bash
bash spotter.sh policy pi05 --gpu 4             # 或：bash spotter.sh policy cosmos --gpu 4
curl -s http://127.0.0.1:8900/health           # 检查
```

**3）冒烟测试：跑一集。** 仿真服务由 driver 自动启动。

```bash
bash spotter.sh run pi05                        # 跑一集：同一场景下对照组和 Spotter 各跑一次
```

**4）在 `sall500` 上完整评测。**

```bash
bash spotter.sh run pi05 sall500 --lanes 8
bash spotter.sh summary
```

| 选项 | 含义 |
|---|---|
| `SET` | `smoke`（1 集，默认）、`s500q96`、`sall500`，或每行一个 `任务:种子:编号` 的文件 |
| `--lanes N` | 并行跑的集数；lane `N` 在端口 `8450+N` 上运行自己的仿真服务 |
| `--sim-gpus N` | 仿真服务共用的卡，即 `0..N-1` 号卡（默认全部） |
| `--arm` | `both`（每集对照组和 Spotter 都跑，默认）、`treat`（只跑 Spotter）、`ctrl`（只跑策略） |
| `--one-shot` | 给判别器加一个示例（需要示例库） |
| `--run NAME` | run 名称（默认 `<策略>_<评测集>`）；同名重跑会接着上次继续 |
| `--dry-run` | 只打印要执行的命令，不真正运行 |

同一个 episode 的两组使用相同的场景、指令和步数预算：

| 组别 | 说明 |
|---|---|
| **对照组（Control）** | 只有策略，不调用判别器。 |
| **实验组（Treatment）** | 策略加判别器，可按需干预。 |

<details>
<summary>直接调用 driver</summary>

`spotter.sh run` 会为每个 lane 启动一个 driver。上面跑一集 pi0.5 的命令等价于：

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

driver 会自己在 `8450+lane` 端口启动和重启仿真服务。同一个 run 的所有 lane 从同一个队列领取 episode，结果数达到 `TARGET` 后各 lane 自动结束。
</details>

## 📊 复现论文结果

论文 Table 1（RoboCasa，1200 集）中 Spotter 的各行，都是在上面完整评测的基础上加以下设置；环境变量在 `spotter.sh run` 之前 export。

| Table 1 中的行 | 设置 |
|---|---|
| 具身模型，1.0× / 1.8× 步数上限 | `--arm ctrl`，配合 `STEP_BUDGET_SCALE=1.0` / `1.8` |
| Spotter (Qwen)，full context | `spotter.sh` 默认值 |
| Spotter (Qwen)，short context | `TEL_CHUNKS=3 QWEN_KEEP_TURNS=3 COMPACT_AT=22000 MAX_INTERVENTIONS=3 SCREEN_CONFIRM2=0`（π0.5 另加 `WINDOW=2`） |
| Spotter (Qwen)，1-shot | `--one-shot` |
| Spotter (GPT)，有 / 无筛子 | `ENGINE=api`，配合 `SCREEN=1` / `SCREEN=0` |
| Spotter (GPT)，1-shot | `ENGINE=api` 加 `--one-shot` |
| Harness VLA | 用其自身代码运行：[RLinf/RPent](https://github.com/RLinf/RPent) |

Table 1 中"固定重试"的两行不包含在本次发布中。

**监督模式**：`spotter.sh` 运行的是同步监督循环（`ASYNC_SCREEN=0`）。如需 3.2 节的有界异步模式，设置 `ASYNC_SCREEN=2 ASYNC_K=2`：策略最多领先上一次检查两个监督窗口（Cosmos Policy 为 4 个 chunk，π0.5 为 2 个）。

**耗时**：`results.jsonl` 记录了每组的总耗时，以及判别器和筛子的调用耗时（见[输出](#-输出)）；本次发布不记录 token 数。

**3.1 节**：`rpc/run_remote_robocasa_cf_branch_collect.py` 采集失败后和正常状态下的候选动作，`cf_bench/` 负责重放这些候选并运行语言反馈实验。测试时打分器（Consistency-Consensus、GeoBoN、RCS）不包含在内。

## 🧠 重建经验库与示例库

仓库自带的经验库（`recovery_explore/memory_bank/global/` 中的 77 条通用经验）和示例库都是冻结的，论文所有结果都直接使用它们；只有在需要重建或覆盖新任务时才需要这一节。

**1）运行学习用的 episode**：只在 seed 195 上运行，不能用测试 seed。只有这一步允许判别器重置并重试；每集的经验会以草稿形式写到 `memory_bank/inbox/<cell>/`。

```bash
ENGINE=api LEARN=1 ALLOW_RESET=1 MAX_RESETS=5 bash spotter.sh run cosmos <seed 195 的 episode 文件> --run learn_cosmos
```

**2）审阅草稿**，把其中通用的经验移到 `memory_bank/global/`，格式见 `recovery_explore/memory_bank/README.md`。去掉所有特权信息和只适用于特定任务的内容。

**3）构建示例库**，从学习 run 中挖掘 1-shot 设置所用的示例：

```bash
python3 recovery_explore/cli/fewshot_build.py --family cosmos --seed 195 --out recovery_explore/fewshot_bank
```

## 🔌 接入你自己的具身模型

Spotter 从不训练策略，所以接入新的具身模型只需要部署服务并注册：

1. **用相同的 RPC 接口提供服务**，参考 `rpc/policy_server_pi05.py`：`GET /health`，以及 `POST /v1/infer`，接收 `InferenceRequest`（任务描述、种子、观测图像和机器人状态），返回包含动作块的 `InferenceResponse`（见 `rpc/protocol.py`）。
2. **注册策略类型**，照着 `cosmos` 和 `pi05` 添加：动作维度和每块步数在 `recovery_explore/cosmos_call.py` 和 `env_service.py`，服务启动在 `launch_svc.sh`，端口和默认值在 `spotter.sh`。
3. **选择监督粒度**：`WINDOW`（每次筛查覆盖的 chunk 数）和 `TEL_CHUNKS`（给判别器看的遥测行数）。
4. **运行** `bash spotter.sh run <你的策略类型> ...`。

## 📈 输出

```text
recovery_explore/runs_<RUN>_<策略>/
├── results.jsonl          # 每集一行
├── lane<N>/driver.log     # 各 lane 的进度日志
└── lane<N>/jd-.../w<k>/   # 第 k 个窗口给判别器看的画面
```

| 字段 | 含义 |
|---|---|
| `task`、`seed`、`episode` | 对应的 episode |
| `control_success` / `treatment_success` | 策略单独运行 / 加上 Spotter 是否成功 |
| `interventions`、`judge_calls`、`screened_windows` | 执行的修复次数、判别器调用次数、被筛子放行的窗口数 |
| `ctrl_s`、`treat_s` | 对照组和 Spotter 组的总耗时（秒；该组被跳过时为 `-1`） |
| `model_s`、`screen_s` | 判别器调用和筛子调用的累计耗时（秒） |
| `stale_steps`、`stale_ivs` | 仅异步模式：基于过时窗口做出的步数和修复次数 |

```bash
bash spotter.sh summary            # 所有 run
bash spotter.sh summary pi05_sall500
```

## 🧩 配置参考

| 设置 | Cosmos | pi0.5 |
|---|---:|---:|
| `FAMILY_OVERRIDE` | `cosmos` | `pi05` |
| 每块步数 | 16 | 25 |
| `WINDOW`（判别窗口，单位：块） | 2 | 1 |
| `TEL_CHUNKS` | 40 | 10 |
| 策略服务端口 | 8800 | 8900 |
| 策略服务 | `python -m rpc.server`（上游 Cosmos Policy） | `python -m rpc.policy_server_pi05` |

| 参数 | 论文取值 | 作用 |
|---|---:|---|
| `STEP_BUDGET_SCALE` | `1.8` | 步数预算倍率，对照组和实验组必须相同。 |
| `SCREEN` | `1` | 开启筛子；`0` 表示每个窗口都调用判别器。 |
| `SCREEN_CONFIRM2` | `1` | 连续两次标记才调用判别器；`0` 表示一次就调用。 |
| `MAX_INTERVENTIONS` | `8` | 每集最多干预次数。 |
| `ALLOW_RESET` / `MAX_RESETS` | `0` / `0` | 关闭回滚，保证各集可比。 |
| `ENGINE` / `MODEL` | `qwen` / `qwen38` | 使用本地 Qwen 判别器；`ENGINE=api` 换成 GPT-6 Astra。 |
| `COMPACT_AT` | `110000` | Qwen 对话压缩阈值（token）。 |
| `QWEN_KEEP_TURNS` | `20` | 完整保留的最近判别轮数。 |
| `FEWSHOT` | `0` | `1` 表示加一个示例（即 `--one-shot`），需要示例库。 |

Qwen 只在介入的修复回合里思考，每次最多 `QWEN_THINK_BUDGET`（默认 `2000`）个 token；筛子和窗口判别都不思考，`QWEN_REPAIR_THINK=0` 则全部不思考，与论文设置一致。

判别器的节奏另有几个开关，默认都是关闭或保守取值：

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `ASYNC_SCREEN` | `0` | `1` 表示判别器思考时策略继续执行，之后从最新完成的窗口接着判。更快，但判别器看到的是已经过去的窗口，这部分代价在 `results.jsonl` 里记为 `stale_steps` 和 `stale_ivs`。`2` 还会把策略限制在上一次检查之后 `ASYNC_K` 个窗口之内。 |
| `MISS_GRACE` | `2` | 策略抓空之后，先给它自己重试的窗口数，之后判别器才可以接管。 |
| `FIRST_IV_FRAC` | `35` | 第一次干预要等到策略至少抓空过一次，或者已经用掉这个百分比的步数预算。`0` 表示不设这个门槛。 |
| `MONITOR_AFTER_BUDGET` | `1` | 干预次数用完后继续调用判别器，只做记录。`0` 表示不再调用。 |

改动 `STEP_BUDGET_SCALE`、`WINDOW` 或策略种类都相当于换了实验；比较不同 run 时应按任务/种子/编号逐集配对，而不只看总成功率。

## 🙏 致谢

- [RoboCasa](https://robocasa.ai)：我们评测所用的仿真基准和场景资源。
- [Cosmos Policy](https://github.com/NVlabs/cosmos-policy)：发布的 RoboCasa 权重及其策略服务。
- [π0.5](https://huggingface.co/DAVIAN-Robotics/pi05-robocasa-H50) 与 [LeRobot](https://github.com/huggingface/lerobot)：RoboCasa 上的 π0.5 权重以及加载它的库。
- [Qwen](https://huggingface.co/Qwen) 与 [vLLM](https://github.com/vllm-project/vllm)：本地部署的筛子和判别器。
- [RPent](https://github.com/RLinf/RPent)：`recovery_explore/` 中部分代码改编自它（Apache-2.0，见 `THIRD_PARTY_NOTICES.md`）。

## 📜 许可证

本项目以 [MIT 许可证](LICENSE) 发布。`recovery_explore/` 中有部分文件改编自 RPent，仍遵循 Apache License 2.0，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 📖 引用

```bibtex
@article{li2026spotter,
  title   = {Spotter: Let the Embodied Model Lead, and the VLM Reflect for It},
  author  = {Li, Long and Zhao, Qichao and Yang, Yue and Xu, Fan and Wang, Zhe and Liew, Alan Wee-Chung and Qu, Chao and Shen, Heng Tao and Pan, Shirui},
  journal = {arXiv preprint arXiv:2609.36808},
  year    = {2026}
}
```
