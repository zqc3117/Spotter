#!/usr/bin/env bash
# spotter.sh -- one entry point for serving the models and running Spotter on RoboCasa and RoboTwin 2.0.
#
#   bash spotter.sh judge  [--gpus 0,1,2,3]                      serve the Qwen judge/screener with vLLM
#   bash spotter.sh policy <pi05|cosmos|robotwin> [--gpu 0]      serve the embodied model
#   bash spotter.sh run    <pi05|cosmos|robotwin> [SET] [options] run episodes with the paper settings
#   bash spotter.sh summary [RUN]                                 success rates of finished runs
#
# SET is smoke (1 episode, default), s500q96 (96), sall500 (1,200, the paper's RoboCasa set), rt50x5q (250) /
# rt50x10n (500) for RoboTwin, or a file with one TASK:SEED:EPISODE per line. Run options:
#   --lanes N        parallel lanes (default 1); lane N uses simulation port 8450+N
#   --sim-gpus N     GPUs the simulation services are spread over, GPU 0..N-1 (default: all visible)
#   --arm A          both (control + treatment on each episode, default) | treat | ctrl
#   --one-shot       add one worked example to the judge (needs the example bank, see README)
#   --run NAME       run name (default <family>_<set>); rerunning a name resumes it
#   --dry-run        print the commands instead of running them
# Every paper setting can still be overridden through the environment (see README, "Configuration").
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
if [ -f env/env.sh ]; then set +u; source env/env.sh; set -u; fi

die() { echo "spotter.sh: $*" >&2; exit 1; }
usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

policy_port() { case "$1" in cosmos) echo 8800 ;; pi05) echo 8900 ;; robotwin) echo 9100 ;; *) die "family must be pi05, cosmos or robotwin, got '$1'" ;; esac; }
# The RoboCasa servers answer GET /health; the RoboTwin server is openpi's websocket server, which answers GET /healthz.
policy_ready() { case "$1" in robotwin) curl -sf "$2/healthz" >/dev/null ;; *) curl -sf "$2/health" >/dev/null ;; esac; }

# ---------------------------------------------------------------- judge
cmd_judge() {
  local gpus="0,1,2,3"
  while [ $# -gt 0 ]; do case "$1" in --gpus) gpus=$2; shift 2 ;; -h|--help) usage ;; *) die "unknown option $1" ;; esac; done
  [ -n "${QWEN_MODEL:-}" ] || die "set QWEN_MODEL in env/env.sh (path or HF id of Qwen3.8-27B-FP8)"
  local vllm="${VLLM_VENV:+$VLLM_VENV/bin/}vllm" port=8300 ports="" g
  mkdir -p logs
  for g in ${gpus//,/ }; do
    port=$((port + 1)); ports="${ports:+$ports,}$port"
    echo "starting Qwen replica on GPU $g -> port $port (log: logs/qwen_$port.log)"
    CUDA_VISIBLE_DEVICES=$g "$vllm" serve "$QWEN_MODEL" \
      --served-model-name qwen38 --port "$port" \
      --max-model-len 131072 --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image":60,"video":2}' \
      --gpu-memory-utilization 0.70 --seed 0 > "logs/qwen_$port.log" 2>&1 &
  done
  echo "waiting for the replicas to come up (this can take a few minutes) ..."
  local p
  for p in ${ports//,/ }; do
    until curl -sf "http://127.0.0.1:$p/v1/models" >/dev/null; do sleep 10; done
    echo "  port $p ready"
  done
  echo "Qwen is up on ports $ports. Keep this terminal open; Ctrl-C stops all replicas."
  [ "$ports" = "8301,8302,8303,8304" ] || echo "Before 'spotter.sh run', export QWEN_PORTS=$ports"
  trap 'kill $(jobs -p) 2>/dev/null' INT TERM
  wait
}

# ---------------------------------------------------------------- policy
cmd_policy() {
  local fam=${1:-}; [ -n "$fam" ] || die "usage: spotter.sh policy <pi05|cosmos|robotwin> [--gpu N]"; shift
  local gpu=0 port; port=$(policy_port "$fam")
  while [ $# -gt 0 ]; do case "$1" in --gpu) gpu=$2; shift 2 ;; --port) port=$2; shift 2 ;; *) die "unknown option $1" ;; esac; done
  export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
  if [ "$fam" = robotwin ]; then
    # RoboTwin 2.0 pi0.5 (motus-robotics/pi0.5_robotwin2 by default), wrapped in RoboTwin's own
    # policy/pi05 adapter and served with openpi's websocket protocol (recovery_explore/robotwin_policy_server.py).
    # It must run from the RoboTwin tree (pi_model.py uses relative checkpoint paths), see README "RoboTwin 2.0".
    local rt_root=${ROBOTWIN_ROOT:-${ROBOTWIN_RUNTIME:+$ROBOTWIN_RUNTIME/worktree/third_party/RoboTwin}}
    [ -f "${rt_root:-}/envs/_base_task.py" ] || die "ROBOTWIN_ROOT (or ROBOTWIN_RUNTIME) does not point at a RoboTwin tree; set it in env/env.sh"
    [ -x "${ROBOTWIN_POLICY_PYTHON:-}" ] || die "ROBOTWIN_POLICY_PYTHON is not an executable python (the venv with RoboTwin's policy/pi05 + openpi); set it in env/env.sh"
    echo "serving $fam on GPU $gpu, port $port (check: curl http://127.0.0.1:$port/healthz)"
    cd "$rt_root"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=${ROBOTWIN_XLA_MEM_FRACTION:-0.05} PYTHONUNBUFFERED=1
    export PYTHONPATH="$rt_root/policy/pi05/src:$rt_root/policy/pi05/packages/openpi-client/src:$rt_root:$REPO"
    exec "$ROBOTWIN_POLICY_PYTHON" "$REPO/recovery_explore/robotwin_policy_server.py" --host 0.0.0.0 --port "$port" \
      --train-config "${ROBOTWIN_TRAIN_CONFIG:-pi05_robotwin2_clean_randomized}" \
      --model-name "${ROBOTWIN_MODEL_NAME:-robotwin2}" --checkpoint-id "${ROBOTWIN_CKPT_ID:-40000}" \
      --pi0-step "${ROBOTWIN_PI0_STEP:-16}" ${ROBOTWIN_SERVER_ARGS:---config-source registered --prompt-prefix-from-config}
  fi
  echo "serving $fam on GPU $gpu, port $port (check: curl http://127.0.0.1:$port/health)"
  if [ "$fam" = pi05 ]; then
    [ -x "${PI05_PYTHON:-}" ] || die "PI05_PYTHON is not an executable python; set it in env/env.sh"
    exec "$PI05_PYTHON" -m rpc.policy_server_pi05 --host 0.0.0.0 --port "$port" --checkpoint "${PI05_CHECKPOINT:?set PI05_CHECKPOINT}"
  else
    [ -d "${COSMOS_POLICY_ROOT:-}" ] || die "COSMOS_POLICY_ROOT is not a directory; set it in env/env.sh"
    # the Cosmos policy server ships with the upstream Cosmos Policy repository
    source "${COSMOS_COMMON_ENV:-$COSMOS_POLICY_ROOT/scripts/common_env.sh}" >/dev/null 2>&1 || true
    export PYTHONPATH="$COSMOS_POLICY_ROOT:${COSMOS_POLICY_SRC:-}:${ROBOCASA_SRC:-}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd "$COSMOS_POLICY_ROOT"
    exec "${COSMOS_POLICY_ENV:?set COSMOS_POLICY_ENV}/bin/python" -m rpc.server --host 0.0.0.0 --port "$port" \
      --checkpoint "${COSMOS_CHECKPOINT:?set COSMOS_CHECKPOINT}" --max-batch-samples 12 --batch-wait-seconds 0.01
  fi
}

# ---------------------------------------------------------------- run
cmd_run() {
  local fam=${1:-}; [ -n "$fam" ] || die "usage: spotter.sh run <pi05|cosmos|robotwin> [SET] [options]"; shift
  local set=smoke
  if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then set=$1; shift; fi
  local lanes=1 simgpus="" arm=both run="" dry=0 oneshot=0
  while [ $# -gt 0 ]; do case "$1" in
    --lanes) lanes=$2; shift 2 ;; --sim-gpus) simgpus=$2; shift 2 ;; --arm) arm=$2; shift 2 ;;
    --run) run=$2; shift 2 ;; --one-shot) oneshot=1; shift ;; --dry-run) dry=1; shift ;; -h|--help) usage ;; *) die "unknown option $1" ;; esac; done

  local port; port=$(policy_port "$fam")
  local queue_src
  case "$set" in
    smoke) queue_src="" ;;
    s500q96|sall500|rt50x5q|rt50x10n|rt_ctlcheck12) queue_src="recovery_explore/episode_sets/$set.txt" ;;
    *) [ -f "$set" ] || die "unknown set '$set' (smoke | s500q96 | sall500 | rt50x5q | rt50x10n | path to a TASK:SEED:EPISODE file)"; queue_src="$set" ;;
  esac
  [ -n "$run" ] || run="${fam}_$(basename "${set%.txt}")"
  local ctrl_only=0 skip_ctrl=0
  case "$arm" in both) ;; treat) skip_ctrl=1 ;; ctrl) ctrl_only=1 ;; *) die "--arm must be both, treat or ctrl" ;; esac
  if [ -z "$simgpus" ]; then simgpus=$( (nvidia-smi -L 2>/dev/null || true) | wc -l | tr -d ' '); [ "${simgpus:-0}" -gt 0 ] || simgpus=1; fi

  # paper settings (Table 3, "full context"); anything already exported in the environment wins
  local window=2 tel=40 keep=20 compact=110000 maxint=8
  [ "$fam" = pi05 ] && { window=1; tel=10; }
  # RoboTwin 2.0 (dual-arm tabletop): the settings of the RoboTwin runs -- 2 chunks per window, the last 10
  # chunks of action history in the telemetry, 5 interventions per episode, a shorter Qwen history (10 turns,
  # compaction at 40k tokens: the judge prompt carries three cameras per chunk), its own lesson bank.
  [ "$fam" = robotwin ] && { window=2; tel=10; keep=10; compact=40000; maxint=5; }
  local -a envs=(
    ON_POD=1 RUN="$run" NGPU="$simgpus" FAMILY_OVERRIDE="$fam"
    SIM_HOST=127.0.0.1 SIM_MANAGED=0 POLICY_URL="${POLICY_URL:-http://127.0.0.1:$port}"
    ROBOCASA_BIN="${ROBOCASA_BIN:-${COSMOS_POLICY_ENV:+$COSMOS_POLICY_ENV/bin}}"
    ENGINE="${ENGINE:-qwen}" MODEL="${MODEL:-qwen38}" SCREEN="${SCREEN:-1}" SCREENER_HOST="${SCREENER_HOST:-127.0.0.1}"
    QWEN_PORTS="${QWEN_PORTS:-8301,8302,8303,8304}" QWEN_KEEP_TURNS="${QWEN_KEEP_TURNS:-$keep}"
    STEP_BUDGET_SCALE="${STEP_BUDGET_SCALE:-1.8}" WINDOW="${WINDOW:-$window}" TEL_CHUNKS="${TEL_CHUNKS:-$tel}"
    COMPACT_AT="${COMPACT_AT:-$compact}" MAX_INTERVENTIONS="${MAX_INTERVENTIONS:-$maxint}"
    ALLOW_RESET="${ALLOW_RESET:-0}" MAX_RESETS="${MAX_RESETS:-0}" LEARN="${LEARN:-0}"
    FEWSHOT="$([ $oneshot = 1 ] && echo 1 || echo "${FEWSHOT:-0}")" FEWSHOT_MAX_IMAGES="${FEWSHOT_MAX_IMAGES:-12}"
    CTRL_ONLY=$ctrl_only SKIP_CONTROL=$skip_ctrl
  )
  if [ "$fam" = robotwin ]; then
    [ -n "${ROBOTWIN_RUNTIME:-}" ] || [ "$dry" = 1 ] || die "set ROBOTWIN_RUNTIME in env/env.sh (the RoboTwin portable runtime: .venv + worktree/third_party/RoboTwin, see README)"
    envs+=(
      RT_TEL_CHUNKS="${RT_TEL_CHUNKS:-$tel}" RT_MODEL="${RT_MODEL:-motus}"
      ROBOTWIN_TASK_CONFIG="${ROBOTWIN_TASK_CONFIG:-demo_randomized}"
      ROBOTWIN_SEED_CACHE="${ROBOTWIN_SEED_CACHE:-$REPO/recovery_explore/episode_sets/robotwin_seed_cache/demo_randomized_seed0_n100.json}"
      RT_MEMORY="${RT_MEMORY:-1}" RT_MEMORY_BANK="${RT_MEMORY_BANK:-memory_bank_robotwin_noreset}"
      JUDGE_DEADLINE="${JUDGE_DEADLINE:-1200}" ACT_DEADLINE="${ACT_DEADLINE:-2400}"
    )
    [ -z "${ROBOTWIN_RUNTIME:-}" ] || envs+=(ROBOTWIN_RUNTIME="$ROBOTWIN_RUNTIME")
    [ -z "${ROBOTWIN_ROOT:-}" ] || envs+=(ROBOTWIN_ROOT="$ROBOTWIN_ROOT")
    [ -z "${ROBOTWIN_PY:-}" ] || envs+=(ROBOTWIN_PY="$ROBOTWIN_PY")
  fi

  if [ $oneshot = 1 ] && [ $dry = 0 ] && [ ! -d "${FEWSHOT_BANK:-recovery_explore/fewshot_bank}" ]; then
    die "--one-shot needs the example bank; download it first (README: Model Preparation)"
  fi

  local qdir="recovery_explore/runs_$run" out="recovery_explore/runs_${run}_$fam"
  local n
  if [ -n "$queue_src" ]; then n=$(grep -v '^#' "$queue_src" | grep -c .); else n=1; fi
  echo "run '$run': $fam, set $set ($n episode(s)), $lanes lane(s), sim on $simgpus GPU(s), arm=$arm"
  echo "results -> $out/results.jsonl"

  if [ "$dry" = 1 ]; then
    echo; echo "# queue"; echo "mkdir -p $qdir && cp ${queue_src:-<smoke episode>} $qdir/episodes_$fam.txt"
    local l; for l in $(seq 0 $((lanes - 1))); do
      echo; echo "# lane $l (simulation port $((8450 + l)), GPU $((l % simgpus)))"
      echo "env ${envs[*]} TARGET=$n bash recovery_explore/judge_driver_v7.sh $l"
    done
    return 0
  fi

  local purl="${POLICY_URL:-http://127.0.0.1:$port}"
  policy_ready "$fam" "$purl" || die "no policy server at $purl -- start it first: bash spotter.sh policy $fam"
  if [ "${ENGINE:-qwen}" = qwen ]; then
    local qp ok=0
    for qp in $(echo "${QWEN_PORTS:-8301,8302,8303,8304}" | tr ',' ' '); do
      curl -sf "http://${SCREENER_HOST:-127.0.0.1}:$qp/v1/models" >/dev/null && ok=$((ok + 1))
    done
    [ "$ok" -gt 0 ] || die "no Qwen replica answers on ${QWEN_PORTS:-8301,8302,8303,8304} -- start it first: bash spotter.sh judge"
  fi

  mkdir -p "$qdir" "$out"
  if [ -n "$queue_src" ]; then grep -v '^#' "$queue_src" | grep . > "$qdir/episodes_$fam.txt"   # episode files may carry a # header
  elif [ "$fam" = robotwin ]; then printf 'adjust_bottle:0:35\n' > "$qdir/episodes_$fam.txt"
  else printf 'PnPCabToCounter:195:0\n' > "$qdir/episodes_$fam.txt"; fi
  if [ "$fam" = robotwin ] && [ ! -f "$out/run_meta.json" ]; then
    # one RoboTwin run = one task_config; the driver refuses a lane (or a reused control) with another one
    printf '{"family": "robotwin", "task_config": "%s", "set": "%s", "arm": "%s", "engine": "%s", "model": "%s", "started": "%s"}\n' \
      "${ROBOTWIN_TASK_CONFIG:-demo_randomized}" "$set" "$arm" "${ENGINE:-qwen}" "${MODEL:-qwen38}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$out/run_meta.json"
  fi

  local -a pids=()
  cleanup() {
    echo; echo "stopping lanes ..."
    [ ${#pids[@]} -gt 0 ] && kill "${pids[@]}" 2>/dev/null || true
    local l p
    for l in $(seq 0 $((lanes - 1))); do
      for p in $(ps -eo pid,args --no-headers | awk -v P="--port $((8450 + l)) " '/recovery_explore.env_servic[e]/ && index($0,P){print $1}'); do
        kill "$p" 2>/dev/null || true
      done
    done
  }
  trap cleanup INT TERM
  local l
  for l in $(seq 0 $((lanes - 1))); do
    mkdir -p "$out"
    env "${envs[@]}" TARGET="$n" bash recovery_explore/judge_driver_v7.sh "$l" > "$out/lane$l.out" 2>&1 &
    pids+=($!)
    echo "  lane $l started (log: $out/lane$l.out, progress: $out/lane$l/driver.log)"
    sleep 5
  done
  echo "running; Ctrl-C stops every lane. Progress: bash spotter.sh summary $run"
  wait || true
  trap - INT TERM
  cmd_summary "$run"
}

# ---------------------------------------------------------------- summary
cmd_summary() {
  local pattern="recovery_explore/runs_${1:-*}_*/results.jsonl" found=0 f
  for f in $pattern; do [ -f "$f" ] && found=1; done
  [ "$found" = 1 ] || { echo "no results yet ($pattern)"; return 0; }
  python3 - $pattern <<'EOF'
import json, sys, os
for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    name = os.path.basename(os.path.dirname(path))[len("runs_"):]
    parts = [f"{len(rows)} episodes"]
    for key, label in (("control_success", "control"), ("treatment_success", "Spotter")):
        v = [bool(r[key]) for r in rows if r.get(key) is not None]
        if v: parts.append(f"{label} {100 * sum(v) / len(v):.1f}% ({sum(v)}/{len(v)})")
    print(f"{name:32s} " + " | ".join(parts))
EOF
}

case "${1:-}" in
  judge) shift; cmd_judge "$@" ;;
  policy) shift; cmd_policy "$@" ;;
  run) shift; cmd_run "$@" ;;
  summary) shift; cmd_summary "$@" ;;
  ""|-h|--help|help) usage ;;
  *) echo "unknown command '$1'" >&2; usage 1 ;;
esac
