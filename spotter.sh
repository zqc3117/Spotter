#!/usr/bin/env bash
# spotter.sh -- one entry point for serving the models and running Spotter on RoboCasa.
#
#   bash spotter.sh judge  [--gpus 0,1,2,3]                  serve the Qwen judge/screener with vLLM
#   bash spotter.sh policy <pi05|cosmos> [--gpu 0]           serve the embodied model
#   bash spotter.sh run    <pi05|cosmos> [SET] [options]      run episodes with the paper settings
#   bash spotter.sh summary [RUN]                             success rates of finished runs
#
# SET is smoke (1 episode, default), s500q96 (96), sall500 (1,200, the paper's set) or a file with
# one TASK:SEED:EPISODE per line. Run options:
#   --lanes N        parallel lanes (default 1); lane N uses simulation port 8450+N
#   --sim-gpus N     GPUs the simulation services are spread over, GPU 0..N-1 (default: all visible)
#   --arm A          both (control + treatment on each episode, default) | treat | ctrl
#   --run NAME       run name (default <family>_<set>); rerunning a name resumes it
#   --dry-run        print the commands instead of running them
# Every paper setting can still be overridden through the environment (see README, "Configuration").
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
if [ -f env/env.sh ]; then set +u; source env/env.sh; set -u; fi

die() { echo "spotter.sh: $*" >&2; exit 1; }
usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

policy_port() { case "$1" in cosmos) echo 8800 ;; pi05) echo 8900 ;; *) die "family must be pi05 or cosmos, got '$1'" ;; esac; }

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
  local fam=${1:-}; [ -n "$fam" ] || die "usage: spotter.sh policy <pi05|cosmos> [--gpu N]"; shift
  local gpu=0 port; port=$(policy_port "$fam")
  while [ $# -gt 0 ]; do case "$1" in --gpu) gpu=$2; shift 2 ;; --port) port=$2; shift 2 ;; *) die "unknown option $1" ;; esac; done
  export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
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
  local fam=${1:-}; [ -n "$fam" ] || die "usage: spotter.sh run <pi05|cosmos> [SET] [options]"; shift
  local set=smoke
  if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then set=$1; shift; fi
  local lanes=1 simgpus="" arm=both run="" dry=0
  while [ $# -gt 0 ]; do case "$1" in
    --lanes) lanes=$2; shift 2 ;; --sim-gpus) simgpus=$2; shift 2 ;; --arm) arm=$2; shift 2 ;;
    --run) run=$2; shift 2 ;; --dry-run) dry=1; shift ;; -h|--help) usage ;; *) die "unknown option $1" ;; esac; done

  local port; port=$(policy_port "$fam")
  local queue_src
  case "$set" in
    smoke) queue_src="" ;;
    s500q96|sall500) queue_src="recovery_explore/episode_sets/$set.txt" ;;
    *) [ -f "$set" ] || die "unknown set '$set' (smoke | s500q96 | sall500 | path to a TASK:SEED:EPISODE file)"; queue_src="$set" ;;
  esac
  [ -n "$run" ] || run="${fam}_$(basename "${set%.txt}")"
  local ctrl_only=0 skip_ctrl=0
  case "$arm" in both) ;; treat) skip_ctrl=1 ;; ctrl) ctrl_only=1 ;; *) die "--arm must be both, treat or ctrl" ;; esac
  if [ -z "$simgpus" ]; then simgpus=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '); [ "${simgpus:-0}" -gt 0 ] || simgpus=1; fi

  # paper settings (Table 3, "full context"); anything already exported in the environment wins
  local window=2 tel=40; [ "$fam" = pi05 ] && { window=1; tel=10; }
  local -a envs=(
    ON_POD=1 RUN="$run" NGPU="$simgpus" FAMILY_OVERRIDE="$fam"
    SIM_HOST=127.0.0.1 SIM_MANAGED=0 POLICY_URL="${POLICY_URL:-http://127.0.0.1:$port}"
    ROBOCASA_BIN="${ROBOCASA_BIN:-${COSMOS_POLICY_ENV:+$COSMOS_POLICY_ENV/bin}}"
    ENGINE="${ENGINE:-qwen}" MODEL="${MODEL:-qwen38}" SCREEN="${SCREEN:-1}" SCREENER_HOST="${SCREENER_HOST:-127.0.0.1}"
    QWEN_PORTS="${QWEN_PORTS:-8301,8302,8303,8304}" QWEN_KEEP_TURNS="${QWEN_KEEP_TURNS:-20}"
    STEP_BUDGET_SCALE="${STEP_BUDGET_SCALE:-1.8}" WINDOW="${WINDOW:-$window}" TEL_CHUNKS="${TEL_CHUNKS:-$tel}"
    COMPACT_AT="${COMPACT_AT:-110000}" MAX_INTERVENTIONS="${MAX_INTERVENTIONS:-8}"
    ALLOW_RESET=0 MAX_RESETS=0 FEWSHOT="${FEWSHOT:-0}" LEARN=0
    CTRL_ONLY=$ctrl_only SKIP_CONTROL=$skip_ctrl
  )

  local qdir="recovery_explore/runs_$run" out="recovery_explore/runs_${run}_$fam"
  local n
  if [ -n "$queue_src" ]; then n=$(grep -c . "$queue_src"); else n=1; fi
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
  curl -sf "$purl/health" >/dev/null || die "no policy server at $purl -- start it first: bash spotter.sh policy $fam"
  if [ "${ENGINE:-qwen}" = qwen ]; then
    local qp ok=0
    for qp in $(echo "${QWEN_PORTS:-8301,8302,8303,8304}" | tr ',' ' '); do
      curl -sf "http://${SCREENER_HOST:-127.0.0.1}:$qp/v1/models" >/dev/null && ok=$((ok + 1))
    done
    [ "$ok" -gt 0 ] || die "no Qwen replica answers on ${QWEN_PORTS:-8301,8302,8303,8304} -- start it first: bash spotter.sh judge"
  fi

  mkdir -p "$qdir"
  if [ -n "$queue_src" ]; then cp "$queue_src" "$qdir/episodes_$fam.txt"
  else printf 'PnPCabToCounter:195:0\n' > "$qdir/episodes_$fam.txt"; fi

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
