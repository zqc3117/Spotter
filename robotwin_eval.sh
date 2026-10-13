#!/bin/bash
# robotwin_eval.sh -- one command for the RoboTwin 2.0 evaluation: serve the Qwen judge and the pi0.5 policy (unless they
# already answer), run the Spotter arm and the policy-only control on the same episodes, then print the summary.
#
#   bash robotwin_eval.sh [--judge-gpus 0,1,2,3] [--policy-gpu 4] [--lanes 12] [--set rt500] [--sim-gpus N] [--dry-run]
#
# Defaults reproduce the released RoboTwin results: 500 episodes (rt500), 1.8x step budget, 1-shot, Qwen thinking in the
# repair turns with QWEN_THINK_BUDGET=5000 (export another value to change it). Logs of the servers go to logs/.
set -euo pipefail
cd "$(dirname "$0")"
[ -f env/env.sh ] || { echo "error: env/env.sh is missing; copy env/env.sh.example and edit it (README: Environment Setup)" >&2; exit 1; }
source env/env.sh
JUDGE_GPUS=0,1,2,3; POLICY_GPU=4; LANES=12; SET=rt500; SIMGPUS=""; DRY=""
while [ $# -gt 0 ]; do case "$1" in
  --judge-gpus) JUDGE_GPUS=$2; shift 2 ;; --policy-gpu) POLICY_GPU=$2; shift 2 ;; --lanes) LANES=$2; shift 2 ;;
  --set) SET=$2; shift 2 ;; --sim-gpus) SIMGPUS="--sim-gpus $2"; shift 2 ;; --dry-run) DRY=--dry-run; shift ;;
  -h|--help) sed -n 2,9p "$0"; exit 0 ;; *) echo "unknown option $1" >&2; exit 1 ;; esac; done
export QWEN_THINK_BUDGET=${QWEN_THINK_BUDGET:-5000}
n=$(echo "$JUDGE_GPUS" | tr ',' '\n' | grep -c .); ports=$(for i in $(seq 1 "$n"); do printf '%s%d' "${sep:-}" $((8300 + i)); sep=,; done)
export QWEN_PORTS=${QWEN_PORTS:-$ports}
mkdir -p logs
up(){ curl -sf -m 5 "$1" >/dev/null; }
if [ -z "$DRY" ]; then
  if ! up "http://${SCREENER_HOST:-127.0.0.1}:${QWEN_PORTS%%,*}/v1/models"; then
    echo "starting Qwen on GPUs $JUDGE_GPUS (ports $QWEN_PORTS) -> logs/judge.log"
    nohup bash spotter.sh judge --gpus "$JUDGE_GPUS" > logs/judge.log 2>&1 &
  fi
  if ! up "${POLICY_URL:-http://127.0.0.1:9100}/healthz"; then
    echo "starting the RoboTwin pi0.5 policy on GPU $POLICY_GPU -> logs/policy_robotwin.log"
    nohup bash spotter.sh policy robotwin --gpu "$POLICY_GPU" > logs/policy_robotwin.log 2>&1 &
  fi
  echo "waiting for Qwen and the policy (first start can take several minutes)"
  for i in $(seq 1 120); do
    q=1; for p in ${QWEN_PORTS//,/ }; do up "http://${SCREENER_HOST:-127.0.0.1}:$p/v1/models" || q=0; done
    [ $q = 1 ] && up "${POLICY_URL:-http://127.0.0.1:9100}/healthz" && break
    [ $i = 120 ] && { echo "error: servers not up after 20 min; see logs/judge.log and logs/policy_robotwin.log" >&2; exit 1; }
    sleep 10
  done
fi
echo "== Spotter arm (QWEN_THINK_BUDGET=$QWEN_THINK_BUDGET) on $SET"
bash spotter.sh run robotwin "$SET" --lanes "$LANES" $SIMGPUS --arm treat --one-shot --run "rt_${SET}_spotter" $DRY
echo "== control arm (policy alone) on $SET"
bash spotter.sh run robotwin "$SET" --lanes "$LANES" $SIMGPUS --arm ctrl --run "rt_${SET}_ctrl" $DRY
[ -n "$DRY" ] || bash spotter.sh summary
