#!/bin/bash
# robodojo_eval.sh -- the RoboDojo evaluation behind the released RoboDojo numbers: VPP2 as the embodied policy,
# Qwen3.8-27B as the screener, GPT-6.1-sol (codex CLI) as the judge, one lane per simulator.
#
#   bash robodojo_eval.sh RUN EPISODES "LANE:GPU:POLICYPORT ..." [both|treat|ctrl] [--dry-run]
#
#   RUN        run name; results go to recovery_explore/runs_<RUN>_robodojo/results.jsonl
#   EPISODES   episode list, one "task:seed:episode" per line (recovery_explore/episode_sets/rd_test540.txt = released set)
#   specs      one lane per spec: lane number, GPU (informational: the driver uses lane mod NGPU), port of a VPP2
#              policy server on $POLICY_HOST
#   arm        both = control rollout then Spotter rollout per episode (default), treat = Spotter only, ctrl = control only
#
# Needs: env/env.sh with ROBODOJO_ENVSH / ROBODOJO_REPO / ROBODOJO_PYTHON (README: RoboDojo), the Qwen screener on
# $SCREENER_HOST:$QWEN_PORTS, VPP2 policy servers (XPolicyLab/policy/VPP2, robodojo.sh server) on $POLICY_HOST, and a
# signed-in `codex` CLI for the GPT judge.
#
# The defaults are exactly the settings of the released run (rdv11, 2026-11). They are NOT the official RoboDojo Agent
# Track protocol: the step budget is 3x the official one (SBS=3.0), the judge prompt carries per-task acceptance notes
# (RD_TASK_NOTES=1, recovery_explore/rd_task_notes.json) on top of the instruction, and the repair tools read the wrist
# and head depth images (RD_DEPTH=1, RD_MARKS=1). Set SBS=1.0 RD_TASK_NOTES=0 for the official budget and instruction.
set -euo pipefail
cd "$(dirname "$0")"
DRY=""; ARGS=()
for x in "$@"; do [ "$x" = --dry-run ] && DRY=1 || ARGS+=("$x"); done
[ ${#ARGS[@]} -ge 3 ] || { sed -n 2,22p "$0"; exit 1; }
RUN=${ARGS[0]}; EPF=${ARGS[1]}; SPECS=${ARGS[2]}; ARM=${ARGS[3]:-both}
[ -f env/env.sh ] || { echo "error: env/env.sh is missing; copy env/env.sh.example and edit it (README: Environment Setup)" >&2; exit 1; }
source env/env.sh
POLICY_HOST=${POLICY_HOST:-127.0.0.1}

# embodied policy and observation (VPP2, end-effector actions, 24-step chunks)
export ROBODOJO_POLICY_NAME=${ROBODOJO_POLICY_NAME:-VPP2} ROBODOJO_ACTION_TYPE=${ROBODOJO_ACTION_TYPE:-ee}
export ROBODOJO_LAYOUT_SEED=${ROBODOJO_LAYOUT_SEED:-1} ROBODOJO_POLICY_CHUNK=${ROBODOJO_POLICY_CHUNK:-24}
export RD_DEPTH=${RD_DEPTH:-1} RD_MARKS=${RD_MARKS:-1} RD_RESTAGE=${RD_RESTAGE:-1} RD_LEAD=${RD_LEAD:-0} RD_FEED_POLICY_HISTORY=${RD_FEED_POLICY_HISTORY:-1}
export RD_CAM_PARAMS=${RD_CAM_PARAMS:-1} RD_CAM_DEPTH=${RD_CAM_DEPTH:-1}
# repair tools and judge prompt
export RD_ESCALATE=${RD_ESCALATE:-1} RD_ESCALATE_MAX_WINDOWS=${RD_ESCALATE_MAX_WINDOWS:-6} RD_ESCALATE_MAX_EPISODE=${RD_ESCALATE_MAX_EPISODE:-2}
export RD_GRASP_VERIFY=${RD_GRASP_VERIFY:-1} RD_GRASP_VERIFY_LIFT=${RD_GRASP_VERIFY_LIFT:-0.03}
export RD_ROBOT_SNAP=${RD_ROBOT_SNAP:-1} RD_ROBOT_SNAP_MAX=${RD_ROBOT_SNAP_MAX:-0.08} RD_PROMPT_DISCIPLINE=${RD_PROMPT_DISCIPLINE:-1}
export RD_HONEST=${RD_HONEST:-1} RD_FIX_FEEDBACK=${RD_FIX_FEEDBACK:-1} RD_STEPWISE=${RD_STEPWISE:-1} RD_STEPWISE_ROUNDS=${RD_STEPWISE_ROUNDS:-8}
export RD_TASK_NOTES=${RD_TASK_NOTES:-1} RD_TRIANGULATE=${RD_TRIANGULATE:-0} RD_DESCENT_TOL=${RD_DESCENT_TOL:-0.015}
export RD_STAGED=${RD_STAGED:-1} RD_PERSIST=${RD_PERSIST:-0} RD_VERIFY_ACT=${RD_VERIFY_ACT:-0}
export RD_REACH_REPORT=${RD_REACH_REPORT:-1} RD_RESTRAINT=${RD_RESTRAINT:-1}
# screener gating
export SCREEN_SHADOW=${SCREEN_SHADOW:-0} SCREEN_CONFIRM2=${SCREEN_CONFIRM2:-0} SCREEN_PROMPT=${SCREEN_PROMPT:-v1}
export SCREEN_THRESHOLD=${SCREEN_THRESHOLD:-3} SCREEN_THINK=${SCREEN_THINK:-0}
export SCREEN_AFTER_IV=${SCREEN_AFTER_IV:-2} SCREEN_MAXGAP=${SCREEN_MAXGAP:-3} SCREEN_STALL_CM=${SCREEN_STALL_CM:-3}
# judge (GPT-6.1-sol through the codex CLI) and budgets
ENGINE=${ENGINE:-codex}; MODEL=${MODEL:-gpt-6.1-sol}; REASONING=${REASONING:-xhigh}
SBS=${SBS:-3.0}; MAXIV=${MAXIV:-4}
export JUDGE_DEADLINE=${JUDGE_DEADLINE:-2700} ACT_DEADLINE=${ACT_DEADLINE:-2700} ADV_DEADLINE=${ADV_DEADLINE:-900}
export CODEX_E2_TIMEOUT=${CODEX_E2_TIMEOUT:-3300} MISS_GRACE=${MISS_GRACE:-0} FIRST_IV_FRAC=${FIRST_IV_FRAC:-0}

RE=recovery_explore; mkdir -p $RE/runs_$RUN $RE/runs_${RUN}_robodojo
[ -f $RE/runs_$RUN/episodes_robodojo.txt ] || grep -v '^#' "$EPF" | sed '/^$/d' > $RE/runs_$RUN/episodes_robodojo.txt
CTRL_ONLY=0; SKIP_CONTROL=0; [ "$ARM" = treat ] && SKIP_CONTROL=1; [ "$ARM" = ctrl ] && CTRL_ONLY=1
N=$(wc -l < $RE/runs_$RUN/episodes_robodojo.txt)
for spec in $SPECS; do
  IFS=: read -r lane gpu pport <<< "$spec"
  cmd=(env ON_POD=1 RUN=$RUN TARGET=$N NGPU=${NGPU:-4} FAMILY_OVERRIDE=robodojo SIM_HOST=127.0.0.1 SIM_MANAGED=0
       POLICY_URL=$POLICY_HOST:$pport ENGINE=$ENGINE MODEL=$MODEL REASONING=$REASONING SCREEN=1
       QWEN_KEEP_TURNS=10 COMPACT_AT=${CPT:-120000} QWEN_THINK_BUDGET=5000 QWEN_REPAIR_THINK=1
       STEP_BUDGET_SCALE=$SBS INTERVENE_CHARGES_STEP_LIM=1 WINDOW=2 TEL_CHUNKS=10 MAX_INTERVENTIONS=$MAXIV
       ALLOW_RESET=0 MAX_RESETS=0 FEWSHOT=0 LEARN=0 CTRL_ONLY=$CTRL_ONLY SKIP_CONTROL=$SKIP_CONTROL
       bash $RE/judge_driver_v7.sh $lane)
  if [ -n "$DRY" ]; then echo "lane $lane (gpu $gpu): ${cmd[*]}"; continue; fi
  setsid nohup "${cmd[@]}" > $RE/runs_${RUN}_robodojo/lane$lane.out 2>&1 < /dev/null &
  echo "lane $lane gpu $gpu policy $POLICY_HOST:$pport pid $!"; sleep 20
done
