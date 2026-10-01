#!/bin/bash
# Full-episode VLM supervision experiment v7: paired comparison of "policy alone" vs "policy + a judge call every N chunks"
# Usage: judge_driver_v7.sh <lane>; env vars TARGET / WINDOW / INTERVENE_BUDGET / PLAN_ROUNDS
#
# The only difference from v6 is **how interventions are delivered**; judging, control and bookkeeping are unchanged:
#   v6: the model typed rex primitives one at a time via Bash; a median intervention took 15 tool turns and 272 s,
#       while the server executed those actions in under 10 s -- 96% of the time went to re-thinking between primitives.
#   v7: judging and planning are one call; the model returns a JSON plan of <=8 primitives, and the server's
#       execute_plan runs it in order, stopping at a **checkpoint** (after a lift) or an **anomaly** (empty grasp, servo missed,
#       target clipped, contact force over threshold, budget exhausted) and handing back the step log and freshly rendered frames.
#       The model then continues with the next plan segment or finishes. No Bash at any point.
# Expected turns: 15 -> 3~4. In the v1 transcripts the model only really used two kinds of feedback: "after the lift, confirm the object
# moved with the hand" and "a step clearly failed, so change course"; the former is the checkpoint, the latter the abort conditions,
# and both are kept.
LANE=${1:?lane}
# ---- Where the driver runs ----
# ON_POD=1 (default): the driver itself runs on a cluster machine (bash 5; with --engine claude,
#          /usr/local/bin/claude is a wrapper with its own proxy and credentials -- do not export http_proxy).
#          Repo root is derived from this script's location (JUDGE_REPO_ROOT overrides); python3 is the robocasa venv.
#          No port-forwarding at all in this mode:
#   SIM_HOST (default 127.0.0.1) host running the simulation service.
#     = 127.0.0.1 -> start the service locally with setsid (recovery_explore/launch_svc.sh, ships with the repo).
#     = anything else (e.g. the simulation host's IP) -> kubectl exec into $SIM_POD to start it, and reach it at http://$SIM_HOST:$PORT
#       (pod IP, direct in-cluster connection). The harness / rex endpoints follow $SIM_HOST.
#   SIM_POD  pod used to start the simulation service when SIM_HOST is not local (default: the simulation host).
#   POLICY_URL policy service address (default http://127.0.0.1:$POLICY, cosmos 8778 / pi05 8877).
#            May point to another pod's IP; the policy service need not share a machine with the simulation service.
#   NGPU     number of GPUs the simulation service is spread across (ON_POD default 2; Mac mode 4).
# ON_POD=0: legacy Mac mode, kubectl exec into $POD to start the service + port-forward. Kept only as a fallback.
ON_POD=${ON_POD:-1}
SIM_HOST=${SIM_HOST:-127.0.0.1}
SIM_POD=${SIM_POD:-<sim-pod>}
# SIM_MANAGED=1: the simulation service is started beforehand by an external launcher (which has kubectl);
# the driver only connects, never starts it. The driver's pod then needs no kubectl, and the simulation can live on any
# machine -- judge CLI on one host, simulation on another, policy inference on a third, each tier using its own resources.
# Cost: if the service dies mid-run the driver cannot restart it; an external watchdog has to.
SIM_MANAGED=${SIM_MANAGED:-0}
# Repo root is derived from this script's location (the driver always lives in <repo>/recovery_explore/),
# so the checkout can live anywhere. JUDGE_REPO_ROOT overrides it.
P=${JUDGE_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# ROBOCASA_BIN: bin dir of the simulation venv, prepended to PATH so python3 is the 3.10 one.
# Only added when set; otherwise the current PATH is used (in single-machine mode the caller activates the venv itself).
if [ -n "${ROBOCASA_BIN:-}" ]; then export PATH="$ROBOCASA_BIN:$PATH"
elif [ -d /workspace/cosmos_policy/envs/robocasa/bin ]; then
  export PATH=/workspace/cosmos_policy/envs/robocasa/bin:$PATH
fi
POD=${SIM_POD:-<sim-pod>}; NS=${JUDGE_NS:-<namespace>}
# 8 lanes in parallel: lanes 0-3 = cosmos, lanes 4-7 = pi0.5
# Simulation services are spread over 4 GPUs; the policy service runs cosmos on GPU0 and pi0.5 on GPU1
PORT=$((8450+LANE))
if [ "$ON_POD" = 1 ]; then NGPU=${NGPU:-2}; else NGPU=${NGPU:-4}; fi
CUDA=$((LANE % NGPU))
# FAMILY_OVERRIDE: once pi0.5 is done, its freed lanes switch to cosmos instead of idling.
# The lane number only picks the port and GPU, no longer the policy family.
# robodojo is the third family (Isaac Sim + ARX X5 dual arm), enabled only via an explicit FAMILY_OVERRIDE:
# it shares the harness subcommands and this driver with the other two families; the differences all live in launch_svc.sh.
if [ "${FAMILY_OVERRIDE:-}" = "robodojo" ]; then
  POLICY=${POLICY:-8950}; FAMILY=robodojo
# robotwin: RoboTwin 2.0 (SAPIEN, aloha-agilex dual arm), same harness surface as robodojo.
elif [ "${FAMILY_OVERRIDE:-}" = "robotwin" ]; then
  POLICY=${POLICY:-9100}; FAMILY=robotwin
elif [ "${FAMILY_OVERRIDE:-}" = "cosmos" ] || { [ -z "${FAMILY_OVERRIDE:-}" ] && [ "$LANE" -lt 4 ]; }; then
  POLICY=${POLICY:-8778}; FAMILY=cosmos
else POLICY=${POLICY:-8877}; FAMILY=pi05; fi
POLICY_URL=${POLICY_URL:-http://127.0.0.1:$POLICY}
SVC_LAUNCH="$P/recovery_explore/launch_svc.sh"
SVC_ARGS="$CUDA $PORT $POLICY_URL $FAMILY rounds"
# Step-budget multiplier. In the three-tier topology the simulation host's launcher passes it through; in single-pod mode restart_svc
# passes it through -- this used to be missing, so single-pod mode always ran at 1.0 whatever was set, and --budget-scale
# is the only parameter that affects the control arm; comparing control and treatment at different multipliers is invalid.
STEP_BUDGET_SCALE=${STEP_BUDGET_SCALE:-1.0}
H="python3 $P/recovery_explore/cli/harness.py --endpoint http://$SIM_HOST:$PORT"
REXPORT=$PORT
# In v7 the driver, not the model, executes the plan (the model has no Bash). It still goes through the model-side rex:
# same endpoint, same server-side limits; only the caller is now the driver.
REX="python3 $P/recovery_explore/cli/rex.py --endpoint http://$SIM_HOST:$PORT"
OUT=$P/recovery_explore/runs_${RUN}_$FAMILY; mkdir -p $OUT/lane$LANE $OUT/claims
RESULTS=$OUT/results.jsonl; touch $RESULTS
QUEUE=$P/recovery_explore/runs_${RUN}/episodes_$FAMILY.txt
RUN=${RUN:-r4}
TARGET=${TARGET:-6}; WINDOW=${WINDOW:-2}; LOOKAHEAD=${LOOKAHEAD:-0}
# ASYNC_SCREEN=1 (2026-09-22, user): the policy keeps running while the screener and the judge look
# at a finished window. A background runner advances window after window; the loop below always
# takes the newest finished window, and only an "intervene" stops the policy (the repair starts
# from wherever the arm is by then). 0 = the synchronous loop, unchanged.
ASYNC=${ASYNC_SCREEN:-0}
# ASYNC_SCREEN=2 (2026-09-22, user): mode 1 took only the newest window and left 50-80% of a failing episode's
# windows unseen. Mode 2 keeps the overlap but not the blind spots:
#   - every window the policy ran since the last check goes to the screener together (montages stacked);
#   - the policy may run at most ASYNC_K windows past the last checked one, then it waits;
#   - the judge runs in the background too (user 2026-09-23): nothing stops the policy except the cap, so a
#     judged window is at most ASYNC_K windows stale, and the policy may fix itself meanwhile.
ASYNC_K=${ASYNC_K:-1}
# LEARN=0 frozen memory bank: retrieval and injection as usual, no new entries written (for formal evaluation).
# LEARN=1 after each trajectory the judge writes that episode's lesson as a draft into inbox/<cell_tag>/;
#         a human runs memory_cli merge-inbox to decide whether it goes in -- the bank is never rewritten automatically.
LEARN=${LEARN:-0}
# Object instance split: A = instances seen in training, B = held out for generalization. harness has always defaulted to B,
# while both policies' published baselines were measured on A -- without aligning, our numbers cannot be compared to theirs.
SPLIT=${SPLIT:-B}
INTERVENE_BUDGET=${INTERVENE_BUDGET:-8}; MAX_INTERVENTIONS=${MAX_INTERVENTIONS:-8}
# Max plan segments per intervention (each segment = one execute_plan + one model call).
# 3 segments * 8 steps is enough for "open -> back to the miss point -> descend -> close -> lift and verify" plus one correction.
PLAN_ROUNDS=${PLAN_ROUNDS:-3}
# Whether rewinding is allowed. With ALLOW_RESET=0 reset_window is never called and retry does not appear among the model's
# options -- it must finish within one intervention or give up. This measures what "starting over" is really worth:
# in episodes with reset, 6 of 11 plan segments came after a reset, and a single episode took 40 min wall clock.
ALLOW_RESET=${ALLOW_RESET:-1}
# Rewinds allowed per episode, passed to env_service (0 is equivalent to ALLOW_RESET=0).
MAX_RESETS=${MAX_RESETS:-5}
[ "$ALLOW_RESET" = 1 ] || MAX_RESETS=0
[ "$MAX_RESETS" = 0 ] && ALLOW_RESET=0
# Local screener (four Qwen replicas on a local GPU host): every window goes through it first; if it passes, the judge is not called.
# **Off by default**: with SCREEN=0 this whole block is skipped and behavior is identical to v7 without a screener
# (SKIPPED empty -> no gap notice; OPUS_CALLED set to 1 at window 1 -> the brief is still inlined in window 1;
#  results gain a screened_windows field that is always 0). The comparison experiment relies on this switch:
# both arms run the same file and the same prompt, differing only in whether the screener runs first.
SCREEN=${SCREEN:-0}; SCREENER_HOST=${SCREENER_HOST:-127.0.0.1}
# Round three switched to opus 5: round-two interventions could not tell "which moment needs fixing", and judging was too conservative.
JUDGE_DEADLINE=${JUDGE_DEADLINE:-600}; ACT_DEADLINE=${ACT_DEADLINE:-1500}   # per-call deadlines (s); RoboTwin thinking runs raise JUDGE_DEADLINE
# Qwen thinking: the screener and the window verdict never think; the repair turns of an intervention
# (exec rounds, push, replan, repeat) think when QWEN_REPAIR_THINK=1, up to QWEN_THINK_BUDGET (2000) tokens
# (cli/qwen_call.py). QWEN_REPAIR_THINK=0 is the paper's setting: no call thinks.
QWEN_REPAIR_THINK=${QWEN_REPAIR_THINK:-1}
# Judge backend: claude = claude code cli, codex = codex cli.
# They differ in three ways, all smoothed over in engine_turn:
#   1. codex does not Read image paths by itself; the overview image must be passed explicitly with -i
#   2. codex exec resume does not accept -s/--sandbox; the sandbox can only be set on the first turn
#   3. codex writes its final answer to the file given by -o; it is wrapped as {"result": ...} so downstream is uniform
ENGINE=${ENGINE:-claude}
if [ "$ENGINE" = "api" ]; then MODEL=${MODEL:-gpt-6-astra}; REASONING=${REASONING:-medium}
elif [ "$ENGINE" = "qwen" ]; then MODEL=${MODEL:-qwen38}; REASONING=${REASONING:-medium}; COMPACT_AT=${COMPACT_AT:-22000}
elif [ "$ENGINE" = "codex" ]; then MODEL=${MODEL:-gpt-5.6-luna}; REASONING=${REASONING:-medium}
else MODEL=${MODEL:-claude-opus-5}; fi
# Do not give up when quota-limited: wait QUOTA_WAIT seconds and retry, up to QUOTA_TRIES times (default: a full 24 hours)
QUOTA_WAIT=${QUOTA_WAIT:-1800}; QUOTA_TRIES=${QUOTA_TRIES:-48}
# Async drops the judge-turn lock (it would refuse the runner's advances). That is only safe for engines
# without a shell (qwen, api); claude/codex have Bash and could reach the simulator.
case "$ENGINE" in qwen|api) ;; *) ASYNC=0 ;; esac
# Context compaction: after each claude turn, look at the context the model actually saw in that turn
# (input + cache_creation + cache_read); above COMPACT_AT, drop the session,
# and the next judging turn resends the W=1 template (brief inlined) plus a driver-generated recap of the episode.
# Once the context exceeds the threshold, the session is reopened at the next window. If a whole trajectory is one conversation, every past
# window's images stay in context -- truncating the telemetry table does not remove them; only reopening does. The threshold here means
# roughly one reopen every three windows, i.e. "only the last ~3 steps of pictures on hand", with a per-window text recap on reopen.
# 2026-09-21 (user): compact every 70k, and inside an intervention too (one rtG2 repair grew
# to 347k across its rounds because only the window boundary reset the session).
COMPACT_AT=${COMPACT_AT:-70000}
# CTRL_ONLY=1 runs only the control arm to build a baseline; REUSE_CONTROL=<results.jsonl> looks control results up in a baseline.
CTRL_ONLY=${CTRL_ONLY:-0}
# The telemetry table only covers the most recent chunks (harness.py reads this env var). 0 = no truncation.
# 2026-09-20 (user): 3 chunks left the judge without history; the table was 20 chunks (10 since 2026-09-21) and the
# earlier ones are sampled into a trajectory summary by cli/harness.py.
TEL_CHUNKS=${TEL_CHUNKS:-20}; export TEL_CHUNKS
# FEWSHOT=N: N worked examples (mined from the learn seed by cli/fewshot_build.py) go into
# the judge brief, pictures interleaved. 0 = the prompt is byte-for-byte what it was before.
FEWSHOT=${FEWSHOT:-0}
# Reuse existing control-arm results instead of rerunning (the control arm does not depend on the judge model)
REUSE_CONTROL=${REUSE_CONTROL:-}

# Startup self-check. Memory retrieval and quota checks run with 2>/dev/null, so a missing file just silently returns empty --
# there were once several whole rounds with zero memory injection and nobody noticed. Exit right away if anything is missing.
NEED_BRIEFS="BRIEF_PNP.md BRIEF_MECH.md"
[ "$FAMILY" = robodojo ] && NEED_BRIEFS="$NEED_BRIEFS BRIEF_RD.md"
[ "$FAMILY" = robotwin ] && NEED_BRIEFS="$NEED_BRIEFS BRIEF_RT.md"
[ "$FAMILY" = robotwin ] && RESTAGE_REACH="goes back there in one planned path, however far"
for f in cli/harness.py cli/rex.py cli/memory_select.py cli/quota_check.py cli/attempt_ledger.py \
         cli/screen_window.py JUDGE_BRIEF_V7.md $NEED_BRIEFS launch_svc.sh; do
  [ -f "$P/recovery_explore/$f" ] || { echo "missing file: $P/recovery_explore/$f" >&2; exit 4; }
done
# RoboTwin keeps its own lesson bank: RoboCasa's kitchen lessons are noise on a dual-arm
# tabletop. memory_select.py reads MEMORY_BANK; the learn turn drafts into its inbox.
if [ "$FAMILY" = robotwin ]; then
  export MEMORY_BANK=$P/recovery_explore/${RT_MEMORY_BANK:-memory_bank_robotwin}   # RT_MEMORY_BANK=memory_bank_robotwin_noreset: 2026-09-22 cleaned copy
  mkdir -p "$MEMORY_BANK/global" "$MEMORY_BANK/inbox"
else export MEMORY_BANK=$P/recovery_explore/memory_bank; fi
[ -d "$MEMORY_BANK/global" ] || { echo "missing memory bank" >&2; exit 4; }
# SCREEN=1 requires the screener module. Without this check it would only surface at the first window, and screen_window.py
# is fail-closed: without a screener it returns flag=true, so the whole run silently becomes --arm a.
if [ "$SCREEN" = 1 ]; then
  [ -f "${SCREENER_DIR:-$P/recovery_explore/cli}/screener.py" ] || {
    echo "missing screener module: ${SCREENER_DIR:-$P/recovery_explore/cli}/screener.py" >&2
    echo "--arm b needs it. Point SCREENER_DIR at it, or use --arm a instead." >&2; exit 4; }
fi
# FEWSHOT>0 requires the example bank. The bank is gitignored (100+ MB of PNGs), and when it is missing
# fewshot_select.py just returns empty; the driver logs one line and continues as 0-shot -- so the run
# is not the configuration you think it is, and the results will not tell you afterwards.
if [ "${FEWSHOT:-0}" -gt 0 ] 2>/dev/null; then
  FS_BANK=${FEWSHOT_BANK:-$P/recovery_explore/fewshot_bank}
  if [ ! -d "$FS_BANK" ]; then
    echo "missing 1-shot example bank: $FS_BANK" >&2
    echo "FEWSHOT=$FEWSHOT needs it. Rebuild it with cli/fewshot_build.py (needs past runs to mine)," >&2
    echo "point FEWSHOT_BANK elsewhere, or set FEWSHOT=0 explicitly to run 0-shot." >&2
    echo "(to proceed anyway, set FEWSHOT_ALLOW_MISSING=1)" >&2
    [ "${FEWSHOT_ALLOW_MISSING:-0}" = 1 ] || exit 4
  fi
  [ -f "$P/recovery_explore/cli/fewshot_select.py" ] || { echo "missing file: cli/fewshot_select.py" >&2; exit 4; }
fi

log(){ echo "$(date +%m-%d\ %H:%M:%S) [judge$LANE] $*" | tee -a $OUT/lane$LANE/driver.log; }
jget(){ python3 -c "import json,sys;print(json.load(sys.stdin).get('$1'))" 2>/dev/null; }

# ---- ASYNC_SCREEN runner ----
# The runner is a background subshell that advances WINDOW chunks at a time into $CD/async/adv_<w>.json
# and stops by itself after a window that succeeded, hit the step limit or failed to advance.
# Every advance and every stretch the loop spends waiting is appended to $CD/async_timing.jsonl,
# which is what the speed report reads: sync-equivalent time = wall + (background advance - waiting).
async_start(){ # $1 = first window the runner executes
  local w0=$1 f k
  mkdir -p "$CD/async"; rm -f "$CD/async/stop" "$CD/async/runner_exit"
  for f in "$CD"/async/adv_*.json; do
    [ -e "$f" ] || continue; k=${f##*adv_}; k=${k%.json}
    [ "$k" -ge "$w0" ] 2>/dev/null && rm -f "$f"
  done
  ( w=$w0
    while [ ! -f "$CD/async/stop" ]; do
      if [ "$ASYNC" = 2 ]; then
        h0=$(date +%s.%N)
        while [ ! -f "$CD/async/stop" ]; do
          c=$(cat "$CD/async/checked" 2>/dev/null); c=${c:-0}
          # c = last window fully checked, so c+1 is the one being checked right now: the policy may run
          # ASYNC_K windows past THAT one. Counting from c alone (the first cut) left no overlap at all.
          [ $((w - c)) -le $((ASYNC_K + 1)) ] && break
          sleep 0.2
        done
        [ -f "$CD/async/stop" ] && break
        echo "{\"kind\":\"hold\",\"w\":$w,\"t0\":$h0,\"t1\":$(date +%s.%N)}" >> "$CD/async_timing.jsonl"
      fi
      t0=$(date +%s.%N)
      A=$(timeout ${ADV_DEADLINE:-600} $H advance --num-chunks $WINDOW --frames-dir "$CD/w$w" --truth-dir "$TD/w$w" --tag w 2>&1 | tail -1)
      t1=$(date +%s.%N)
      printf '%s\n' "$A" > "$CD/async/adv_$w.tmp" && mv "$CD/async/adv_$w.tmp" "$CD/async/adv_$w.json"
      echo "{\"kind\":\"adv\",\"w\":$w,\"t0\":$t0,\"t1\":$t1}" >> "$CD/async_timing.jsonl"
      st=$(printf '%s' "$A" | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: print('stop'); sys.exit()
c=d.get('committed_timestep')
print('stop' if c in (None,'') or d.get('task_success') is True or d.get('done') is True else 'go')" 2>/dev/null)
      if [ "$st" != go ]; then
        # 2026-09-23 (user): the episode is over the moment the policy succeeds; do not wait for a screener
        # or judge call that is still looking at an older window. Only this episode's calls match $CD.
        if [ "$(printf '%s' "$A" | jget task_success)" = "True" ]; then
          echo "$(date +%s.%N)" > "$CD/async/success"
          pkill -f "screen_window[.]py .*$CD/" 2>/dev/null; pkill -f "qwen_call[.]py .*$CD/" 2>/dev/null
          pkill -f "gpt6_call[.]py .*$CD/" 2>/dev/null
        fi
        break
      fi
      w=$((w+1))
    done
    touch "$CD/async/runner_exit" ) &
  RUNNER_PID=$!; RUNNER_ON=1
}
async_next(){ # $1 = first window the loop has not seen; sets ADV, W (newest finished window), ASKIP
  local want=$1 k last="" t0
  t0=$(date +%s.%N)
  while :; do
    last=""; k=$want
    while [ -f "$CD/async/adv_$k.json" ]; do last=$k; k=$((k+1)); done
    [ -n "$last" ] && break
    [ -f "$CD/async/runner_exit" ] && break
    sleep 0.2
  done
  echo "{\"kind\":\"wait\",\"w\":$want,\"t0\":$t0,\"t1\":$(date +%s.%N)}" >> "$CD/async_timing.jsonl"
  if [ -f "$CD/async/runner_exit" ]; then
    # the runner may have finished one more window between the scan and this check: rescan
    wait "$RUNNER_PID" 2>/dev/null; RUNNER_ON=0
    k=$want; while [ -f "$CD/async/adv_$k.json" ]; do last=$k; k=$((k+1)); done
  fi
  ASKIP=""
  if [ -z "$last" ]; then ADV=""; W=$want; return 0; fi
  ADV=$(cat "$CD/async/adv_$last.json")
  for ((k=want; k<last; k++)); do ASKIP="$ASKIP $k"; done
  W=$last
}
async_montage(){ # $1 = output png, rest = window indices (oldest first): stack their montages, newest 3 at most
  python3 - "$CD" "$@" << 'PYM' 2>/dev/null
import json, sys
from PIL import Image
cd, out, ws = sys.argv[1], sys.argv[2], sys.argv[3:][-3:]
ims = []
for w in ws:
    try:
        ims.append(Image.open(json.load(open(f"{cd}/async/adv_{w}.json"))["montage"]).convert("RGB"))
    except Exception:
        pass
if not ims: sys.exit(1)
W = max(i.width for i in ims)
ims = [i if i.width == W else i.resize((W, int(i.height * W / i.width))) for i in ims]
canvas = Image.new("RGB", (W, sum(i.height for i in ims) + 6 * (len(ims) - 1)), (255, 255, 255))
y = 0
for i in ims:
    canvas.paste(i, (0, y)); y += i.height + 6
canvas.save(out); print(out)
PYM
}
async_stop(){ # stop the runner at the end of the window it is executing
  [ "${RUNNER_ON:-0}" = 1 ] || return 0
  local t0; t0=$(date +%s.%N)
  touch "$CD/async/stop"; wait "$RUNNER_PID" 2>/dev/null; RUNNER_ON=0
  echo "{\"kind\":\"wait\",\"w\":-1,\"t0\":$t0,\"t1\":$(date +%s.%N),\"why\":\"stop\"}" >> "$CD/async_timing.jsonl"
}

# robotwin: one run = one RoboTwin task_config. judge.sh writes $OUT/run_meta.json at start; a
# lane whose ROBOTWIN_TASK_CONFIG differs, a --reuse-control baseline from another config, or a
# sim service that reports a different task_config in `meta` refuses to run (exit 5), so control
# and treatment can never mix demo_clean with demo_randomized.
if [ "$FAMILY" = robotwin ]; then
  export ROBOTWIN_TASK_CONFIG=${ROBOTWIN_TASK_CONFIG:-demo_randomized}
  rt_cfg_of(){ python3 -c "import json,sys
try: print(json.load(open(sys.argv[1])).get('task_config') or '')
except Exception: print('')" "$1" 2>/dev/null; }
  if [ -f "$OUT/run_meta.json" ]; then
    _c=$(rt_cfg_of "$OUT/run_meta.json")
    [ -n "$_c" ] && [ "$_c" != "$ROBOTWIN_TASK_CONFIG" ] && { log "run_meta task_config=$_c but this lane has $ROBOTWIN_TASK_CONFIG; refusing"; exit 5; }
  fi
  if [ -n "$REUSE_CONTROL" ]; then
    _c=$(rt_cfg_of "$(dirname "$REUSE_CONTROL")/run_meta.json")
    [ "$_c" = "$ROBOTWIN_TASK_CONFIG" ] || { log "REUSE_CONTROL baseline task_config='${_c}' != $ROBOTWIN_TASK_CONFIG; refusing"; exit 5; }
  fi
  _c=$($H meta 2>/dev/null | tail -1 | python3 -c "import json,sys
try: print(json.load(sys.stdin).get('task_config') or '')
except Exception: print('')" 2>/dev/null)
  [ -n "$_c" ] && [ "$_c" != "$ROBOTWIN_TASK_CONFIG" ] && { log "sim service reports task_config=$_c, lane wants $ROBOTWIN_TASK_CONFIG; refusing"; exit 5; }
fi
run_with_deadline(){ local s="$1" lg="$2"; shift 2; "$@" > "$lg" 2>&1 < /dev/null & local pid=$!; local w=0
  while kill -0 $pid 2>/dev/null; do sleep 5; w=$((w+5))
    [ $w -ge $s ] && { echo "[watchdog] killed ${s}s" >> "$lg"; kill -9 $pid 2>/dev/null; return 124; }
  done; wait $pid 2>/dev/null; return $?; }

# In ON_POD mode kubectl gets 3 retries (the k8s API is flaky); Mac mode keeps the original single call.
kx(){ local i; for i in 1 2 3; do kubectl "$@" && return 0; sleep 4; done; return 1; }
restart_svc(){
  # The service start command is shared by all three modes: Mac -> kubectl exec $POD; ON_POD with local sim -> plain bash;
  # ON_POD with sim on another pod -> kubectl exec $SIM_POD.
  # First kill the old service by process name and wait for it to disappear. This is best effort, **not** a guarantee the port is free:
  # once the process is a zombie its cmdline is empty, so this loop can conclude the old service is gone in the window where it
  # "no longer matches but its fds are not yet released". The real gate is in launch_svc.sh (it waits on the port itself and, failing that,
  # finds the port holder via /proc and kills it). This used to be a fixed sleep 2, sometimes enough, sometimes not, showing up as
  # "a lane occasionally dies": the new service hits the unreleased port -> no free port -> driver reports "service will not start".
  local svc_cmd="for i in \$(seq 1 40); do
      pids=\$(ps -eo pid,cmd --no-headers | awk '/recovery_explore.env_servic[e]/ && /--port $PORT /{print \$1}')
      [ -z \"\$pids\" ] && break
      for p in \$pids; do kill -9 \$p 2>/dev/null; done
      sleep 1
    done
    MAX_RESETS=$MAX_RESETS STEP_BUDGET_SCALE=$STEP_BUDGET_SCALE setsid bash $SVC_LAUNCH $SVC_ARGS > /tmp/judge_svc_$PORT.out 2>&1 </dev/null & disown; sleep 14"
  local tries=6
  if [ "$SIM_MANAGED" = 1 ]; then
    # The service is kept alive by a supervisor loop on the simulation host. Each arm needs a fresh process
    # (episode-begin refuses to build a second episode in a process that already holds an env); the driver has no kubectl,
    # so it asks the service to exit and the supervisor loop relaunches it 3 s later. First wait for the old process to really be gone, then for
    # the new one to come up; otherwise meta may still be probing the old, exiting process.
    $H quit >/dev/null 2>&1
    # Wait up to 60 s for the old process to be truly gone. After quit it may take a dozen seconds or more to exit (it finishes writing video);
    # waiting only 10 s sends episode-begin to the exiting process -> episode creation fails. That is how 32 of 37 cells
    # in one round were lost (2026-09-21). The same root cause in single-machine mode is handled by the port wait in launch_svc.sh.
    for i in $(seq 1 30); do $H meta >/dev/null 2>&1 || break; sleep 2; done
    tries=${SVC_WAIT_TRIES:-18}
  elif [ "$ON_POD" = 1 ]; then
    if [ "$SIM_HOST" = 127.0.0.1 ]; then bash -c "$svc_cmd" >/dev/null 2>&1
    else kx exec -n $NS "$SIM_POD" -- bash -c "$svc_cmd" >/dev/null 2>&1; fi
    # Direct in-cluster connection to http://$SIM_HOST:$PORT, no port-forward
  else
    kubectl exec -n $NS "$POD" -- bash -c "$svc_cmd" >/dev/null 2>&1
    for p in $(pgrep -f "port-forward.*$PORT:$PORT"); do kill $p 2>/dev/null; done; sleep 1
    nohup kubectl port-forward -n $NS "$POD" $PORT:$PORT >/dev/null 2>&1 &
    sleep 6
  fi
  for i in $(seq 1 $tries); do $H meta >/dev/null 2>&1 && return 0; sleep 10; done
  log "service will not start"; return 1; }

# Let the policy run the rest of the episode alone; returns task_success
# The control arm must use exactly the same call granularity as the treatment arm. This used to hard-code --num-chunks 8 while the
# treatment arm used WINDOW=2, so the two arms' request_id sequences differed and the generated actions were not the same. In m2astra
# 3 cells where the judge never intervened (int=0) had ctrl=True treat=False for exactly this reason:
# the flip had nothing to do with the judge, just different sampling in the two arms. With matched granularity, int=0 cells should be
# identical, and any remaining flips are really caused by the judge.
# The step cap must be converted too: the old acc>30 at 8 chunks/call = 240 chunks; keeping 30 would cut the control arm
# to 60 chunks at WINDOW=2, trading an old bug for a new one. Now counted in total chunks.
CTRL_MAX_CHUNKS=${CTRL_MAX_CHUNKS:-240}
run_to_end(){ local acc=0 chunks=0 td=""
  # For RoboDojo the control arm needs more than success/failure: the official leaderboard score is the mean episode_score, and failed
  # episodes still earn partial credit on the 35 tasks with staged scoring (run_eval records process_score/100 for them).
  # truth is written to disk only when --truth-dir is passed. The third argument is passed only for the robodojo family; for the other two
  # $td is empty and expands to no argument at all, so argv is exactly as before this change.
  { [ "$FAMILY" = robodojo ] || [ "$FAMILY" = robotwin ]; } && [ -n "${3:-}" ] && td="--truth-dir $3"
  while true; do
    # advance must have a deadline: once Isaac hits a CUDA crash (cudaErrorIllegalAddress) the RPC never returns;
    # lane30 of rd380 hung for 48 minutes unnoticed that way. A timeout / no reply ends the episode as a failure,
    # and restart_svc gives the next episode a clean service.
    local r; r=$(timeout ${ADV_DEADLINE:-600} $H advance --num-chunks $WINDOW --frames-dir "$1" --tag "$2$acc" $td 2>&1 | tail -1)
    [ -z "$r" ] && { log "  control advance timed out / no reply (${ADV_DEADLINE:-600}s), ending as failure"; echo False; return; }
    local done_=$(echo "$r" | jget done); local ok=$(echo "$r" | jget task_success)
    acc=$((acc+1)); chunks=$((chunks+WINDOW))
    [ "$ok" = "True" ] && { echo True; return; }
    [ "$done_" = "True" ] && { echo False; return; }
    [ $chunks -ge $CTRL_MAX_CHUNKS ] && { echo False; return; }
  done; }

# Quota-limit detection: both CLIs' wording goes here; better to wait an extra round than to waste a trajectory.
# Rate-limit detection only looks at structured fields and never greps the model's text for keywords: during the t1 rerun "429"
# appeared in a passage the judge itself wrote, was mistaken for a rate limit, and 30 minutes were wasted waiting.
quota_blocked(){ python3 "$P/recovery_explore/cli/quota_check.py" "$1" 2>/dev/null; }

# 2026-09-22 (user): Codex CLI (ChatGPT Pro login) first; after CODEX_FALLBACK_AFTER (default 3) failed
# attempts on one call -- empty reply, "at capacity"/network error, or an error object -- the rest of this
# episode goes to the OFFICIAL OpenAI API (cli/gpt6_call.py, same model; key OPENAI_API_TOKEN_FILE).
# A CLI session cannot be resumed over the API, so the first API call gets a rebrief (brief + task parse +
# episode recap, or the intervention memo) and later calls chain on previous_response_id. Next episode
# tries the CLI again. CODEX_FALLBACK_AFTER=0 turns the fallback off.
codex_fallback_due(){
  [ "${CODEX_FALLBACK_AFTER:-3}" -gt 0 ] 2>/dev/null || return 1
  [ "$ENGINE" = codex ] && [ "${CODEX_ALT:-0}" = 0 ] || return 1
  CX_FAILS=$((CX_FAILS+1))
  [ "$CX_FAILS" -ge "${CODEX_FALLBACK_AFTER:-3}" ] || return 1
  CODEX_ALT=1; SID=""; ALT_REBRIEF=1; tries=0; caps=0; NALT=$((${NALT:-0}+1))
  log "  Codex CLI call failed $CX_FAILS times -> rest of this episode uses the official API key (gpt6_call.py, same model, brief resent)"
  return 0; }
alt_rebrief(){ local pf="$1" out="$1.alt"
  if [ "${IN_IV:-0}" = 1 ] && [ -s "$CD/brief_cache.txt" ]; then cp "$pf" "$out"; iv_rebrief "$out"; echo "$out"; return; fi
  { [ -s "$CD/brief_cache.txt" ] && cat "$CD/brief_cache.txt"
    echo "CONTEXT RESET: your earlier conversation in this episode is not available (the client changed)."
    echo "The brief is above; here is what happened so far in this episode."
    echo "Task instruction: $INSTR"
    [ -s "$CD/task_plan.json" ] && { echo "Your task parse:"; cat "$CD/task_plan.json"; echo; }
    build_recap 2>/dev/null
    echo; echo "------------------------------------------------------------"; echo
    cat "$pf"; } > "$out"; echo "$out"; }
claude_turn(){ # $1=prompt file $2=output json $3=deadline $4=optional image $5=allowed tools (default: Read only)
  # Compaction inside an intervention too: previous turn over threshold while intervening -> drop the session and prefix this turn's prompt with the brief + this intervention's memo.
  if [ "${IN_IV:-0}" = 1 ] && [ "${COMPACT_PENDING:-0}" = 1 ] && [ -s "$CD/brief_cache.txt" ]; then
    SID=""; COMPACT_PENDING=0; NCOMPACT=$((NCOMPACT+1)); COMPACTED_W=1
    iv_rebrief "$1"
    log "  window $W mid-intervention compaction: new session, brief + intervention memo resent"
  fi
  # v7 only ever grants Read (to view images); interventions come back as JSON plans executed server-side; only the learn turn adds Write.
  # A privilege audit found that a judge given Glob/Grep would crawl the whole repo
  # and happen to read truth.json from other run directories. The working directory is also the episode dir now, not the repo root.
  local tools="${5:-Read}"
  local tries=0 rc=0 caps=0; CX_FAILS=0
  # Fail on the spot if the prompt file is missing or empty. Otherwise `claude -p ""` reports on resume
  # "No deferred tool marker found", the empty result is retried three times, and a filename typo kills the whole lane.
  if [ ! -s "$1" ]; then
    log "  prompt file missing or empty: $1 (no call made)"
    printf '{"result":"","error":"missing prompt %s"}' "$1" > "$2"
    return 2
  fi
  while :; do
    if [ "$ENGINE" = "api" ] || { [ "$ENGINE" = codex ] && [ "${CODEX_ALT:-0}" = 1 ]; }; then
      local _pf="$1"
      if [ "${ALT_REBRIEF:-0}" = 1 ]; then _pf=$(alt_rebrief "$1"); ALT_REBRIEF=0; fi
      # Talk to the Responses API directly, bypassing the codex CLI. Images and sessions are kept: --image goes as multimodal
      # input, --resume via previous_response_id, so this channel is equivalent to the codex branch;
      # the only difference is "which client it goes through" -- which is exactly the variable under test.
      ( cd "$CD" && run_with_deadline "$3" /dev/null \
          python3 "$P/recovery_explore/cli/gpt6_call.py" \
            --prompt "$_pf" --out "$2" --model "$MODEL" --effort "$REASONING" \
            --timeout "$3" ${SID:+--resume "$SID"} ${4:+--image "$4"} ); rc=$?
    elif [ "$ENGINE" = "qwen" ]; then
      # Local Qwen (vLLM on a local GPU host): same interface as the api branch; the session continues via a local transcript file.
      ( cd "$CD" && run_with_deadline "$3" /dev/null \
          python3 "$P/recovery_explore/cli/qwen_call.py" \
            --prompt "$1" --out "$2" --model "$MODEL" --effort "$REASONING" \
            --think "${QWEN_TURN_THINK:-$QWEN_REPAIR_THINK}" \
            --timeout "$3" ${SID:+--resume "$SID"} ${4:+--image "$4"} ); rc=$?
    elif [ "$ENGINE" = "codex" ]; then
      local txt="${2%.json}.out" ev="${2%.json}.jsonl"
      local args=()
      if [ -n "$SID" ]; then
        # Note: the resume subcommand has no -C/--cd; passing it gives a usage error and an empty output file,
        # and downstream treats an empty result as "ok". The working directory is handled by the cd in run_with_deadline.
        args=(exec resume "$SID" "$(cat "$1")" --skip-git-repo-check
              --dangerously-bypass-approvals-and-sandbox -m "$MODEL"
              -c "model_reasoning_effort=$REASONING" -o "$txt" --json)
      else
        args=(exec "$(cat "$1")" --skip-git-repo-check
              --dangerously-bypass-approvals-and-sandbox -s danger-full-access -m "$MODEL"
              -c "model_reasoning_effort=$REASONING" -C "$CD" -o "$txt" --json)
      fi
      [ -n "${4:-}" ] && [ -f "${4:-}" ] && args+=(-i "$4")
      rm -f "$txt"
      local t0=$(date +%s)
      ( cd "$CD" && run_with_deadline "$3" "$ev" codex "${args[@]}" ); rc=$?
      local t1=$(date +%s)
      # codex does not report duration_ms the way claude does, so time it here;
      # otherwise comparing per-call latency across the three models later would mean back-computing from episode wall clock.
            # The compaction threshold applies to codex too, provided the event stream has turn.completed usage:
      # if present, map it to claude's field names in usage (cached_input_tokens -> cache_read);
      # without usage nothing is written, ctx below comes out 0, and the codex path behaves as before.
      python3 - "$txt" "$ev" "$2" "$(( (t1 - t0) * 1000 ))" << 'PY'
import json, sys, os
txt, ev, out, ms = sys.argv[1:5]
result = open(txt).read() if os.path.exists(txt) else ""
tid = ""; usage = None
if os.path.exists(ev):
    for line in open(ev):
        try: d = json.loads(line)
        except Exception: continue
        tid = d.get("thread_id") or tid
        if not result and d.get("type") == "error":
            result = json.dumps(d)[:2000]
        u = d.get("usage") if d.get("type") == "turn.completed" else None
        if isinstance(u, dict) and "input_tokens" in u:
            cached = int(u.get("cached_input_tokens") or 0)
            usage = {"input_tokens": max(int(u.get("input_tokens") or 0) - cached, 0),
                     "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached,
                     "output_tokens": int(u.get("output_tokens") or 0)}
rec = {"result": result, "session_id": tid, "duration_ms": int(ms)}
if usage: rec["usage"] = usage
json.dump(rec, open(out, "w"), ensure_ascii=False)
PY
    else
      local args=(-p "$(cat "$1")" --model $MODEL --output-format json \
                  --allowedTools "$tools")
      # No --permission-mode bypassPermissions: on the cluster claude runs as root, and that flag is
      # rejected outright ("cannot be used with root/sudo privileges"), so every turn would emit only that error.
      # In non-interactive -p mode the tools in --allowedTools never prompt anyway, which is enough.
      [ -n "$SID" ] && args+=(--resume "$SID")
      ( cd "$CD" && run_with_deadline "$3" "$2" claude "${args[@]}" ); rc=$?
    fi
    # Treat every empty result as a failure and retry: when the judge says nothing, downstream defaults to ok,
    # i.e. the whole trajectory goes unsupervised without anyone noticing (that is how luna's first run was wasted).
    local got; got=$(python3 -c "import json,sys;print(len((json.load(open(sys.argv[1])).get('result') or '').strip()))" "$2" 2>/dev/null || echo 0)
    if [ "${got:-0}" -lt 5 ] && codex_fallback_due; then sleep 2; continue; fi
    if [ "${got:-0}" -lt 5 ] && [ $tries -lt 3 ]; then
      # Drop the session before retrying. The most common cause of an empty result is a broken --resume itself
      # ("No deferred tool marker found in the resumed session"); retrying with the same
      # session id just fails the same way again, and three failures kill the whole lane.
      tries=$((tries+1)); SID=""
      log "  judge returned empty, session dropped, retry $tries"; sleep 10; continue
    fi
    # When the CLI returns an error object instead of a verdict, result is non-empty and passes the length check;
    # downstream cannot parse a verdict and defaults to ok -- the trajectory goes unsupervised while the scores look normal.
    # That is how rstage1 was wasted: all 104 judge calls were token_revoked, and 16 trajectories were never looked at.
    # Retrying does not help with these (credential problems), so kill the lane and make it shout in the log.
    if python3 - "$2" << 'PYERR'
import json, sys
try:
    r = (json.load(open(sys.argv[1])).get("result") or "")
except Exception:
    sys.exit(1)
try:
    obj = json.loads(r)
except Exception:
    sys.exit(1)
sys.exit(0 if isinstance(obj, dict) and obj.get("type") == "error" else 1)
PYERR
    then
      # Capacity / rate-limit errors are transient (gpt-6-astra "Selected model is at capacity" once killed
      # all 20 lanes of g408); back off and retry instead of killing the lane; only credential errors abort.
      if python3 - "$2" << 'PYCAP'
import json, sys
try: r = (json.load(open(sys.argv[1])).get("result") or "")
except Exception: sys.exit(1)
m = ""
try:
    o = json.loads(r)
    if isinstance(o, dict): m = str(o.get("message") or o.get("error") or "")
except Exception: m = r
m = m.lower()
sys.exit(0 if any(k in m for k in ("at capacity", "rate limit", "429", "overloaded",
                                   "temporarily", "try again", "timeout", "unavailable",
                                   "503", "502", "500",
                                   # at 48-way concurrency the proxy also drops streams; these are transient too
                                   "reconnect", "stream disconnected", "tls handshake",
                                   "connection", "eof", "broken pipe", "reset by peer",
                                   "network", "socket")) else 1)
PYCAP
      then
        if codex_fallback_due; then continue; fi
        if [ $caps -lt ${CAP_TRIES:-30} ]; then
          caps=$((caps+1)); SID=""
          log "  CLI reports capacity/rate limit, waiting ${CAP_WAIT:-90}s before retry $caps; raw: $(python3 -c "import json,sys;print((json.load(open(sys.argv[1])).get('result') or '')[:160])" "$2" 2>/dev/null)"
          sleep ${CAP_WAIT:-90}; continue
        fi
        log "  CLI capacity error persists after $caps retries, aborting this lane"
      fi
      if codex_fallback_due; then continue; fi
      log "  CLI returned an error object instead of a verdict, aborting this lane; raw:"
      python3 -c "import json,sys;print((json.load(open(sys.argv[1])).get('result') or '')[:300])" "$2" | tee -a $OUT/lane$LANE/driver.log
      exit 3
    fi
    if [ "${got:-0}" -lt 5 ]; then
      # Three empty results in a row: stop here. Defaulting to ok means the whole episode goes unsupervised while the results look normal.
      # That is what happened in the smoke test -- each turn was a single CLI error line, and all 15 windows were "judged ok".
      log "  judge returned empty 3 times in a row, aborting the whole lane; raw output:"; head -c 400 "$2" | tee -a $OUT/lane$LANE/driver.log
      exit 3
    fi
    if quota_blocked "$2" && [ $tries -lt $QUOTA_TRIES ]; then
      tries=$((tries+1))
      log "  quota limited, waiting $((QUOTA_WAIT/60)) min before retry $tries"
      sleep $QUOTA_WAIT; SID=""; continue
    fi
    break
  done
  local sid; sid=$(python3 -c "import json;print(json.load(open('$2')).get('session_id',''))" 2>/dev/null)
  [ -n "$sid" ] && SID="$sid"
  # Context compaction: the context the model saw this turn = input + cache_creation + cache_read.
  # Over the threshold we only set a flag (COMPACT_PENDING); the session is actually dropped at the start of the next window --
  # intervention turns after the verdict in the same window still need this session, and swapping it mid-way would lose the brief.
  # Note: top-level usage is the sum over the whole tool loop (one 32-turn intervention in the smoke test summed to 870k);
  # the real "current context" is what the last message saw -- usage.iterations[-1] (falling back to top level).
  local ctx; ctx=$(python3 -c "import json;u=json.load(open('$2')).get('usage') or {};it=u.get('iterations') or [];u=it[-1] if it else u;print(sum(int(u.get(k) or 0) for k in ('input_tokens','cache_creation_input_tokens','cache_read_input_tokens')))" 2>/dev/null || echo 0)
  if [ "${ctx:-0}" -gt "$COMPACT_AT" ]; then
    if [ "${IN_IV:-0}" = 1 ]; then log "  window $W context ${ctx} tokens exceeds $COMPACT_AT, compacting next turn: new session"
    else log "  window $W context ${ctx} tokens exceeds $COMPACT_AT, compacting next window: new session"; fi
    COMPACT_PENDING=1
  fi
  return $rc; }

# Recap after compaction: generated from $EVENTS, one line per past window, capped at 40 lines.
# Structured memo after an intervention: what was done last time, what was expected, what to answer first now.
# Rebuilt every window, so it survives context compaction (compaction happened 98 times across 408 episodes).
# Prompt prefix for mid-intervention compaction: brief (incl. few-shot) + task parse + this window's verdict and expectation + a record of each turn of this intervention.
iv_rebrief(){ local pf="$1" tmp="$1.rebrief"
  { cat "$CD/brief_cache.txt"
    echo "CONTEXT RESET IN THE MIDDLE OF AN INTERVENTION. Your earlier conversation was dropped to keep the"
    echo "context short. The brief is repeated above; what you need about this intervention is below."
    echo "Task instruction: $INSTR"
    [ -s "$CD/task_plan.json" ] && { echo "Your task parse:"; cat "$CD/task_plan.json"; echo; }
    python3 - "$CD/judge_w$W.json" "$W" << 'PYRB' 2>/dev/null
import json, re, sys
try:
    r = json.load(open(sys.argv[1])).get("result") or ""
    o = json.loads(re.search(r"\{.*\}", r, re.S).group(0))
except Exception:
    o = {}
if o.get("diagnosis"): print("At window " + sys.argv[2] + ", where this intervention started, your diagnosis was: " + str(o["diagnosis"])[:1200])
if o.get("expect"): print("You said you expected to see: " + str(o["expect"])[:600])
PYRB
    echo "Intervention $NINT of this episode, attempt $ATT, $((MAX_INTERVENTIONS - NINT)) intervention(s) left after this one."
    python3 $P/recovery_explore/cli/attempt_ledger.py "$CD" "$W" "$ATT" 2>/dev/null
    python3 - "$CD" "$W" "$ATT" << 'PYCUR' 2>/dev/null
import glob, json, re, sys
cd, w, a = sys.argv[1:4]
fs = sorted(glob.glob(f"{cd}/plan_w{w}_a{a}_r*.json"), key=lambda f: int(re.search(r"_r(\d+)\.json$", f).group(1)))
if fs:
    print(f"THE CURRENT ATTEMPT ({a}) SO FAR -- not rewound, the scene is where these segments left it:")
    for f in fs:
        r = re.search(r"_r(\d+)\.json$", f).group(1)
        try: plan = json.load(open(f))
        except Exception: plan = []
        try: ex = json.load(open(f"{cd}/exec_w{w}_a{a}_r{r}.json"))
        except Exception: ex = {}
        ops = " | ".join(str(st.get("op")) + (" " + str(st.get("arm")) if st.get("arm") else "") for st in plan if isinstance(st, dict))
        print(f"  segment {int(r)+1}: {ops} -> stopped because: {ex.get('stop_reason')}")
PYCUR
    echo; echo "------------------------------------------------------------"; echo
    cat "$pf"; } > "$tmp" && mv "$tmp" "$pf"
}

build_iv_block(){
  IV_BLOCK=""
  [ -n "${IV_LAST_W:-}" ] || return 0
  local head="WHAT YOU DID AT WINDOW $IV_LAST_W: ${IV_MEMO:-(not recorded)}"
  local exp="WHAT YOU SAID YOU EXPECTED TO SEE NOW: ${IV_EXPECT:-(you did not state one)}"
  if [ "${VERIFY_PENDING:-0}" = 1 ]; then
    IV_BLOCK="$head
$exp
THIS IS A VERIFICATION WINDOW. You intervened in the window just before this one, so
you may not intervene again here. Open your diagnosis by answering, from the numbers
and the image: did that expectation hold -- yes, no, or cannot tell yet -- and on what
evidence. Then answer ok. If it did not hold, say plainly what the repair failed to
achieve; you get to act again in the window after this one, and that answer is what
you will have to build on."
  else
    IV_BLOCK="$head
$exp
Before anything else, say whether that expectation held. If it did, do not repair the
same thing twice. If it did not, a further intervention has to change category, not offset."
  fi
}

build_recap(){ MAX_RESETS="$MAX_RESETS" python3 - "$EVENTS" "$NINT" "$MAX_INTERVENTIONS" "$GAVEUP" << 'PY'
import json, os, sys
ev, nint, maxint, gaveup = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
resets = sum(int(e.get("resets") or 0) for e in ev)
lines = []
for e in ev:
    ln = f"- window {e['window']}: {e.get('committed', 0)} steps committed, judge {e.get('judge', '?')}"
    if e.get("judge") == "intervene" and int(e.get("attempts") or 0) > 0:
        ln += (f"; intervention: {e.get('attempts')} attempt(s), {e.get('actions_used', 0)} actions used, "
               f"{e.get('resets', 0)} reset(s), result {e.get('result') or '?'}")
    lines.append(ln)
LIMIT = 34
if len(lines) > LIMIT:
    lines = [f"- ({len(lines) - LIMIT} earlier windows omitted)"] + lines[-LIMIT:]
print("Recap of this episode so far (your earlier session was compacted):")
print("\n".join(lines) if lines else "- (no windows judged yet)")
print(f"Interventions so far: {nint} of {maxint}; gave up: {'yes' if gaveup == '1' else 'no'}; resets used: {resets} of {os.environ.get('MAX_RESETS', '5')}.")
PY
}

# Output spec for the judging turn. Verdict and plan come in one call: ok carries no plan, intervene
# comes with <=8 primitives directly, saving the second round trip of "ask for a verdict, then ask for a plan".
PLAN_SPEC=$(cat << 'SPEC'
Answer with ONE json object and nothing else (no prose around it, no code fence):

{"verdict": "ok" | "intervene" | "giveup",
 "diagnosis": "one or two sentences on what you see",
 "expect": "when verdict is intervene: one sentence naming the observable you expect in the NEXT window if this repair worked -- a number or a visible fact, not a hope. Example: gripper width stays above 20 mm while the arm lifts, or the kettle leaves the burner. Omit for ok and giveup.",
 "plan": []}

- verdict "ok": the policy is coping. plan MUST be empty. Most windows are ok.
- verdict "intervene": something is wrong that you can fix. plan is 1 to 8 steps,
  executed IN ORDER by the simulator without asking you again, so write the whole
  repair, not just its first move.
- verdict "giveup": beyond rescue. plan MUST be empty. This is final for the episode.

Each step is one of:
  {"op": "gripper", "state": "open" | "close"}
  {"op": "move_to", "target": {"xyz": [x, y, z]}, "gripper": "hold" | "open" | "close"}
  {"op": "move_to", "target": {"pixel": [row, col], "camera": "primary" | "secondary" | "wrist", "dz": 0.05}}
  {"op": "nudge",  "dxyz": [dx, dy, dz], "gripper": "hold"}
  {"op": "lift",   "dz": 0.08, "gripper": "hold"}
  {"op": "rotate", "drot": [rx, ry, rz], "gripper": "hold"}
  {"op": "policy", "chunks": 2}
  {"op": "retreat"}
  {"op": "retreat", "to": "pre_grasp"}

Three rules the simulator enforces on every plan (a plan that breaks one is rejected
whole, nothing runs, and the error names the step): on a pick-and-place task a manual
close is refused until a policy hand-back in this intervention has failed, and any
manual close must be followed by a lift in the same plan; on microwave, coffee-machine
and stove tasks a move_to farther than 6 cm from the current pose is refused; and once
a step in this intervention has missed, overshot or been clipped, only gripper and
policy steps are accepted for the rest of it.

retreat with "to": "pre_grasp" goes somewhere better than the takeover pose: the
point on the approach the policy itself took, just before its last empty closure,
with the fingers opened. The telemetry names it when it exists. From there a policy
hand-back of two chunks lets the policy redo the grasp on its own -- that is the
default repair for a missed grasp: look at the object first (the miss may have
moved it), then {"op": "retreat", "to": "pre_grasp"}, then {"op": "policy", "chunks": 2}.
If the object is no longer where the policy was reaching (it slid, tipped or rolled),
the pre-grasp point is aimed at the old position: aim the OPEN fingers at the current
position of the object instead, with a pixel move_to (dz 0.05 above it), then hand back.
Never send the same repair twice for the same missed grasp: if a repair was followed by
another empty closure, the next one must come in differently (pixel re-aim on the
current position of the object, or rotate the wrist about the vertical axis by about 0.4 rad
before the hand-back, or grasp yourself as in ESCALATE AFTER TWO FAILED GRASPS).

retreat without "to" is the soft undo: one action that servos the hand back to the pose it had
when you took over and puts the fingers back the way they were, dropping anything
your repair picked up. Nothing else in the scene is touched. Execution stops after
it so you can look. Use it when a repair has failed and you want to hand the policy
back what it had, or before a second attempt on a different line. It is a Cartesian
move like any other, so from a configuration that has just blown up it can blow up
too; then open the fingers and stop.

The policy step hands control back to the robot policy for that many chunks (1 or 2,
and at most 4 chunks in total across one intervention),
starting from wherever the arm is now, then stops and shows you what happened. It
costs one action per chunk. Use it once you have put the hand somewhere workable:
the policy is usually better at the last few centimetres of a grasp than a string
of small moves you write, and it is the only step here that can finish the task.
Where it sits in the plan is your call (2026-09-21): a plan may open with it and two
hand-backs may sit next to each other. Buying each hand-back with a repair of your own
is still the better habit -- the controller resumes from the pose that was already
failing -- but the shape is no longer refused.
Each chunk handed back is taken from the fixed policy-step budget of the episode, which
the policy needs to finish the task; your other steps are not. Hand back once, at
the end of the repair, for one or two chunks -- not as a way to see what the policy
will do.

A pixel target is unprojected server-side and then offset by dz metres, so you can
point at what you see instead of computing world coordinates. Pick pixels only from
the newest images you were given, and convert to the native frame of that camera first:
every camera renders at 512 x 512 and row/col are in that frame. The per-window
pictures are 256 x 256 copies, so double what you read off them. The checkpoint
picture is the three native 512-wide tiles side by side -- subtract 0, 512 or 1024
from the column for primary, secondary or wrist, and keep the row as is. A target
that unprojects far from where you pointed is almost always this conversion missed.

Execution stops early and hands back to you when: the fingers close on nothing
(width under 12 mm), a step travels several times the distance you asked for
(overshoot: the controller is misbehaving in that arm configuration), a servo does
not reach its target, a move is clipped because it was too far, wrist contact force
spikes, the action budget or the policy hand-back allowance runs out, or a `lift`
completes -- that is the checkpoint where you look at whether the object came up
with the hand. You then get the step-by-step log and fresh images.
SPEC
)

# RoboDojo is bimanual (ARX X5 x2): every action names which hand moves, the other holds still. There is no wrist force
# sensor, and the gripper reports a 0..1 normalised opening rather than millimetres; there is no pre_grasp point. These differences
# are stated only in this patch; the PLAN_SPEC body is shared by both families -- so lessons the judge learns can transfer across benchmarks.
# robotwin (RoboTwin 2.0, aloha-agilex) takes the same PLAN_SPEC filter -- the robotwin
# intervention mixin mirrors the RoboDojo one: no pixel targets, no pre_grasp anchor, no force
# sensor, normalised gripper opening -- and gets its own bimanual addendum below.
if [ "$FAMILY" = robodojo ] || [ "$FAMILY" = robotwin ]; then
  # First strip three things from PLAN_SPEC that do not exist on RoboDojo, then append the addendum. Keeping them is worse than removing them:
  # the judge writes plans exactly as the prompt says, and they fail every time. That is how the first real intervention was lost --
  # Qwen followed the spec to the letter with {"target": {"pixel": ...}}, two plan segments in a row errored, and the episode was judged giveup.
  #   1. Pixel targeting: needs a depth map. RoboDojo's arx_x5.yml sets depth / approximate_depth /
  #      intrinsic_matrix all to false, and the depth annotator in camera_config.yml is commented out.
  #      Enabling it changes the benchmark environment, and the control arm's score could no longer be compared to the official
  #      leaderboard -- both arms must share the environment, so it stays off. The judge aims with relative offsets instead, which also proved steadier on RoboCasa.
  #   2. retreat's pre_grasp anchor: relies on "the approach point before the last empty closure", a state machine RoboCasa
  #      wrote around finger separation in millimetres; this robot has no finger separation.
  #   3. Wrist contact-force abort: this robot has no force sensor.
  PLAN_SPEC=$(printf '%s' "$PLAN_SPEC" | awk '
    BEGIN { RS = ""; ORS = "\n\n" }
    /A pixel target is unprojected/ { next }
    /retreat with "to": "pre_grasp"/ { next }
    {
      # The trailing \n? is required: in paragraph mode the last line has no newline, and the pre_grasp entry
      # happens to be the last line of the primitive list. With a hard-coded \n it always stayed in the list.
      gsub(/[ ]*\{"op": "move_to", "target": \{"pixel"[^\n]*\n?/, "")
      gsub(/[ ]*\{"op": "retreat", "to": "pre_grasp"\}\n?/, "")
      # In the original this sentence wraps as "wrist contact force\nspikes,"; a single-line pattern never matches.
      gsub(/wrist contact force\nspikes, /, "")
      # On this robot an empty grasp is judged by the normalised opening, not finger separation in millimetres.
      gsub(/\(width under 12 mm\)/, "(the opening comes back at or below 0.05)")
      print
    }')
  if [ "$FAMILY" = robotwin ]; then
  # robotwin only: the shared "Three rules" paragraph talks about pick-and-place /
  # microwave / stove; the robotwin gate rules are restated (with examples) below.
  PLAN_SPEC=$(printf '%s' "$PLAN_SPEC" | awk 'BEGIN { RS = ""; ORS = "\n\n" } /^Three rules the simulator enforces/ { next } /^Execution stops early/ { next } { print }')
  RD_ADDENDUM=$(cat << 'RTSPEC'

THIS ROBOT IS BIMANUAL (aloha-agilex: two 6-DoF arms with parallel grippers).
Every step except "policy" takes an "arm": "left" | "right" field; omit it and
the step goes to the arm the telemetry names as the one your steps drive by
default. The other arm holds its joints exactly where they are -- including
whatever it is holding. The "state" block lists left_ee_pose and right_ee_pose
as [x, y, z, qw, qx, qy, qz] in the world frame.
There is no force or contact sensing on this robot, so execution never stops
for contact; and the gripper reports a normalised opening from 0 (shut) to 1
(open), not a width in millimetres -- "closed on nothing" means the opening
came back at or below 0.05 after a close.
MOVING THE ARM (2026-09-22). Every move (move_to, nudge, lift, rotate, retreat, move_both) is
planned ONCE by the robot motion planner from the current joints to the target and the whole
path is executed; the reply then says where the fingertips actually ended up (final_dist_m, eef).
If the residual is under 2 cm the plan goes on; otherwise the plan stops with move_residual so you
can look again -- nothing is locked, your next plan may move the arm again, from where it is now.
A target the planner cannot reach at all stops with plan_failed. Every control step of a repair
now counts against the step limit of the episode, exactly like the steps of the policy.

TWO-ARM TASKS. Many tasks here need both hands: a handover (one hand gives,
the other takes), a lift with both hands on one object, or each hand placing
its own object. Before you move anything, decide from the pictures which arm
owns which part of the task at this moment. Never open a hand that is holding
something the other hand has not yet closed on. A handover fails most often
because the receiving hand closes too early, too far from the object, or
before the giving hand has stopped moving.

Two more step types move BOTH hands in the same control steps:
  {"op": "move_both", "left": {"dxyz": [dx, dy, dz]} or {"xyz": [x, y, z]},
   "right": {...}, "gripper_left": "hold", "gripper_right": "hold"}
  {"op": "lift_both", "dz": 0.08, "gripper": "hold"}
Use them whenever both grippers hold the same object: lifting a pot by its two
handles, carrying a jointly held object, or bringing two hands together for a
handover. A one-arm lift there drags the object against the other hand and
stalls or fails; with the same dxyz for both hands the grip is kept. Example,
pot held by both handles: {"op": "lift_both", "dz": 0.08, "gripper": "hold"}.

HOW TO AIM ON THIS ROBOT. Small corrections: RELATIVE. Grasping an object yourself: a BOX
(or a PIXEL) on the object -- the camera depth then says where the object really is.

  * BOX (preferred), with move_to and {"bbox": [row0, col0, row1, col1], "camera": ..., "dz": d}:
    a box tightly around the object in one of the 256 x 256 pictures. The simulator back-projects
    every surface pixel inside it from the camera depth and aims at their MEDIAN, which is far
    steadier than one pixel. {"op": "locate", "bbox": [...], "camera": ...} runs the same depth
    query without moving (no action spent) and reports median_xyz, the 3-D extent of the box and
    its top_z, so you can plan the next segment from real numbers.

  * PIXEL, with move_to and {"pixel": [row, col], "camera": ..., "dz": d}. Point
    at the centre of the object in one of the 256 x 256 camera pictures you were given
    ("primary" = head, "secondary" = left wrist, "wrist" = right wrist; in a
    side-by-side picture subtract 0, 256 or 512 from the column). The simulator
    looks up the 3D surface point under that pixel and sends the FINGERTIP CENTRE
    to it, raised by dz metres. Grasp from above: first dz 0.05 with the fingers
    open, then a nudge down of about 4-5 cm, then close. The reply gives the 3D
    point it found, so you can check it landed on the object and not the table.

  * RELATIVE, with nudge and lift. "The hand is 3 cm short and 2 cm to the left
    of the object" becomes {"op": "nudge", "dxyz": [0.02, 0.03, 0.0]}. This
    keeps the wrist orientation the policy chose and only corrects the offset,
    so the hand stays in a pose the policy can carry on from. Read the offset
    off the pictures: you are given the head view and both wrist views, and the
    wrist view of the working hand shows what that hand is about to touch.
  * ABSOLUTE, with move_to and an explicit {"xyz": [x, y, z]} in world metres.
    The xyz is where the FINGERTIP CENTRE goes. In the window telemetry,
    "End-effector is now at" is the wrist point, 12 cm behind the fingertips
    along the direction the gripper points; the executed-step log and the
    re-stage point give fingertip positions. When in doubt, use a relative move.

Give the real target: any move is one planned path in one step (one action), however far. The
reply says where it actually got.

Execution stops early and hands back to you when: the fingers close on nothing (the opening
comes back at or below 0.05), a planned move ends more than 2 cm from its target
(move_residual) or no path to it exists (plan_failed), the action budget or the policy hand-back
allowance runs out, the step limit of the episode is reached, or a lift completes -- that is the
checkpoint where you look at whether the object came up with the hand. You then get the
step-by-step log and fresh images.

PLAN SHAPES ON THIS ROBOT -- this section overrides anything earlier in the brief that
says a plan is "rejected whole". Nothing here is a rejection any more (2026-09-21): the simulator repairs a plan
it cannot run as written and tells you what it changed in the execution report. You decide the
shape -- a plan may start with a policy step, may hand back twice in a row, and may close the
fingers yourself in its very first repair when that is what the scene calls for. What the
simulator still does for you:
  * a manual close gets a verifying lift appended if you did not write one, so a grasp is always
    checked by seeing whether the object comes up;
  * a plan identical to one that already failed in this intervention is raised by one centimetre
    so the retry is at least different -- but a centimetre is rarely the difference, so change the
    approach, the arm, or who makes the close;
  * steps beyond the actions left in this round are cut off rather than failing the whole plan.
Read the "rewritten" lines of the execution report: they say exactly what the simulator changed.


INTERVENE EARLY. When a clear, fixable fault is visible -- the wrong object
grasped, a grasp missed with the fingers shut on nothing on the arm that owns
the grasp, an object knocked over or dropped, an arm jammed against the table or
another object (large joint error) for two or more chunks -- intervene in this window.
Waiting until the last windows leaves too few policy steps for the hand-back to
finish the task, and giving up late wastes the whole budget.

THE GRASP, DONE PROPERLY (user, 2026-09-21). When you take a grasp yourself, plan the whole thing
first and write it as ONE plan, in this order -- do not creep in with one- and two-centimetre moves:
  1. move the hand to directly ABOVE the object, a few centimetres up, squared to it (a pixel
     move_to on the centre of the object with dz 0.05 is the usual way);
  2. OPEN the fingers wide -- wider than the object; a small object that shifts when touched is
     knocked away by fingers that are only half open;
  3. if the wrist is not square to the object, rotate it so the fingers straddle the narrow side
     of the object;
  4. descend STRAIGHT DOWN with a nudge of about 4-5 cm;
  5. close;
  6. lift a few centimetres and check in the pictures that the object came up with the hand.
Only move on after step 6 shows a hold. If it did not come up, rewind and change step 1 or 3
(position or orientation), not the depth by a centimetre. The simulator puts an "open" at the
start of a plan that closes a hand that is not open.

THE PLACE, DONE PROPERLY. Releasing is the mirror of grasping: carry the object to directly ABOVE
the destination, check in the head view (and the wrist view of the working hand) that it is squarely
over it, descend, and only then open the fingers; then lift away. Opening before the object is
over the destination drops it beside the target.

LOST FROM VIEW IS A FAULT. If the object, or the working hand, has left the head camera picture --
lifted too high, pushed to the edge, knocked off to one side -- do not wait for it to come back.
Intervene: find it in the other cameras first (the other wrist view often still shows it), then
move the arm so that the head view and the wrist view of the working hand see the target again, and
continue from there. When the task needs the hand pointed at something (scanning, aiming), use the
wrist view to tell which way to turn: the target should move toward the centre of that picture.

NAME THE OBJECTS BEFORE YOU JUDGE. In your diagnosis, say which thing in the picture is
the object the instruction names and which thing is the destination it must end up on or
in (the mat, the plate, the stand, the big bin rather than the small one). The policy
can run a clean-looking trajectory to the WRONG thing: an object set down beside its mat
instead of on it, or dropped into the wrong bin. That is a fault worth repairing even
though nothing is stuck and no grasp was missed -- pick the object back up and place it
on the destination you named. If you cannot tell the two candidates apart in the head
view, look at the wrist view of the working hand before you answer.

ONE OBJECT, TWO HANDS. Some instructions mean both hands act on the SAME object at the
same time: a pot lifted by its two handles, a bin tipped into another bin, a wide or
heavy object, a block handed from one hand to the other. If the policy is working one
hand at a time there -- one hand pushing or lifting alone while the other waits -- the
semantics are wrong, not just the timing. Repair it with the both-hand steps
(move_both / lift_both) so the two hands move together, then hand back.

A DROPPED OR TOPPLED OBJECT IS A FAULT. If the object falls out of a hand (a handover
where the receiving hand never really held it), or the arm knocks a neighbouring object
over on its way in, intervene: the object is now somewhere else, in a different
orientation, and the policy will keep reaching for where it used to be. Re-aim at where
it lies NOW (a pixel move_to on its current position) and grasp it in its new
orientation -- a bottle on its side is grasped across the body, not around the cap. Do
not treat "the object moved" as a reason to give up.

WHEN NOT TO INTERVENE. Past episodes on this robot show that repairs made while
the policy was still coping usually broke the episode. Answer ok when:
  * a hand hovers OPEN and still near or above its target, or waits while the
    other hand works -- the policy is lining up, it has not missed anything yet;
  * the task is to press, stamp, click or push something (a stapler, a bell, a
    seal, a switch): the policy closes the gripper on purpose and presses with
    the closed fingers, so a shut hand with nothing in it is NOT a missed grasp;
  * the empty closure is on the idle arm, not on the arm that owns the grasp. In a
    two-hand task judge only the hand the telemetry says is doing the work RIGHT NOW:
    the other hand closing on nothing while its partner carries the object is not a fault;
  * the working hand has missed once: the policy usually re-opens and tries again, so
    give it two windows to retry by itself before you step in;
  * the hand is still travelling toward the object, however slowly, or has paused or
    held still for fewer than four windows -- slow is not stalled.
If you cannot name the fault in one sentence and point at it in the pictures,
answer ok and look again in the next window.

ESCALATE AFTER TWO FAILED GRASPS. Re-opening the fingers and handing back from
almost the same pose usually reproduces the same miss. Once the telemetry lists two or more empty closures in this EPISODE
(any windows), or two hand-backs in this intervention ended with that hand shut on
nothing (a rewind does NOT reset either count), stop handing back for the close and
grasp yourself -- starting with the first plan of this intervention: aim the OPEN fingers at the object with a PIXEL move_to (see HOW
TO AIM: dz 0.05 above it, then a nudge down), close ({"op": "gripper",
"state": "close", "arm": ...}) and lift ({"op": "lift", "dz": 0.05, "arm": ...}) in
ONE plan, then hand back 1-2 chunks so the policy carries on from a verified hold. Never send the same plan shape
a third time. The execution report says per hand "SHUT ON NOTHING (empty hand)",
"open" or "partly closed": an opening of 0.00 is an EMPTY hand, never a grasp.
RTSPEC
)
  else
  RD_ADDENDUM=$(cat << 'RDSPEC'

THIS ROBOT IS BIMANUAL (two ARX X5 arms). Every step except "policy" takes an
"arm": "left" | "right" field; omit it and the step goes to the right arm. The
other arm holds its joints exactly where they are. The "state" block lists
left_ee_pose and right_ee_pose as [x, y, z, qw, qx, qy, qz] in the world frame.
There is no wrist force sensor on this robot, so execution never stops for
contact; and the gripper reports a normalised opening from 0 (shut) to 1
(open), not a width in millimetres -- "closed on nothing" means the opening
came back at or below 0.05 after a close.
Cartesian moves are solved by inverse kinematics from the current arm
configuration: a target the IK cannot reach stops the plan with servo_missed,
exactly like a missed servo, and disables further Cartesian steps for this
intervention.

HOW TO AIM ON THIS ROBOT. There is no pixel targeting here: this benchmark's
cameras deliver colour only, so the simulator cannot turn a pixel into a 3D
point. You aim two ways instead, and the first is almost always the right one:

  * RELATIVE, with nudge and lift. "The hand is 3 cm short and 2 cm to the left
    of the object" becomes {"op": "nudge", "dxyz": [0.02, 0.03, 0.0]}. This
    keeps the wrist orientation the policy chose and only corrects the offset,
    so the hand stays in a pose the policy can carry on from. Read the offset
    off the pictures: you are given the head view and both wrist views, and the
    wrist view of the working hand shows what that hand is about to touch.
  * ABSOLUTE, with move_to and an explicit {"xyz": [x, y, z]} in world metres.
    Both hands' world positions are printed in the telemetry every window, so
    you can compute a target relative to where a hand already is. Use this to
    go back somewhere the arm has been, not to reach somewhere new: an absolute
    move to a pose the policy has never seen is what a hand-back cannot recover
    from.

A single move is capped at 5 cm unless the target is a point this arm has
already occupied, so plan a long correction as two or three steps, and expect
the reply to tell you how far it actually got.
RDSPEC
)
  fi
  PLAN_SPEC="$PLAN_SPEC
$RD_ADDENDUM"
fi

if [ "$ALLOW_RESET" = 1 ]; then
  RESULT_OPTS='"fixed" | "retry" | "giveup" | "continue"'
  RESTAGE_TAIL='it is the cheap first retry. If the scene itself has been disturbed, or a step
travelled several times what you asked, answer retry instead: the simulator rewinds
to the state you took over in.'
  BLOWUP_ACTION='Answer retry: the rewind puts the arm back where you took over. After the rewind,
unless you have a plan that needs no Cartesian move from that region, answer fixed
with an empty plan -- that hands the policy back exactly what it had.'
  RETRY_DOC='- "retry": not fixed; the simulator is rewound to the start of this intervention and
  you plan again from scratch. plan MUST be empty.'
else
  RESULT_OPTS='"fixed" | "giveup" | "continue"'
  RESTAGE_TAIL='it is the only kind of retry that exists -- nothing rewinds the scene, and whatever
you have already moved stays moved. When a repair has failed once, the honest ending
is retreat, then fixed with an empty plan: the policy gets its own pose back and all
the steps it has left. Two attempts from one takeover is the limit; a third small
correction from wherever the hand now is, is how interventions run an episode out.'
  BLOWUP_ACTION='Send one plan containing only {"op": "retreat"}. If that step also travels several
times what it asked for, open the fingers and end the intervention as fixed with a
note saying what happened.'
  RETRY_DOC='There is no rewind in this run: whatever you have already done to the scene stays
done. Finish the repair with further "continue" segments, or stop. An intervention
whose last executed step was a successful retreat, followed by fixed with an empty
plan, hands the policy back exactly what it had and is not counted against your
intervention allowance -- provided the policy was not handed back in between: a
retreat after a hand-back throws away the re-approach the policy just made, and
that intervention counts in full.'
fi

# Extract the single JSON object from the model's reply. The model occasionally adds a code fence or preamble; both are handled.
extract_json(){ python3 - "$1" << 'JSONPY'
import json, re, sys
try: t = json.load(open(sys.argv[1])).get("result") or ""
except Exception: t = ""
m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.S) or re.search(r"(\{.*\})", t, re.S)
if not m:
    print("{}"); raise SystemExit
try: print(json.dumps(json.loads(m.group(1)), ensure_ascii=False))
except Exception: print("{}")
JSONPY
}
jfield(){ python3 -c "
import json,sys
try: d=json.loads(sys.argv[1])
except Exception: d={}
v=d.get(sys.argv[2])
print('' if v is None else (json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v))
" "$1" "$2" 2>/dev/null; }


while true; do
  GOT=$(wc -l < $RESULTS | xargs); [ "$GOT" -ge $TARGET ] && { log "reached $TARGET results, done"; break; }
  NEXT=""
  while read -r spec; do
    [ -z "$spec" ] && continue
    mkdir "$OUT/claims/$spec" 2>/dev/null && { NEXT="$spec"; break; }
  done < $QUEUE
  [ -z "$NEXT" ] && { log "queue exhausted, stopping"; break; }
  TASK=${NEXT%%:*}; REST=${NEXT#*:}; SEED=${REST%%:*}; EP=${REST##*:}
  CELL="jd-${TASK}_s${SEED}_ep${EP}"; CD=$OUT/lane$LANE/$CELL
  # Clear a same-named directory first: when rerunning a cell, leftover w*/ frames from the last run would mix with the new ones,
  # and the judge would see pictures from another episode (this really happened during the t1 rerun).
  # Privileged ground truth goes into a separate tree the judge cannot see ($CD's path is written into the prompt).
  TD=$OUT/_truth/lane$LANE/$CELL
  rm -rf "$CD" "$TD"; mkdir -p $CD $TD
  log "═══ $NEXT ═══ ($ENGINE/$MODEL${REASONING:+/$REASONING} LEARN=$LEARN)"

  # ---------- Control arm: policy alone ----------
  restart_svc || continue
  BEG=$($H episode-begin --task "$TASK" --seed "$SEED" --episode-index "$EP" --obj-instance-split "$SPLIT" 2>&1 | tail -1)
  if [ "$(echo "$BEG" | jget ok)" != "True" ]; then
    # A single failure may just be a service restart; restart the service and retry once, and only then count it as a failed episode-begin.
    log "  episode-begin failed once ($(echo "$BEG" | head -c 160)), restarting service and retrying"
    restart_svc && BEG=$($H episode-begin --task "$TASK" --seed "$SEED" --episode-index "$EP" --obj-instance-split "$SPLIT" 2>&1 | tail -1)
  fi
  [ "$(echo "$BEG" | jget ok)" != "True" ] && { log "  episode-begin failed, skipping: $(echo "$BEG" | head -c 200)"; continue; }
  INSTR=$(echo "$BEG" | jget instruction); TOTC=$(echo "$BEG" | jget total_chunks)
  MAXSTEPS=$(echo "$BEG" | jget max_steps); case "$MAXSTEPS" in ""|None) MAXSTEPS=0;; esac
  if [ -n "$REUSE_CONTROL" ] && [ -f "$REUSE_CONTROL" ]; then
    # The baseline lookup cannot go by cell name alone: the control arm is only reproducible for "the same policy family + the same window granularity".
    # m2astra tripped on granularity (control --num-chunks 8 vs treatment WINDOW=2: different request_id sequences
    # mean different trajectories). If the family or granularity does not match, treat it as not found, run it live, and say so in the log.
    CTRL=$(python3 - "$REUSE_CONTROL" "$CELL" "$FAMILY" "$WINDOW" << 'PY'
import json, sys
want, fam, win = sys.argv[2], sys.argv[3], int(sys.argv[4])
for line in open(sys.argv[1]):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("cell") != want:
        continue
    if d.get("policy_family") != fam:
        print(f"MISMATCH family={d.get('policy_family')}!={fam}"); break
    if int(d.get("window_chunks") or -1) != win:
        print(f"MISMATCH window={d.get('window_chunks')}!={win}"); break
    if d.get("control_success") is None:
        print("MISMATCH control_success=null"); break
    print("True" if d["control_success"] else "False"); break
else:
    print("MISSING")
PY
)
    case "$CTRL" in MISMATCH*) log "  baseline mismatch (${CTRL#MISMATCH }), running control live"; CTRL=MISSING ;; esac
    if [ "$CTRL" = "MISSING" ] && [ "${SKIP_CONTROL:-0}" = 1 ]; then
      log "  no control for this episode in the baseline, SKIP_CONTROL=1: skipping control, treatment arm only"; CTRL=None; CTRL_S=-1
    elif [ "$CTRL" = "MISSING" ]; then
      log "  could not reuse control, running it live"
      _t0=$(date +%s); CTRL=$(run_to_end "$CD/ctrl" c "$TD/ctrl"); CTRL_S=$(( $(date +%s) - _t0 ))
    else
      log "  control (reused baseline): $CTRL"; CTRL_S=-1
    fi
  elif [ "${SKIP_CONTROL:-0}" = 1 ]; then
    # Treatment arm only: control runs in another run (ctrl-only baseline) and is merged by cell afterwards.
    CTRL=None; CTRL_S=-1; log "  SKIP_CONTROL=1: skipping control, treatment arm only"
  else
    _t0=$(date +%s); CTRL=$(run_to_end "$CD/ctrl" c "$TD/ctrl"); CTRL_S=$(( $(date +%s) - _t0 ))
    log "  control (policy alone): $CTRL (${CTRL_S}s)"
  fi

  # Control only: builds a baseline bank for harness experiments. With matched granularity the control arm is fully reproducible
  # (55 int=0 cells measured identical across arms), so one baseline run is enough;
  # later experiments look it up by (family, checkpoint, WINDOW, task, seed, episode).
  # This path calls no CLI and no screener and builds no treatment arm; an episode is just the policy-alone run.
  if [ "${CTRL_ONLY:-0}" = 1 ]; then
    python3 - "$CELL" "$TASK" "$SEED" "$EP" "$CTRL" "$LANE" "$FAMILY" "$WINDOW" "${CTRL_S:--1}" "$INSTR" << 'PY' >> $RESULTS
import json,sys
c,t,sd,e,ctrl,lane,fam,win,cs,instr=sys.argv[1:11]
print(json.dumps({"cell":c,"task":t,"seed":int(sd),"episode":int(e),
 "control_success":ctrl=="True","treatment_success":None,
 "judge_calls":0,"interventions":0,"events":[],
 "window_chunks":int(win),"lane":int(lane),"policy_family":fam,"lookahead":0,
 "compactions":0,"screened_windows":0,
 "ctrl_s":int(cs),"treat_s":-1,"model_s":0.0,"screen_s":0.0,
 "ctrl_only":True,"instruction":instr},ensure_ascii=False))
PY
    log "  control only: $CTRL (${CTRL_S}s)"
    continue
  fi

  # ---------- Treatment arm: same episode, one judge call every WINDOW chunks ----------
  restart_svc || continue
  # The treatment arm's episode-begin result must be checked. It used to be discarded: if env_service had not been replaced by a new process
  # (episode-begin refuses to build a second episode in a process that already holds an env), the treatment arm kept advancing
  # on the control arm's finished env, producing garbage that looked fine.
  BEG2=$($H episode-begin --task "$TASK" --seed "$SEED" --episode-index "$EP" --obj-instance-split "$SPLIT" 2>&1 | tail -1)
  [ "$(echo "$BEG2" | jget ok)" != "True" ] && { log "  treatment-arm episode-begin failed, skipping: $(echo "$BEG2" | cut -c1-200)"; continue; }
  TREAT_T0=$(date +%s)
  SID=""; W=0; NINT=0; JUDGED=0; EVENTS="[]"; TS=False; GAVEUP=0; BADADV=0; GIVEUP_PUSHED=0; EVER_CLOSED=0; NREFUND=0; FIRSTMISS_W=""; NEARLY=0; PREV_NGW=0; MISS_W=-99; NGW=0; CODEX_ALT=0; ALT_REBRIEF=0
  IV_LAST_W=""; IV_MEMO=""; IV_EXPECT=""; VERIFY_PENDING=0; NVERIFY=0
  # Screener bookkeeping: how many windows were passed, which ones (the judge must be told about the gap), and flag history.
  SCREENED=0; SKIPPED=""; SKIPINFO=""; SFLAGS=""; OPUS_CALLED=0; PREV_IV_SHAPE=""; PREV_IV_NGRASP=""; REPEAT_PUSHED=0
  NCOMPACT=0; COMPACT_NEXT=0; COMPACTED_W=0; COMPACT_PENDING=0
  RUNNER_ON=0; RUNNER_PID=""; ASKIP=""; ASYNC_JUMPW=""; NASKIP=0; NARAN=0; ABATCH=""; NBATCH=0; ASTALE=0; NASTALE=0
  if [ "$ASYNC" != 0 ]; then rm -rf "$CD/async"; : > "$CD/async_timing.jsonl"; fi
  while true; do
    W=$((W+1))
    COMPACTED_W=0; IN_IV=0
    if [ "$COMPACT_PENDING" = 1 ]; then
      SID=""; COMPACT_NEXT=1; COMPACT_PENDING=0; NCOMPACT=$((NCOMPACT+1)); COMPACTED_W=1
    fi
    if [ "$ASYNC" != 0 ]; then
      [ "$ASYNC" = 2 ] && { mkdir -p "$CD/async"; echo $((W-1)) > "$CD/async/checked"; }
      [ "${RUNNER_ON:-0}" = 1 ] || async_start $W
      async_next $W
      if [ "$ASYNC" = 2 ]; then ABATCH="$ASKIP"; ASKIP=""; fi
      if [ -n "$ASKIP" ]; then
        # windows the policy ran through while an earlier one was being checked: never shown to the
        # screener or the judge; the next prompt says so, like it does for screened-out windows.
        NASKIP=$((NASKIP + $(echo $ASKIP | wc -w)))
        EVENTS=$(python3 - "$EVENTS" "$CD" $ASKIP << 'PYA'
import json,sys
l=json.loads(sys.argv[1]); cd=sys.argv[2]
for k in sys.argv[3:]:
    try: c=json.load(open(f"{cd}/async/adv_{k}.json")).get("committed_timestep")
    except Exception: c=None
    l.append({'window':int(k),'committed':c,'judge':'async_skipped'})
print(json.dumps(l))
PYA
)
        for _k in $ASKIP; do
          SKIPPED="${SKIPPED:+$SKIPPED,}$_k"; SKIPINFO="${SKIPINFO:+$SKIPINFO; }$_k (not checked: the policy kept running while an earlier window was being checked)"
        done
      fi
    else
    ADV=$(timeout ${ADV_DEADLINE:-600} $H advance --num-chunks $WINDOW --frames-dir "$CD/w$W" --truth-dir "$TD/w$W" --tag w 2>&1 | tail -1)
    fi
    MONT=$(echo "$ADV" | jget montage); DONE=$(echo "$ADV" | jget done)
    # When advance fails, committed_timestep is empty/None. DONE is not True either,
    # so the old driver kept looping and called the judge every window -- one lane in t1 spun idle for
    # 44 windows, burning 44 judge calls before anyone noticed. 2 in a row restarts the service; 2 more abandons the episode.
    CTCHK=$(echo "$ADV" | jget committed_timestep)
    if [ -z "$CTCHK" ] || [ "$CTCHK" = "None" ]; then
      BADADV=$((BADADV+1))
      log "  window $W advance returned no step count (attempt $BADADV)"
      if [ "$BADADV" -ge 4 ]; then log "  repeated failures, abandoning this episode"; TS=False; break; fi
      if [ "$BADADV" -ge 2 ]; then restart_svc || true; fi
      sleep 5; W=$((W-1)); continue
    fi
    BADADV=0
    TELF=$(echo "$ADV" | jget telemetry_file)
    if [ "$ASYNC" = 2 ] && [ -n "${ABATCH:-}" ]; then
      # windows run since the last check are shown together: stacked montages, the newest telemetry
      # (its table already carries their chunks)
      _bm=$(async_montage "$CD/w$W/batch_montage.png" $ABATCH $W); [ -n "$_bm" ] && MONT=$_bm
      NBATCH=$((NBATCH + $(echo $ABATCH | wc -w)))
      EVENTS=$(python3 - "$EVENTS" "$CD" $ABATCH << 'PYA'
import json,sys
l=json.loads(sys.argv[1]); cd=sys.argv[2]
for k in sys.argv[3:]:
    try: c=json.load(open(f"{cd}/async/adv_{k}.json")).get("committed_timestep")
    except Exception: c=None
    l.append({'window':int(k),'committed':c,'judge':'async_batched'})
print(json.dumps(l))
PYA
)
      ABATCH=""
    fi
    # 2026-09-22 (user): after an empty closure let the policy retry for MISS_GRACE windows before
    # acting. Count only the hand doing the work right now (harness names it); remember the window
    # in which that count last went up.
    NGW=$(sed -n 's/^Empty closures on the working hand this episode: \([0-9]*\)\.$/\1/p' "${TELF:-/dev/null}" 2>/dev/null | tail -1); NGW=${NGW:-0}
    if [ "$NGW" -gt "$PREV_NGW" ] 2>/dev/null; then MISS_W=$W; PREV_NGW=$NGW; fi
    TEL=""; [ -f "$TELF" ] && TEL=$(cat "$TELF")
    # Memory retrieval: match symptom/applies_when using the task instruction plus current-state keywords, take 3.
    # Do not stuff in the whole index -- 23 entries of prose is a wall the model skips.
    NGRASP=$(echo "$ADV" | jget grasp_attempts)
    # Situation words depend on task type: for mechanism tasks the situation is "did the fixture change, did the hand reach the control", not grasping;
    # grasp words used to be applied to everything, so old gripper/grasp entries always filled the slots and none of the 19 lessons from two rounds was ever injected
    # (103 of the 122 prompts in l24c carried the same closed-gripper entry).
    case "$TASK" in PnP*|CoffeeSetupMug|CoffeeServeMug) MKIND=pnp ;; *) MKIND=mech ;; esac
    FLAGS="judging whether the trajectory is failing"
    # Phase words: aperture (open / narrow / medium = possibly holding), whether there has been an intervention, late progress -- lessons about carrying,
    # placing, and after hand-back are only retrieved with these words; otherwise it is always the same approach/missed-grasp entries.
    # harness_advance has no top-level gripper_width (that is a call_cosmos key); the aperture is in telemetry.width_mm;
    # jget used to return None -> 0 mm -> "gripper has closed" from the very start, making the PnP gate useless.
    # "Attempted a grasp" = the policy issued a close command (cmd_gripper > 0.5) or the aperture was once below 30 mm.
    GWMM=$(echo "$ADV" | python3 -c "
import json,sys
d=json.load(sys.stdin); t=d.get('telemetry') or {}
print(int(float(t.get('width_mm') or 99)))" 2>/dev/null || echo 99)
    CLOSECMD=$(echo "$ADV" | python3 -c "
import json,sys
d=json.load(sys.stdin); t=d.get('telemetry') or {}
print(1 if any(float(c.get('cmd_gripper') if c.get('cmd_gripper') is not None else -1)>0.5 for c in (t.get('chunks') or [])) else 0)" 2>/dev/null || echo 0)
    if [ "$GWMM" -lt 12 ]; then FLAGS="$FLAGS narrow aperture closed empty";
    elif [ "$GWMM" -lt 45 ]; then FLAGS="$FLAGS mid-range aperture holding carrying transport object with the hand";
    else FLAGS="$FLAGS gripper open"; fi
    [ "${NINT:-0}" -gt 0 ] && FLAGS="$FLAGS after a recovery hand back preserve progress verified grasp drop after handoff"
    [ "$W" -ge 6 ] && FLAGS="$FLAGS destination placement release delivered still scene stall near the end already on the destination"
    if [ "$MKIND" = mech ]; then
      if [ "${NGRASP:-0}" != "0" ] && [ "${NGRASP:-0}" != "None" ]; then
        FLAGS="$FLAGS hand at the control, empty closure, fixture unchanged, engagement, contact, stall windows, knob index rotation, obscured burner indicator, wrist view"
      else
        FLAGS="$FLAGS approaching the fixture, reduced travel near the control, fixture unchanged, pressing, contact, stall windows, knob index rotation, obscured burner indicator"
      fi
    else
      if [ "${NGRASP:-0}" != "0" ] && [ "${NGRASP:-0}" != "None" ]; then
        FLAGS="$FLAGS gripper closed on nothing empty close missed grasp transport placement dropped fallen gone wrist view"
      else
        FLAGS="$FLAGS no grasp attempt yet approaching gripper open slow descent"
      fi
    fi
    # RoboDojo task names do not match PnP*, so they all fall into mech -- in the lesson bank mech covers stove knobs,
    # microwave doors and faucets, pnp covers pixel unprojection and sink routes; on a tabletop dual-arm robot every one of them is
    # noise (a 2026-09-18 offline Qwen replay showed prompts full of "neighbouring burner"). Until the bank
    # has RoboDojo entries of its own, this family gets no lessons and no full-bank index.
    if [ "$FAMILY" = robodojo ]; then MEM=""
    elif [ "$FAMILY" = robotwin ]; then
      MEM=$(python3 $P/recovery_explore/cli/memory_select.py --kind mech --instruction "$INSTR" --situation "$FLAGS two arms left right handover lift both hands grasp missed stalled" --limit 5 2>/dev/null)
      [ "${RT_MEMORY:-1}" = 0 ] && MEM=""   # 0-shot: no lesson library at all (2026-09-22)
    else MEM=$(python3 $P/recovery_explore/cli/memory_select.py --kind $MKIND --instruction "$INSTR" --situation "$FLAGS" --limit 5 2>/dev/null); fi
    OKW=$(echo "$ADV" | jget task_success); CT=$(echo "$ADV" | jget committed_timestep)
    [ "$OKW" = "True" ] && { TS=True; log "  window $W policy completed the task on its own"; break; }
    if [ "$DONE" = "True" ]; then log "  window $W reached the step limit, ending"; break; fi
    # --- PnP hard gate: do not judge until the policy has attempted a grasp ---
    # Five rounds of data: 4 of 5 broken PnP episodes and all 5 wasted interventions happened before the policy had attempted a grasp;
    # 11 of 12 rescues came after the first missed grasp. "Has grasped" = an empty closure was recorded, or the gripper once closed below 40 mm
    # (a successful grasp produces no empty-closure event and is recognised by aperture; otherwise the carrying phase would be gated forever).
    if [ "$CLOSECMD" = 1 ] || [ "${GWMM:-99}" -lt 30 ]; then EVER_CLOSED=1; fi
    # Second PnP gate: after the first missed grasp, let the policy try again on its own. In control runs the policy usually re-grasps
    # by itself after a miss; both episodes broken in the first 53 of qwen1200 were interventions at the first miss, then the same recipe three times.
    # Rule: only one miss recorded and less than 2 windows since the first miss -> do not judge; a second miss,
    # or two windows after the miss (whatever was going to happen has happened), calls the judge.
    NGN=$(python3 -c "
import sys,json
v=sys.argv[1]
try:
    x=json.loads(v); print(len(x) if isinstance(x,list) else int(x))
except Exception:
    try: print(int(float(v)))
    except Exception: print(0)" "${NGRASP:-0}" 2>/dev/null); NGN=${NGN:-0}
    if [ "$NGN" -ge 1 ] && [ -z "${FIRSTMISS_W:-}" ]; then FIRSTMISS_W=$W; fi
    if [ "$MKIND" = pnp ] && [ "$NGN" -eq 1 ] && [ -n "${FIRSTMISS_W:-}" ] && [ $((W - FIRSTMISS_W)) -lt 1 ]; then
      log "  window $W PnP gate: policy just missed once, letting it retry on its own, not judging"
      EVENTS=$(python3 -c "
import json;l=json.loads('$EVENTS')
l.append({'window':$W,'committed':int('${CT:-0}'),'judge':'gated','why':'first-miss-grace'})
print(json.dumps(l))")
      continue
    fi
    # PNP_GATE_MAX_W: this gate applies only for the first N windows (default 999 = always, identical to before).
    # Relaxed setting (RELAX=1) uses 5: a policy that keeps not grasping is itself a failure; after 5 windows the judge should look.
    if [ "$MKIND" = pnp ] && [ "$EVER_CLOSED" = 0 ] && [ "$W" -le "${PNP_GATE_MAX_W:-999}" ] && { [ "${NGRASP:-0}" = "0" ] || [ "${NGRASP:-0}" = "None" ]; }; then
      log "  window $W PnP gate: policy has not attempted a grasp yet, not judging"
      EVENTS=$(python3 -c "
import json;l=json.loads('$EVENTS')
l.append({'window':$W,'committed':int('${CT:-0}'),'judge':'gated'})
print(json.dumps(l))")
      continue
    fi
    # --- Tier 1: local screener ---
    # fail-closed lives in screen_window.py: an exception / no connection / empty reply all return flag=true.
    # Only the flag field counts here, and a missing one is treated as true -- a pass must be the screener's explicit answer, never a default.
    if [ "$SCREEN" = 1 ] && [ -n "$TELF" ] && [ -n "$MONT" ]; then
      # stderr goes to driver.log instead of being discarded -- that is where screener config errors used to be swallowed.
      SCR=$(python3 $P/recovery_explore/cli/screen_window.py \
        --montage "$MONT" --telemetry "$TELF" --task "$INSTR" \
        --prev-flags "$SFLAGS" --host "$SCREENER_HOST" --tag "w$W" --family "$FAMILY" \
        --out "$CD/screen_w$W.json" 2>>$OUT/lane$LANE/driver.log | tail -1)
      # In async mode the policy may finish the task while the screener runs; if so, do not ask the judge.
      if [ "$ASYNC" != 0 ] && [ -f "$CD/async/success" ]; then
        TS=True; log "  window $W policy finished the task in the background, cancelling in-flight calls, episode over"; break
      fi
      # Screener module failing to import = a config error, not a runtime fault. Left alone, the whole run would alarm on every
      # window, silently turning --arm b into --arm a with nothing visible in the log. Stop right here.
      if [ "$(echo "$SCR" | jget source)" = "screener_missing" ]; then
        log "  ✗ screener module import failed: $(echo "$SCR" | jget why)"
        log "  ✗ --arm b needs the screener. Point SCREENER_DIR at screener.py, or use --arm a instead."
        log "  ✗ (to proceed anyway, set SCREEN_ALLOW_MISSING=1; fail-closed then calls the judge on every window)"
        [ "${SCREEN_ALLOW_MISSING:-0}" = 1 ] || exit 1
      fi
      SFLAG=$(echo "$SCR" | jget flag); SRISK=$(echo "$SCR" | jget risk); SRULE=$(echo "$SCR" | jget rule_hit)
      [ "$SFLAG" = "False" ] && SFLAG=false || SFLAG=true
      PREVF=$(printf '%s' "$SFLAGS" | awk -F, '{print $NF}')
      SFLAGS="${SFLAGS:+$SFLAGS,}$SFLAG"
      # 2026-09-20 (user): with the screener the judge intervened 1.6x as often as without it (189 vs
      # 115 on the same 235 cells), mostly re-intervening while the policy was retrying after a repair.
      # A flag now reaches the judge only when the previous window was flagged too, except a fresh
      # empty-closure rule hit (no intervention in the last 2 windows): that is the rescuable miss
      # and waiting a window lets the policy carry nothing away. SCREEN_CONFIRM2=0 restores the old rule.
      if [ "$SFLAG" = "true" ] && [ "${SCREEN_CONFIRM2:-1}" = 1 ] && [ "$PREVF" != "true" ]; then
        RECENT_IV=0; [ -n "${IV_LAST_W:-}" ] && [ $((W - IV_LAST_W)) -le 2 ] && RECENT_IV=1
        if ! { [ "$SRULE" = "True" ] && [ "$RECENT_IV" = 0 ]; }; then
          SFLAG=pending
        fi
      fi
      if [ "$SFLAG" = "pending" ]; then
        SCREENED=$((SCREENED+1)); SKIPPED="${SKIPPED:+$SKIPPED,}$W"; SKIPINFO="${SKIPINFO:+$SKIPINFO; }$W (flagged once, risk ${SRISK:-?}, not confirmed)"
        log "  window $W screener flagged once (risk ${SRISK:-?}, rule ${SRULE:-?}), waiting for confirmation next window, not calling $MODEL"
        EVENTS=$(python3 -c "
import json;l=json.loads('$EVENTS')
l.append({'window':$W,'committed':int('${CT:-0}'),'judge':'screen_pending','risk':'${SRISK:-}'})
print(json.dumps(l))")
        continue
      fi
      if [ "$SFLAG" = "false" ]; then
        SCREENED=$((SCREENED+1)); SKIPPED="${SKIPPED:+$SKIPPED,}$W"; SKIPINFO="${SKIPINFO:+$SKIPINFO; }$W (routine, risk ${SRISK:-?})"
        log "  window $W screener passed (risk ${SRISK:-?}), not calling $MODEL"
        EVENTS=$(python3 -c "
import json;l=json.loads('$EVENTS')
l.append({'window':$W,'committed':int('${CT:-0}'),'judge':'screened','risk':'${SRISK:-}'})
print(json.dumps(l))")
        continue
      fi
      log "  window $W screener flagged (risk ${SRISK:-?}) -> calling $MODEL"
    fi

    # Once interventions are used up (or after giveup) the judge is no longer called: its verdict could not be executed, and without
    # the execution log the model assumes the infrastructure is broken -- measured: 7 idle calls, ~250k tokens, in one episode.
    # 2026-09-21 (user): after the budget is spent the judge keeps WATCHING. rtF1 left 4584
    # windows unjudged once three interventions were used, so the whole late part of an episode
    # went unseen. MONITOR_AFTER_BUDGET=0 restores the old "stop calling it" behaviour.
    MONITOR=0
    if [ "$GAVEUP" = 1 ] || [ "$NINT" -ge "$MAX_INTERVENTIONS" ]; then
      if [ "${MONITOR_AFTER_BUDGET:-1}" != 1 ]; then
        log "  window $W interventions exhausted ($NINT/$MAX_INTERVENTIONS), no longer calling $MODEL, policy runs to the end"
        EVENTS=$(python3 -c "
import json;l=json.loads('$EVENTS')
l.append({'window':$W,'committed':int('${CT:-0}'),'judge':'budget_exhausted'})
print(json.dumps(l))")
        continue
      fi
      MONITOR=1
    fi

    # --- Tier 2: judge verdict + plan ---
    # The brief is inlined when "the judge has not been called yet", not by window number -- the screener may pass window 1,
    # and the judge's first call may already be at window 5.
    if [ "$OPUS_CALLED" = 0 ] || [ "$COMPACT_NEXT" = 1 ]; then
      # The whole brief is inlined into the prompt instead of giving a repo path to read; <episode dir>/<port> are substituted in place.
      # The first verdict after compaction also comes through here: the same brief plus a recap of the episode.
      # v7 has its own brief: 40% of v6's taught shell usage of $REX, and v7 grants no Bash,
      # so a disclaimer "actually you have no shell" had to be tacked on -- a manual teaching shell usage
      # plus "you have no shell" wastes context and contradicts itself. JUDGE_BRIEF_V7.md was written against the
      # plan interface from the start, with nothing dropped (limits, long-distance rules, force signals, all repair lessons are there).
      # A different criteria addendum per task type: PnP is "did the object move with the hand"; mechanism tasks (door / drawer /
      # knob / faucet / button) need nothing in hand at all, and gripper aperture is not a failure signal there.
      # Both episodes broken last round came from repairing coffee mugs with PnP criteria.
      # RoboDojo is a different robot (tabletop, dual arm, normalised opening, no force sensor); both RoboCasa
      # criteria addenda are wrong there line by line -- MECH is about microwave doors and stove knobs. Split by family first;
      # cosmos / pi05 still take the original case, byte for byte unchanged.
      if [ "$FAMILY" = robodojo ]; then BRIEF_TASK=BRIEF_RD.md
      elif [ "$FAMILY" = robotwin ]; then BRIEF_TASK=BRIEF_RT.md
      else
      case "$TASK" in PnP*|CoffeeSetupMug|CoffeeServeMug) BRIEF_TASK=BRIEF_PNP.md ;; *) BRIEF_TASK=BRIEF_MECH.md ;; esac
      fi
      build_iv_block
      { echo "Read this brief in full first. It is everything you know about the interface."
        echo
        # Pick one rewind section by mode: ALLOW_RESET=1 (learning round, sim can rewind) keeps the REWIND block,
        # otherwise the NO-REWIND block. Both used to go into the prompt; the model believed "no rewind" and did not retry once in three rounds.
        awk -v f="$P/recovery_explore/$BRIEF_TASK" -v rw="$ALLOW_RESET" \
            '/<!--TASK-SECTION-->/{while((getline l < f)>0) print l; next}
             /<!--NO-REWIND-->/{skip=(rw==1); next} /<!--\/NO-REWIND-->/{skip=0; next}
             /<!--REWIND-->/{skip=(rw!=1); next} /<!--\/REWIND-->/{skip=0; next}
             !skip {print}' \
            "$P/recovery_explore/JUDGE_BRIEF_V7.md"
        [ "${RELAX:-0}" = 1 ] && cat << 'RELAXEOF'

OVERRIDE FOR THIS RUN (it replaces the matching lines above wherever they differ):

When a repair attempt does not produce what you set out to get, you do NOT have to
end the intervention. You may make a second attempt inside the same intervention as
long as it is materially different in kind, not in offset: a different line in, a
lower or higher grasp, letting the policy make the close instead of you, a different
control surface. Only after a second attempt has also failed do you retreat and
answer fixed with an empty plan. A third attempt is still not allowed.

Being too cautious costs as much as being too eager. If the numbers and the image
together say the policy is not going to recover on its own, act; do not wait for one
more window of confirmation that you already have.
RELAXEOF
        echo; echo "------------------------------------------------------------"; echo; } > $CD/p_w$W.txt
      if [ "$COMPACT_NEXT" = 1 ]; then
        { build_recap; echo; echo "------------------------------------------------------------"; echo; } >> $CD/p_w$W.txt
        COMPACT_NEXT=0
      fi
      # Full-bank title index, given once with the brief (resent when the brief is resent after compaction). Filtered by task type.
      [ "$FAMILY" = robodojo ] || [ "$FAMILY" = robotwin ] || { python3 $P/recovery_explore/cli/memory_select.py --index --kind $MKIND 2>/dev/null; echo; echo "------------------------------------------------------------"; echo; } >> $CD/p_w$W.txt
      if [ "${FEWSHOT:-0}" -gt 0 ] 2>/dev/null; then
        FS_AS=markers; { [ "$ENGINE" = api ] || [ "$ENGINE" = qwen ]; } || FS_AS=paths
        # Qwen vLLM: 20 pictures per request and a 32k context -> keep the block's pictures few.
        FS_MAXIMG=${FEWSHOT_MAX_IMAGES:-0}; [ "$ENGINE" = qwen ] && FS_MAXIMG=${FEWSHOT_MAX_IMAGES:-6}
        FS_TASK="$TASK"
        FS_BLOCK=$(python3 $P/recovery_explore/cli/fewshot_select.py --family "$FAMILY" --task "$FS_TASK" --n "$FEWSHOT" --images-as $FS_AS --max-images $FS_MAXIMG 2>>$OUT/lane$LANE/driver.log)
        if [ -n "$FS_BLOCK" ]; then
          { printf '%s\n' "$FS_BLOCK"; echo "------------------------------------------------------------"; echo; } >> $CD/p_w$W.txt
          log "  few-shot: $(printf '%s\n' "$FS_BLOCK" | grep -c '^=== Worked example') example(s) added to the brief (FEWSHOT=$FEWSHOT, task=$FS_TASK)"
        else log "  few-shot: no usable examples (FEWSHOT=$FEWSHOT, task=$FS_TASK), running 0-shot"; fi
      fi
      cp $CD/p_w$W.txt $CD/brief_cache.txt
      cat >> $CD/p_w$W.txt << PEOF
Task instruction: $INSTR
This is window $W ($WINDOW chunks per window, about $TOTC chunks in the episode;
$CT steps executed so far).
You have $((MAX_INTERVENTIONS - NINT)) intervention(s) left in this episode (of
$MAX_INTERVENTIONS). Spend them on windows where you can actually change the outcome.

Overview image (one row per camera, left to right in time):
$MONT

$TEL

$MEM

Read the numbers above before you look at the image. Keep the bar high: most
windows should be ok.
$IV_BLOCK

$PLAN_SPEC
PEOF
    else build_iv_block; cat > $CD/p_w$W.txt << PEOF
Window $W ($CT steps executed, about $TOTC chunks in the episode). You remember
every earlier window. $((MAX_INTERVENTIONS - NINT)) intervention(s) left of $MAX_INTERVENTIONS.
$IV_BLOCK

Overview image:
$MONT

$TEL

$MEM

Read the numbers before the image.

$PLAN_SPEC
PEOF
    fi
    # The judge never saw screened-out windows, yet the prompt says "you remember every window". The gap must be stated,
    # or it will take "the previous window" to mean the immediately preceding one.
    if [ -n "$SKIPPED" ]; then
      printf '\nWindows you were not shown (a fast local check passed them): %s.\nTheir chunks are in the action table and the "Earlier in this episode" trajectory above, so you can see what the arm did in them. Most windows of an episode look like those: the policy approaching, re-approaching, opening and closing, pausing. A policy that is retrying after a repair looks unusual for a window or two; that alone is not a new fault. Intervene only for a fault you can name and point at in this window.\n' "$SKIPINFO" >> $CD/p_w$W.txt
      SKIPINFO=""; SKIPPED=""
    fi
    OPUS_CALLED=1
    # Judging-turn lock: the budget is closed while judging, so the _round_budget gate guards nothing, yet the model still has Bash and the endpoint.
    JTOK=$(python3 -c "import secrets;print(secrets.token_hex(12))")
    # async: the runner is advancing the policy right now, and the lock refuses harness_advance. The
    # Qwen judge has no tool that reaches the simulator, so the lock guards nothing there.
    [ "$ASYNC" != 0 ] || $H judge-lock --control-token "$JTOK" >/dev/null 2>&1
    # 2026-09-21 (user feedback on rtG1): read the task before judging anything. The first judged
    # window of an episode must return a structured task parse; every later window gets it back
    # and judges the policy against it (lift_pot was worked with one hand for a whole episode,
    # blocks_ranking_size was ordered by the wrong size, and nobody noticed).
    if [ "$FAMILY" = robotwin ]; then
      if [ -s "$CD/task_plan.json" ]; then
        { echo; echo "YOUR TASK PARSE (written at the start of this episode; judge every window against it):";
          cat "$CD/task_plan.json"; echo;
          echo "If what the policy is doing contradicts this parse -- the wrong object, the wrong destination,";
          echo "one hand where the parse says both, the wrong order -- that is a fault even if nothing is stuck."; } >> "$CD/p_w$W.txt"
      else
        { echo; echo "FIRST, READ THE TASK. This is the first window you judge in this episode, so add one more field";
          echo 'to your json, "task_plan": {"objects": [...], "destination": "...", "order": "...", "hands": "one" | "both",';
          echo '"done_when": "what the head camera shows when the task is complete"}. Work it out from the instruction';
          echo "and the first pictures: which object in the scene is which (the big one and the small one, which bin,";
          echo "which mat), whether one hand or both must act on the same object, and in what order things happen."; } >> "$CD/p_w$W.txt"
      fi
    fi
    QWEN_TURN_THINK=0 claude_turn "$CD/p_w$W.txt" "$CD/judge_w$W.json" $JUDGE_DEADLINE "$MONT" "Read"
    if [ "$ASYNC" != 0 ] && [ -f "$CD/async/success" ]; then
      TS=True; log "  window $W policy finished the task in the background, cancelling in-flight calls, episode over"; break
    fi
    if [ "$FAMILY" = robotwin ] && [ ! -s "$CD/task_plan.json" ]; then
      python3 - "$CD/judge_w$W.json" "$CD/task_plan.json" << 'PY2' 2>/dev/null
import json, re, sys
try:
    txt = json.load(open(sys.argv[1])).get("result", "") or ""
    m = re.search(r"\{.*\}", txt, re.S)
    tp = json.loads(m.group(0)).get("task_plan") if m else None
    if isinstance(tp, dict) and tp:
        json.dump(tp, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
except Exception:
    pass
PY2
      [ -s "$CD/task_plan.json" ] && log "  window $W task parse: $(tr -d '\n' < "$CD/task_plan.json" | cut -c1-160)"
    fi
    [ "$ASYNC" != 0 ] || $H judge-unlock --control-token "$JTOK" >/dev/null 2>&1
    JUDGED=$((JUDGED+1))
    JJ=$(extract_json "$CD/judge_w$W.json")
    echo "$JJ" > "$CD/verdict_w$W.json"
    V=$(jfield "$JJ" verdict); V=${V:-ok}
    case "$V" in ok|intervene|giveup) ;; *) log "  window $W invalid verdict ($V), treating as ok"; V=ok ;; esac
    if [ "$V" = "giveup" ]; then GAVEUP=1; log "  window $W verdict giveup -> no further interventions this episode"; V=ok; fi
    USED=0; ATT=0; RSETS=0; RES=""; NPLAN=0; RETREATED=0
    if [ "$MONITOR" = 1 ]; then
      [ "$V" = "intervene" ] && log "  window $W monitor only (interventions exhausted $NINT/$MAX_INTERVENTIONS): judge wanted to intervene, logged without acting" \
                            || log "  window $W monitor only: verdict $V"
      EVENTS=$(python3 -c "
import json;l=json.loads('$EVENTS')
l.append({'window':$W,'committed':int('${CT:-0}'),'judge':'monitor','would':'$V'})
print(json.dumps(l))")
      V=ok
    fi
    # Per human review: once the judge says giveup, no more interventions this episode; the policy runs to the end.
    # In round three one episode judged intervene ten times but acted once; the other nine were blocked by the cap and idled, wasting money.
    if [ "$V" = "intervene" ] && [ "${VERIFY_PENDING:-0}" = 1 ]; then
      log "  window $W verification window: intervened in the previous window, observe only (judge wanted to intervene, blocked)"
      NVERIFY=$((NVERIFY+1)); V=ok
    fi
    [ "${VERIFY_PENDING:-0}" = 1 ] && VERIFY_PENDING=0
    # 2026-09-21 (user): the FIRST intervention of an episode came at 22% of the way through in
    # rtF1, before the policy had shown a real fault, and those early repairs are where most of
    # the damage was. The first one now waits until either the policy has closed on nothing at
    # least once, or the episode is past FIRST_IV_FRAC of its step budget. Later interventions are
    # unaffected. FIRST_IV_FRAC=0 turns the gate off.
    # Hold 1 (2026-09-22, user): an empty closure on the working hand is not yet a reason to take over --
    # the policy often re-opens and gets it on the next try (in the 50x20 run most "harmed" episodes
    # were taken over at the first miss, ~13% into the episode). While the working hand has missed
    # only once, wait MISS_GRACE windows after that miss; a second miss, or still failing after the
    # grace, lets the intervention through.
    if [ "$V" = "intervene" ] && [ "${MISS_GRACE:-2}" -gt 0 ] && [ "$NGW" = 1 ] && [ $((W - MISS_W)) -lt "${MISS_GRACE:-2}" ] 2>/dev/null; then
      log "  window $W verdict intervene, but the working hand has missed only once (window $MISS_W) -> letting the policy retry first (grace ${MISS_GRACE:-2} windows)"
      NEARLY=$((NEARLY+1)); V=ok
    fi
    # Hold 2 (2026-09-21): the FIRST intervention waits until the working hand has missed at least
    # once, or the episode is past FIRST_IV_FRAC of its step budget. FIRST_IV_FRAC=0 turns it off.
    if [ "$V" = "intervene" ] && [ "$NINT" = 0 ] && [ "${FIRST_IV_FRAC:-35}" -gt 0 ] 2>/dev/null; then
      _frac=$(python3 -c "
import sys
ct, mx = float('${CT:-0}' or 0), float('${MAXSTEPS:-0}' or 0)
print(int(100 * ct / mx) if mx > 0 else 100)" 2>/dev/null)
      if [ "$NGW" = "0" ] && [ "${_frac:-100}" -lt "${FIRST_IV_FRAC:-35}" ] 2>/dev/null; then
        log "  window $W verdict intervene, but it is the first of the episode, progress only ${_frac}%, and the working hand has not missed yet -> observing first"
        NEARLY=$((NEARLY+1)); V=ok
      fi
    fi
    if [ "$V" = "intervene" ] && [ "$GAVEUP" = "0" ] && [ "$NINT" -lt "$MAX_INTERVENTIONS" ]; then
      NINT=$((NINT+1)); IN_IV=1
      STALE=""
      if [ "$ASYNC" != 0 ]; then
        # The judge decided on window $W, but the policy kept going. Stop it at the end of the window it is
        # in, then repair from where the arm is now (same staleness LOOKAHEAD simulates, but real).
        async_stop
        _al=$W; while [ -f "$CD/async/adv_$((_al+1)).json" ]; do _al=$((_al+1)); done
        if [ "$_al" -gt "$W" ]; then
          _aa=$(cat "$CD/async/adv_$_al.json")
          STALE=$(echo "$_aa" | jget committed_timestep)
          NARAN=$((NARAN + _al - W))
          EVENTS=$(python3 - "$EVENTS" "$CD" $(seq $((W+1)) $_al) << 'PYA'
import json,sys
l=json.loads(sys.argv[1]); cd=sys.argv[2]
for k in sys.argv[3:]:
    try: c=json.load(open(f"{cd}/async/adv_{k}.json")).get("committed_timestep")
    except Exception: c=None
    l.append({'window':int(k),'committed':c,'judge':'async_ran_while_judging'})
print(json.dumps(l))
PYA
)
          ASYNC_JUMPW=$_al
          if [ "$(echo "$_aa" | jget task_success)" = "True" ]; then
            NINT=$((NINT-1)); TS=True; log "  window $W verdict intervene, but the policy finished on its own while judging (window $_al)"; break
          fi
          if [ "$(echo "$_aa" | jget done)" = "True" ]; then
            NINT=$((NINT-1)); log "  window $W verdict intervene, but the step limit was reached while judging (window $_al)"; break
          fi
          ASTALE=$((ASTALE + ${STALE:-$CT} - CT)); NASTALE=$((NASTALE+1))
          log "  window $W async: the policy ran $((_al-W)) more window(s) while judging ($CT -> ${STALE:-?} steps), intervening from the current state"
        fi
        IV_T0=$(date +%s.%N)
      fi
      if [ "${LOOKAHEAD:-0}" -gt 0 ]; then
        # Reproduce the staleness of the async scheme: the verdict is based on the window's final frames, but the policy has run L more chunks
        SA=$(timeout ${ADV_DEADLINE:-600} $H advance --num-chunks $LOOKAHEAD --frames-dir "$CD/la$W" --truth-dir "$TD/la$W" --tag la 2>&1 | tail -1)
        STALE=$(echo "$SA" | jget committed_timestep)
        if [ "$(echo "$SA" | jget task_success)" = "True" ]; then
          TS=True; log "  window $W policy finished on its own during the $LOOKAHEAD-chunk lookahead"; break; fi
        log "  window $W lookahead $LOOKAHEAD chunk(s): $CT -> $STALE steps"
      fi
      # Situation words during an intervention cover the whole repair: re-stage, servo missed, clipping, pixel targets, nearby objects,
      # wrist orientation, lift verification, drop after hand-back, one change at a time, three failures -- primitive/infrastructure
      # lessons get 2 reserved slots (--prim-slots), otherwise they always rank behind grasp lessons.
      case "$TASK" in PnP*|CoffeeSetupMug|CoffeeServeMug) MKIND=pnp; MSIT="repairing the missed grasp, arm retreated, empty close, approach and close the gripper, verify lift, hand back, avoid knocking over neighbours, re-stage retry, servo missed did not reach escalate offset, clipped waypoint, pixel target unprojection surface centre, wrist orientation rotate, drop after handoff, one change per attempt, three failed repairs, aperture settle, sink container route, topple upright" ;;
                      *) MKIND=mech; MSIT="repairing engagement with the control, hand at the fixture, reseat, reposition, lever knob button, verify the fixture changed, hand back, re-stage retry, servo missed did not reach escalate offset, clipped waypoint, pixel target unprojection, wrist orientation rotate, one change per attempt, three failed repairs, knob index panel, obscured burner" ;; esac
      MEMI=$(python3 $P/recovery_explore/cli/memory_select.py --kind $MKIND --instruction "$INSTR" --situation "$MSIT" --limit 8 --prim-slots 3 2>/dev/null)
      # robotwin: the lesson library is RoboCasa's (microwave, stove knob, sink); on a
      # RoboTwin tabletop it misleads the repair loop, as it does the window prompt (MEM="").
      if [ "$FAMILY" = robotwin ]; then
        MEMI=$(python3 $P/recovery_explore/cli/memory_select.py --kind mech --instruction "$INSTR" --situation "repairing a missed grasp, closed on nothing, which arm, left right both hands, handover, lift together, pixel aim, grasp yourself, hand back, rewind, change something, servo stalled, clipped, give up" --limit 8 --prim-slots 3 2>/dev/null)
        [ "${RT_MEMORY:-1}" = 0 ] && MEMI=""
      fi
      # The pose at takeover = this intervention's "re-stage anchor". While the policy advances, every chunk records an entry
      # in _waypoints; execute_plan during an intervention does not, so this point is always
      # in visited_points -- a move_to back here hits the 8 cm relaxation and may travel 60 cm in one step
      # without clipping. A real robot has no reset, but "remember where the hand was at takeover, go back and redo it if the repair fails"
      # is something a real robot can do too; this is the soft reset.
      STAGE_XYZ=$(printf '%s' "$TEL" | sed -n 's/.*End-effector is now at \(\[[^]]*\]\).*/\1/p' | tail -1)
      if [ "$FAMILY" = robotwin ] && [ -n "$STAGE_XYZ" ]; then
        # robotwin: the telemetry position is the wrist (RoboTwin endpose), 12 cm behind
        # the fingertip centre that move_to servos; hand the judge the fingertip point.
        STAGE_XYZ=$(printf '%s' "$TEL" | python3 -c "
import re,sys,math
ls=[l for l in sys.stdin.read().splitlines() if 'End-effector is now at' in l]
m=re.search(r'now at \[([^]]*)\].*quaternion \(w,x,y,z\) \[([^]]*)\]', ls[-1]) if ls else None
if not m: print(''); sys.exit()
p=[float(v) for v in m.group(1).split(',')]; w,x,y,z=[float(v) for v in m.group(2).split(',')]
n=math.sqrt(w*w+x*x+y*y+z*z) or 1.0; w,x,y,z=w/n,x/n,y/n,z/n
ax=[1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)]
print([round(p[i]+0.12*ax[i],4) for i in range(3)])" 2>/dev/null)
      fi
      # RoboCasa reports finger separation (mm); this RoboDojo robot has no such quantity and reports a
      # 0..1 normalised opening. Value and unit are taken separately; the cosmos / pi05 branch is byte-identical to before.
      if [ "$FAMILY" = robodojo ] || [ "$FAMILY" = robotwin ]; then
        STAGE_MM=$(printf '%s' "$TEL" | sed -n 's/.*opening \([0-9.]*\),.*/\1/p' | tail -1)
        STAGE_UNIT="(normalised, 1 = open)"
      else
        STAGE_MM=$(printf '%s' "$TEL" | sed -n 's/.*aperture \([0-9.]*\) mm.*/\1/p' | tail -1)
        STAGE_UNIT="mm"
      fi
      RESTAGE=""
      if [ -n "$STAGE_XYZ" ]; then
        RESTAGE="Re-stage point for this intervention: the end-effector was at $STAGE_XYZ with the
fingers at ${STAGE_MM:-?} $STAGE_UNIT when you took over. The arm reached that point under its own
power, so it counts as visited: a single move_to with that target ${RESTAGE_REACH:-may travel up to
60 cm and will not be clipped}.

If the attempt you just made left the hand somewhere you cannot work from, do not
push on from there. Go back to that point, put the fingers back the way they were,
and come in again on a different line -- a single step {\"op\": \"retreat\"} does
exactly that in one action. That costs actions from the same budget and
$RESTAGE_TAIL"
        # robotwin: STAGE_XYZ was converted to the fingertip centre above; say so.
        if [ "$FAMILY" = robotwin ]; then RESTAGE=${RESTAGE/the end-effector was at/the fingertip centre was at}; fi
      fi
      # Token for closing the budget: the model never sees it, so it cannot remove the server-side privilege gate itself.
      WTOK=$(python3 -c "import secrets;print(secrets.token_hex(12))")
      $H open-budget --budget $INTERVENE_BUDGET --control-token "$WTOK" >/dev/null 2>&1
      # The first plan is the one from the judging turn; no extra question.
      PLAN=$(jfield "$JJ" plan)
      SHAPE=$(python3 -c "
import json,sys
try: print('>'.join(str(s.get('op'))+('/'+str(s.get('to')) if s.get('to') else '') for s in json.loads(sys.argv[1])))
except Exception: print('')" "$PLAN" 2>/dev/null)
      # 2026-09-20 (user): rescues stall at 10-14% because the repair after a failed repair is the same
      # retreat->hand-back from the same pose, and the policy misses the same way again. If the last
      # intervention had this op shape and the gripper has closed on nothing again since, push back once.
      if [ -n "$SHAPE" ] && [ "$SHAPE" = "$PREV_IV_SHAPE" ] && [ "${NGRASP:-0}" != "None" ] && [ -n "$PREV_IV_NGRASP" ] \
         && [ "${NGRASP:-0}" -gt "$PREV_IV_NGRASP" ] 2>/dev/null && [ "$REPEAT_PUSHED" != 1 ]; then
        REPEAT_PUSHED=1
        log "  window $W plan has the same shape as the last failed intervention ($SHAPE), pushing back once for a different repair"
        printf '%s\n' "Your last intervention (window $IV_LAST_W) used this same repair: $SHAPE.
Since then the gripper has closed on nothing again, so that repair did not work, and
sending it again from the same place usually fails the same way.

Change the approach, not the offset:
  - if you can see the object, aim the OPEN fingers at its CURRENT position with a pixel
    move_to (dz 0.05 above it, camera primary or wrist), then hand back {\"op\": \"policy\", \"chunks\": 2};
  - or rotate the wrist about the vertical axis by about 0.4 rad before handing back,
    so the fingers meet the object from a different side;
  - or, if the telemetry lists two or more empty closures, grasp it yourself as in
    ESCALATE AFTER TWO FAILED GRASPS.
Answer again with the same JSON schema (verdict, diagnosis, expect, plan)." > "$CD/p_repeat_w$W.txt"
        claude_turn "$CD/p_repeat_w$W.txt" "$CD/repeat_w$W.json" $ACT_DEADLINE "" "Read"
        RJJ=$(extract_json "$CD/repeat_w$W.json")
        NPL=$(jfield "$RJJ" plan)
        if [ -n "$NPL" ] && [ "$NPL" != "[]" ]; then PLAN=$NPL; JJ=$RJJ; log "  window $W plan changed after pushback"; else log "  window $W no new plan after pushback, keeping the original"; fi
        SHAPE=$(python3 -c "
import json,sys
try: print('>'.join(str(s.get('op'))+('/'+str(s.get('to')) if s.get('to') else '') for s in json.loads(sys.argv[1])))
except Exception: print('')" "$PLAN" 2>/dev/null)
      fi
      PREV_IV_SHAPE=$SHAPE; PREV_IV_NGRASP=${NGRASP:-0}; [ "$PREV_IV_NGRASP" = "None" ] && PREV_IV_NGRASP=0
      ATT=1; R=0; RES=""; NPLAN=0; USED=0; RSETS=0; NEMPTY=0; RETREAT_OK=0; RETREATED=0; POLICY_USED=0
      while :; do
        if [ -z "$PLAN" ] || [ "$PLAN" = "[]" ]; then
          log "  window $W attempt $ATT has no executable plan, closing as giveup"; RES=${RES:-giveup}; break
        fi
        PF="$CD/plan_w${W}_a${ATT}_r$R.json"; printf '%s' "$PLAN" > "$PF"
        FD="$CD/act$W/a${ATT}r$R"
        EX=$($REX exec-plan --plan-file "$PF" --frames-dir "$FD" 2>&1 | tail -1)
        echo "$EX" > "$CD/exec_w${W}_a${ATT}_r$R.json"
        NPLAN=$((NPLAN+1))
        SR=$(echo "$EX" | jget stop_reason); NLEFT=$(echo "$EX" | jget remaining_count)
        if [ -z "$SR" ] || [ "$SR" = "None" ]; then
          # No stop_reason from the server = the whole call failed (bad arguments, RPC error).
          # None must not be passed on to the model as a normal result; this intervention ends here.
          log "  window $W a${ATT}r$R execute_plan failed: $(echo "$EX" | head -c 200)"
          RES=error; break
        fi
        USED=$(echo "$EX" | jget actions_used)
        log "  window $W a${ATT}r$R executed $(python3 -c "
import json,sys
try: print(len(json.loads(sys.argv[1]).get('executed') or []))
except Exception: print('?')" "$EX") step(s) -> $SR (${NLEFT:-?} step(s) not executed, ${USED:-?} action(s) used)"
        # How many empty grasps so far in this intervention. Consecutive empty grasps = the hand has drifted from where the grasp was first missed,
        # and more small corrections in place just keep closing on nothing -- 4 of 11 interventions ran out this way.
        [ "$SR" = closed_empty ] && NEMPTY=$((NEMPTY+1))
        NEXTHINT=""
        case "$SR" in
          overshoot)
            NEXTHINT="That step travelled several times the distance you asked for. The arm is in a
configuration where Cartesian moves cannot be trusted, so for the rest of this
intervention the simulator refuses move_to, nudge, lift, rotate and retreat. Open the
fingers if they are shut on nothing and end the intervention as fixed; a hand-back of
one chunk is allowed if the policy is likely to recover from where the hand is." ;;
          move_residual)
            NEXTHINT="The last move ended farther from its target than 2 cm (final_dist_m above). Look at where the
fingertips are now in the pictures, then send the next move from here -- correct the target, come in on
another line, or use a bbox locate to re-measure the object. Nothing is locked." ;;
          servo_missed|move_clipped)
            NEXTHINT="The last step did not reach where you sent it. From this arm configuration
further Cartesian moves fail the same way, so for the rest of this intervention the
simulator refuses move_to, nudge, lift, rotate and retreat. What remains: open the
fingers if they are shut on nothing, hand back one or two chunks so the policy
re-approaches on its own, or end the intervention as fixed." ;;
          policy_cap)
            NEXTHINT="The policy hand-back allowance for this intervention is used up. Finish the
repair with your own steps -- close, lift, look -- or answer fixed if the state is
one the policy can continue from." ;;
        esac
        if [ "${NEMPTY:-0}" -ge 2 ]; then
          NEXTHINT="$NEXTHINT
That is empty closure number $NEMPTY in this one intervention. Stop correcting from
where the hand is now: it has drifted from the point where the grasp was first missed,
which is why the corrections keep landing on nothing. Return to the re-stage point
({\"op\": \"retreat\"}), reopen, and approach again from above."
        fi
        # Whether the policy was handed back in this segment: retreating after a hand-back throws away the policy's fresh re-approach;
        # that is how the sink avocado and the fish on the cutting board were lost in n90abc. Interventions with a hand-back no longer get an undo refund.
        if python3 -c "
import json,sys
ex=(json.loads(sys.argv[1]).get('executed') or [])
sys.exit(0 if any(e.get('op')=='policy' for e in ex) else 1)" "$EX" 2>/dev/null; then POLICY_USED=1; fi
        RETREAT_OK=$(python3 -c "
import json,sys
try: ex=(json.loads(sys.argv[1]).get('executed') or [])
except Exception: ex=[]
e=ex[-1] if ex else {}
print(1 if e.get('op')=='retreat' and e.get('servo_ok') is not False and not e.get('overshoot') and not e.get('error') and sys.argv[2]!='1' else 0)" "$EX" "${POLICY_USED:-0}")
        if [ "$RETREAT_OK" = 1 ]; then
          NEXTHINT="You are back at the takeover pose with the fingers as they were. The object is
NOT back: it is wherever your last contact left it, so look at it again in the fresh
image before planning -- re-derive any target from what you see now, never from the
numbers you used before. If the policy had been handed back in this intervention,
you have just undone its re-approach as well; do not make that a habit. If you have a materially different approach -- a different
line in, a lower grasp, letting the policy make the close -- send it as continue.
Otherwise answer fixed with an empty plan: that hands the policy back what it had,
and an intervention ended this way is not counted against your allowance."
        fi
        R=$((R+1))
        # Hand the execution result back to the model: continue with the next segment or finish. It Reads the image itself.
        IMGS=$(python3 -c "
import json,sys
try: d=json.loads(sys.argv[1])
except Exception: d={}
img=d.get('image')
print(f\"{img}  (one picture, three cameras side by side: {d.get('image_layout','primary | secondary | wrist')})\" if img else '(no images)')" "$EX")
        if [ "$FAMILY" = robotwin ]; then
        # robotwin: the shared formatter reads RoboCasa keys (eef_after, moved_cm,
        # width_mm, force) that this robot does not report -- it printed "None" for every
        # step, so the judge never saw whether a move arrived. Report what the mixin returns.
        LOGTXT=$(python3 -c "
import json,sys
try: d=json.loads(sys.argv[1])
except Exception: d={}
def r(v): return [round(float(x),3) for x in v] if isinstance(v,(list,tuple)) else v
for note in d.get('rewritten') or []:
    print('the simulator changed your plan before running it: ' + str(note))
for e in d.get('executed') or []:
    op=e.get('op'); bits=[f\"step {e.get('step')} {op}\"]
    if e.get('arm') and op!='policy': bits.append(f\"arm {e.get('arm')}\")
    for k in ('target','dxyz','dz','drot','state'):
        if e.get(k) is not None: bits.append(f'{k}={e[k]}')
    if e.get('error'): bits.append('ERROR ' + str(e['error']))
    elif op=='policy':
        bits.append(f\"{e.get('chunks')} chunk(s) handed back, policy chunks left this intervention {e.get('policy_chunks_left')}\")
    elif op=='gripper':
        bits.append(f\"opening now {e.get('opening_norm')}\" + (' -- CLOSED ON NOTHING' if e.get('closed_empty') else ''))
    else:
        eef=e.get('eef')
        eef=({k: r(v) for k,v in eef.items()} if isinstance(eef,dict) else r(eef))
        bits.append(f\"asked {e.get('requested_dist_m')} m, travelled {e.get('travelled_m')} m, ended {e.get('final_dist_m')} m from the target\")
        if (e.get('segments') or 1)>1: bits.append(f\"{e.get('segments_done')}/{e.get('segments')} segments\")
        bits.append(f'fingertips now {eef}')
        if e.get('hand_offset_drift_m') is not None and op in ('move_both','lift_both'): bits.append(f\"hand-to-hand offset changed {e['hand_offset_drift_m']} m\")
        if e.get('ok') is False: bits.append(f\"DID NOT REACH TARGET ({e.get('servo_stop')}\" + (f\", {e['failed_arm']} arm could not be planned\" if e.get('failed_arm') else '') + ')')
        if e.get('clipped'): bits.append(f\"CLIPPED: asked {e.get('asked_dist_m')} m, longest allowed {e.get('clipped_to_m')} m\")
        if e.get('overshoot'): bits.append('OVERSHOOT')
        if e.get('task_success_latched'): bits.append('TASK SUCCESS REGISTERED')
    print('  ' + '; '.join(str(b) for b in bits))
st=(d.get('state') or {}); op_=st.get('opening') or {}; cmd_=st.get('gripper') or {}
def _hand(side):
    o, c = op_.get(side), cmd_.get(side)
    if o is None: return f'{side} ?'
    o = float(o)
    if o <= 0.05 and (c is None or float(c) <= 0.5): return f'{side} {o:.2f} = SHUT ON NOTHING (empty hand)'
    if o >= 0.85: return f'{side} {o:.2f} = open'
    return f'{side} {o:.2f} = partly closed (holding something only if it rose with the hand)'
print(f\"  now: fingers {_hand('left')}; {_hand('right')}; resets left {d.get('resets_left')}; policy steps left {st.get('policy_steps_left')} of {st.get('policy_steps_max')} (hand-backs come out of these)\")" "$EX")
        else
        LOGTXT=$(python3 -c "
import json,sys
try: d=json.loads(sys.argv[1])
except Exception: d={}
for e in d.get('executed') or []:
    bits=[f\"step {e.get('step')} {e.get('op')}\"]
    for k in ('target','dxyz','dz','drot','state'):
        if e.get(k) is not None: bits.append(f'{k}={e[k]}')
    bits.append(f\"eef {e.get('eef_after')}\")
    bits.append(f\"moved {e.get('moved_cm')} cm\")
    bits.append(f\"width {e.get('width_mm')} mm\")
    bits.append(f\"force {e.get('wrist_contact_force')} N\")
    if e.get('servo_ok') is False: bits.append('SERVO DID NOT REACH TARGET')
    if e.get('clipped_to_m') is not None: bits.append(f\"CLIPPED to {e['clipped_to_m']} m (asked {e.get('requested_dist_m')} m)\")
    if e.get('overshoot'): bits.append(f\"OVERSHOOT: asked {e.get('requested_cm')} cm, travelled {e.get('moved_cm')} cm\")
    if e.get('policy_chunks_left') is not None: bits.append(f\"policy chunks left this intervention {e.get('policy_chunks_left')}\")
    if e.get('retreat'): r=e['retreat']; bits.append(f\"RETREAT to takeover pose: fingers {r.get('fingers')}, {r.get('final_dist_m')} m off, wrist {r.get('rot_before_deg')} -> {r.get('rot_after_deg')} deg off\")
    if e.get('error'): bits.append('ERROR ' + str(e['error']))
    print('  ' + '; '.join(str(b) for b in bits))
st=(d.get('state') or {})
print(f\"  now: eef {st.get('eef_pos')}, aperture {round(1000*float(st.get('gripper_width') or 0),1)} mm, resets left {st.get('resets_left')}, policy steps left {st.get('policy_steps_left')} of {st.get('policy_steps_max')} (hand-backs come out of these)\")" "$EX")
        fi
        if [ "$R" -ge "$PLAN_ROUNDS" ] || [ "${NLEFT:-0}" = "0" ] && [ "$SR" = "plan_complete" ]; then TAIL_HINT="This was your last segment; finish now."; else TAIL_HINT="You may send one more segment if the repair is not finished."; fi
        if [ "$FAMILY" = robotwin ]; then
          APERTURE_NOTE="The opening (0 shut .. 1 open) is the only thing that tells you the fingers shut on
nothing (at or below 0.05). It cannot tell holding from open, so confirm a grasp by
whether the object came up with the hand in the images, not by the number alone."
        else
          APERTURE_NOTE="The aperture is the only thing that tells you the fingers shut on nothing (under
12 mm). It cannot tell holding from open, so confirm a grasp by whether the object
came up with the hand in the images, not by the number alone."
        fi
        cat > "$CD/p_exec_w${W}_a${ATT}_r$((R-1)).txt" << PEOF
Your plan was executed. It stopped because: $SR

$LOGTXT

Fresh image (read it -- one file, three cameras side by side):
$IMGS

$MEMI

$RESTAGE

$NEXTHINT

$TAIL_HINT Answer with ONE json object and nothing else:

{"result": $RESULT_OPTS,
 "note": "one or two sentences",
 "plan": []}

- "continue": the repair needs more steps; plan is the next 1 to 8 of them and gets
  executed the same way. Use this only if you can say what went wrong with what you
  just saw.
- "fixed": the state is one the policy can carry on from. plan MUST be empty.
  If this repair was about getting the object into the hand, do not answer fixed on
  the strength of a close alone -- lift a few centimetres first and say whether the
  object came up with the fingers. A close with nothing lifted is not a fixed state.
$RETRY_DOC
- "giveup": the object can no longer be acted on at all -- it has left the reachable
  workspace, or no camera row still shows it. plan MUST be empty. This ends every
  intervention for the episode, so it is not a way to close out a difficult repair.
  **Accumulated drift, a servo that missed, and repeated empty closures are not
  reasons to give up -- they are reasons to go back to the re-stage point and come in
  again.** While the object is still sitting where it was and still within reach, a
  further attempt exists; if you have not re-staged at least once in this
  intervention, you have not run out of attempts.
  The one exception is a step that moved several times what you asked for: that is
  not drift, it is the controller misbehaving in this arm configuration, and neither
  a return move nor a hand-back from here is reliable. $BLOWUP_ACTION

$APERTURE_NOTE
PEOF
        EJ="$CD/act_w${W}_a${ATT}_r$((R-1)).json"
        # The 4th argument is the image attachment: codex does not read file paths by itself, so it must be passed with -i.
        # The claude branch ignores it and opens the path in the prompt with the Read tool as usual; neither side is affected.
        CKIMG=$(echo "$EX" | jget image)
        claude_turn "$CD/p_exec_w${W}_a${ATT}_r$((R-1)).txt" "$EJ" $ACT_DEADLINE "${CKIMG:-}" "Read"
        RJ=$(extract_json "$EJ")
        RES=$(jfield "$RJ" result); RES=${RES:-fixed}
        PLAN=$(jfield "$RJ" plan)
        # Push back once on a giveup in the first segment. Measured: 8 of 11 interventions ended in giveup, most with the hand still next to
        # the object and the object still in place -- not a lost cause, just no further attempt.
        if [ "$RES" = giveup ] && [ "$R" -le 1 ] && [ "$GIVEUP_PUSHED" != 1 ]; then
          GIVEUP_PUSHED=1
          log "  window $W giveup on the first segment, pushing back once: return to the re-stage point and retry"
          printf '%s\n' "You answered giveup after a single attempt. Unless the object has left the reachable
workspace or no longer appears in any camera row, that is premature: one failed
descent is the normal first outcome of a repair, not proof that the state cannot be
recovered.

Go back to the re-stage point given above, reopen the fingers, and approach on a
different line -- higher, or offset to one side. Then answer again. If after that
attempt the object really is unreachable, say giveup and name what makes it so." > "$CD/p_push_w${W}_a${ATT}.txt"
          claude_turn "$CD/p_push_w${W}_a${ATT}.txt" "$CD/push_w${W}_a${ATT}.json" $ACT_DEADLINE "" "Read"
          PJ=$(extract_json "$CD/push_w${W}_a${ATT}.json")
          RES=$(jfield "$PJ" result); RES=${RES:-giveup}
          PLAN=$(jfield "$PJ" plan)
          if [ "$RES" = continue ] && [ -n "$PLAN" ] && [ "$PLAN" != "[]" ]; then
            log "  window $W switched to continue after pushback"; continue
          fi
        fi
        case "$RES" in
          continue)
            if [ "$R" -ge "$PLAN_ROUNDS" ]; then
              log "  window $W a$ATT used all $PLAN_ROUNDS plan segments, closing as fixed"; RES=fixed; break
            fi
            [ -z "$PLAN" ] || [ "$PLAN" = "[]" ] && { log "  window $W said continue without a plan, closing as fixed"; RES=fixed; break; }
            continue ;;
          retry)
            if [ "$ALLOW_RESET" != 1 ]; then
              # The model wrote retry anyway (the option is no longer in the prompt). With no rewind available,
              # this intervention ends here, leaving whatever actions it already made in the scene.
              log "  window $W said retry but rewind is disabled in this run, intervention ends"; RES=unfixed; break
            fi
            RW=$($H reset-window 2>&1 | tail -1)
            if [ "$(echo "$RW" | jget ok)" != "True" ]; then
              log "  window $W wanted to retry but reset is unavailable: $(echo "$RW" | jget error)"; RES=fixed; break
            fi
            LEFT=$(echo "$RW" | jget resets_left); RSETS=$((RSETS+1))
            log "  window $W attempt $ATT reported not fixed -> reset, $LEFT left this episode"
            # Increment first, then write the prompt: the name written and the name read below must be the same.
            # It used to write a$ATT and, after incrementing, read a$((ATT+1)); the file never existed,
            # claude got an empty prompt, resume errored, and three empty-result retries killed the whole lane.
            ATT=$((ATT+1)); R=0
            # The rewind undoes the scene, but the history of the failure must stay: list for it, turn by turn, what was sent in this intervention,
            # each step's result, the hand's final state and its own verdicts, and require it to change based on them (user 2026-09-19: without a
            # record of failures it never learns and just repeats the same moves). The sim side also has a gate that rejects a plan identical to a failed attempt.
            LEDGER=$(python3 $P/recovery_explore/cli/attempt_ledger.py "$CD" "$W" "$ATT" 2>/dev/null)
            cat > "$CD/p_replan_w${W}_a$ATT.txt" << PEOF
You said it was not fixed. The simulator is back at the state this intervention
started from, every action you took is undone, and your budget is $INTERVENE_BUDGET
actions again. This is attempt $ATT; $LEFT resets left this episode.

$LEDGER

Change at least one thing based on what the attempts above taught you; never repeat one verbatim.
If what you saw means the policy should simply be left alone from here, answer
{"result": "fixed", "note": "...", "plan": []}: the scene is exactly as you took it
over, and that hands it back untouched.

$TEL

$MEMI

$PLAN_SPEC
PEOF
            claude_turn "$CD/p_replan_w${W}_a$ATT.txt" "$CD/replan_w${W}_a$ATT.json" $ACT_DEADLINE "$MONT" "Read"
            RJ=$(extract_json "$CD/replan_w${W}_a$ATT.json")
            PLAN=$(jfield "$RJ" plan); RRES=$(jfield "$RJ" result)
            if [ "$RRES" = giveup ]; then GAVEUP=1; break; fi
            if [ "$RRES" = fixed ] || [ -z "$PLAN" ] || [ "$PLAN" = "[]" ]; then
              log "  window $W chose to hand back untouched after rewind (intervention undone)"; RES=fixed; RETREAT_OK=1; break
            fi
            continue ;;
          giveup) GAVEUP=1; break ;;
          *) RES=fixed; break ;;
        esac
      done
      CB=$($H close-budget --control-token "$WTOK" 2>&1 | tail -1); USED=$(echo "$CB" | jget actions_used); RSETS=$(echo "$CB" | jget resets)
      [ "$RES" = "giveup" ] && { GAVEUP=1; log "  window $W verdict giveup -> no further interventions this episode, policy runs to the end"; }
      # An intervention that ends in retreat and hands back untouched is not counted (at most 2 refunds per episode); in rewind rounds,
      # "hand back untouched after a rewind" is refunded as well, so quota accounting is the same in both modes.
      # Since 2026-09-17 retreat counts against the quota too (-lt 0 = never refund): on seed 500 refunds let the judge back out 3-4 times
      # per episode on cup tasks, each time undoing a grasp that might have succeeded, breaking 12 episodes.
      if [ "$RES" = fixed ] && [ "${RETREAT_OK:-0}" = 1 ] && [ "${NREFUND:-0}" -lt 0 ]; then
        NINT=$((NINT-1)); NREFUND=$((NREFUND+1)); RETREATED=1
        log "  window $W ended with retreat and handed back untouched, not counted against the intervention quota (refunded $NREFUND times this episode)"
      fi
      log "  window $W verdict intervene -> $ATT attempt(s) / $NPLAN plan segment(s), ${USED:-0} action(s) used, ${RSETS:-0} reset(s), result $RES"
      # Force a "verification window" after an intervention: the next judged window may only observe and report, not act.
      # Rationale: across 408 episodes the "ok" after an intervention carried no information (ok -> 9% eventual success, re-intervene -> 6%);
      # the judge would let go right after repairing, unable to say whether it had fixed anything.
      IV_LAST_W=$W
      IV_EXPECT=$(jfield "$JJ" expect)
      IV_MEMO=$(python3 - "$CD" "$W" << 'PYM'
import json,sys,glob,os
cd,w=sys.argv[1],sys.argv[2]
ops=[]
for f in sorted(glob.glob(f"{cd}/plan_w{w}_a*_r*.json")):
    try:
        pl=json.load(open(f)); st=pl.get("plan") or pl.get("steps") or []
        ops+= [s.get("op","?") if isinstance(s,dict) else str(s) for s in st]
    except Exception: pass
stops=[]
for f in sorted(glob.glob(f"{cd}/exec_w{w}_a*_r*.json")):
    try:
        j=json.load(open(f)); r=j.get("stop_reason") or j.get("status")
        if r: stops.append(str(r))
    except Exception: pass
print(("steps: "+" + ".join(ops[:8]) if ops else "steps: (none recorded)") + ("; how they ended: "+", ".join(stops[:6]) if stops else ""))
PYM
)
      VERIFY_PENDING=1
      [ "$ASYNC" != 0 ] && echo "{\"kind\":\"iv\",\"w\":$W,\"t0\":${IV_T0:-0},\"t1\":$(date +%s.%N)}" >> "$CD/async_timing.jsonl"
    else
      [ "$ASYNC" != 0 ] || $H close-budget >/dev/null 2>&1
      log "  window $W verdict $V"
    fi
    EVENTS=$(python3 -c "
import json;l=json.loads('$EVENTS')
l.append({'window':$W,'committed':int('${CT:-0}'),'judge':'$V','actions_used':int('${USED:-0}'),'attempts':int('${ATT:-0}'),'resets':int('${RSETS:-0}'),'result':'${RES:-}','intervened_at':(int('${STALE}') if '${STALE:-}' not in ('','None') else None),'compactions':int('${COMPACTED_W:-0}'),'plan_segments':int('${NPLAN:-0}'),'retreated':int('${RETREATED:-0}'),'verify_blocked':int('${NVERIFY:-0}')})
print(json.dumps(l))")
    # async: after a repair the next window is the first one past what the policy ran while judging
    if [ -n "$ASYNC_JUMPW" ]; then W=$ASYNC_JUMPW; ASYNC_JUMPW=""; fi
  done
  [ "$ASYNC" != 0 ] && async_stop
  [ "$TS" != "True" ] && TS=$(run_to_end "$CD/tail" t)
  if [ "$LEARN" = "1" ]; then
    INBOX=$MEMORY_BANK/inbox/${TASK}_s${SEED}_ep${EP}
    mkdir -p "$INBOX"
    # API engines have no Write tool (gpt6_call.py talks to the Responses API directly). Have it return the draft between
    # <draft> tags and let the driver write it to disk below, naming the file after the id in the frontmatter
    # to satisfy the "id must equal the filename" check. codex / claude engines still Write it themselves.
    if [ "$ENGINE" = "api" ] || [ "$ENGINE" = "qwen" ]; then
      DRAFT_HOW="reply with the complete draft file -- frontmatter and body -- between a line
containing only <draft> and a line containing only </draft>, and nothing else outside
those two lines. If there is no transferable lesson, reply with the single line
NO-DRAFT followed by one sentence saying why."
    else
      DRAFT_HOW="write it as ONE memory-bank draft into $INBOX/<id>.md, where <id> is the id in
its frontmatter (lowercase words joined by hyphens, derived from the title -- never
the word draft). Otherwise write nothing and say so in one line."
    fi
    # The learn turn must see this episode's intervention ledger: verdicts, plans, each segment's stop reason, how it changed after a rewind,
    # and the next window's conclusion. After context compaction the model only has the last few windows; in l30a both failed episodes answered
    # "no early repair details in the record", so nothing was learned. The ledger is rebuilt from the JSON on disk, not from context.
    python3 - "$CD" "$EVENTS" << 'PY' > $CD/ledger.txt 2>/dev/null
import json, sys, glob, re, os
cd, events = sys.argv[1], json.loads(sys.argv[2])
def J(p):
    try: return json.load(open(p))
    except Exception: return None
def short(t, n=220): return (t or "").replace("\n", " ")[:n]
out = []
for e in events:
    if e.get("judge") != "intervene": continue
    W = e["window"]
    out.append(f"Window {W}: intervened; attempts {e.get('attempts')}, rewinds {e.get('resets')}, actions kept {e.get('actions_used')}, ended {e.get('result')}" + (", ended by retreat/undo (handed back untouched)" if e.get("retreated") else ""))
    v = J(f"{cd}/verdict_w{W}.json")
    if v: out.append(f"  diagnosis: {short(v.get('diagnosis'))}\n  first plan: {json.dumps(v.get('plan'))[:260]}")
    for a in range(1, int(e.get("attempts") or 1) + 1):
        segs = []
        for ex in sorted(glob.glob(f"{cd}/exec_w{W}_a{a}_r*.json"), key=lambda p: int(re.search(r"_r(\d+)", p).group(1))):
            x = J(ex)
            if not x: continue
            segs.append("/".join(str(st.get("op")) for st in x.get("executed", [])) + "->" + str(x.get("stop_reason")))
        if segs: out.append(f"  attempt {a}: " + " | ".join(segs))
        rp = J(f"{cd}/replan_w{W}_a{a+1}.json")
        if rp:
            try:
                rr = json.loads(rp["result"]); out.append(f"  after rewind {a}: {rr.get('result') or rr.get('verdict')} :: {short(rr.get('note') or rr.get('diagnosis'))}")
            except Exception: pass
    for w2 in range(W + 1, W + 4):
        nv = J(f"{cd}/verdict_w{w2}.json")
        if nv: out.append(f"  next judged window {w2}: {nv.get('verdict')} :: {short(nv.get('diagnosis'), 180)}"); break
print("\n".join(out) if out else "(no intervention in this episode)")
PY
    LEDGER=$(cat $CD/ledger.txt 2>/dev/null)
    RT_LEARN_NOTE=""
    if [ "$FAMILY" = robotwin ]; then RT_LEARN_NOTE="

This is a dual-arm tabletop robot (left and right arm, head camera, one wrist camera
per arm), not a kitchen. Good lessons here name the arm roles and the kind of task:
handovers, lifting one object with both hands, two objects placed at once, stacking,
articulated objects such as a laptop lid, tools such as a hammer. Useful subjects are
which arm to repair, when a missed grasp is worth repairing yourself instead of
handing back, how to tell an empty closure from a hold in the telemetry, when to rewind
and what to change, and when an episode is really lost. Never write anything you could
only know from the simulator's success check."
    fi
    cat > $CD/p_learn.txt << PEOF
This episode is over. Control arm succeeded: $CTRL. Treatment arm (with you): $TS.

Record of your interventions in this episode, rebuilt from the logs (the transcript
you can see may have been compacted; trust this):
$LEDGER

Read those two outcomes before deciding what the lesson is:
- Control succeeded and treatment failed: the policy would have finished on its own,
  and your intervention is what broke it. The lesson is about the signal that should
  have kept you out, or about what your repair did that the policy could not recover
  from -- not about counting windows in the abstract.
- Both failed and you intervened: your repair did not work. The lesson is a concrete
  alternative to try next time on this kind of object or fixture -- a different
  approach line, grasp height, who should make the close, when to hand back -- with
  Falsify saying what result next time would show it was wrong.
- Both succeeded, or control failed and treatment succeeded: say what the decisive
  observation or move was, only if it is not already obvious from the rules you were
  given. A repeat of a rule you already had is not a lesson; reply NO-DRAFT.
- Control succeeded, treatment failed, and you never intervened: the policy is not
  deterministic and this run of it failed on its own. There is nothing to learn from
  the outcome; reply NO-DRAFT unless a window you passed showed a signal worth a rule.

If this trajectory taught you something that would transfer to a different episode,
$DRAFT_HOW

A lesson has to transfer to be worth keeping. Advice about a kind of object or a kind
of task is fine: what to watch for with a fish or a mug, how a faucet lever behaves,
how tightly to close on something soft, what tends to drop during transport. What is
not fine is anything tuned to this one scene: coordinates, pixel positions, offsets in
metres, a hover height or an approach angle that worked here, "the X is left of the
Y". A number that only makes sense in this scene is not a lesson; a rule someone
could apply in a different scene with a different object is. If all you have is the
former, reply NO-DRAFT.${RT_LEARN_NOTE}

Schema, copy it exactly (the validator rejects anything else):
- frontmatter keys in this order: id, scope, kind, title, applies_when, symptom,
  evidence (cells / attempts / source), confidence, related
- symptom is a bracketed list of 5 to 12 short search keywords, e.g.
  [empty close, closed on nothing, slipped, transport, aperture] -- not a sentence
- related is a bracketed list (may be empty: [])
- id is lowercase words joined by hyphens and must equal the filename stem
- scope: global ; kind: one of primitive perception strategy failure infra
- confidence: single-shot (one episode), probable (two), verified (three or more)
- evidence.cells: [${TASK}_s${SEED}_ep${EP}] ; evidence.attempts: a single integer (how many
  episodes support it -- here 1) ; evidence.source: a short tag such as judge-learn
- body must contain the sections **Why:**, **How to apply:**, **Falsify:** in that order
- English only

Do not edit anything under memory_bank/global/ — drafts go to the inbox and a human
decides whether they are published.
PEOF
    QWEN_TURN_THINK=0 claude_turn "$CD/p_learn.txt" "$CD/learn.json" 600 "" "Read,Write"
    if [ "$ENGINE" = "api" ] || [ "$ENGINE" = "qwen" ]; then
      python3 - "$CD/learn.json" "$INBOX" << 'PY' >> $OUT/lane$LANE/driver.log 2>&1
import json, re, sys, os
r = json.load(open(sys.argv[1])).get("result", "") or ""
m = re.search(r"^<draft>\s*$(.*?)^</draft>\s*$", r, re.S | re.M)
if not m:
    print("  LEARN=1 no draft (%s)" % (r.strip().splitlines()[0][:80] if r.strip() else "empty reply")); sys.exit(0)
body = m.group(1).strip() + "\n"
mid = re.search(r"^id:\s*([a-z0-9-]+)\s*$", body, re.M)
name = (mid.group(1) if mid else "draft") + ".md"
open(os.path.join(sys.argv[2], name), "w").write(body)
print("  LEARN=1 draft written: %s" % name)
PY
    fi
    log "  LEARN=1 draft directory $INBOX"
  fi
  TREAT_S=$(( $(date +%s) - TREAT_T0 ))
  log "  treatment (with judge): $TS | screener passed ${SCREENED:-0} windows, $JUDGED judge calls, $NINT interventions, $NCOMPACT compactions (control ${CTRL_S:--1}s / treatment ${TREAT_S}s)"
  # Note: the heredoc is quoted (<< 'PY'), so the shell does no expansion; every value must come in via argv.
  # $FAMILY / ${LOOKAHEAD} used to be embedded directly in the Python, ended up as literal strings, int() threw,
  # and the result line for a whole episode was never written.
  python3 - "$CELL" "$TASK" "$SEED" "$EP" "$CTRL" "$TS" "$JUDGED" "$NINT" "$EVENTS" "$WINDOW" "$LANE" "$FAMILY" "${LOOKAHEAD:-0}" "${NCOMPACT:-0}" "${SCREENED:-0}" "${CTRL_S:--1}" "${TREAT_S:--1}" "$CD" "${ASTALE:-0}" "${NASTALE:-0}" << 'PY' >> $RESULTS
import json,sys,glob,os
c,t,s,e,ctrl,ts,nj,ni,ev,win,lane,fam,la,nc,nscr,cs,tsec,cd=sys.argv[1:19]
# Split the treatment arm's wall clock into: CLI calls (judge + intervention), screener, and the rest (sim + disk writes).
# Each turn's duration_ms is timed by claude_turn itself; the screener's dt is recorded by screen_window.py.
def _sum(pat, key, scale):
    tot = 0.0
    for f in glob.glob(os.path.join(cd, pat)):
        try:
            v = (json.load(open(f)) or {}).get(key)
            if v: tot += float(v) / scale
        except Exception: pass
    return round(tot, 1)
model_s = _sum("judge_w*.json", "duration_ms", 1000.0) + _sum("act_w*.json", "duration_ms", 1000.0) \
        + _sum("act*/*/exec_*.json", "duration_ms", 1000.0)
screen_s = _sum("screen_w*.json", "dt", 1.0)
# ASYNC_SCREEN: split the wall clock. adv_s = policy+sim time spent by the background runner,
# wait_s = time the loop sat waiting for it, so the part of adv_s hidden behind screening/judging is
# adv_s - wait_s, and the same trajectory run synchronously would have taken treat_s + that.
asy = {}
tf = os.path.join(cd, "async_timing.jsonl")
if os.path.exists(tf):
    adv = wait = iv = hold = 0.0; nadv = 0; nrule = 0
    for line in open(tf):
        try: r = json.loads(line)
        except Exception: continue
        d = float(r.get("t1", 0)) - float(r.get("t0", 0))
        if d < 0 or d > 1e5: continue
        if r.get("kind") == "adv": adv += d; nadv += 1
        elif r.get("kind") == "wait": wait += d
        elif r.get("kind") == "iv": iv += d
        elif r.get("kind") == "hold": hold += d
        elif r.get("kind") == "rule_stop": nrule += 1
    evl = json.loads(ev)
    asy = {"async": 1, "adv_s": round(adv, 1), "adv_windows": nadv, "wait_s": round(wait, 1), "iv_s": round(iv, 1),
           "hidden_s": round(adv - wait, 1), "sync_equiv_s": round(int(tsec) + adv - wait, 1),
           "async_skipped": sum(1 for x in evl if x.get("judge") == "async_skipped"),
           "async_ran_while_judging": sum(1 for x in evl if x.get("judge") == "async_ran_while_judging"),
           "async_batched": sum(1 for x in evl if x.get("judge") == "async_batched"),
           "hold_s": round(hold, 1), "rule_stops": nrule,
           "stale_steps": int(sys.argv[19]) if len(sys.argv) > 19 else None,
           "stale_ivs": int(sys.argv[20]) if len(sys.argv) > 20 else None}
print(json.dumps({**asy, "cell":c,"task":t,"seed":int(s),"episode":int(e),
 "control_success":(None if ctrl=="None" else ctrl=="True"),"treatment_success":ts=="True",
 "judge_calls":int(nj),"interventions":int(ni),"events":json.loads(ev),
 "window_chunks":int(win),"lane":int(lane),"policy_family":fam,"lookahead":int(la),
 "compactions":int(nc),"screened_windows":int(nscr),
 "ctrl_s":int(cs),"treat_s":int(tsec),"model_s":model_s,"screen_s":screen_s},ensure_ascii=False))
PY
done
log "════ judge lane$LANE finished, $(wc -l < $RESULTS | xargs) results total ════"
