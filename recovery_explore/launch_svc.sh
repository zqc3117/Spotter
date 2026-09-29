#!/bin/bash
# Start one env_service. Works on any host and is shipped with the repo.
# Binds 0.0.0.0: in the multi-host topology the driver runs on another host and connects by IP.
# env_service binds only 127.0.0.1 by default, so local health checks pass while every
# cross-host connection gets "Connection refused".
#   $1 = cuda device   $2 = port   $3 = policy endpoint URL (may be on another host)
#   $4 = policy family (cosmos | pi05 | robodojo | robotwin, default cosmos)   $5 = run-id prefix (default rounds)
#
# The robodojo family uses a different module (env_service_robodojo) and a different
# interpreter: RoboCasa is MuJoCo + the robocasa venv, RoboDojo is Isaac Sim + its own conda
# env. Otherwise it looks identical to the layers above -- same port, same harness
# subcommands, same 0.0.0.0 binding -- so the lane launcher / judge driver only need the
# family name, not the differences between the two.
set -u
cuda=$1; port=$2; policy_url=$3; family=${4:-cosmos}; prefix=${5:-rounds}
# MAX_RESETS is passed in by the caller as an environment variable (the judge driver's reset limit), default 5.
export MAX_RESETS=${MAX_RESETS:-5}
# STEP_BUDGET_SCALE likewise comes from the caller (--budget-scale), default 1.0 = official TASK_MAX_STEPS.
export STEP_BUDGET_SCALE=${STEP_BUDGET_SCALE:-1.0}
# The repo root is derived from this script's location, so the checkout can live anywhere.
# JUDGE_REPO_ROOT overrides it.
P=${JUDGE_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

# ---- Before starting the service, make sure the port is actually free ----------
# env_service uses --port-tries 1 (the caller expects exactly 8450+lane and cannot shift),
# so if the previous process releases the port a little late, the new one fails with
#   RuntimeError: no free port in [<port>, <port>+1)
# and the caller only sees "service failed to start" -- the whole lane is wasted.
#
# Why not wait on the process name: PID 1 in these containers is often `sleep infinity` or
# similar, which does not reap children, so zombies can pile up into the hundreds; and once
# a process is a zombie its cmdline is empty -- `ps | grep -- "--port N"` would conclude
# "old service gone" in the window where it no longer matches but its fds are not yet
# released, and the new service would hit the occupied port.
# So wait on the port itself. If it never frees up, kill whoever holds it (the port is
# dedicated to this system).
#   SVC_PORT_WAIT=<seconds>  maximum wait (default 45)
#   SVC_PORT_KILL=0          wait only, never kill
svc_wait_port(){
  python3 - "${SVC_HOST:-0.0.0.0}" "$1" "${SVC_PORT_WAIT:-45}" "${SVC_PORT_KILL:-1}" <<'PY'
import os, signal, socket, sys, time

host, port, budget, may_kill = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == "1"

def listeners(port):
    """PIDs holding a LISTEN socket on this port, found through /proc (not the process table)."""
    want = set()
    try:
        with open("/proc/net/tcp") as fh:
            for line in fh.read().splitlines()[1:]:
                f = line.split()
                if len(f) > 9 and f[3] == "0A" and int(f[1].split(":")[1], 16) == port and f[9] != "0":
                    want.add("socket:[%s]" % f[9])
    except OSError:
        return []
    if not want:
        return []
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        fd_dir = "/proc/%s/fd" % entry
        try:
            names = os.listdir(fd_dir)
        except OSError:
            continue
        for name in names:
            try:
                if os.readlink(os.path.join(fd_dir, name)) in want:
                    pids.append(int(entry))
                    break
            except OSError:
                continue
    return pids

deadline = time.time() + budget
killed = set()
while True:
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((host, port))
        probe.close()
        sys.exit(0)
    except OSError as exc:
        probe.close()
        last = exc
    if may_kill:
        for pid in listeners(port):
            if pid == os.getpid() or pid in killed:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                sys.stderr.write("port %d was held by pid %d, killed it\n" % (port, pid))
                killed.add(pid)
            except OSError:
                pass
    if time.time() >= deadline:
        sys.stderr.write("port %d still busy after %.0fs: %s\n" % (port, budget, last))
        sys.exit(1)
    time.sleep(1)
PY
}
svc_wait_port "$port" || { echo "port $port never became free; not starting the service" >&2; exit 1; }

if [ "$family" = robodojo ]; then
  # RoboDojo (Isaac Sim 6.0.1 + Isaac Lab 3.0, ARX X5 dual-arm), installed on shared storage.
  #
  # The environment is not reassembled here; we source /workspace/robodojo/isaac6_rd6_env.sh
  # directly -- the single source of truth for the setup that passed official validation
  # (12.50 vs 11.41 on the leaderboard): conda activate RoboDojo6, LD_LIBRARY_PATH pointing
  # at that env's lib, Vulkan ICD, EULA, and unset CUDA_VISIBLE_DEVICES.
  # An earlier hand-copied version used Isaac 5.1-era defaults (RoboDojo + envs/RoboDojo,
  # python 3.11) and always segfaulted in librtx.scenedb -- Isaac 5.1 does not run on driver
  # 595. A hand copy would drift again, so reference the script instead.
  RD_INSTALL=${ROBODOJO_ROOT:-/workspace/robodojo}
  RD_ENVSH=${ROBODOJO_ENVSH:-$RD_INSTALL/isaac6_rd6_env.sh}
  [ -f "$RD_ENVSH" ] || { echo "RoboDojo6 env script not found: $RD_ENVSH" >&2; exit 1; }
  # shellcheck disable=SC1090
  . "$RD_ENVSH"
  # Proxies only matter for downloads; the simulator talks to in-cluster IPs, so a proxy
  # (pulled in by the env script) would only add a detour.
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  RD_ROOT=${ROBODOJO_REPO:-${ROBODOJO6:-$RD_INSTALL/RoboDojo6}}
  RD_PY=${ROBODOJO_PYTHON:-$RD_INSTALL/envs/RoboDojo6/bin/python}
  [ -x "$RD_PY" ] || { echo "RoboDojo interpreter not found: $RD_PY (override with ROBODOJO_PYTHON)" >&2; exit 1; }
  [ -d "$RD_ROOT" ] || { echo "RoboDojo repo not found: $RD_ROOT (override with ROBODOJO_REPO)" >&2; exit 1; }
  [ -d "$RD_ROOT/src/eval_client" ] || { echo "$RD_ROOT does not look like a RoboDojo repo (missing src/eval_client)" >&2; exit 1; }
  cd "$P"
  mkdir -p "recovery_explore_runs/${prefix}_$port"
  # CUDA_VISIBLE_DEVICES must stay unset: with it set, Omniverse marks every GPU as
  # "CUDA in bad state" and skips them all, and the renderer then segfaults (observed on RTX 5090).
  # The GPU is selected via --cuda-device, which the module turns into Isaac's --device cuda:N.
  unset CUDA_VISIBLE_DEVICES
  # The two "family" values are different things: the judge driver's family (robodojo) names
  # the benchmark, env_service's --policy-family names the policy, and the policy run on
  # RoboDojo is pi0.5. Passing the family name straight through is rejected by argparse
  # (invalid choice: 'robodojo').
  # PYTHONDONTWRITEBYTECODE: code on shared storage changes often, and stale .pyc files do bite.
  exec env PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$P:$RD_ROOT:$RD_ROOT/XPolicyLab" \
    ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES \
    MAX_RESETS="$MAX_RESETS" "$RD_PY" \
    -u -m recovery_explore.env_service_robodojo \
    --output-dir "recovery_explore_runs/${prefix}_$port" --run-id "${prefix}_$port" \
    --host "${SVC_HOST:-0.0.0.0}" --cuda-device "$cuda" --port "$port" --port-tries 1 --step-budget-scale "$STEP_BUDGET_SCALE" \
    --robodojo-root "$RD_ROOT" ${ROBODOJO_ACTION_TYPE:+--action-type "$ROBODOJO_ACTION_TYPE"} \
    ${ROBODOJO_ENV_CFG:+--env-cfg-type "$ROBODOJO_ENV_CFG"} ${ROBODOJO_LAYOUT_SEED:+--layout-seed "$ROBODOJO_LAYOUT_SEED"} \
    --policy-family "${ROBODOJO_POLICY_FAMILY:-pi05}" --policy-endpoint "$policy_url"
fi

if [ "$family" = robotwin ]; then
  # RoboTwin 2.0 (SAPIEN 3, aloha-agilex dual-arm). Interpreter and tree reuse LingBot's
  # validated runtime (py3.10 / sapien 3.0.0b1 / mplib 0.2.1 / torch 2.9 cu128); nothing extra installed.
  #   ROBOTWIN_PY          simulator interpreter (default: the portable runtime's .venv)
  #   ROBOTWIN_ROOT        RoboTwin tree (portable worktree: code is symlinked, assets are real files)
  #   ROBOTWIN_TASK_CONFIG demo_clean | demo_randomized (default demo_randomized)
  #   ROBOTWIN_RT_DENOISER default optix: on RTX 5090 the official oidn fails with "OIDN Error: invalid handle";
  #                        LingBot's own RTX 5090 evaluation also uses optix
  # One service per port; several can share a GPU (~7.5 GB VRAM each).
  RT_WS=${ROBOTWIN_RUNTIME:?set ROBOTWIN_RUNTIME to the RoboTwin portable runtime workspace}
  RT_ROOT=${ROBOTWIN_ROOT:-$RT_WS/worktree/third_party/RoboTwin}
  RT_PY=${ROBOTWIN_PY:-$RT_WS/.venv/bin/python}
  [ -x "$RT_PY" ] || { echo "RoboTwin interpreter not found: $RT_PY (override with ROBOTWIN_PY)" >&2; exit 1; }
  [ -f "$RT_ROOT/envs/_base_task.py" ] || { echo "$RT_ROOT does not look like a RoboTwin tree (missing envs/_base_task.py)" >&2; exit 1; }
  [ -d "$RT_ROOT/assets/embodiments" ] || { echo "$RT_ROOT/assets is missing embodiments (assets not downloaded or symlink broken)" >&2; exit 1; }
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  # GPU selection: SAPIEN's render device follows the CUDA device, and the Mesa
  # device-select layer pins Vulkan by UUID. The module's --cuda-device does the same;
  # setting it here too means child processes (cuRobo etc.) see the same GPU from the start.
  export CUDA_VISIBLE_DEVICES="$cuda" ROBOTWIN_VULKAN_GPU="$cuda" NVIDIA_VISIBLE_DEVICES="$cuda"
  if [ -z "${VK_ICD_FILENAMES:-}" ]; then
    for icd in /usr/share/vulkan/icd.d/nvidia_icd.json ${VK_ICD_FALLBACK:-/dev/null}; do
      [ -f "$icd" ] && { export VK_ICD_FILENAMES="$icd"; break; }
    done
  fi
  export VK_LOADER_LAYERS_ENABLE=${VK_LOADER_LAYERS_ENABLE:-VK_LAYER_MESA_device_select}
  export MESA_VK_DEVICE_SELECT=${MESA_VK_DEVICE_SELECT:-$(nvidia-smi --id="$cuda" --query-gpu=uuid --format=csv,noheader 2>/dev/null)}
  export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}
  export ROBOTWIN_RT_DENOISER=${ROBOTWIN_RT_DENOISER:-optix}
  cd "$P"
  mkdir -p "recovery_explore_runs/${prefix}_$port"
  # PYTHONPATH: this repo, the RoboTwin tree, the in-tree cuRobo sources (the venv's nvidia_curobo
  # .pth points at /source/... on the machine that built it, which is broken here), and the
  # openpi-client bundled with pi05 (pure python).
  exec env PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore::UserWarning \
    PYTHONPATH="$P:$RT_ROOT:$RT_ROOT/envs/curobo/src:$RT_ROOT/policy/pi05/packages/openpi-client/src" \
    MAX_RESETS="$MAX_RESETS" "$RT_PY" \
    -u -m recovery_explore.env_service_robotwin \
    --output-dir "recovery_explore_runs/${prefix}_$port" --run-id "${prefix}_$port" \
    --host "${SVC_HOST:-0.0.0.0}" --cuda-device "$cuda" --port "$port" --port-tries 1 --step-budget-scale "$STEP_BUDGET_SCALE" \
    --robotwin-root "$RT_ROOT" --task-config "${ROBOTWIN_TASK_CONFIG:-demo_randomized}" \
    ${ROBOTWIN_SEED_CACHE:+--seed-cache "$ROBOTWIN_SEED_CACHE"} \
    --policy-family "${ROBOTWIN_POLICY_FAMILY:-pi05}" --policy-endpoint "$policy_url"
fi

source "${COSMOS_COMMON_ENV:-${COSMOS_POLICY_ROOT:-/workspace/cosmos_policy}/scripts/common_env.sh}" >/dev/null 2>&1
cd "$P"
mkdir -p "recovery_explore_runs/${prefix}_$port"
# The shared robocasa venv's bin/python -> /usr/bin/python3, which is 3.12 on Ubuntu 24.04 hosts
# -> "No module named numpy". Such hosts can carry a local shim venv at /opt/robocasa_env
# (bin/python -> /usr/bin/python3.10, lib -> the shared venv); prefer it when present.
RC_ENV=${COSMOS_POLICY_ENV:-/workspace/cosmos_policy/envs/robocasa}
# (common_env.sh above exports COSMOS_POLICY_ENV=the shared venv, so test the path, not "unset")
case "$RC_ENV" in /workspace/cosmos_policy/envs/robocasa|/workspace/cosmos_policy/envs/robocasa/)
  [ -x /opt/robocasa_env/bin/python ] && RC_ENV=/opt/robocasa_env ;; esac
exec env PYTHONFAULTHANDLER=1 MAX_RESETS="$MAX_RESETS" PYTHONPATH="$PWD" "$RC_ENV/bin/python" \
  -u -m recovery_explore.env_service \
  --output-dir "recovery_explore_runs/${prefix}_$port" --run-id "${prefix}_$port" \
  --host "${SVC_HOST:-0.0.0.0}" --cuda-device "$cuda" --port "$port" --port-tries 1 --debug-resets 0 --step-budget-scale "$STEP_BUDGET_SCALE" \
  --policy-family "$family" --policy-endpoint "$policy_url"
