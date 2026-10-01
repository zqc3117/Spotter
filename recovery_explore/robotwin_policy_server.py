"""RoboTwin 2.0 pi0.5 policy inference server (openpi websocket protocol).

Wraps RoboTwin's official adapter ``policy/pi05/deploy_policy.py`` / ``pi_model.py::PI0``
so that each request reproduces ``deploy_policy.eval`` semantics exactly:

    reset=True  -> deploy_policy.reset_model(model); model.set_language(instruction)
    every call  -> model.update_observation_window(input_rgb_arr, input_state)
                   actions = model.get_action()[:model.pi0_step]

Request (msgpack-numpy via openpi_client.WebsocketClientPolicy.infer):
    {"input_rgb_arr": [head, right, left] uint8 HxWx3  (= deploy_policy.encode_obs()[0]),
     "input_state": float32[14]                         (= deploy_policy.encode_obs()[1]),
     "instruction": str, "reset": bool}
Response: {"actions": float32[H,14], "infer_ms": float, "n_calls": int} (+ server_timing added by
WebsocketPolicyServer).

Must be started with cwd = RoboTwin root (pi_model.py uses relative checkpoint paths) and with
PYTHONPATH containing policy/pi05/src, policy/pi05/packages/openpi-client/src and the RoboTwin root.
See ops/podlib/policy_replica.sh (family ``robotwin``).

Train-config reconciliation
---------------------------
The checkpoint's ``metadata.pt`` stores the config it was actually trained with. The registered
openpi config ``pi05_robotwin_clean_multitask`` was reconstructed by hand and (as of 2026-09-19)
disagrees with metadata on ``action_horizon`` (32 vs 50) and ``adapt_to_pi`` (True vs False).
By default (``--config-source metadata``) this server overrides those fields from metadata.pt by
wrapping ``openpi.training.config.get_config`` before PI0 is built — PI0 itself is untouched.
``--config-source registered`` uses the registered config verbatim.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import logging
import os
import re
import sys
import time
import types

import numpy as np

log = logging.getLogger("robotwin_policy_server")


def _shim_lerobot() -> None:
    """openpi.training.data_loader imports ``lerobot.common.datasets.lerobot_dataset`` (lerobot<0.3
    layout); the venv has lerobot 0.4 where it is ``lerobot.datasets``. Only training uses it."""
    try:
        importlib.import_module("lerobot.common.datasets.lerobot_dataset")
        return
    except Exception:
        pass
    mod = importlib.import_module("lerobot.datasets.lerobot_dataset")
    for name in ("lerobot.common", "lerobot.common.datasets"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["lerobot.common.datasets.lerobot_dataset"] = mod


def _metadata_overrides(ckpt_dir: str) -> dict:
    """Parse action_horizon / adapt_to_pi / use_delta_joint_actions from metadata.pt's config repr."""
    path = os.path.join(ckpt_dir, "metadata.pt")
    if not os.path.exists(path):
        log.warning("no metadata.pt in %s; cannot reconcile config", ckpt_dir)
        return {}
    import torch

    meta = torch.load(path, map_location="cpu", weights_only=False)
    cfg = str(meta.get("config", ""))
    out = {}
    m = re.search(r"'model': \{[^}]*'action_horizon': (\d+)", cfg)
    if m:
        out["action_horizon"] = int(m.group(1))
    m = re.search(r"'model': \{[^}]*'max_token_len': (\d+)", cfg)
    if m:
        out["max_token_len"] = int(m.group(1))
    data_part = cfg[cfg.find("'data': {"):] if "'data': {" in cfg else ""
    for key in ("adapt_to_pi", "use_delta_joint_actions"):
        # the data-level key (not base_config's) is the last top-level occurrence before repack
        ms = re.findall(rf"'{key}': (True|False)", data_part)
        if ms:
            out[key] = ms[-1] == "True"
    log.info("metadata.pt (%s, step %s) -> %s", cfg[:80], meta.get("global_step"), out)
    return out


def _install_config_override(train_config_name: str, overrides: dict) -> dict:
    from openpi.training import config as _config

    base = _config.get_config(train_config_name)
    model_kw, data_kw = {}, {}
    for k in ("action_horizon", "max_token_len"):
        if k in overrides and getattr(base.model, k) != overrides[k]:
            model_kw[k] = overrides[k]
    for k in ("adapt_to_pi", "use_delta_joint_actions"):
        if k in overrides and hasattr(base.data, k) and getattr(base.data, k) != overrides[k]:
            data_kw[k] = overrides[k]
    if not model_kw and not data_kw:
        log.info("registered config %s already matches metadata", train_config_name)
        return {}
    fixed = dataclasses.replace(
        base,
        model=dataclasses.replace(base.model, **model_kw) if model_kw else base.model,
        data=dataclasses.replace(base.data, **data_kw) if data_kw else base.data,
    )
    orig = _config.get_config

    def get_config(name):
        return fixed if name == train_config_name else orig(name)

    _config.get_config = get_config
    diff = {**{f"model.{k}": (getattr(base.model, k), v) for k, v in model_kw.items()},
            **{f"data.{k}": (getattr(base.data, k), v) for k, v in data_kw.items()}}
    log.warning("overriding registered config %s from metadata.pt: %s", train_config_name, diff)
    return {k: v[1] for k, v in diff.items()}


def _patch_robotwin_repo_id() -> None:
    """policy_config.create_trained_policy(robotwin_repo_id=X) does ``data_config.asset_id = X`` on a
    frozen dataclass -> FrozenInstanceError. Intended behaviour: load norm stats from
    <ckpt>/assets/<X>. Do exactly that and pass them as ``norm_stats=`` instead."""
    import pathlib

    from openpi.policies import policy_config as _pc
    from openpi.training import checkpoints as _ck

    orig = _pc.create_trained_policy
    if getattr(orig, "_robotwin_patched", False):
        return

    def create_trained_policy(train_config, checkpoint_dir, *, robotwin_repo_id=None, norm_stats=None, **kw):
        if robotwin_repo_id is not None and norm_stats is None:
            norm_stats = _ck.load_norm_stats(pathlib.Path(checkpoint_dir) / "assets", robotwin_repo_id)
            log.info("norm stats loaded from %s/assets/%s", checkpoint_dir, robotwin_repo_id)
        return orig(train_config, checkpoint_dir, norm_stats=norm_stats, **kw)

    create_trained_policy._robotwin_patched = True
    _pc.create_trained_policy = create_trained_policy


class RoboTwinPi05Policy:
    """openpi BasePolicy-compatible wrapper around RoboTwin's PI0 adapter."""

    def __init__(self, deploy_policy, model, prompt_prefix: str = ""):
        self._dp = deploy_policy
        self._model = model
        self._n = 0
        # Some checkpoints (motus-robotics/pi0.5_robotwin2) were trained with a fixed scene prefix
        # prepended to every task instruction; openpi's InjectDefaultPrompt never applies it when a
        # prompt is present, so it has to be added here. Empty = unchanged behaviour.
        self._prefix = prompt_prefix or ""

    def infer(self, obs: dict) -> dict:
        t0 = time.monotonic()
        rgb = [np.asarray(x, dtype=np.uint8) for x in obs["input_rgb_arr"]]
        if len(rgb) != 3 or any(x.ndim != 3 or x.shape[-1] != 3 for x in rgb):
            raise ValueError(f"input_rgb_arr must be 3 HxWx3 uint8 arrays, got {[x.shape for x in rgb]}")
        state = np.asarray(obs["input_state"], dtype=np.float32).reshape(-1)
        if state.shape != (14,):
            raise ValueError(f"input_state must be 14-dim, got {state.shape}")
        instruction = obs.get("instruction")
        if instruction is not None and self._prefix:
            instruction = self._prefix + str(instruction)
        if bool(obs.get("reset", False)):
            self._dp.reset_model(self._model)
            self._model.set_language(instruction)
        elif getattr(self._model, "instruction", None) != instruction and instruction is not None:
            # Only happens if several clients share one replica; stay correct per request.
            log.warning("instruction changed without reset; re-setting language")
            self._model.set_language(instruction)
        self._model.update_observation_window(rgb, state)
        actions = np.asarray(self._model.get_action()[: self._model.pi0_step], dtype=np.float32)
        self._n += 1
        return {"actions": actions, "infer_ms": (time.monotonic() - t0) * 1000.0, "n_calls": self._n}

    def reset(self) -> None:
        self._dp.reset_model(self._model)

    @property
    def metadata(self) -> dict:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--train-config", default="pi05_robotwin_clean_multitask")
    ap.add_argument("--model-name", default="crelf_democlean")
    ap.add_argument("--checkpoint-id", default="35000")
    ap.add_argument("--pi0-step", type=int, default=50)
    ap.add_argument("--config-source", choices=["metadata", "registered"], default="metadata")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--prompt-prefix-from-config", action="store_true",
                    help="prepend the train config's data.default_prompt to every instruction "
                         "(for checkpoints trained with a scene prefix; off by default)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", force=True)

    robotwin_root = os.getcwd()
    pi05_dir = os.path.join(robotwin_root, "policy", "pi05")
    if not os.path.isfile(os.path.join(pi05_dir, "deploy_policy.py")):
        sys.exit(f"cwd must be the RoboTwin root (no policy/pi05/deploy_policy.py under {robotwin_root})")
    if pi05_dir not in sys.path:
        sys.path.append(pi05_dir)

    _shim_lerobot()
    ckpt_dir = os.path.join("policy", "pi0", "checkpoints", args.train_config, args.model_name, str(args.checkpoint_id))
    applied = {}
    if args.config_source == "metadata":
        applied = _install_config_override(args.train_config, _metadata_overrides(ckpt_dir))

    _patch_robotwin_repo_id()
    import pi_model  # noqa: E402  (RoboTwin adapter)

    if not hasattr(pi_model, "os"):  # pi_model.py calls os.listdir without importing os
        pi_model.os = os
    deploy_policy = importlib.import_module("deploy_policy")

    t0 = time.monotonic()
    model = deploy_policy.get_model({
        "train_config_name": args.train_config,
        "model_name": args.model_name,
        "checkpoint_id": args.checkpoint_id,
        "pi0_step": args.pi0_step,
    })
    log.info("model loaded in %.1fs (config overrides: %s)", time.monotonic() - t0, applied)
    prefix = ""
    if args.prompt_prefix_from_config:
        from openpi.training import config as _config

        prefix = getattr(_config.get_config(args.train_config).data, "default_prompt", None) or ""
        if not prefix:
            sys.exit(f"--prompt-prefix-from-config: {args.train_config} has no data.default_prompt")
        log.warning("prompt prefix ON: %r", prefix)
    policy = RoboTwinPi05Policy(deploy_policy, model, prompt_prefix=prefix)

    if not args.no_warmup:
        t0 = time.monotonic()
        dummy = {"input_rgb_arr": [np.zeros((240, 320, 3), np.uint8)] * 3,
                 "input_state": np.zeros(14, np.float32), "instruction": "warmup", "reset": True}
        a = policy.infer(dummy)["actions"]
        policy.reset()
        log.info("warmup ok: actions %s in %.1fs", a.shape, time.monotonic() - t0)

    from openpi.serving import websocket_policy_server

    server = websocket_policy_server.WebsocketPolicyServer(
        policy, host=args.host, port=args.port,
        metadata={"family": "robotwin", "train_config": args.train_config, "model_name": args.model_name,
                  "checkpoint_id": str(args.checkpoint_id), "pi0_step": args.pi0_step,
                  "config_overrides": {k: str(v) for k, v in applied.items()}, "prompt_prefix": prefix})
    log.info("serving on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
