"""Actor worker for the full DeepSeek-V4.1-Flash on one RTX 4090 (``dsv41_4090.stream``).

Same surface as ``MegatronActorWorker`` (init/train/onload/offload/update_weights/save_model).
The model keeps every decoder layer on disk (one shared GPU pool, paged per layer; see
``streaming.py``) and trains LoRA adapters on the dense parts of the tail layers.

Each step runs in a short-lived subprocess (``streaming.main``) so that the colocated SGLang
engine gets the whole GPU back afterwards: its weights alone need ~19.5 of the 22 GB. Weight sync
is file based: the step merges the adapters into the dense FP8 weights of ``hf_model_path`` (the
serving checkpoint), and the rollout engines reload it from disk (see ``patch.py``). Routed experts never change, so the engine's file-backed expert pages stay valid.

Environment (``trainer.env_vars``): ``STREAM_BASE_CKPT`` is the pristine pruned checkpoint the
model is built from and the merge starts from; ``STREAM_ENGRAM_CKPT`` holds the engram tables.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time

import ray
import torch

logger = logging.getLogger(__name__)

# coda/custom/dsv41_4090/stream_worker.py -> repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STEP_ENTRY = "import sys; from coda.custom.dsv41_4090.streaming import main; sys.exit(main())"


def to_trajectories(batch):
    """LoongSage RolloutBatch -> [{group, prompt_ids, resp_ids, old_logp, mask, reward}]."""
    out = []
    old_all = batch.get("rollout_log_probs")
    for i in range(len(batch["tokens"])):
        tokens = batch["tokens"][i].tolist()
        resp_len = int(batch["response_lengths"][i])
        total = int(batch["total_lengths"][i]) if "total_lengths" in batch else len(tokens)
        tokens = tokens[:total]
        old = old_all[i].float().tolist()[:resp_len] if old_all else [0.0] * resp_len
        mask = batch["loss_masks"][i].float().tolist()[:resp_len]
        out.append({"group": str(batch["prompt_id"][i]), "prompt_ids": tokens[:total - resp_len],
                    "resp_ids": tokens[total - resp_len:], "old_logp": old, "mask": mask,
                    "has_old": bool(old_all), "reward": float(batch["rewards"][i])})
    return out


class MegatronStreamActorWorker:
    def __init__(self, world_size: int = 1, rank: int = 0):
        assert world_size == 1, "the streaming actor runs on a single GPU"
        self.world_size, self.rank = world_size, rank

    @staticmethod
    def validate_config(config):
        if config.critic.enable:
            raise ValueError("MegatronStreamActorWorker does not support a critic")
        return config

    def get_ip_port(self):
        return "127.0.0.1", 0

    def init(self, config, master_addr=None, master_port=None):
        self.config = config
        self.serve_ckpt = str(config.hf_model_path)
        self.base_ckpt = os.environ["STREAM_BASE_CKPT"]
        self.engram_ckpt = os.environ["STREAM_ENGRAM_CKPT"]
        self.work_dir = os.path.join(str(config.checkpoint_path), "stream_state")
        os.makedirs(self.work_dir, exist_ok=True)
        return {"world_size": 1, "rank": 0}

    def train(self, step, rollout_data_ref):
        t0 = time.time()
        batch = ray.get(rollout_data_ref[0].ref)
        trajectories = to_trajectories(batch)
        torch.save(trajectories, os.path.join(self.work_dir, "trajectories.pt"))
        log_path = os.path.join(self.work_dir, f"train_step{step}.log")
        cmd = [sys.executable, "-c", STEP_ENTRY,
               "--work-dir", self.work_dir, "--serve-ckpt", self.serve_ckpt,
               "--base-ckpt", self.base_ckpt, "--engram-ckpt", self.engram_ckpt,
               "--lr", str(float(self.config.megatron.optimizer.lr))]
        with open(log_path, "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                cwd=REPO_ROOT, env=dict(os.environ)).returncode
        if rc != 0:
            tail = open(log_path).read()[-3000:]
            raise RuntimeError(f"stream train step {step} failed (rc={rc}); log {log_path}:\n{tail}")
        metrics = json.load(open(os.path.join(self.work_dir, "metrics.json")))
        metrics["perf/train_total_seconds"] = time.time() - t0
        line = " - ".join(f"{k}:{v}" for k, v in metrics.items())
        print(f"step:{step} - {line}", flush=True)
        logger.info("[stream] step %d: %s", step, metrics)
        with open(os.path.join(self.work_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
        return metrics

    def update_weights(self, channel_meta=None):
        return None

    def onload(self):
        return None

    def offload(self, move_params: bool = False):
        return None

    def save_model(self, step, checkpoint=True, hf=False):
        src = os.path.join(self.work_dir, "adapter.pt")
        if step is not None and os.path.exists(src):
            import shutil

            shutil.copyfile(src, os.path.join(self.work_dir, f"adapter_step{step}.pt"))
        return None
