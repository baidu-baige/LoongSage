"""Actor worker for the full DeepSeek-V4.1-Flash on one RTX 4090 (``dsv41_4090.stream``).

Same surface as ``MegatronActorWorker`` (init/train/onload/offload/update_weights/save_model).
The model keeps every decoder layer on disk (one shared GPU pool, paged per layer; see
``streaming.py``) and trains LoRA adapters on the dense parts of the tail layers.

Each step runs in a short-lived subprocess (``streaming.main``) so that the colocated SGLang
engine gets the whole GPU back afterwards: its weights alone need ~19.5 of the 22 GB. Weight sync
is file based: the step merges the adapters into the dense FP8 weights of ``hf_model_path`` (the
serving checkpoint), and the rollout engines reload it from disk (see ``patch.py``). Routed experts never change, so the engine's file-backed expert pages stay valid.

The saved config supplies ``dsv41_4090.base_ckpt`` (pristine model/merge source) and
``dsv41_4090.orig_ckpt`` (engram tables); no separate path arguments or environment are needed.
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
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# coda/custom/dsv41_4090/stream_worker.py -> repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STEP_ENTRY = "import sys; from coda.custom.dsv41_4090.streaming import main; sys.exit(main())"


class MegatronStreamActorWorker:
    def __init__(self, world_size: int = 1, rank: int = 0):
        assert world_size == 1, "the streaming actor runs on a single GPU"
        self.world_size, self.rank = world_size, rank

    @staticmethod
    def validate_config(config):
        if config.critic.enable:
            raise ValueError("MegatronStreamActorWorker does not support a critic")
        if config.opd.enable or config.algorithm.ref_kl.enable:
            raise ValueError("Streaming currently supports policy training without OPD/ref-KL")
        if config.trainer.forward_only or config.trainer.qat or config.trainer.use_fp32_lm_head:
            raise ValueError("Streaming uses adapter-only training and merged-weight fake quantization; "
                             "forward_only, trainer.qat and use_fp32_lm_head are not supported")
        if config.trainer.num_nodes * config.trainer.num_gpus_per_node != 1:
            raise ValueError("Streaming requires exactly one training GPU")
        for key in ("tensor_model_parallel_size", "pipeline_model_parallel_size",
                    "context_parallel_size", "expert_model_parallel_size", "expert_tensor_parallel_size"):
            if config.megatron.model.get(key, 1) != 1:
                raise ValueError(f"Streaming requires {key}=1")
        from coda.backends.megatron.megatron_actor_worker import MegatronActorWorker

        return MegatronActorWorker.validate_config(config)

    def get_ip_port(self):
        return "127.0.0.1", 0

    def init(self, config, master_addr=None, master_port=None):
        self.config = config
        self.work_dir = os.path.join(str(config.checkpoint_path), "stream_state")
        os.makedirs(self.work_dir, exist_ok=True)
        return {"world_size": 1, "rank": 0}

    def train(self, step, rollout_data_ref):
        t0 = time.time()
        batch = ray.get(rollout_data_ref[0].ref)
        # Preserve the standard batch, including routing records and segment/group metadata.
        torch.save(batch, os.path.join(self.work_dir, "rollout_batch.pt"))
        OmegaConf.save(self.config, os.path.join(self.work_dir, "config.yaml"))
        log_path = os.path.join(self.work_dir, f"train_step{step}.log")
        cmd = [sys.executable, "-c", STEP_ENTRY, "--work-dir", self.work_dir, "--step", str(step)]
        with open(log_path, "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                cwd=REPO_ROOT).returncode
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
