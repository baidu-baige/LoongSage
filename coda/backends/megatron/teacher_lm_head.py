"""Teacher ``lm_head`` management for OPD full-vocabulary KL methods.

The full-vocab KL methods (``full_kl`` / ``full_jsd``) reconstruct teacher
full logits from teacher hidden states using a TP-sharded copy of the teacher's
``lm_head``. :class:`TeacherLMHeads` owns the full lifecycle of those weights —
loading, on/offloading, and hidden→logits reconstruction — as a per-process
singleton owned by the train worker, so the math-only policy layer
(:mod:`coda.algorithms.kl_policy`) stays unaware of these Megatron/TP details.
"""

from __future__ import annotations

import json
import logging
import os

import torch
import torch.nn as nn
from omegaconf import DictConfig
from safetensors.torch import load_file
from transformers import AutoConfig

from megatron.core import parallel_state as mpu

from coda.backends.megatron.checkpoint import load_tensor_from_checkpoint
from coda.utils.checkpoint_utils import resolve_dist_ckpt_dir

logger = logging.getLogger(__name__)

# Megatron ties output_layer to the word embedding when
# share_embeddings_and_output_weights is set (the GPTModelProvider default), and
# then only stores the embedding key. Order matters: an untied checkpoint has
# both, and output_layer.weight is the authoritative one.
_CKPT_LM_HEAD_KEYS = ("output_layer.weight", "embedding.word_embeddings.weight")


def _load_lm_head_weight(model_path: str) -> torch.Tensor:
    """Load lm_head weight from a local HF checkpoint.

    Supports tied weights (falls back to ``model.embed_tokens.weight``) and
    sharded checkpoints (resolved through ``*.index.json``).  Returns the
    weight in whatever dtype it was stored in; the caller is responsible for
    any dtype conversion (e.g., FP32 promotion).
    """
    config = AutoConfig.from_pretrained(model_path)

    # Resolve weight_map for sharded checkpoints
    weight_map = None
    use_safetensors = True
    for index_name, st in [("model.safetensors.index.json", True), ("pytorch_model.bin.index.json", False)]:
        index_path = os.path.join(model_path, index_name)
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                weight_map = json.load(f)["weight_map"]
            use_safetensors = st
            break

    # Determine target key (handle tied embeddings)
    target_key = "lm_head.weight"
    if weight_map and target_key not in weight_map:
        if getattr(config, "tie_word_embeddings", True):
            target_key = "model.embed_tokens.weight"

    # Locate checkpoint file containing the target key
    if weight_map:
        checkpoint_file = os.path.join(model_path, weight_map[target_key])
    else:
        for name, st in [("model.safetensors", True), ("pytorch_model.bin", False)]:
            candidate = os.path.join(model_path, name)
            if os.path.exists(candidate):
                checkpoint_file = candidate
                use_safetensors = st
                break
        else:
            raise FileNotFoundError(f"No checkpoint file found in {model_path}")

    state_dict = load_file(checkpoint_file) if use_safetensors else torch.load(checkpoint_file, map_location="cpu")

    if "lm_head.weight" in state_dict:
        return state_dict["lm_head.weight"]
    if "model.embed_tokens.weight" in state_dict:
        return state_dict["model.embed_tokens.weight"]
    raise ValueError(f"Neither 'lm_head.weight' nor 'model.embed_tokens.weight' found in {checkpoint_file}")


class TeacherLMHeads:
    """Per-process owner of teacher ``lm_head`` modules + hidden→logits rebuild.

    Singleton: built once on the last PP stage by the train worker when an
    active KL policy needs teacher logits. One TP-sharded ``nn.Linear`` per
    distinct teacher weight *source* -- its dist checkpoint when
    ``dist_ckpt_path`` is set, else its ``hf_path`` (teachers sharing a source
    share a module; ``_by_source`` dedups device moves), with ``_by_idx`` mapping
    each teacher index to its module for reconstruction.
    """

    _instance: "TeacherLMHeads | None" = None

    def __init__(self, config: DictConfig):
        self.config = config
        self._by_source: dict[str, nn.Linear] = {}
        self._by_idx: dict[int, nn.Linear] = {}

    # ── singleton access ──────────────────────────────────────────────
    @classmethod
    def get(cls) -> "TeacherLMHeads | None":
        """Return the process-wide instance (or ``None`` if not set up)."""
        return cls._instance

    @classmethod
    def setup(cls, config: DictConfig) -> "TeacherLMHeads":
        """Build + load the teacher lm_heads once and store as the singleton."""
        inst = cls(config)
        inst._load()
        cls._instance = inst
        return inst

    @classmethod
    def reset(cls) -> None:
        """Clear the singleton (re-init safety / tests)."""
        cls._instance = None

    # ── lifecycle ─────────────────────────────────────────────────────
    def _load(self) -> None:
        """Build one TP-sharded lm_head per distinct teacher weight source.

        Keying the dedup cache on the resolved source (checkpoint when set, else
        ``hf_path``) keeps two teachers that share an HF config but differ in
        checkpoint from collapsing into one head.
        """
        for idx, teacher in enumerate(self.config.opd.teachers):
            ckpt_dir = resolve_dist_ckpt_dir(
                teacher.get("dist_ckpt_path"), f"opd.teachers[{idx}].dist_ckpt_path"
            )
            source = ckpt_dir or teacher.hf_path
            if source not in self._by_source:
                self._by_source[source] = self._build(source, from_ckpt=ckpt_dir is not None)
            self._by_idx[idx] = self._by_source[source]

    def _build(self, source: str, from_ckpt: bool = False) -> nn.Linear:
        """Build a TP-sharded teacher lm_head on GPU from *source*."""
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        if from_ckpt:
            _, weight = load_tensor_from_checkpoint(source, _CKPT_LM_HEAD_KEYS)
        else:
            weight = _load_lm_head_weight(source)  # [vocab_size, hidden_size]

        # TP slice: partition vocab dim (dim=0) like ColumnParallelLinear
        per_partition_size = weight.size(0) // tp_size
        start = tp_rank * per_partition_size
        end = start + per_partition_size
        weight_local = weight[start:end].contiguous()

        lm_head = nn.Linear(weight_local.size(1), weight_local.size(0), bias=False)
        lm_head.weight = nn.Parameter(weight_local, requires_grad=False)
        lm_head = lm_head.to(device=torch.cuda.current_device(), dtype=self._dtype()).eval()

        logger.info(
            f"Loaded teacher lm_head from {source}, local shape: {tuple(weight_local.shape)}"
        )
        return lm_head

    def _dtype(self) -> torch.dtype:
        """Target dtype for the lm_head weight.

        Must be stated explicitly rather than inherited from the weight file: it
        has to match the dtype of the teacher hidden states it consumes, and the
        source dtype does not. HF safetensors happen to be bf16, but a Megatron
        dist checkpoint stores ``output_layer.weight`` in whatever dtype the run
        that produced it used (fp32 when it had ``use_fp32_lm_head`` on), which
        would otherwise reach ``F.linear`` as a bf16-vs-fp32 mismatch.
        """
        if self.config.trainer.use_fp32_lm_head:
            return torch.float32
        if self.config.opd.model.bf16:
            return torch.bfloat16
        if self.config.opd.model.fp16:
            return torch.float16
        return torch.float32

    def onload(self, device) -> None:
        """Move every distinct lm_head to *device*."""
        for lm in self._by_source.values():
            lm.to(device)

    def offload(self) -> None:
        """Move every distinct lm_head to CPU."""
        for lm in self._by_source.values():
            lm.to("cpu")
