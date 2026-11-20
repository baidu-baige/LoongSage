"""Tree training (``trainer.packer=tree``): pack a trajectory's rows so shared prefixes are
computed once.

In a multi-turn agent trajectory, each training row (one Segment) repeats the whole
conversation so far, so consecutive rows share long prefixes. The tree packer packs a
micro-batch's rows into one tree-shaped node buffer: every token is computed once in the
forward and backward passes, and per-row log-probs / entropy match the row-wise path.

How it works:
* ``core/plan.py``: consecutive rows of a trajectory form a chain where row k shares a
  prefix of length ``bp_k`` with row k-1. Each chain is cut into trees under a node
  budget, at the cheapest estimated cut points (``cut: min_cost``) or greedily.
* ``core/batch.py``: every node is stored once; its position id is its root-path depth and
  it carries the R3 routing of the last row that contains it.
* ``megatron/adapters/``: only sequence mixers need the tree layout. Attention gathers
  each piece's ancestors as K/V (FlashAttention), GDN chains ``final_state`` into
  ``initial_state`` along the tree; everything else is per-token and runs unchanged.
* ``megatron/loss.py``: one CE over the nodes plus a small extra CE for branch-point
  nodes whose target differs between rows. The LM head runs on the response rows only.

Requirements: ``context_parallel_size == 1``, OPD/GKD off, no image inputs,
``megatron.model.attention_backend: flash``, ``gdn_pre_gated_delta_rule_fusion: false``,
``trainer.use_dynamic_batch_size: true``, a model with explicit mRoPE position ids, and
every layer's sequence mixer covered by an adapter. When one does not hold, the trainer
warns and falls back to rows. The critic and forward passes with a PG-KL policy always
use rows.

Options (``trainer.packer_options``):
* ``cut`` (default ``min_cost``): tree cut points, ``min_cost`` or ``greedy``.
* ``max_nodes_per_gpu`` (default null = ``max_tokens_per_gpu``): node budget per
  micro-batch. Trees hold fewer activations per token than rows, so a larger budget
  usually fits; tune per parallel config and GPU memory.
* ``max_lm_rows_per_gpu`` (default null = ``max_tokens_per_gpu``): logit rows per tree and
  micro-batch, bounding the last PP stage under large node budgets.
* ``lm_head_rows`` (default true): LM head on the loss rows only; off automatically with MTP.
* ``r3_stats`` (default true): log ``tree/r3_mismatch_*`` in training.
* ``gdn_level_tokens`` (default 16000): GDN tokens one serial kernel call is worth vs
  re-read prefix tokens (the same math either way).
* ``dp_balance`` (default true): balance DP shards by estimated tree cost.

New model families: add a module under ``megatron/adapters/`` that runs their sequence
mixer on micro-batches whose ``packed_seq_params`` carries ``tree_pieces``, and register
it with ``register_layer_adapter(LayerAdapter("name", matches, install))``.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from coda.utils.multimodal_config import multimodal_enabled
from coda.utils.packing import MicroBatchPacker, PackedInputs, PackPlan, register_packer

from coda.custom.tree_pack.core.cost import chain_cost

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS = {
    "cut": "min_cost",            # tree cut points: min_cost (cheapest estimate) or greedy
    "max_nodes_per_gpu": None,    # node budget per micro-batch (None: max_tokens_per_gpu)
    "max_lm_rows_per_gpu": None,  # logit rows per tree and micro-batch (None: max_tokens_per_gpu)
    "lm_head_rows": True,         # LM head on the loss rows only (off with MTP)
    "r3_stats": True,             # log tree/r3_mismatch_* in training
    "gdn_level_tokens": 16000,    # GDN: tokens one serial call is worth vs re-read tokens
    "dp_balance": True,           # balance DP shards by estimated tree cost
}


class _ChunkCache:
    """Under VPP every model chunk runs ``forward_step`` on the same micro-batch: build its
    tree once and share it. ``first`` marks the building call; entries are dropped after the
    last chunk."""

    def __init__(self, n_chunks: int):
        self.n_chunks = n_chunks
        self._entries: dict[int, list] = {}

    def get(self, key: int, build: Callable[[], Any]) -> tuple[Any, bool]:
        """Return ``(tree, first)`` for micro-batch ``key``, calling ``build`` on first use."""
        entry = self._entries.get(key)
        first = entry is None
        if first:
            entry = self._entries[key] = [build(), 0]
        entry[1] += 1
        if entry[1] >= self.n_chunks:
            del self._entries[key]
        return entry[0], first


class _RowOutputs:
    """Per-row log-probs / entropy from the node logits (``packed_seq_params.row_outputs``)."""

    def __init__(self, index):
        self.index = index

    def log_probs(self, logits, temperature):
        from coda.custom.tree_pack.megatron.loss import tree_compute_log_probs
        return tree_compute_log_probs(self.index, logits, temperature)

    def entropy(self, logits, temperature, with_grad):
        from coda.custom.tree_pack.megatron.loss import tree_compute_entropy
        return tree_compute_entropy(self.index, logits, temperature, metric_only=not with_grad)


def _trainable_rows(traj) -> list[np.ndarray]:
    tokens = np.asarray(traj.tokens)
    return [tokens[s.token_start:s.token_end] for s in traj.segments if s.trainable]


@register_packer("tree")
class TreePacker(MicroBatchPacker):
    """Packs each micro-batch's rows into one tree node buffer (shared prefixes once)."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        unknown = set(config.trainer.get("packer_options", {})) - set(DEFAULT_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown trainer.packer_options for tree: {sorted(unknown)}")
        self.options = {**DEFAULT_OPTIONS, **config.trainer.get("packer_options", {})}
        if self.options["cut"] not in ("min_cost", "greedy"):
            raise ValueError(f"tree cut must be 'min_cost' or 'greedy', got {self.options['cut']!r}")
        if self.options["dp_balance"]:
            self.dp_loads = self._dp_loads
        self._lm_rows_only = False
        self._cache = _ChunkCache(1)
        self._stats: list[dict[str, float]] = []

    @classmethod
    def unsupported_reason(cls, config: DictConfig) -> str | None:
        m = config.megatron.model
        if int(m.get("context_parallel_size", 1) or 1) > 1:
            return "context_parallel_size > 1"
        if config.opd.enable:
            return "OPD/GKD enabled"
        if m.get("gdn_pre_gated_delta_rule_fusion", False):
            return "gdn_pre_gated_delta_rule_fusion=True"
        if str(m.get("attention_backend", "")) != "flash":
            return f"attention_backend={m.get('attention_backend')!r} (needs 'flash')"
        if not config.trainer.use_dynamic_batch_size:
            return "use_dynamic_batch_size=False (node budgets need dynamic batching)"
        if multimodal_enabled(config):
            return "multimodal.enable=True (image inputs are not supported by tree packing)"
        return None

    def _dp_loads(self, groups) -> list[int]:
        """Estimated tree cost per group (shared prefixes counted once)."""
        return [int(sum(chain_cost(_trainable_rows(t)) for t in g.trajectories)) for g in groups]

    def setup(self, model) -> str | None:
        from megatron.core.utils import get_model_config

        from coda.custom.tree_pack.megatron.adapters import match_adapters
        from coda.custom.tree_pack.megatron.lm_head import install_lm_head_row_select

        mc = get_model_config(model[0])
        # node positions are the root-path depth, passed as explicit position_ids
        if getattr(mc, "position_embedding_type", None) != "mrope":
            return "the model has no explicit mRoPE position_ids"
        adapters, reason = match_adapters(model)
        if reason:
            return reason
        for adapter in adapters:
            adapter.install(self.options)
        # MTP reads every position, so the LM head keeps them all
        self._lm_rows_only = bool(self.options["lm_head_rows"]) and not getattr(mc, "mtp_num_layers", None)
        if self._lm_rows_only:
            for chunk in model:
                install_lm_head_row_select(chunk)
        self._cache = _ChunkCache(len(model))
        logger.info("[tree_pack] enabled (%s): shared prefixes are computed once per micro-batch",
                    ", ".join(a.name for a in adapters))
        return None

    def applies(self, *, pg_policy=None) -> bool:
        # a PG-KL policy reads row-wise logits
        return pg_policy is None

    def extra_keys(self) -> list[str]:
        return ["dp_local_traj_idx"]

    def plan(self, rollout_data: dict, mini_batch_ranges: list[tuple[int, int]],
             max_budget: int) -> PackPlan:
        """Each trajectory's chained rows cut into trees of <= the node budget (and the
        logit-row budget); the trees are the items bin-packed into micro-batches."""
        from coda.backends.megatron.data import _trajectory_row_ranges

        from coda.custom.tree_pack.core.batch import chain_branch_points
        from coda.custom.tree_pack.core.plan import group_nodes, plan_sequences, plan_sequences_min_cost

        opts = self.options
        if opts["max_nodes_per_gpu"]:
            max_budget = int(opts["max_nodes_per_gpu"])        # CP = 1
        lm_budget = None
        if opts["lm_head_rows"]:
            # fp32 logits follow the logit rows: cap them at what a row-wise micro-batch holds
            lm_budget = int(opts["max_lm_rows_per_gpu"] or self.config.trainer.max_tokens_per_gpu)
        cut = plan_sequences_min_cost if opts["cut"] == "min_cost" else plan_sequences

        tokens = rollout_data["tokens"]
        resp_lens = rollout_data["response_lengths"]
        traj_ranges = _trajectory_row_ranges(rollout_data["dp_local_traj_idx"])
        all_rows, all_costs, all_lms = [], [], []
        stats = {"trajectories": 0, "trees": 0, "ideal_nodes": 0, "recompute_nodes": 0}
        for s, e in mini_batch_ranges:
            rows, costs, lms = [], [], []
            for ts, te in traj_ranges:
                if ts < s or te > e:
                    continue
                lens = [int(t.numel()) for t in tokens[ts:te]]
                bp = chain_branch_points(tokens[ts:te], [0] * len(lens))
                resp = [int(resp_lens[r]) for r in range(ts, te)]
                cuts = cut(lens, bp, max_budget, resp_lens=resp, lm_cap=lm_budget)
                for a, b in cuts:
                    rows.append(list(range(ts + a, ts + b)))
                    costs.append(group_nodes(lens, bp, a, b))
                    lms.append(sum(resp[a:b]))
                # every tree after the first recomputes its first row's shared prefix
                stats["trajectories"] += 1
                stats["trees"] += len(cuts)
                stats["ideal_nodes"] += lens[0] + sum(n - p for n, p in zip(lens[1:], bp[1:]))
                stats["recompute_nodes"] += sum(bp[a] for a, _ in cuts[1:])
            all_rows.append(rows)
            all_costs.append(costs)
            all_lms.append(lms)
        nodes = stats["ideal_nodes"] + stats["recompute_nodes"]
        logger.info(
            "[tree_pack] plan: budget=%d trajectories=%d trees=%d row_tokens=%d nodes=%d "
            "ideal_nodes=%d recompute_nodes=%d (%.1f%% of nodes)",
            max_budget, stats["trajectories"], stats["trees"], sum(int(t.numel()) for t in tokens),
            nodes, stats["ideal_nodes"], stats["recompute_nodes"],
            100.0 * stats["recompute_nodes"] / max(nodes, 1),
        )
        return PackPlan(all_rows, all_costs, max_budget, all_lms if lm_budget else None, lm_budget)

    def pack(self, batch: dict, model_chunk, mb_key: int, *, training: bool,
             routed_rows: Callable[[], list]) -> PackedInputs:
        from coda.backends.megatron.cp_utils import prepare_packed_seq_params

        from coda.custom.tree_pack.core.batch import build_tree_microbatch
        from coda.custom.tree_pack.megatron.lm_head import lm_head_rows

        cfg = self.config

        def build():
            return build_tree_microbatch(
                batch["tokens"], batch["response_lengths"], batch["dp_local_traj_idx"],
                pack=lambda t: prepare_packed_seq_params(
                    [t], cp_partition_mode=cfg.megatron.model.cp_partition_mode),
                routed_rows=routed_rows() if cfg.trainer.use_rollout_routing_replay else None,
                restrict_lm_rows=self._lm_rows_only,
                r3_stats=training and bool(self.options["r3_stats"]),
            )

        tmb, first = self._cache.get(mb_key, build)
        if training and first:
            self._stats.append(tmb.stats)
        psp = tmb.packed_seq_params
        psp.row_outputs = _RowOutputs(psp.tree_loss_index)
        replay_batch = batch if tmb.node_routing is None else {
            **batch, "rollout_routed_experts": [tmb.node_routing.flatten(1)]}
        return PackedInputs(tmb.tokens, psp, tmb.position_ids, replay_batch,
                            lm_head_rows(psp.tree_loss_index.lm_rows))

    def step_metrics(self, step: int, metrics: dict) -> None:
        """Sum the per-micro-batch tree stats, over DP ranks too on the last PP stage (like the
        loss metrics); ratios are recomputed from the sums."""
        from coda.backends.megatron.model import reduce_dict
        from megatron.core import parallel_state as mpu

        stats, self._stats = self._stats, []
        if not stats:
            return
        tot: dict[str, float] = {}
        for st in stats:
            for k, v in st.items():
                if not k.endswith("_ratio"):
                    tot[k] = tot.get(k, 0.0) + v
        if metrics:
            tot = reduce_dict(tot, group=mpu.get_data_parallel_group())
        tot["tree/node_ratio"] = tot["tree/nodes"] / max(tot["tree/row_tokens"], 1.0)
        if "tree/r3_shared_tokens" in tot:
            tot["tree/r3_mismatch_token_ratio"] = (
                tot["tree/r3_mismatch_tokens"] / max(tot["tree/r3_shared_tokens"], 1.0))
        logger.info("[tree_pack] step=%s micro_batches=%d %s", step, len(stats),
                    " ".join(f"{k}={v:.4g}" for k, v in sorted(tot.items())))
        if metrics:
            metrics.update(tot)