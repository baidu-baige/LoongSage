"""Megatron forward / training / teacher-forward routines.

* ``forward_only``    – inference pass whose ``output_fn`` collects per-token
  outputs: log-probs (and, when a PG KL policy is active, per-token PG KL) for
  the actor, values for the critic.
* ``train_minibatch`` – single gradient step through the Megatron scheduler,
  with the actor's policy loss or the critic's value loss.
* ``forward_teacher`` – teacher inference pass for OPD; always collects
  ``teacher_entropy`` plus the subset of {hidden states, top-k log_softmax,
  full log probs} that the active GKD / pg KL methods request.

All three share CP-slicing and sequence packing logic via
:mod:`coda.backends.megatron.cp_utils`.
"""

from __future__ import annotations

import logging
from functools import partial
from collections.abc import Sequence
from typing import Any, Callable

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

from megatron.core import parallel_state as mpu
from megatron.core.utils import get_model_config, get_attr_wrapped_model
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.distributed import DistributedDataParallel as DDP
from omegaconf import DictConfig

from coda.backends.megatron.cp_utils import (
    prepare_packed_mrope_position_ids,
    prepare_packed_seq_params,
    prepare_routing_replay_data,
    gather_and_slice_response,
)
from coda.backends.megatron.data import DataIterator
from coda.backends.megatron.tree_batch import build_tree_microbatch
from coda.backends.megatron.tree_loss import lm_head_rows, tree_compute_log_probs
from coda.backends.megatron.loss import loss_function
from coda.backends.megatron.logits_utils import compute_log_probs
from coda.utils.multimodal_tensors import merge_multimodal_batch
from coda.backends.megatron.kl_ctx import (
    KLCtx,
    TeacherCtx,
)
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction


# ════════════════════════════════════════════════════════════════════════
# Distributed helpers
# ════════════════════════════════════════════════════════════════════════

def reduce_dict(
    input_dict: dict[str, float],
    group: dist.ProcessGroup | None = None,
) -> dict[str, float]:
    """All-reduce a ``{str: float}`` dict across a distributed group (sum)."""
    if dist.get_world_size(group) == 1:
        return input_dict

    keys = sorted(input_dict.keys())
    values = [input_dict[k] for k in keys]

    tensor_values = torch.tensor(values, dtype=torch.float32, device=torch.cuda.current_device())

    dist.all_reduce(tensor_values, op=dist.ReduceOp.SUM, group=group)

    reduced_values = tensor_values.tolist()
    return dict(zip(keys, reduced_values))

def is_main_rank(last_pp=True):
    """Check if the current process is the main rank."""
    is_pp_stage = mpu.is_pipeline_last_stage() if last_pp else mpu.is_pipeline_first_stage()
    return (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        and is_pp_stage
    )

# Metrics reported as the ``loss_agg_mode``-aggregated loss (policy loss, PPO
# value loss); normalized by num_sequences in seq-mean modes.
_AGGREGATED_LOSS_KEYS = ("train/loss", "train/value_loss")


def _collect_loss_metrics(
    config: DictConfig,
    losses: list[dict[str, float]],
    grad_norm: float,
) -> dict[str, float]:
    """Aggregate per-micro-batch loss dicts, all-reduce across DP, and normalize."""
    loss_dict: dict[str, float] = {}

    if not mpu.is_pipeline_last_stage():
        return loss_dict

    # Sum metrics across micro-batches
    for micro_batch_metrics in losses:
        for k, v in micro_batch_metrics.items():
            loss_dict[k] = loss_dict.get(k, 0.0) + v

    # All-reduce across DP group
    dp_group = mpu.get_data_parallel_group()
    loss_dict = reduce_dict(loss_dict, group=dp_group)

    # Normalize: the aggregated losses follow loss_agg_mode (divide by
    # num_sequences for seq-mean); every other metric is a per-token sum.
    loss_agg_mode = config.algorithm.loss_agg_mode
    num_tokens = max(loss_dict.pop("num_tokens"), 1)
    num_sequences = max(loss_dict.pop("num_sequences"), 1)
    for k in loss_dict:
        if k in _AGGREGATED_LOSS_KEYS and loss_agg_mode != "token-mean":
            loss_dict[k] /= float(num_sequences)
        else:
            loss_dict[k] /= float(num_tokens)
    loss_dict["train/grad_norm"] = grad_norm

    return loss_dict


# ════════════════════════════════════════════════════════════════════════
# Routing Replay Helpers
# ════════════════════════════════════════════════════════════════════════

def _live_replays(module: torch.nn.Module) -> dict[int, str]:
    """`id(RouterReplay) -> module name` for every RouterReplay held by a module under `module`."""
    return {
        id(rr): name
        for name, m in module.named_modules()
        if isinstance(rr := getattr(m, "router_replay", None), RouterReplay)
    }


def collect_replay_routers(model_chunk: torch.nn.Module) -> dict[int, RouterReplay]:
    """Map 0-based global decoder layer id -> its `RouterReplay` in `model_chunk`.

    Walks the module tree instead of indexing `RouterReplay.global_router_replay_instances`: that list is
    append-only and may contain orphans. For example, Qwen3-VL's text model builds a decoder in
    `GPTModel.__init__` and then replaces it, so the first set of instances never runs and positional
    indexing would hit the wrong routers.

    MTP layers are skipped because their `layer_number` restarts at 1 and would collide with decoder layer 1.
    They get no replay data (MTP is disabled in replay-enabled recipes).

    Raises if the chunk holds a RouterReplay that is neither mapped nor in an MTP layer (e.g. an MoE layer
    that is not a `TransformerLayer`). Otherwise it would reach the forward pass with no target indices and
    fail there with an opaque AttributeError. The result is cached on the chunk (its structure is fixed).
    """
    cached = getattr(model_chunk, "_replay_routers", None)
    if cached is not None:
        return cached

    routers: dict[int, RouterReplay] = {}
    accounted: set[int] = set()  # RouterReplay ids that are mapped or deliberately skipped
    for module in model_chunk.modules():
        if not isinstance(module, TransformerLayer):
            continue
        if module.is_mtp_layer:
            accounted.update(_live_replays(module))
            continue
        router_replay = getattr(getattr(module.mlp, "router", None), "router_replay", None)
        if router_replay is None:
            continue
        layer_id = module.layer_number - 1  # layer_number is 1-based and includes the PP/VP offset
        assert layer_id not in routers, f"Duplicate RouterReplay for decoder layer {layer_id}"
        routers[layer_id] = router_replay
        accounted.add(id(router_replay))

    unmapped = [name for rid, name in _live_replays(model_chunk).items() if rid not in accounted]
    if unmapped:
        raise RuntimeError(
            f"R3: {len(unmapped)} RouterReplay instance(s) in the model chunk are not under a "
            f"TransformerLayer's `mlp.router`, so they would get no replay data: {unmapped[:5]}. "
            "Extend collect_replay_routers to map them."
        )
    model_chunk._replay_routers = routers
    return routers


def _r3_rows(batch: dict, model_chunk: torch.nn.Module) -> list[torch.Tensor]:
    """Per-row routed experts reshaped to ``[seq, num_layers, topk]``.

    Rows arrive as read-only numpy views of the Ray object store (see put_dp_shards_to_ray).
    """
    mc = get_model_config(model_chunk)
    return [torch.as_tensor(seq).reshape(len(seq), mc.num_layers, mc.moe_router_topk)
            for seq in batch["rollout_routed_experts"]]


def setup_routing_replay(config: DictConfig, batch: dict, model_chunk: torch.nn.Module):
    """
    Extract routing data from batch and setup Megatron RouterReplay for the current model chunk.
    Only sets replay data for the routers that live in `model_chunk` (its PP/VP stage).
    """
    model_config = get_model_config(model_chunk)
    packed_experts = prepare_routing_replay_data(_r3_rows(batch, model_chunk), model_config)

    routers = collect_replay_routers(model_chunk)
    assert routers, "Routing replay is enabled but no live RouterReplay was found in the model chunk"
    for layer_id, router_replay in routers.items():
        router_replay.set_target_indices(packed_experts[:, layer_id])

    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)


# ════════════════════════════════════════════════════════════════════════
# MRoPE position ids
# ════════════════════════════════════════════════════════════════════════

def _uses_mrope(model_chunk: torch.nn.Module) -> bool:
    """Whether this model needs explicit 3D MRoPE position_ids (Qwen VL family).

    Plain rope models ignore ``position_ids`` entirely, so they keep passing None.
    """
    return getattr(get_model_config(model_chunk), "position_embedding_type", None) == "mrope"


def _make_rope_index_fn(model_chunk: torch.nn.Module) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Full-sequence MRoPE positions `[3, len]` of one trajectory, via Bridge's own `get_rope_index`.

    Reusing Bridge's function keeps CP>1 positions identical to what the CP=1 path computes
    internally. Only Qwen3-VL family models reach here (Qwen2.5-VL with CP>1 is rejected by
    `validate_config`), so the import is lazy.
    """
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import get_rope_index

    cfg = get_model_config(model_chunk)

    def rope_index_fn(tokens: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        positions, _ = get_rope_index(
            cfg.spatial_merge_size,
            cfg.image_token_id,
            cfg.video_token_id,
            cfg.vision_start_token_id,
            input_ids=tokens.unsqueeze(0),
            image_grid_thw=image_grid_thw,
        )
        return positions[:, 0]

    return rope_index_fn


def _get_position_ids(
    model_chunk: torch.nn.Module,
    tokens_list: list[torch.Tensor],
    packed_tokens: torch.Tensor,
    cp_partition_mode: str,
    multimodal_rows: list[list[str] | None] | None = None,
    image_table: dict[str, dict[str, torch.Tensor]] | None = None,
) -> torch.Tensor | None:
    """Rank-local 3D MRoPE position ids for this micro-batch, or None when not needed."""
    if mpu.get_context_parallel_world_size() <= 1 or not _uses_mrope(model_chunk):
        # Without CP the model derives MRoPE itself from the full input_ids.
        return None
    grids = [
        torch.cat([image_table[h]["image_grid_thw"] for h in row], dim=0) if row else None
        for row in (multimodal_rows or [])
    ] or None
    position_ids = prepare_packed_mrope_position_ids(
        tokens_list, cp_partition_mode=cp_partition_mode,
        image_grid_thw_list=grids,
        rope_index_fn=_make_rope_index_fn(model_chunk) if grids and any(g is not None for g in grids) else None,
    )
    assert position_ids.size(-1) == packed_tokens.size(0), (
        f"MRoPE position_ids ({position_ids.size(-1)}) must align token-for-token "
        f"with CP-local packed tokens ({packed_tokens.size(0)})"
    )
    return position_ids


def _multimodal_forward_kwargs(batch: dict, image_table: dict | None) -> dict[str, torch.Tensor]:
    """Per-image tensors (`pixel_values`, `image_grid_thw`) to splat into the model forward.

    `batch["multimodal_train_inputs"]` holds image hashes per row; the tensors come from `image_table`.
    Empty for text-only micro-batches. Every PP stage must pass the same kwargs, so this is
    called unconditionally in the forward step; loss_mask is deliberately not forwarded.
    """
    return merge_multimodal_batch(batch.get("multimodal_train_inputs"), image_table) or {}


# ════════════════════════════════════════════════════════════════════════
# Tree (shared-prefix) helpers
# ════════════════════════════════════════════════════════════════════════

def forward_only_uses_tree(config: DictConfig, pg_policy=None) -> bool:
    """Whether forward-only passes use tree micro-batches.

    Args:
        config: Unified training configuration.
        pg_policy: Active PG KL policy; it needs row-wise logits, so it disables tree.

    Returns:
        True when ``trainer.tree_pack`` is on and no PG policy is active. The train
        worker's data iterator must use the same decision.
    """
    return bool(config.trainer.get("tree_pack", False)) and pg_policy is None


def _install_tree_patches(config: DictConfig, model: Sequence) -> bool:
    """Install the tree monkeypatches (idempotent). Returns whether the LM head runs on the
    loss rows only (``tree_lm_head_rows``; off with MTP, which reads every position)."""
    from coda.backends.megatron.tree_attn import install_attention_tree_patch
    from coda.backends.megatron.tree_gdn import install_gdn_tree_patch
    from coda.backends.megatron.tree_loss import install_lm_head_row_select

    install_gdn_tree_patch(config.trainer.get("tree_gdn_level_tokens", None))
    install_attention_tree_patch()
    if not _uses_mrope(model[0]):
        raise NotImplementedError("tree_pack requires a model with explicit mRoPE position_ids")
    if not bool(config.trainer.get("tree_lm_head_rows", True)) or getattr(
            get_model_config(model[0]), "mtp_num_layers", None):
        return False
    for chunk in model:
        install_lm_head_row_select(chunk)
    return True


class _TreeChunkCache:
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


def _tree_lm_rows(packed_seq_params) -> torch.Tensor | None:
    """LM-head rows of a tree micro-batch (None: every position)."""
    idx = getattr(packed_seq_params, "tree_loss_index", None)
    return None if idx is None else idx.lm_rows


def _tree_microbatch(config, batch, tokens_list, model_chunk, cache, key, lm_rows_only,
                     r3_stats):
    """Tree micro-batch for ``batch`` (shared across VPP chunks) and the routing-replay batch
    carrying its node routing. Returns ``(tmb, first, replay_batch)``."""
    tmb, first = cache.get(key, lambda: build_tree_microbatch(
        tokens_list, batch["response_lengths"], batch["dp_local_traj_idx"],
        routed_rows=_r3_rows(batch, model_chunk) if config.trainer.use_rollout_routing_replay else None,
        cp_partition_mode=config.megatron.model.cp_partition_mode,
        restrict_lm_rows=lm_rows_only, r3_stats=r3_stats,
    ))
    replay_batch = batch if tmb.node_routing is None else {
        **batch, "rollout_routed_experts": [tmb.node_routing.flatten(1)]}
    return tmb, first, replay_batch



# ════════════════════════════════════════════════════════════════════════
# Forward-only
# ════════════════════════════════════════════════════════════════════════

def compute_forward_only_outputs(
    packed_targets: torch.Tensor,
    batch: dict,
    packed_seq_params,
    config: DictConfig,
    pg_policy,
    output_tensor: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Unified forward-only loss closure: log_probs + optional PG KL."""
    temperature = config.trainer.temperature
    total_lengths = batch["total_lengths"]
    response_lengths = batch["response_lengths"]
    cp_partition_mode = config.megatron.model.cp_partition_mode

    collected = {}
    tree_idx = getattr(packed_seq_params, "tree_loss_index", None)
    if tree_idx is not None:
        if pg_policy is not None:
            raise NotImplementedError("tree forward-only does not support PG-KL policies")
        collected["log_probs"] = tree_compute_log_probs(tree_idx, output_tensor, temperature)
        return torch.tensor(0.0, device=output_tensor.device), collected
    collected["log_probs"] = compute_log_probs(
        packed_targets,
        total_lengths,
        response_lengths,
        # compute_log_probs mutates logits in-place; clone only when a PG policy
        # below still needs the original output_tensor to compute KL.
        output_tensor.clone() if pg_policy else output_tensor,
        temperature,
        cp_partition_mode=cp_partition_mode,
    )

    # Compute PG KL via the active PG policy
    if pg_policy is not None:
        ctx = KLCtx(
            batch, output_tensor, packed_seq_params,
            temperature=temperature, cp_partition_mode=cp_partition_mode,
        )
        per_token_kl, _ = pg_policy.compute_kl(config, ctx)
        per_token_kl = gather_and_slice_response(
            per_token_kl, total_lengths, response_lengths,
            cp_partition_mode=cp_partition_mode,
        )
        collected["per_token_kl"] = per_token_kl

    return torch.tensor(0.0, device=output_tensor.device), collected


@torch.no_grad()
def forward_only(
    config: DictConfig,
    model: Sequence[DDP],
    data_iterator: Sequence[DataIterator],
    num_microbatches: int,
    pg_policy=None,
    output_fn: Callable = compute_forward_only_outputs,
) -> list[dict[str, list[torch.Tensor]]]:
    """Run forward passes only and collect non-loss outputs (e.g., logprobs, PG KL).

    Args:
        config: Unified training configuration.
        model: Sequence of DDP-wrapped model chunks.
        data_iterator: The iterable yielding forward-only microbatches.
        num_microbatches: Count of micro-batches to exhaust in the single forward pass.
        pg_policy: The active PG KL policy instance (or None).
        output_fn: Output closure with the signature of
            :func:`compute_forward_only_outputs` (the default), i.e.
            ``(packed_targets, batch, packed_seq_params, config, pg_policy,
            output_tensor) -> (loss, collected)``.

    Returns:
        List of per-microbatch dicts, each mapping output names (``"log_probs"``
        and, when a PG policy is active, ``"per_token_kl"``) to lists of
        per-trajectory tensors.
    """
    # Detect teacher fields from rollout_data (populated by collect_teacher).
    teacher_keys = [k for k in data_iterator[0].rollout_data if k.startswith("teacher_")]
    # Tree micro-batches need no PG-KL policy (it reads row-wise logits) and the policy
    # outputs; callers with another closure (e.g. the critic's value head) keep rows.
    tree_pack = forward_only_uses_tree(config, pg_policy)
    lm_rows_only = _install_tree_patches(config, model) if tree_pack else False
    tree_cache = _TreeChunkCache(len(model))

    def forward_step_func(data_iter, model_chunk):
        # 1. Fetch micro-batch
        fetch_keys = [
            "tokens", "total_lengths", "response_lengths", "rollout_routed_experts", "multimodal_train_inputs",
        ]
        fetch_keys += teacher_keys
        if tree_pack:
            fetch_keys.append("dp_local_traj_idx")
        batch = data_iter.get_next(fetch_keys)
        tokens_list: list[torch.Tensor] = batch["tokens"]

        # 2. CP-slice and pack (partition mode threaded to keep zigzag/contiguous consistent)
        cp_partition_mode = config.megatron.model.cp_partition_mode
        replay_batch = batch
        if tree_pack:
            # forward-only does not report tree stats
            tmb, _, replay_batch = _tree_microbatch(config, batch, tokens_list, model_chunk,
                                                    tree_cache, data_iter.offset, lm_rows_only, False)
            packed_tokens, packed_seq_params = tmb.tokens, tmb.packed_seq_params
            position_ids = tmb.position_ids
            packed_targets = None   # the tree loss index carries the targets
        else:
            packed_tokens, packed_seq_params = prepare_packed_seq_params(
                tokens_list, cp_partition_mode=cp_partition_mode,
            )
            target_list = [
                torch.cat([t[1:], t.new_full((1,), 0)])
                for t in tokens_list
            ]
            packed_targets, _ = prepare_packed_seq_params(
                target_list, cp_partition_mode=cp_partition_mode,
            )
            position_ids = _get_position_ids(
                model_chunk, tokens_list, packed_tokens, cp_partition_mode,
                multimodal_rows=batch.get("multimodal_train_inputs"),
                image_table=data_iter.image_table,
            )

        # Handle Routing Replay (Forward Only)
        if config.trainer.use_rollout_routing_replay:
            setup_routing_replay(config, replay_batch, model_chunk)

        # 3. Model forward (tree micro-batches: LM head on the loss rows only)
        with lm_head_rows(_tree_lm_rows(packed_seq_params)):
            output = model_chunk(
                input_ids=packed_tokens.unsqueeze(0),
                position_ids=position_ids,
                attention_mask=None,
                packed_seq_params=packed_seq_params,
                **_multimodal_forward_kwargs(batch, data_iter.image_table),
            )

        if config.trainer.use_rollout_routing_replay:
            RouterReplay.clear_global_indices()
            RouterReplay.clear_global_router_replay_action()

        # 4. Unified loss closure
        return output, partial(
            output_fn,
            packed_targets, batch, packed_seq_params, config, pg_policy,
        )

    # Run through Megatron scheduler
    forward_backward_func: Callable[..., list[Any]] | Callable[..., Any | list[Any]] = get_forward_backward_func()

    forward_data_store: list[dict[str, list[torch.Tensor]]] = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=1,
        micro_batch_size=1,
        forward_only=True,
    )

    return forward_data_store


# ════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════

def train_minibatch(
    config: DictConfig,
    step: int,
    data_iterator: Sequence[DataIterator],
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    num_microbatches: int,
    gkd_policy=None,
    qat_plan=None,
    loss_fn: Callable = loss_function,
) -> dict[str, float]:
    """Execute a single pipeline-parallel training step.

    Defines an inner ``forward_step_func`` that:
    1. Fetches a micro-batch from the data iterator.
    2. CP-slices and packs tokens.
    3. Runs the model forward.
    4. Returns a loss closure (``loss_fn``, default ``loss_function``) that
       computes the loss from the model output. It is called as
       ``loss_fn(config, batch, packed_seq_params, gkd_policy, output_tensor)``
       and returns ``(loss, denom, metrics)``.

    Then hands off to Megatron's ``forward_backward_func`` for gradient
    computation, and finishes with an optimizer + scheduler step (followed by a
    QAT re-snap of the params when ``qat_plan`` is given).

    Returns:
        ``loss_dict`` — aggregated/normalized metrics with ``grad_norm`` folded
        in (empty on non-last PP stages).
    """
    # Detect teacher fields from rollout_data (populated by collect_teacher).
    teacher_keys = [k for k in data_iterator[0].rollout_data if k.startswith("teacher_")]
    tree_pack = bool(config.trainer.get("tree_pack", False))
    tree_r3_stats = bool(config.trainer.get("tree_r3_stats", True))
    tree_stats: list[dict[str, float]] = []
    lm_rows_only = _install_tree_patches(config, model) if tree_pack else False
    tree_cache = _TreeChunkCache(len(model))

    # ── forward_step callback ──────────────────────────────────────────

    def forward_step_func(data_iter, model_chunk):
        """Megatron forward-step callback.

        *data_iter* is one of the VPP DataIterators; each call to
        ``get_next`` advances it by one micro-batch.
        """
        # 1. Fetch micro-batch. Keys absent from the rollout data come back as
        # None, so one list serves both the policy loss and the value loss.
        keys = [
            "tokens",
            "loss_masks",
            "raw_loss_masks",
            "response_lengths",
            "total_lengths",
            "advantages",
            "is_weights",
            "old_log_probs",
            "rollout_log_probs",
            "rollout_routed_experts",
            "ref_log_probs",
            "old_values",
            "returns",
            "multimodal_train_inputs",
        ]
        if tree_pack:
            keys.append("dp_local_traj_idx")
        keys += teacher_keys

        batch = data_iter.get_next(keys)
        tokens_list: list[torch.Tensor] = batch["tokens"]
        mb_key = data_iter.offset          # same micro-batch on every VPP chunk

        # 2. CP-slice and pack (partition mode threaded to keep zigzag/contiguous consistent)
        cp_partition_mode = config.megatron.model.cp_partition_mode
        replay_batch = batch
        if tree_pack:
            # the loss closure detects tree micro-batches via packed_seq_params
            tmb, first, replay_batch = _tree_microbatch(
                config, batch, tokens_list, model_chunk, tree_cache, mb_key, lm_rows_only,
                tree_r3_stats)
            packed_tokens, packed_seq_params = tmb.tokens, tmb.packed_seq_params
            position_ids = tmb.position_ids
            if first:
                tree_stats.append(tmb.stats)
        else:
            packed_tokens, packed_seq_params = prepare_packed_seq_params(
                tokens_list,
                cp_partition_mode=cp_partition_mode,
            )
            position_ids = _get_position_ids(
                model_chunk, tokens_list, packed_tokens, cp_partition_mode,
                multimodal_rows=batch.get("multimodal_train_inputs"),
                image_table=data_iter.image_table,
            )

        # Handle Routing Replay (Train)
        if config.trainer.use_rollout_routing_replay:
            setup_routing_replay(config, replay_batch, model_chunk)

        # 3. Model forward (tree micro-batches: LM head on the loss rows only)
        with lm_head_rows(_tree_lm_rows(packed_seq_params)):
            output = model_chunk(
                input_ids=packed_tokens.unsqueeze(0),
                position_ids=position_ids,
                attention_mask=None,
                packed_seq_params=packed_seq_params,
                **_multimodal_forward_kwargs(batch, data_iter.image_table),
            )
        if config.trainer.use_rollout_routing_replay:
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

        # 4. Loss callback
        return output, partial(loss_fn, config, batch, packed_seq_params, gkd_policy)

    # Chunked optimizer-state offload: push fp32 master weights (and any resident
    # optimizer-state chunks) to CPU before the activation-heavy forward/backward so
    # they don't occupy GPU memory during compute. ``optimizer.step()`` restores them
    # (masters + chunk-by-chunk) on its own — see ChunkedOptimizerStateOffloader.step.
    optimizer.offload_optimizer_state_for_forward()

    # Set grad to zero.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    # ── Run Megatron forward + backward ────────────────────────────────
    forward_backward_func = get_forward_backward_func()
    losses = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        # When variable_seq_lengths=True, seq_length and micro_batch_size are
        # unused; actual tensor shapes are communicated dynamically via P2P.
        seq_length=1,
        micro_batch_size=1,
        forward_only=False,
    )

    if config.trainer.use_rollout_routing_replay:
        RouterReplay.clear_global_indices()
        RouterReplay.clear_global_router_replay_action()

    # ── Optimizer step ─────────────────────────────────────────────────
    update_successful, grad_norm, num_zeros = optimizer.step()
    logger.info(
        f"optimizer.step(): update_successful={update_successful}, "
        f"grad_norm={grad_norm.item()}, num_zeros={num_zeros}"
    )
    if update_successful:
        opt_param_scheduler.step(increment=1)
        if qat_plan is not None:
            # Re-snap the freshly updated bf16 params onto the rollout quantization grid; the fp32
            # master keeps the un-quantized value (straight-through QAT). Needs a separate fp32
            # master: validate_config turns off store_param_remainders under QAT.
            qat_plan.snap_()

    metrics = _collect_loss_metrics(
        config,
        losses,
        grad_norm.item(),
    )
    if tree_stats:
        _log_tree_stats(step, tree_stats, metrics)
    return metrics


def _log_tree_stats(step: int, tree_stats: list[dict[str, float]], metrics: dict) -> None:
    """Sum the per-micro-batch tree stats, over DP ranks too on the last PP stage (like the
    loss metrics); ratios are recomputed from the sums."""
    tot: dict[str, float] = {}
    for st in tree_stats:
        for k, v in st.items():
            if not k.endswith("_ratio"):
                tot[k] = tot.get(k, 0.0) + v
    if metrics:
        tot = reduce_dict(tot, group=mpu.get_data_parallel_group())
    tot["tree/node_ratio"] = tot["tree/nodes"] / max(tot["tree/row_tokens"], 1.0)
    if "tree/r3_shared_tokens" in tot:
        tot["tree/r3_mismatch_token_ratio"] = (
            tot["tree/r3_mismatch_tokens"] / max(tot["tree/r3_shared_tokens"], 1.0))
    logger.info("[tree_pack] step=%s micro_batches=%d %s", step, len(tree_stats),
                " ".join(f"{k}={v:.4g}" for k, v in sorted(tot.items())))
    if metrics:
        metrics.update(tot)


# ════════════════════════════════════════════════════════════════════════
# Teacher forward (OPD inference)
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def forward_teacher(
    config: DictConfig,
    model: Sequence,
    data_iterator: Sequence[DataIterator],
    num_microbatches: int,
    policies: list = None,
) -> dict[str, list[torch.Tensor]]:
    """Run a teacher forward pass and collect the fields required by OPD.

    Args:
        config: Unified training configuration. ``config.opd`` (specifically
            ``gkd_kl_method`` / ``pg_kl_method``) decides which
            optional fields are collected.
        model: Sequence of model chunks (one per VPP stage; no DDP wrapping
            for teacher inference).
        data_iterator: Pre-built per-VPP-stage iterators yielding microbatches.
        num_microbatches: Number of microbatches to consume in this pass.

    Collected fields (always ``teacher_entropy``; the rest are the keys each
    active policy's ``collect_teacher`` returns):
        - ``teacher_hidden_states`` — when an output_layer pre-hook is needed.
        - ``teacher_topk_logprobs`` / ``teacher_topk_indices`` — TP-distributed
          top-k log_softmax over the vocabulary.
        - ``teacher_log_probs`` — per-token teacher log probs at the targets.

    Aggregates outputs across microbatches and, when dynamic batching has
    reordered trajectories, restores the original trajectory order on the last PP stage.

    Returns:
        On the last PP stage: ``dict[str, list[Tensor per trajectory]]``.
        On every other PP stage: empty dict.
    """
    if policies is None:
        policies = []
    need_hidden = any(p.need_teacher_logits() for p in policies)

    # Register hook to capture hidden states before output_layer.
    # The hook fires once per microbatch; the driver pops entries in order so
    # each loss closure gets the correct microbatch's hidden tensor.
    hidden_cache: list[torch.Tensor] = []
    hook_handle = _register_hidden_states_hook(model, hidden_cache) if need_hidden else None

    def forward_step_func(data_iter, model_chunk):
        batch = data_iter.get_next(["tokens", "total_lengths", "response_lengths", "multimodal_train_inputs"])
        tokens_list = batch["tokens"]
        cp_partition_mode = config.megatron.model.cp_partition_mode
        packed_tokens, packed_seq_params = prepare_packed_seq_params(
            tokens_list, cp_partition_mode=cp_partition_mode,
        )
        position_ids = _get_position_ids(
            model_chunk, tokens_list, packed_tokens, cp_partition_mode,
            multimodal_rows=batch.get("multimodal_train_inputs"),
            image_table=data_iter.image_table,
        )
        output = model_chunk(
            input_ids=packed_tokens.unsqueeze(0),
            position_ids=position_ids,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
            **_multimodal_forward_kwargs(batch, data_iter.image_table),
        )

        target_list = [
            torch.cat([t[1:], t.new_full((1,), 0)])
            for t in tokens_list
        ]
        packed_targets = prepare_packed_seq_params(
            target_list, cp_partition_mode=cp_partition_mode,
        )[0]

        # The output_layer pre-hook has already fired during the forward above,
        # so this microbatch's hidden is now at the front of the FIFO.
        hidden = hidden_cache.pop(0) if hook_handle else None

        return output, partial(
            _collect_teacher_outputs,
            batch["total_lengths"], batch["response_lengths"],
            policies=policies,
            hidden=hidden,
            packed_targets=packed_targets,
            temperature=config.trainer.temperature,
            cp_partition_mode=cp_partition_mode,
        )

    forward_backward_func = get_forward_backward_func()

    forward_data_store = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=1,
        micro_batch_size=1,
        forward_only=True,
    )

    if hook_handle:
        hook_handle.remove()

    # Aggregate results across microbatches on the last pipeline stage
    collected: dict[str, list] = {}
    if mpu.is_pipeline_last_stage():
        assert forward_data_store, "all_forward_data is empty on the last PP stage"
        for key in forward_data_store[0].keys():
            values: list = []
            for mb_result in forward_data_store:
                values += mb_result[key]
            collected[key] = values

        # Undo the dynamic-batch permutation / -1 padding so outputs align with
        # the Segment rows of rollout_data.
        order = sum(data_iterator[0].micro_batch_indices, [])
        n_rows = len(data_iterator[0].rollout_data["tokens"])
        for key in collected:
            reordered: list = [None] * n_rows
            for value, idx in zip(collected[key], order):
                if idx >= 0:
                    reordered[idx] = value
            collected[key] = reordered

    return collected


def _register_hidden_states_hook(model: Sequence, cache: list):
    """Register a forward_pre_hook on output_layer to capture hidden states.

    Only registers on the last pipeline stage. The hook fires once per
    microbatch, appending the hidden states tensor to cache in execution order.
    """
    if not mpu.is_pipeline_last_stage():
        return None
    last_chunk = model[-1]
    try:
        output_layer = get_attr_wrapped_model(last_chunk, 'output_layer')
    except RuntimeError:
        return None

    def hook_fn(module, args):
        cache.append(args[0].detach())  # [seq_len, batch, hidden_dim]

    return output_layer.register_forward_pre_hook(hook_fn)


def _collect_teacher_outputs(
    total_lengths: list[int],
    response_lengths: list[int],
    output_tensor: torch.Tensor,
    policies: list,
    hidden: torch.Tensor | None = None,
    packed_targets: torch.Tensor | None = None,
    temperature: float = 1.0,
    cp_partition_mode: str = "zigzag",
):
    """Teacher-forward loss closure: drive each active policy's collect_teacher.

    Builds one :class:`TeacherCtx` per microbatch (the caller passes this
    microbatch's hidden tensor, already popped from the hook FIFO), lets every
    active policy collect its primitives, and always adds method-independent
    ``teacher_entropy`` for the entropy_gap metric.

    Policies that declare ``need_teacher_logits()`` consume reconstructed teacher
    logits on the student side; the ``teacher_hidden_states`` they are rebuilt
    from are a framework transport detail, so they are collected here (when any
    such policy is active) rather than named/produced by the policy itself.

    Args:
        cp_partition_mode: Threaded from ``config.megatron.model.cp_partition_mode``
            so teacher CP gather/reconstruct matches the attention layout.
    """
    ctx = TeacherCtx(
        output_tensor=output_tensor,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        packed_targets=packed_targets,
        hidden=hidden,
        temperature=temperature,
        cp_partition_mode=cp_partition_mode,
    )

    collected: dict[str, list[torch.Tensor]] = {}
    for policy in policies:
        collected.update(policy.collect_teacher(ctx))

    # teacher_hidden_states is a framework transport detail behind
    # need_teacher_logits(): collected here when any active policy needs teacher
    # logits, so the policy contract stays at "needs teacher logits" and never
    # names this field. Reconstructed on the student side via TeacherLMHeads.
    if any(p.need_teacher_logits() for p in policies):
        collected["teacher_hidden_states"] = ctx.hidden_states()

    # teacher_entropy is method-independent: collected unconditionally so the
    # entropy_gap metric is available regardless of which KL methods are active.
    collected["teacher_entropy"] = ctx.entropy()

    return torch.tensor(0.0, device=output_tensor.device), collected
