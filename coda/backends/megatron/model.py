"""Megatron forward / training / teacher-forward routines.

* ``forward_only``    – student inference pass collecting log-probs (and, when
  a PG KL policy is active, per-token PG KL) for the subsequent RL update.
* ``train_minibatch`` – single gradient step through the Megatron scheduler.
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
    prepare_packed_seq_params,
    prepare_routing_replay_data,
    gather_and_slice_response,
)
from coda.backends.megatron.data import DataIterator
from coda.backends.megatron.loss import loss_function
from coda.backends.megatron.logits_utils import compute_log_probs
from coda.backends.megatron.kl_ctx import (
    KLCtx,
    TeacherCtx,
)
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
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
    return mpu.get_tensor_model_parallel_rank() == 0 and \
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and \
        mpu.is_pipeline_last_stage() if last_pp else mpu.is_pipeline_first_stage()

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

    # Normalize: divide by num_tokens (token-mean) or num_sequences (seq-mean)
    loss_agg_mode = config.algorithm.loss_agg_mode
    num_tokens = max(loss_dict.pop("num_tokens"), 1)
    num_sequences = max(loss_dict.pop("num_sequences"), 1)
    for k in loss_dict:
        if k == "train/loss" and loss_agg_mode != "token-mean":
            loss_dict[k] /= float(num_sequences)
        else:
            loss_dict[k] /= float(num_tokens)
    loss_dict["train/grad_norm"] = grad_norm

    return loss_dict

# ════════════════════════════════════════════════════════════════════════
# Routing Replay Helpers
# ════════════════════════════════════════════════════════════════════════

def setup_routing_replay(config: DictConfig, batch: dict, model_chunk: torch.nn.Module, model: Sequence):
    """
    Extract routing data from batch and setup Megatron RouterReplay for the current model chunk.
    Only sets replay data for the RouterReplay instances belonging to the current model_chunk's vp_stage. 
    """
    model_config = get_model_config(model_chunk)
    rollout_routed_experts = [
        seq.reshape(seq.size(0), model_config.num_layers, model_config.moe_router_topk)
        for seq in batch.get("rollout_routed_experts")
    ]

    packed_experts = prepare_routing_replay_data(rollout_routed_experts, model_config)

    # Determine which vp_stage this model_chunk corresponds to.
    # Megatron scheduler passes model[model_chunk_id] to forward_step_func,
    # so identity comparison is reliable.
    vp_stage = next(i for i, m in enumerate(model) if m is model_chunk)

    def _is_moe_layer(layer_id):
        if isinstance(model_config.moe_layer_freq, int):
            return layer_id % model_config.moe_layer_freq == 0
        elif isinstance(model_config.moe_layer_freq, list):
            return model_config.moe_layer_freq[layer_id] != 0
        return True

    # Compute instance offset: count MoE layers in all preceding vp stages
    instance_offset = 0
    for vp in range(vp_stage):
        num_layers = get_num_layers_to_build(model_config, vp_stage=vp)
        layer_offset = get_transformer_layer_offset(model_config, vp_stage=vp)
        for layer_id in range(layer_offset, layer_offset + num_layers):
            if _is_moe_layer(layer_id):
                instance_offset += 1

    # Set target indices only for this vp_stage's MoE layers
    num_layers = get_num_layers_to_build(model_config, vp_stage=vp_stage)
    layer_offset = get_transformer_layer_offset(model_config, vp_stage=vp_stage)
    instances = RouterReplay.global_router_replay_instances
    idx = instance_offset
    for layer_id in range(layer_offset, layer_offset + num_layers):
        if not _is_moe_layer(layer_id):
            continue
        instances[idx].set_target_indices(packed_experts[:, layer_id])
        idx += 1

    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

# ════════════════════════════════════════════════════════════════════════
# Forward-only
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def forward_only(
    config: DictConfig,
    model: Sequence[DDP],
    data_iterator: Sequence[DataIterator],
    num_microbatches: int,
    pg_policy=None,
) -> list[dict[str, list[torch.Tensor]]]:
    """Run forward passes only and collect non-loss outputs (e.g., logprobs, PG KL).

    Args:
        config: Unified training configuration.
        model: Sequence of DDP-wrapped model chunks.
        data_iterator: The iterable yielding forward-only microbatches.
        num_microbatches: Count of micro-batches to exhaust in the single forward pass.
        pg_policy: The active PG KL policy instance (or None).

    Returns:
        List of per-microbatch dicts, each mapping output names (``"log_probs"``
        and, when a PG policy is active, ``"per_token_kl"``) to lists of
        per-trajectory tensors.
    """
    # Detect teacher fields from rollout_data (populated by collect_teacher).
    teacher_keys = [k for k in data_iterator[0].rollout_data if k.startswith("teacher_")]

    def forward_step_func(data_iter, model_chunk):
        # 1. Fetch micro-batch
        fetch_keys = ["tokens", "total_lengths", "response_lengths", "rollout_routed_experts"]
        fetch_keys += teacher_keys
        batch = data_iter.get_next(fetch_keys)
        tokens_list: list[torch.Tensor] = batch["tokens"]

        # 2. CP-slice and pack (partition mode threaded to keep zigzag/contiguous consistent)
        cp_partition_mode = config.megatron.model.cp_partition_mode
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

        # Handle Routing Replay (Forward Only)
        if config.trainer.use_rollout_routing_replay:
            setup_routing_replay(config, batch, model_chunk, model)

        # 3. Model forward
        output = model_chunk(
            input_ids=packed_tokens.unsqueeze(0),
            position_ids=None,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
        )

        if config.trainer.use_rollout_routing_replay:
            RouterReplay.clear_global_indices()
            RouterReplay.clear_global_router_replay_action()

        # 4. Unified loss closure
        return output, partial(
            compute_forward_only_outputs,
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
) -> dict[str, float]:
    """Execute a single pipeline-parallel training step.

    Defines an inner ``forward_step_func`` that:
    1. Fetches a micro-batch from the data iterator.
    2. CP-slices and packs tokens.
    3. Runs the model forward.
    4. Returns a loss closure (``loss_function``) that computes log-probs and
       loss from the model output.

    Then hands off to Megatron's ``forward_backward_func`` for gradient
    computation, and finishes with an optimizer + scheduler step.

    Returns:
        ``loss_dict`` — aggregated/normalized metrics with ``grad_norm`` folded
        in (empty on non-last PP stages).
    """
    # Detect teacher fields from rollout_data (populated by collect_teacher).
    teacher_keys = [k for k in data_iterator[0].rollout_data if k.startswith("teacher_")]

    # ── forward_step callback ──────────────────────────────────────────

    def forward_step_func(data_iter, model_chunk):
        """Megatron forward-step callback.

        *data_iter* is one of the VPP DataIterators; each call to
        ``get_next`` advances it by one micro-batch.
        """
        # 1. Fetch micro-batch
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
        ]
        keys += teacher_keys

        batch = data_iter.get_next(keys)
        tokens_list: list[torch.Tensor] = batch["tokens"]

        # 2. CP-slice and pack (partition mode threaded to keep zigzag/contiguous consistent)
        packed_tokens, packed_seq_params = prepare_packed_seq_params(
            tokens_list,
            cp_partition_mode=config.megatron.model.cp_partition_mode,
        )

        # Handle Routing Replay (Train)
        if config.trainer.use_rollout_routing_replay:
            setup_routing_replay(config, batch, model_chunk, model)

        # 3. Model forward
        output = model_chunk(
            input_ids=packed_tokens.unsqueeze(0),
            position_ids=None,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
        )
        if config.trainer.use_rollout_routing_replay:
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

        # 4. Loss callback
        return output, partial(loss_function, config, batch, packed_seq_params, gkd_policy)

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

    return _collect_loss_metrics(
        config,
        losses,
        grad_norm.item(),
    )


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
        batch = data_iter.get_next(["tokens", "total_lengths", "response_lengths"])
        tokens_list = batch["tokens"]
        cp_partition_mode = config.megatron.model.cp_partition_mode
        packed_tokens, packed_seq_params = prepare_packed_seq_params(
            tokens_list, cp_partition_mode=cp_partition_mode,
        )
        output = model_chunk(
            input_ids=packed_tokens.unsqueeze(0),
            position_ids=None,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
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
