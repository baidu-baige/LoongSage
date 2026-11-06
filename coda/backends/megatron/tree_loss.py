"""Per-row log-probs / entropy of tree micro-batches, and LM-head row selection.

Row r's response token j is predicted from the logits at node ``path_r[j-1]``; a shared
node has the same context in every row, so only its target can differ (at branch points).
One vocab-parallel CE runs over the nodes with each node's default target (next token of
its emitting row), plus a small extra CE for the (node, target) pairs that differ. Outputs
match ``compute_log_probs`` / ``compute_entropy`` (per-row lists).
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from coda.backends.megatron.tree_pack import TreeSeq, row_path_nodes


@dataclass
class TreeLossIndex:
    """Gather plan from node logits to per-row response log-probs / entropy."""

    node_targets: torch.Tensor   # [T] default CE target per node
    extra_nodes: torch.Tensor    # [E] logit node of each extra (node, target) pair
    extra_targets: torch.Tensor  # [E]
    logp_src: torch.Tensor       # [R] index into cat([node_logp (T), extra_logp (E)])
    logit_nodes: torch.Tensor    # [R] logit node of every response token
    resp_lens: list[int]         # per row; sum == R
    lm_rows: torch.Tensor | None = None  # if set, indices above are relative to these rows


def _owner_index(ts: TreeSeq) -> tuple[torch.Tensor, torch.Tensor]:
    """(emitting row, offset in that row) of every node of ``ts``."""
    rows = torch.tensor([r for r, _ in ts.node_order], dtype=torch.long)
    offs = torch.tensor([o for _, o in ts.node_order], dtype=torch.long)
    return rows, offs


def build_loss_index(
    seqs: list[TreeSeq],
    row_tokens: list[torch.Tensor],
    resp_lens: list[int],
    total_nodes: int,
) -> TreeLossIndex:
    """Loss index for a ``merge_pieces`` buffer.

    Args:
        seqs: TreeSeqs packed back-to-back in the buffer.
        row_tokens: 1-D CPU token tensor of every row, indexed like the TreeSeqs' rows.
        resp_lens: Response length of every row (its last ``resp_lens[r]`` tokens).
        total_nodes: Buffer length including tail padding.

    Returns:
        The index on CPU (see ``index_to``).
    """
    node_targets = torch.zeros(total_nodes, dtype=torch.long)
    lnodes, tgts = [], []
    node_off = 0
    for ts in seqs:
        rb, re = ts.row_begin, ts.row_end
        rows, offs = _owner_index(ts)
        # default target: the next token in the row that emitted the node (0 past the end)
        toks = [row_tokens[r].long() for r in range(rb, re)]
        tlen = torch.tensor([t.numel() for t in toks], dtype=torch.long)
        start = tlen.cumsum(0) - tlen
        nxt, rel = offs + 1, rows - rb
        ok = nxt < tlen[rel]
        tgt = torch.zeros_like(nxt)
        tgt[ok] = torch.cat(toks)[start[rel[ok]] + nxt[ok]]
        node_targets[node_off:node_off + ts.num_nodes] = tgt
        for r in range(rb, re):
            path = torch.tensor(row_path_nodes(ts, r), dtype=torch.long) + node_off
            L, n = path.numel(), resp_lens[r]
            if L != row_tokens[r].numel():
                raise ValueError(f"row {r}: tree path {L} != row length {row_tokens[r].numel()}")
            if not 0 <= n < L:
                raise ValueError(f"row {r}: response length {n} out of range for length {L}")
            lnodes.append(path[L - n - 1:L - 1])
            tgts.append(row_tokens[r][L - n:L].long())
        node_off += ts.num_nodes

    logit_nodes, targets = torch.cat(lnodes), torch.cat(tgts)
    miss = (node_targets[logit_nodes] != targets).nonzero(as_tuple=True)[0]
    src = logit_nodes.clone()
    src[miss] = total_nodes + torch.arange(miss.numel())
    return TreeLossIndex(node_targets, logit_nodes[miss], targets[miss], src, logit_nodes,
                         [t.numel() for t in lnodes])


def index_to(idx: TreeLossIndex, device: torch.device | str) -> TreeLossIndex:
    """``idx`` with all tensors on ``device``."""
    return TreeLossIndex(idx.node_targets.to(device), idx.extra_nodes.to(device),
                         idx.extra_targets.to(device), idx.logp_src.to(device),
                         idx.logit_nodes.to(device), idx.resp_lens,
                         None if idx.lm_rows is None else idx.lm_rows.to(device))


def restrict_to_lm_rows(idx: TreeLossIndex) -> TreeLossIndex:
    """``idx`` re-expressed over the logit nodes of the response tokens only, so the LM head
    computes ``[len(lm_rows), V/tp]`` logits instead of ``[T, V/tp]``."""
    if idx.lm_rows is not None:
        raise ValueError("index is already restricted")
    rows = torch.unique(idx.logit_nodes)                    # sorted node ids
    n_total, n_rows = idx.node_targets.numel(), rows.numel()

    def pos(nodes: torch.Tensor) -> torch.Tensor:
        return torch.searchsorted(rows, nodes)

    src = idx.logp_src.clone()
    is_node = src < n_total
    src[is_node] = pos(src[is_node])
    src[~is_node] = src[~is_node] - n_total + n_rows        # extra CE results follow
    return TreeLossIndex(idx.node_targets[rows], pos(idx.extra_nodes), idx.extra_targets,
                         src, pos(idx.logit_nodes), idx.resp_lens, rows)


# LM-head row selection: forward pre-hook on the output layer.
_LM_ROWS: ContextVar = ContextVar("coda_tree_lm_rows", default=None)


@contextmanager
def lm_head_rows(rows: torch.Tensor | None):
    """Within this block the hooked output layer computes logits only for ``rows`` of its
    (SP-gathered) input; ``None`` leaves it untouched."""
    token = _LM_ROWS.set(rows)
    try:
        yield
    finally:
        _LM_ROWS.reset(token)


def install_lm_head_row_select(model_chunk: torch.nn.Module) -> bool:
    """Hook the chunk's output layer for ``lm_head_rows``; False if the chunk has none
    (non-last pipeline stage). Idempotent.

    Under SP the hook all-gathers the hidden states and runs the layer with
    ``sequence_parallel`` off for that call; the layer then all-reduces the input gradient,
    so the gather treats its output gradient as replicated and only splits it back.
    """
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

    heads = [m for n, m in model_chunk.named_modules()
             if n.endswith("output_layer") and "mtp" not in n]
    if not heads:
        return False
    if len(heads) != 1:
        raise RuntimeError(f"expected one output_layer, found {len(heads)}")
    head = heads[0]
    if getattr(head, "_coda_lm_rows_hooked", False):
        return True

    def pre_hook(module, args, kwargs):
        """Feed only the selected rows to the layer (all-gathered first under SP)."""
        rows = _LM_ROWS.get()
        if rows is None:
            return None
        if args:
            hidden, args = args[0], args[1:]
        else:
            hidden = kwargs.pop("input_")
        if module.sequence_parallel:
            hidden = gather_from_sequence_parallel_region(
                hidden, tensor_parallel_output_grad=False, group=module.tp_group)
            module.sequence_parallel = False
            module._coda_restore_sp = True
        hidden = hidden.index_select(0, rows)
        return (hidden, *args), kwargs

    def post_hook(module, args, output):
        if getattr(module, "_coda_restore_sp", False):
            module.sequence_parallel = True
            module._coda_restore_sp = False

    # prepend: run before other pre-hooks (e.g. the fp32 LM-head cast)
    head.register_forward_pre_hook(pre_hook, prepend=True, with_kwargs=True)
    head.register_forward_hook(post_hook, always_call=True)   # restore SP on errors too
    head._coda_lm_rows_hooked = True
    return True


def _logits_2d(output_tensor: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """``[T, V/tp]`` logits divided by ``temperature`` (a view when it is 1)."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    logits = output_tensor.squeeze(0) if output_tensor.dim() == 3 else output_tensor
    return logits / temperature if temperature != 1.0 else logits


def tree_compute_log_probs(idx: TreeLossIndex, output_tensor: torch.Tensor,
                           temperature: float = 1.0) -> list[torch.Tensor]:
    """Per-row response log-probs of a tree micro-batch.

    Like ``compute_log_probs`` it modifies the logits in place (fused CE).

    Args:
        idx: Loss index on the logits' device.
        output_tensor: LM-head output, ``[1, T, V/tp]`` or ``[T, V/tp]``.
        temperature: Sampling temperature.

    Returns:
        One ``[resp_len]`` tensor per row.
    """
    from megatron.core import parallel_state as mpu
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    tp_group = mpu.get_tensor_model_parallel_group()
    logits = _logits_2d(output_tensor, temperature).contiguous()
    # copy the branch-point rows before the in-place CE
    extra_logits = logits.index_select(0, idx.extra_nodes)
    node_logp = -fused_vocab_parallel_cross_entropy(
        logits.unsqueeze(1), idx.node_targets.unsqueeze(1), tp_group).squeeze(-1)
    parts = [node_logp]
    if idx.extra_nodes.numel():
        parts.append(-fused_vocab_parallel_cross_entropy(
            extra_logits.unsqueeze(1), idx.extra_targets.unsqueeze(1),
            tp_group).squeeze(-1))
    flat = torch.cat(parts).index_select(0, idx.logp_src)
    return list(flat.split(idx.resp_lens))


def tree_compute_entropy(idx: TreeLossIndex, output_tensor: torch.Tensor,
                         temperature: float = 1.0, metric_only: bool = False) -> list[torch.Tensor]:
    """Per-row response entropy of a tree micro-batch; ``output_tensor`` is left untouched.

    Args:
        idx: Loss index on the logits' device.
        output_tensor: LM-head output, ``[1, T, V/tp]`` or ``[T, V/tp]``.
        temperature: Sampling temperature.
        metric_only: No gradient; computed in token chunks without full-size temporaries.

    Returns:
        One ``[resp_len]`` tensor per row.
    """
    from megatron.core import parallel_state as mpu

    from coda.backends.megatron.logits_utils import (
        _VocabParallelEntropy,
        _vocab_parallel_entropy_chunked,
    )

    tp_group = mpu.get_tensor_model_parallel_group()
    if metric_only:
        ent = _vocab_parallel_entropy_chunked(_logits_2d(output_tensor), tp_group, temperature)
        return list(ent.index_select(0, idx.logit_nodes).split(idx.resp_lens))
    # response rows only; the copy also shields the logits from the in-place backward
    logits = _logits_2d(output_tensor, temperature).index_select(0, idx.logit_nodes)
    ent = _VocabParallelEntropy.apply(logits, tp_group)
    return list(ent.split(idx.resp_lens))
