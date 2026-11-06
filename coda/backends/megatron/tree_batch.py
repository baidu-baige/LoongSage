"""Tree micro-batch assembly: per-node tokens and R3 routing, loss index, packing.

A node is stored once, so per-row data is taken from one row. Tokens come from the emitting
row (shared prefixes are value-equal). R3 routing of a shared node may differ between rows:
rollouts copy the parent's routing for the shared prefix, except that the first segment
after the GRPO root may get a different routing for the root prompt. Using the LAST row that
contains a node therefore agrees with every row but possibly the root row.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from coda.backends.megatron.tree_loss import _owner_index
from coda.backends.megatron.tree_pack import (
    TreeSeq, build_tree_seq, merge_pieces, merge_positions, row_path_nodes,
)


def _gather(ts: TreeSeq, per_row: list[torch.Tensor], src_row: torch.Tensor,
            src_pos: torch.Tensor) -> torch.Tensor:
    """``out[n] = per_row[src_row[n]][src_pos[n]]`` for the nodes of ``ts``."""
    proto = per_row[ts.row_begin]
    out = proto.new_empty((ts.num_nodes, *proto.shape[1:]))
    for r in range(ts.row_begin, ts.row_end):
        sel = (src_row == r).nonzero(as_tuple=True)[0]
        if sel.numel():
            out[sel] = per_row[r][src_pos[sel]]
    return out


def gather_nodes(seqs: list[TreeSeq], per_row: list[torch.Tensor]) -> torch.Tensor:
    """Per-node values taken from each node's emitting row.

    Args:
        seqs: TreeSeqs in buffer order.
        per_row: One tensor per row, first dim = position in the row.

    Returns:
        ``[N, ...]`` over the nodes of all ``seqs``.
    """
    return torch.cat([_gather(ts, per_row, *_owner_index(ts)) for ts in seqs])


def tree_routed_experts(
    seqs: list[TreeSeq], routed_rows: list[torch.Tensor], with_stats: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """R3 routing of every node, taken from the last row containing it.

    Args:
        seqs: TreeSeqs in buffer order.
        routed_rows: Per-row routing ``[len, layers, topk]``.
        with_stats: Also compute mismatch stats (diagnostic; routing is the same either way).
            They cover (row, node) pairs where the node is shared and the row's routing was
            not the one kept; a token mismatches if any layer's top-k expert set differs.

    Returns:
        ``(node_routing [N, layers, topk], stats)``.
    """
    parts = []
    shared = mism_tok = mism_layer = 0
    for ts in seqs:
        src_row, src_pos = _owner_index(ts)
        uses = torch.zeros(ts.num_nodes, dtype=torch.long) if with_stats else None
        paths = {}
        for r in range(ts.row_begin, ts.row_end):   # later rows overwrite: last row wins
            path = torch.tensor(row_path_nodes(ts, r), dtype=torch.long)
            src_row[path] = r
            src_pos[path] = torch.arange(path.numel())
            if with_stats:
                uses[path] += 1
                paths[r] = path
        out = _gather(ts, routed_rows, src_row, src_pos)
        parts.append(out)
        if not with_stats:
            continue
        for r, path in paths.items():
            sel = ((uses[path] > 1) & (src_row[path] != r)).nonzero(as_tuple=True)[0]
            if not sel.numel():
                continue
            own, used = routed_rows[r][sel], out[path[sel]]
            differ = (own != used).flatten(1).any(-1).nonzero(as_tuple=True)[0]
            shared += sel.numel()
            if differ.numel():       # compare top-k as sets
                d = own[differ].sort(-1).values != used[differ].sort(-1).values
                d = d.any(-1) if d.dim() > 2 else d
                mism_tok += int(d.reshape(d.shape[0], -1).any(-1).sum())
                mism_layer += int(d.sum())
    node_routing = torch.cat(parts)
    if not with_stats:
        return node_routing, {}
    n_layers = node_routing.shape[1] if node_routing.dim() > 1 else 1
    stats = {
        "tree/r3_shared_tokens": float(shared),
        "tree/r3_mismatch_tokens": float(mism_tok),
        "tree/r3_mismatch_token_ratio": mism_tok / shared if shared else 0.0,
        "tree/r3_mismatch_layer_ratio": mism_layer / (shared * n_layers) if shared else 0.0,
    }
    return node_routing, stats


def lcp_len(a: torch.Tensor, b: torch.Tensor) -> int:
    """Longest common prefix length of two 1-D tensors ``a`` and ``b``."""
    m = min(a.numel(), b.numel())
    ne = (a[:m] != b[:m]).nonzero()
    return int(ne[0, 0]) if ne.numel() else m


def chain_branch_points(tokens: list[torch.Tensor], groups: list[int]) -> list[int]:
    """Branch points of consecutive rows.

    Args:
        tokens: 1-D token tensor of every row.
        groups: Group id of every row; only consecutive rows of a group are chained.

    Returns:
        Per row, its shared prefix with the previous row of the group (0 = new chain),
        capped at ``len - 1`` so each row owns a node. A chain restarts where ``bp`` would
        decrease (the packer needs it non-decreasing; restarting only recomputes a prefix).
    """
    bp = [0] * len(tokens)
    for k in range(1, len(tokens)):
        if groups[k] != groups[k - 1]:
            continue
        b = min(lcp_len(tokens[k], tokens[k - 1]), tokens[k].numel() - 1)
        bp[k] = b if b >= bp[k - 1] else 0
    return bp


def plan_trees(lens: list[int], bp: list[int], resp: list[int]) -> list[TreeSeq]:
    """Split rows into TreeSeqs.

    Args:
        lens, bp, resp: Row lengths, branch points and response lengths.

    Returns:
        One TreeSeq per maximal run of rows chained by ``bp > 0``.
    """
    seqs, a = [], 0
    for k in range(1, len(lens) + 1):
        if k == len(lens) or bp[k] == 0:
            seqs.append(build_tree_seq(a, k, lens, bp, resp))
            a = k
    return seqs


@dataclass
class TreeMicroBatch:
    """Model inputs of one tree micro-batch."""

    tokens: torch.Tensor            # [T] node tokens, tail padded like row-wise packing
    packed_seq_params: object       # carries tree_pieces / tree_loss_index
    position_ids: torch.Tensor      # [3, 1, T] mRoPE positions (node depth)
    node_routing: torch.Tensor | None  # [N, layers, topk] R3 routing (unpadded)
    stats: dict[str, float]
    seqs: list = None               # TreeSeqs in micro-batch row order


def build_tree_microbatch(
    tokens: list[torch.Tensor],
    response_lengths: list[int],
    groups: list[int],
    routed_rows: list[torch.Tensor] | None = None,
    cp_partition_mode: str = "zigzag",
    restrict_lm_rows: bool = False,
    r3_stats: bool = True,
) -> TreeMicroBatch:
    """Pack a micro-batch of rows into one tree node buffer.

    Args:
        tokens: 1-D token tensor of every row.
        response_lengths: Response length of every row.
        groups: Trajectory id of every row; rows of an id must be consecutive and in
            trajectory order.
        routed_rows: Per-row R3 routing ``[len, layers, topk]`` (None: no R3).
        cp_partition_mode: As for ``prepare_packed_seq_params``.
        restrict_lm_rows: Express the loss index over the response logit rows only.
        r3_stats: Compute the R3 mismatch stats.

    Returns:
        The tree micro-batch.
    """
    from coda.backends.megatron.cp_utils import prepare_packed_seq_params
    from coda.backends.megatron.tree_loss import build_loss_index, index_to, restrict_to_lm_rows

    dev = tokens[0].device
    lens = [int(t.numel()) for t in tokens]
    bp = chain_branch_points(tokens, groups)
    seqs = plan_trees(lens, bp, list(response_lengths))
    cpu_rows = [t.detach().to("cpu", torch.long) for t in tokens]
    node_tok = gather_nodes(seqs, cpu_rows)
    packed, psp = prepare_packed_seq_params([node_tok.to(dev)], cp_partition_mode=cp_partition_mode)
    T = packed.numel()
    psp.tree_pieces = merge_pieces(seqs, T)
    loss_index = build_loss_index(seqs, cpu_rows, list(response_lengths), T)
    if restrict_lm_rows:
        loss_index = restrict_to_lm_rows(loss_index)
    psp.tree_loss_index = index_to(loss_index, dev)
    pos = torch.tensor(merge_positions(seqs, T), dtype=torch.long, device=dev)
    stats = {
        "tree/rows": float(len(tokens)),
        "tree/seqs": float(len(seqs)),
        "tree/row_tokens": float(sum(lens)),
        "tree/nodes": float(node_tok.numel()),
        "tree/lm_rows": float(T if loss_index.lm_rows is None else loss_index.lm_rows.numel()),
    }
    node_routing = None
    if routed_rows is not None:
        node_routing, r3 = tree_routed_experts(seqs, routed_rows, with_stats=r3_stats)
        stats.update(r3)
    return TreeMicroBatch(packed, psp, pos.view(1, 1, -1).expand(3, 1, -1), node_routing, stats,
                          seqs)
