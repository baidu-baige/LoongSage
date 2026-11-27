"""Gather plan from tree node logits to per-row response log-probs / entropy.

Row r's response token j is predicted from the logits at node ``path_r[j-1]``; a shared
node has the same context in every row, so only its target can differ (at branch points).
One CE runs over the nodes with each node's default target (next token of its emitting
row), plus a small extra CE for the (node, target) pairs that differ.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from coda.custom.tree_pack.index.plan import TreeSeq, row_path_nodes


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


def owner_index(ts: TreeSeq) -> tuple[torch.Tensor, torch.Tensor]:
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
        rows, offs = owner_index(ts)
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

