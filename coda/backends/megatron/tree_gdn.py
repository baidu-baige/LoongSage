"""Tree execution of the two sequence-mixing kernels of a GatedDeltaNet layer.

Everything else in Megatron's GDN ``_forward_compute`` is per-token. Both kernels below call
the same fla functions Megatron uses; only the sequence layout changes, so every node sees
exactly the causal context of its own row while being computed once:

  * conv: each piece is one varlen sequence with its W-1 root-path predecessors prepended
    (their outputs dropped); all pieces go in one call.
  * GDN core: serial varlen calls chained by ``final_state`` -> ``initial_state``. Chains of
    pieces are cut at anchors; a branch off a non-anchor re-reads the run since the last
    anchor instead of adding a call (``_gdn_levels``). Autograd sums the gradients of
    re-read tokens.
"""
from __future__ import annotations

import inspect
from contextvars import ContextVar

import torch

from coda.backends.megatron.tree_pack import Piece


# Tokens one serial GDN call is worth when trading calls for re-read tokens (0 = one call
# per tree depth). Set from trainer.tree_gdn_level_tokens.
GDN_LEVEL_TOKENS = 16000.0


def _cached(cache, key, build):
    """Micro-batch scoped layout cache (shared by all layers and the recompute pass)."""
    if cache is None:
        return build()
    if key not in cache:
        cache[key] = build()
    return cache[key]


def context_indices(pieces: list[Piece], width: int) -> list[list[int]]:
    """Conv context of every piece.

    Args:
        pieces: Pieces of a node buffer (parents before children).
        width: Conv kernel width.

    Returns:
        Per piece, the ``width - 1`` nodes preceding it on its root path (fewer near the root).
    """
    need = width - 1
    out: list[list[int]] = []
    for p in pieces:
        ctx: list[int] = []
        cur = p.parent_piece
        while cur != -1 and len(ctx) < need:
            q = pieces[cur]
            take = min(need - len(ctx), q.node_end - q.node_start)
            ctx = list(range(q.node_end - take, q.node_end)) + ctx
            cur = q.parent_piece
        out.append(ctx)
    return out


def _cu(lens: list[int], device) -> torch.Tensor:
    t = torch.zeros(len(lens) + 1, dtype=torch.long, device=device)
    t[1:] = torch.tensor(lens, dtype=torch.long, device=device).cumsum(0)
    return t


def _check_cover(pieces: list[Piece], T: int) -> None:
    """The (non-empty) piece ranges tile ``[0, T)``."""
    end = 0
    for p in sorted((p for p in pieces if p.node_end > p.node_start), key=lambda p: p.node_start):
        if p.node_start != end:
            raise ValueError(f"tree pieces leave a gap at node {end}")
        end = p.node_end
    if end != T:
        raise ValueError(f"tree pieces cover {end} nodes, buffer has {T}")


def _conv_layout(pieces: list[Piece], width: int, T: int, dev):
    gather, lens, src, dst = [], [], [], []
    for p, ctx in zip(pieces, context_indices(pieces, width)):
        src.extend(range(len(gather) + len(ctx), len(gather) + len(ctx) + p.node_end - p.node_start))
        dst.extend(range(p.node_start, p.node_end))
        gather.extend(ctx + list(range(p.node_start, p.node_end)))
        lens.append(len(ctx) + p.node_end - p.node_start)
    _check_cover(pieces, T)
    perm = torch.empty(T, dtype=torch.long)
    perm[torch.tensor(dst)] = torch.tensor(src)
    return torch.tensor(gather, dtype=torch.long, device=dev), _cu(lens, dev), perm.to(dev)


def tree_causal_conv1d(x, weight, bias, activation, pieces: list[Piece], conv_fn,
                       cache=None) -> torch.Tensor:
    """Causal conv over a node buffer.

    Args:
        x: Node buffer ``[1, T, D]``.
        weight, bias, activation: As for ``conv_fn``.
        pieces: Pieces tiling the buffer.
        conv_fn: fla ``causal_conv1d``.
        cache: Micro-batch layout cache (None: build every call).

    Returns:
        ``[1, T, D]`` in node order.
    """
    gather_t, cu, perm = _cached(cache, ("conv", weight.shape[-1], x.shape[1], x.device),
                                 lambda: _conv_layout(pieces, weight.shape[-1], x.shape[1], x.device))
    y, _ = conv_fn(x=x.index_select(1, gather_t), weight=weight, bias=bias,
                   activation=activation, initial_state=None, output_final_state=False,
                   cu_seqlens=cu)
    return y.index_select(1, perm)


def _gdn_levels(pieces: list[Piece], dev, level_tokens: float = 0.0):
    """Layout of the serial GDN calls (one per level).

    Pieces form chains (each continued by its largest child). A segment is a chain run ending
    at an anchor, run as one varlen sequence from the previous anchor's final state; a branch
    off a non-anchor re-reads the run since that anchor. Anchors are chosen per chain to
    minimize ``level_tokens * calls + re-read tokens`` (0: anchor every piece).

    Returns ``[(gidx, cu, init, keep, own_pos, cu_cpu)]`` per level and the permutation from the
    concatenated ``own_pos`` outputs to node order. ``init``: per sequence, index into the
    finals kept so far (-1: zero state); ``keep``: sequences whose final state is kept.
    """
    n = len(pieces)
    size = [p.node_end - p.node_start for p in pieces]
    kids: list[list[int]] = [[] for _ in range(n)]
    for i, p in enumerate(pieces):
        if p.parent_piece != -1:
            kids[p.parent_piece].append(i)
    sub = size[:]                              # subtree node counts (parents precede kids)
    for i in reversed(range(n)):
        if pieces[i].parent_piece != -1:
            sub[pieces[i].parent_piece] += sub[i]
    cont = [max(kids[i], key=lambda c: sub[c]) if kids[i] else -1 for i in range(n)]

    seg_of = [-1] * n                          # segment holding each piece's own nodes
    upto = [0] * n                             # its sequence length through the piece's end
    segs: list[dict] = []
    for s in range(n):
        p = pieces[s].parent_piece
        if p != -1 and cont[p] == s:
            continue                           # not a chain start
        chain = [s]
        while cont[chain[-1]] != -1:
            chain.append(cont[chain[-1]])
        if p == -1:
            ctx, init_seg = [], -1
        elif segs[seg_of[p]]["end"] == p:      # parent ends its segment: start from its state
            ctx, init_seg = [], seg_of[p]
        else:                                  # re-read the run since the parent's anchor
            ctx, init_seg = segs[seg_of[p]]["nodes"][:upto[p]], segs[seg_of[p]]["init_seg"]
        ends = _chain_anchors([size[c] for c in chain],
                              [len(kids[c]) - (cont[c] != -1) for c in chain],
                              len(ctx), level_tokens)
        start = 0
        for e in ends:
            nodes = list(ctx) if start == 0 else []
            own = []
            for c in chain[start:e + 1]:
                own.append((len(nodes), c))
                nodes.extend(range(pieces[c].node_start, pieces[c].node_end))
                upto[c] = len(nodes)
                seg_of[c] = len(segs)
            level = 0 if init_seg == -1 else segs[init_seg]["level"] + 1
            segs.append(dict(level=level, init_seg=init_seg, nodes=nodes, own=own,
                             end=chain[e]))
            init_seg, start = len(segs) - 1, e + 1

    needed = {sg["init_seg"] for sg in segs if sg["init_seg"] != -1}
    final_idx: dict[int, int] = {}             # segment -> index into the stacked finals
    levels, own_order, placed = [], [], []
    for L in range(max(sg["level"] for sg in segs) + 1):
        ids = [j for j, sg in enumerate(segs) if sg["level"] == L]
        gidx, lens, own_pos, keep = [], [], [], []
        for j in ids:
            sg = segs[j]
            for pos, c in sg["own"]:
                own_pos.extend(range(len(gidx) + pos, len(gidx) + pos + size[c]))
                own_order.extend(range(pieces[c].node_start, pieces[c].node_end))
                placed.append(c)
            gidx.extend(sg["nodes"])
            lens.append(len(sg["nodes"]))
            if j in needed:
                keep.append(len(lens) - 1)
        init = [final_idx[segs[j]["init_seg"]] if segs[j]["init_seg"] != -1 else -1
                for j in ids]
        base = len(final_idx)
        for t, slot in enumerate(keep):
            final_idx[ids[slot]] = base + t
        cu_cpu = _cu(lens, "cpu")
        levels.append((torch.tensor(gidx, dtype=torch.long, device=dev), cu_cpu.to(dev),
                       init, keep, torch.tensor(own_pos, dtype=torch.long, device=dev), cu_cpu))
    T = sum(size)
    assert sorted(placed) == list(range(n)), "every piece must be placed exactly once"
    _check_cover(pieces, T)
    perm = torch.empty(T, dtype=torch.long)
    perm[torch.tensor(own_order)] = torch.arange(T)
    return levels, perm.to(dev)


def _chain_anchors(lens: list[int], branches: list[int], base: int,
                   level_tokens: float) -> list[int]:
    """Anchor positions along a chain (the last is the chain end), exact O(m^2) DP.
    ``lens``: piece sizes; ``branches``: side children per piece (each re-reads the run since
    the last anchor if the piece is not one); ``base``: context the first segment re-reads.

    Segment ``s..i`` (anchor at i) costs ``level_tokens + sum_t branches[t] * run_t`` over
    its non-anchor pieces; ``acc[s]`` / ``run[s]`` grow by one piece per step."""
    m = len(lens)
    best, back, acc, run = [], [], [], []
    for i in range(m):
        for s in range(i):                     # extend segment s.. by piece i-1
            run[s] += lens[i - 1]
            acc[s] += branches[i - 1] * run[s]
        acc.append(level_tokens)
        run.append(base if i == 0 else 0)
        b, bj = float("inf"), -1
        for s in range(i + 1):
            cost = acc[s] + (0.0 if s == 0 else best[s - 1])
            if cost <= b:                      # ties: more anchors (shorter segments)
                b, bj = cost, s - 1
        best.append(b)
        back.append(bj)
    ends, i = [], m - 1
    while i != -1:
        ends.append(i)
        i = back[i]
    return ends[::-1]


def tree_gated_delta_rule(q, k, v, g, beta, pieces: list[Piece], gdr_fn, cache=None,
                          level_tokens: float | None = None, cpu_cu: bool = False,
                          **kw) -> torch.Tensor:
    """GDN core over a node buffer.

    Args:
        q, k: ``[1, T, H, K]``; v: ``[1, T, H, V]``; g, beta: ``[1, T, H]``.
        pieces: Pieces tiling the buffer.
        gdr_fn: fla ``chunk_gated_delta_rule``.
        cache: Micro-batch layout cache (None: build every call).
        level_tokens: See ``GDN_LEVEL_TOKENS`` (None: use it).
        cpu_cu: Also pass ``cu_seqlens_cpu`` (fla then builds its chunk indices without a
            device sync).
        **kw: Passed through to ``gdr_fn``.

    Returns:
        ``[1, T, H, V]`` in node order.
    """
    lt = GDN_LEVEL_TOKENS if level_tokens is None else level_tokens
    levels, perm = _cached(cache, ("gdn", lt, q.device), lambda: _gdn_levels(pieces, q.device, lt))
    outs, finals = [], []
    for gidx, cu, init_ids, keep, own_pos, cu_cpu in levels:
        sel = [t.index_select(1, gidx) for t in (q, k, v, g, beta)]
        # level 0 starts from zero; later sequences from a kept final of a lower level
        init = None if init_ids[0] == -1 else torch.stack([finals[i] for i in init_ids])
        if cpu_cu:
            kw["cu_seqlens_cpu"] = cu_cpu
        o, ht = gdr_fn(*sel, initial_state=init, output_final_state=True, cu_seqlens=cu, **kw)
        outs.append(o.index_select(1, own_pos))
        finals.extend(ht[j] for j in keep)
    return torch.cat(outs, dim=1).index_select(1, perm)


# Megatron integration (monkeypatch). Tree mode is per micro-batch: ``packed_seq_params``
# carries ``tree_pieces``; without it every call takes the original path.
_TREE_PIECES: ContextVar = ContextVar("coda_tree_pieces", default=None)


_GDN_FORWARD_COMPUTE_PARAMS = (
    "self", "hidden_states", "batch", "seq_len_post_headwise", "cp_size_headwise",
    "cp_group_headwise", "cp_size_chunkwise", "cp_group_chunkwise", "cu_seqlens_q",
    "packed_seq_params", "chunkwise_cp_context",
)


def _check_gdn_patch_points(gdn_mod) -> None:
    """Fail loudly if Megatron changed what the patch hooks (otherwise row-wise kernels
    would silently run on a tree buffer)."""
    params = tuple(inspect.signature(gdn_mod.GatedDeltaNet._forward_compute).parameters)
    problems = []
    if params != _GDN_FORWARD_COMPUTE_PARAMS:
        problems.append(f"GatedDeltaNet._forward_compute params {params}")
    if not callable(getattr(gdn_mod, "causal_conv1d", None)):
        problems.append("gdn.causal_conv1d missing")
    try:
        if "causal_conv1d(" not in inspect.getsource(gdn_mod.GatedDeltaNet.pre_gated_delta_rule):
            problems.append("pre_gated_delta_rule no longer calls causal_conv1d")
        if "self.gated_delta_rule(" not in inspect.getsource(gdn_mod.GatedDeltaNet._forward_compute):
            problems.append("_forward_compute no longer calls self.gated_delta_rule")
    except (OSError, TypeError):
        problems.append("cannot read GatedDeltaNet source")
    if problems:
        raise RuntimeError("tree_pack GDN patch is incompatible with this Megatron: "
                           + "; ".join(problems))


def install_gdn_tree_patch(level_tokens: float | None = None) -> None:
    """Route GatedDeltaNet's conv1d and GDN core through the tree kernels for micro-batches
    with ``tree_pieces``. Idempotent; ``level_tokens`` sets ``GDN_LEVEL_TOKENS``."""
    global GDN_LEVEL_TOKENS
    from megatron.core.ssm.gated_delta_net import gdn as gdn_mod

    if level_tokens is not None:
        GDN_LEVEL_TOKENS = float(level_tokens)
    if getattr(gdn_mod, "_coda_tree_patched", False):
        return
    _check_gdn_patch_points(gdn_mod)
    fla_gdr = gdn_mod.chunk_gated_delta_rule   # deterministic_mode uses a torch fallback
    fla_cpu_cu = "cu_seqlens_cpu" in inspect.signature(fla_gdr).parameters
    orig_conv = gdn_mod.causal_conv1d
    orig_fc = gdn_mod.GatedDeltaNet._forward_compute

    def conv_dispatch(*args, **kwargs):
        """``causal_conv1d`` on the node layout inside a tree ``_forward_compute``."""
        pieces = _TREE_PIECES.get()
        if pieces is None:
            return orig_conv(*args, **kwargs)
        if kwargs.get("cp_context") is not None or kwargs.get("initial_state") is not None:
            raise NotImplementedError("tree GDN conv: CP / initial_state unsupported")
        pieces, cache = pieces
        y = tree_causal_conv1d(kwargs["x"], kwargs["weight"], kwargs.get("bias"),
                               kwargs.get("activation"), pieces, orig_conv, cache=cache)
        return y, None

    def forward_compute(self, hidden_states, batch, seq_len_post_headwise, cp_size_headwise,
                        cp_group_headwise, cp_size_chunkwise, cp_group_chunkwise,
                        cu_seqlens_q, packed_seq_params, chunkwise_cp_context):
        """Original ``_forward_compute``, with both kernels on the tree layout when
        ``packed_seq_params`` carries ``tree_pieces``."""
        args = (hidden_states, batch, seq_len_post_headwise, cp_size_headwise,
                cp_group_headwise, cp_size_chunkwise, cp_group_chunkwise, cu_seqlens_q,
                packed_seq_params, chunkwise_cp_context)
        pieces = getattr(packed_seq_params, "tree_pieces", None)
        if pieces is None:
            return orig_fc(self, *args)
        if self.gdn_pre_gated_delta_rule_fusion:
            raise NotImplementedError("tree GDN requires gdn_pre_gated_delta_rule_fusion=False")
        # SP is fine: in_proj all-gathers the sequence, both kernels see the whole buffer
        if cp_size_headwise > 1 or cp_size_chunkwise > 1:
            raise NotImplementedError("tree GDN supports CP=1 only")
        if batch != 1:
            raise NotImplementedError("tree GDN expects a THD buffer with batch == 1")

        orig_gdr = self.gated_delta_rule
        cpu_cu = fla_cpu_cu and orig_gdr is fla_gdr
        cache = packed_seq_params.__dict__.setdefault("tree_kernel_cache", {})

        def gdr_dispatch(q, k, v, g, beta, initial_state=None, output_final_state=False,
                         cu_seqlens=None, cp_context=None, **kw):
            o = tree_gated_delta_rule(q, k, v, g, beta, pieces, orig_gdr, cache=cache,
                                      cpu_cu=cpu_cu, **kw)
            return o, None

        token = _TREE_PIECES.set((pieces, cache))
        self.gated_delta_rule = gdr_dispatch
        try:
            return orig_fc(self, *args)
        finally:
            self.gated_delta_rule = orig_gdr
            _TREE_PIECES.reset(token)

    gdn_mod.causal_conv1d = conv_dispatch
    gdn_mod.GatedDeltaNet._forward_compute = forward_compute
    gdn_mod._coda_tree_patched = True
