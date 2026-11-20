"""Tree packing for shared-prefix training (index math only, no torch).

A trajectory's trainable rows form a caterpillar: row k shares a prefix of length
``bp_k = LCP(row_k, row_{k-1})`` with row k-1 (``bp`` non-decreasing) and stores only
``row_k[bp_k:]``. Row k's root path is

    row_0[0:bp_1] ++ row_1[bp_1:bp_2] ++ ... ++ row_k[bp_k:]

so each token is stored once. ``TreeSeq`` is one packed node buffer; it is cut into
*pieces* (contiguous node runs with a single parent) that GDN / attention process from the
parent's state, and ``row_nodes`` maps each row's response back to nodes for the loss.

Common arguments: ``row_lens`` (row lengths), ``bp`` (branch point of every row against the
previous one, 0 = new chain) and ``resp_lens`` (response length, the row's last tokens).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from coda.utils.tree_pack_utils import TREE_ATTN_PAIR_COST, TREE_LEVEL_COST


@dataclass
class Piece:
    """Nodes ``[node_start, node_end)`` continuing piece ``parent_piece`` (-1: root).
    ``depth_start`` is the root-path length before the first node."""

    node_start: int
    node_end: int
    parent_piece: int
    depth_start: int


@dataclass
class TreeSeq:
    """One packed sequence: rows ``[row_begin, row_end)`` of a trajectory."""

    row_begin: int
    row_end: int
    node_order: list[tuple[int, int]] = field(default_factory=list)  # node -> (row, offset)
    pieces: list[Piece] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)     # node -> root-path depth (RoPE)
    row_nodes: dict[int, list[int]] = field(default_factory=dict)  # row -> response nodes
    row_last_piece: dict[int, int] = field(default_factory=dict)   # row -> its leaf piece

    @property
    def num_nodes(self) -> int:
        return len(self.node_order)


def _seq_cost(row_lens: list[int], bp: list[int], a: int, k: int) -> int:
    """Nodes a row adds to a sequence whose first row is ``a``."""
    return row_lens[k] if k == a else row_lens[k] - bp[k]


def plan_sequences(row_lens: list[int], bp: list[int], cap: int,
                   resp_lens: list[int] | None = None,
                   lm_cap: int | None = None) -> list[tuple[int, int]]:
    """Greedily group consecutive rows into tree sequences.

    Args:
        row_lens, bp, resp_lens: See the module docstring.
        cap: Max nodes per sequence.
        lm_cap: Max response tokens (logit rows) per sequence; None = no limit.

    Returns:
        ``(row_begin, row_end)`` per sequence; a row over a cap forms its own sequence.
    """
    seqs: list[tuple[int, int]] = []
    a = 0
    n = len(row_lens)
    while a < n:
        nodes = row_lens[a]
        lm = resp_lens[a] if lm_cap is not None else 0
        b = a + 1
        while b < n and nodes + _seq_cost(row_lens, bp, a, b) <= cap and (
                lm_cap is None or lm + resp_lens[b] <= lm_cap):
            nodes += _seq_cost(row_lens, bp, a, b)
            if lm_cap is not None:
                lm += resp_lens[b]
            b += 1
        seqs.append((a, b))
        a = b
    return seqs


@dataclass(frozen=True)
class TreeCostWeights:
    """Cost of a tree sequence in node-equivalents: nodes + weighted attention pairs and
    serial GDN levels (defaults in ``coda.utils.tree_pack_utils``)."""

    attn_pair: float = TREE_ATTN_PAIR_COST
    level: float = TREE_LEVEL_COST


def group_nodes(row_lens: list[int], bp: list[int], a: int, b: int) -> int:
    """Nodes of one tree sequence.

    Args:
        row_lens, bp: See the module docstring.
        a, b: Rows ``[a, b)`` of the sequence.

    Returns:
        Its node count (row ``a`` in full, later rows from their branch point).
    """
    return row_lens[a] + sum(row_lens[k] - bp[k] for k in range(a + 1, b))


def group_cost(row_lens: list[int], bp: list[int], a: int, b: int,
               w: TreeCostWeights) -> float:
    """Estimated cost of one tree sequence.

    Args:
        row_lens, bp: See the module docstring.
        a, b: Rows ``[a, b)`` of the sequence.
        w: Cost weights.

    Returns:
        Nodes + causal attention pairs (row k owns depths ``[lo_k, L_k)``) + GDN levels
        (~ rows + 1), in node-equivalents.
    """
    pairs = 0.0
    for k in range(a, b):
        lo = 0 if k == a else bp[k]
        L = row_lens[k]
        pairs += (L * (L + 1) - lo * (lo + 1)) / 2
    return group_nodes(row_lens, bp, a, b) + w.attn_pair * pairs + w.level * (b - a + 1)


def plan_sequences_min_cost(row_lens: list[int], bp: list[int], cap: int,
                            w: TreeCostWeights | None = None,
                            resp_lens: list[int] | None = None,
                            lm_cap: int | None = None) -> list[tuple[int, int]]:
    """Like ``plan_sequences`` but with the cut points minimizing the total ``group_cost``.

    Each new sequence recomputes its first row's prefix, so cutting early where the shared
    prefix is still short can be cheaper.

    Args:
        row_lens, bp, resp_lens: See the module docstring.
        cap: Max nodes per sequence.
        w: Cost weights (None: defaults).
        lm_cap: Max response tokens per sequence; None = no limit.

    Returns:
        ``(row_begin, row_end)`` per sequence.
    """
    w = w or TreeCostWeights()
    n = len(row_lens)
    inf = float("inf")
    best = [0.0] + [inf] * n
    arg = [0] * (n + 1)
    for j in range(1, n + 1):
        # running group_nodes and 2*pairs of rows [i, j): integer sums, so the cost is
        # bit-equal to group_cost(row_lens, bp, i, j, w)
        lm = nodes = pairs2 = 0
        for i in range(j - 1, -1, -1):
            if i < j - 1:       # row i+1 no longer leads the group: it owns [bp, L) only
                lo = bp[i + 1]
                nodes -= lo
                pairs2 -= lo * (lo + 1)
            L = row_lens[i]
            nodes += L
            pairs2 += L * (L + 1)
            if lm_cap is not None:
                lm += resp_lens[i]
            if j - i > 1 and (nodes > cap or (lm_cap is not None and lm > lm_cap)):
                break       # both only grow as the group extends back
            c = best[i] + (nodes + w.attn_pair * (pairs2 / 2) + w.level * (j - i + 1))
            if c < best[j]:
                best[j], arg[j] = c, i
    out, j = [], n
    while j > 0:
        out.append((arg[j], j))
        j = arg[j]
    return out[::-1]


def build_tree_seq(
    row_begin: int,
    row_end: int,
    row_lens: list[int],
    bp: list[int],
    resp_lens: list[int],
) -> TreeSeq:
    """Build one TreeSeq. Each row emits a spine piece (up to the next row's branch point)
    and a thorn piece (the rest).

    Args:
        row_begin, row_end: Rows ``[row_begin, row_end)`` of the sequence.
        row_lens, bp, resp_lens: See the module docstring (indexed by trajectory row).

    Returns:
        The TreeSeq, with node indices local to it.
    """
    ts = TreeSeq(row_begin, row_end)

    def child_bp(r: int) -> int:
        return bp[r + 1] if r + 1 < row_end else row_lens[r]

    # depth_owner[d] = spine piece holding root-path depth d (bp is non-decreasing, so only
    # already-recorded branch depths are queried)
    depth_owner: dict[int, int] = {}

    def emit_piece(r: int, c0: int, c1: int, parent: int) -> int:
        """Append row ``r``'s offsets ``[c0, c1)`` as a piece; returns its index."""
        pstart = len(ts.node_order)
        for off in range(c0, c1):
            ts.node_order.append((r, off))
            ts.positions.append(off)
        ts.pieces.append(Piece(pstart, len(ts.node_order), parent, c0))
        return len(ts.pieces) - 1

    for r in range(row_begin, row_end):
        lo = 0 if r == row_begin else bp[r]
        hi = row_lens[r]
        branch = min(child_bp(r), hi)
        resp_start = row_lens[r] - resp_lens[r]
        branch_parent = -1 if lo == 0 else depth_owner.get(lo - 1, -1)
        n0 = len(ts.node_order)

        if branch > lo:  # spine [lo, branch)
            thorn_parent = emit_piece(r, lo, branch, branch_parent)
            depth_owner[branch - 1] = thorn_parent
        else:            # the next row branches where this one does
            thorn_parent = branch_parent
        if hi > branch:  # thorn [branch, hi)
            emit_piece(r, branch, hi, thorn_parent)

        n1 = len(ts.node_order)  # row r emitted offsets lo..hi-1 as nodes [n0, n1)
        ts.row_nodes[r] = list(range(n0 + max(resp_start, lo) - lo, n1))
        ts.row_last_piece[r] = len(ts.pieces) - 1 if n1 > n0 else thorn_parent
    return ts


def row_path_nodes(ts: TreeSeq, r: int) -> list[int]:
    """Root path of one row.

    Args:
        ts: The TreeSeq holding the row.
        r: Row index.

    Returns:
        Node indices of the row's tokens, in row order.
    """
    chain = []
    cur = ts.row_last_piece[r]
    while cur != -1:
        chain.append(cur)
        cur = ts.pieces[cur].parent_piece
    chain.reverse()
    out: list[int] = []
    for pi in chain:
        p = ts.pieces[pi]
        out.extend(range(p.node_start, p.node_end))
    return out


def merge_pieces(seqs: list[TreeSeq], total_nodes: int) -> list[Piece]:
    """Pieces of TreeSeqs packed back-to-back into one buffer.

    Args:
        seqs: TreeSeqs in buffer order.
        total_nodes: Buffer length; trailing padding becomes its own root piece.

    Returns:
        The pieces in global node coordinates.
    """
    out: list[Piece] = []
    node_off = 0
    for ts in seqs:
        piece_off = len(out)
        for p in ts.pieces:
            out.append(Piece(p.node_start + node_off, p.node_end + node_off,
                             -1 if p.parent_piece == -1 else p.parent_piece + piece_off,
                             p.depth_start))
        node_off += ts.num_nodes
    if total_nodes < node_off:
        raise ValueError(f"buffer of {total_nodes} nodes < packed {node_off}")
    if total_nodes > node_off:
        out.append(Piece(node_off, total_nodes, -1, 0))
    return out


def merge_positions(seqs: list[TreeSeq], total_nodes: int) -> list[int]:
    """RoPE positions of a ``merge_pieces`` buffer.

    Args:
        seqs: TreeSeqs in buffer order.
        total_nodes: Buffer length.

    Returns:
        Every node's depth in its row; padding counts from 0 like a row-wise sequence.
    """
    out: list[int] = []
    for ts in seqs:
        out.extend(ts.positions)
    return out + list(range(total_nodes - len(out)))

