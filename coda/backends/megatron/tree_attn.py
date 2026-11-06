"""Tree execution of full attention: each node attends to its root path.

Each piece is one THD query sequence whose K/V is gathered as ``ancestor nodes ++ piece
nodes``; bottom-right causal alignment then equals the row-wise causal mask. Shared K/V
gradients accumulate through ``index_select``. FlashAttention only: TE's cuDNN fused
attention returns wrong dK/dV for this layout at head_dim 256.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from coda.backends.megatron.tree_pack import Piece


@dataclass
class TreeAttnLayout:
    """THD varlen layout of the tree attention: one query sequence per piece."""

    kv_index: torch.Tensor       # [sum_kv] node index of every gathered K/V entry
    cu_seqlens_q: torch.Tensor   # [P+1] int32
    cu_seqlens_kv: torch.Tensor  # [P+1] int32
    max_seqlen_q: int
    max_seqlen_kv: int


def build_attn_layout(pieces: list[Piece], device: torch.device | str) -> TreeAttnLayout:
    """Attention layout of a node buffer.

    Args:
        pieces: Pieces in ``merge_pieces`` order (parents before children).
        device: Device of the layout tensors.

    Returns:
        Per piece: queries = its nodes, keys = its root path.
    """
    own = [torch.arange(p.node_start, p.node_end) for p in pieces]
    paths: list[list[int]] = []
    idx_parts: list[torch.Tensor] = []
    q_lens, kv_lens = [], []
    for i, p in enumerate(pieces):
        par = p.parent_piece
        paths.append((paths[par] if par != -1 else []) + [i])
        idx_parts.extend(own[c] for c in paths[i])
        q_lens.append(p.node_end - p.node_start)
        kv_lens.append(q_lens[i] + (kv_lens[par] if par != -1 else 0))

    def cu(lens):
        t = torch.zeros(len(lens) + 1, dtype=torch.int32)
        t[1:] = torch.tensor(lens, dtype=torch.int32).cumsum(0)
        return t.to(device)

    return TreeAttnLayout(torch.cat(idx_parts).to(device), cu(q_lens), cu(kv_lens),
                          max(q_lens), max(kv_lens))


def _check_pieces_in_order(pieces: list[Piece], total: int) -> None:
    pos = 0
    for p in pieces:
        if p.node_start != pos:
            raise ValueError("tree pieces must partition the node buffer in order")
        pos = p.node_end
    if pos != total:
        raise ValueError(f"tree pieces cover {pos} nodes, buffer has {total}")


def get_attn_layout(packed_seq_params, total: int, device: torch.device | str) -> TreeAttnLayout:
    """Attention layout of a tree micro-batch, cached on it for all layers.

    Args:
        packed_seq_params: The micro-batch's params carrying ``tree_pieces``.
        total: Buffer length.
        device: Device of the layout tensors.

    Returns:
        The cached ``TreeAttnLayout``.
    """
    lay = getattr(packed_seq_params, "tree_attn_layout", None)
    if lay is None:
        pieces = packed_seq_params.tree_pieces
        _check_pieces_in_order(pieces, total)
        lay = build_attn_layout(pieces, device)
        packed_seq_params.tree_attn_layout = lay
    return lay


def _require_flash_backend() -> None:
    if os.environ.get("NVTE_FUSED_ATTN", "1") != "0" or os.environ.get("NVTE_FLASH_ATTN", "1") == "0":
        raise RuntimeError(
            "tree attention requires the FlashAttention backend "
            "(NVTE_FUSED_ATTN=0, NVTE_FLASH_ATTN=1): cuDNN FusedAttention gives wrong "
            "dK/dV for this layout at head_dim 256")


def install_attention_tree_patch() -> None:
    """Route TEDotProductAttention through the tree layout for micro-batches with
    ``tree_pieces`` (monkeypatch). Idempotent."""
    import transformer_engine.pytorch as te
    from megatron.core.extensions import transformer_engine as mte

    cls = mte.TEDotProductAttention
    if getattr(cls, "_coda_tree_patched", False):
        return
    import inspect

    params = tuple(inspect.signature(cls.forward).parameters)
    expected = ("self", "query", "key", "value", "attention_mask", "attn_mask_type",
                "attention_bias", "packed_seq_params", "num_splits")
    if params != expected:
        raise RuntimeError(
            f"tree_pack attention patch is incompatible with this Megatron: "
            f"TEDotProductAttention.forward params {params}, expected {expected}")
    orig_forward = cls.forward

    def forward(self, query, key, value, attention_mask, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None, num_splits=None):
        """Original forward, or tree attention when ``packed_seq_params`` has ``tree_pieces``."""
        if getattr(packed_seq_params, "tree_pieces", None) is None:
            return orig_forward(self, query, key, value, attention_mask, attn_mask_type,
                                attention_bias=attention_bias,
                                packed_seq_params=packed_seq_params, num_splits=num_splits)
        if packed_seq_params.qkv_format != "thd":
            raise NotImplementedError("tree attention expects THD inputs")
        if attention_bias is not None or self.config.context_parallel_size > 1:
            raise NotImplementedError("tree attention: attention_bias / CP unsupported")
        if self.config.qk_clip or self.config.log_max_attention_logit:
            raise NotImplementedError("tree attention: qk_clip / max-logit logging unsupported")
        _require_flash_backend()
        lay = get_attn_layout(packed_seq_params, query.shape[0], query.device)
        k = key.index_select(0, lay.kv_index)
        v = value.index_select(0, lay.kv_index)
        return te.DotProductAttention.forward(
            self, query, k, v, None, qkv_format="thd",
            attn_mask_type="padding_causal_bottom_right",
            cu_seqlens_q=lay.cu_seqlens_q, cu_seqlens_kv=lay.cu_seqlens_kv,
            max_seqlen_q=lay.max_seqlen_q, max_seqlen_kv=lay.max_seqlen_kv)

    cls.forward = forward
    cls._coda_tree_patched = True
