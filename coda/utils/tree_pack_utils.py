"""Tree training settings shared by the controller and the Megatron backend.

Kept free of Megatron imports so the controller can use it.
"""

from __future__ import annotations

from omegaconf import DictConfig

# Default cost of a tree sequence in node-equivalents (empirical; tune per model and hardware).
TREE_ATTN_PAIR_COST = 9.15e-6   # one (query, key) pair of causal full attention
TREE_LEVEL_COST = 4200.0        # one serial GDN level (a round of varlen kernel calls)


def tree_pack_unsupported_reason(config: DictConfig) -> str | None:
    """Why ``trainer.tree_pack`` cannot run with this config (None if it can).

    The model check (mRoPE) needs the built model and is done by the train worker.
    """
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
    return None
