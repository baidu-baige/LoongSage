"""LM-head row selection: the output layer computes logits only for the loss rows."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import torch

# forward pre-hook on the output layer
_LM_ROWS: ContextVar = ContextVar("tree_pack_lm_rows", default=None)


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
    if getattr(head, "_tree_pack_lm_rows_hooked", False):
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
            module._tree_pack_restore_sp = True
        hidden = hidden.index_select(0, rows)
        return (hidden, *args), kwargs

    def post_hook(module, args, output):
        if getattr(module, "_tree_pack_restore_sp", False):
            module.sequence_parallel = True
            module._tree_pack_restore_sp = False

    # prepend: run before other pre-hooks (e.g. the fp32 LM-head cast)
    head.register_forward_pre_hook(pre_hook, prepend=True, with_kwargs=True)
    head.register_forward_hook(post_hook, always_call=True)   # restore SP on errors too
    head._tree_pack_lm_rows_hooked = True
    return True

