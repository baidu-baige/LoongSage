"""
Value head for the PPO critic: replaces the GPT model's ``output_layer``.
"""

from __future__ import annotations

import torch
from megatron.core import tensor_parallel

# Adapt from https://github.com/THUDM/slime/blob/18868a79b0b083a093798a94ed0f8cb4f71b9c14/slime/backends/megatron_utils/model_provider.py#L67
class LinearValueHead(torch.nn.Linear):
    """``hidden_size -> 1`` value projection with the ``output_layer`` call signature.

    Args:
        hidden_size: Model hidden size.
        sequence_parallel: Whether the incoming hidden states are split over
            the sequence dimension (TP > 1 in coda). The parameters are then
            tagged ``sequence_parallel`` so ``finalize_model_grads`` sums their
            per-TP-rank gradients.
        init_std: Std of the normal weight init; the bias starts at 0.
        seed: Seed of a dedicated generator for the weight init, so every rank
            holding a replica starts from the same head (DDP does not
            broadcast parameters).
    """

    def __init__(self, hidden_size: int, *, sequence_parallel: bool, init_std: float, seed: int):
        super().__init__(hidden_size, 1, bias=True)
        self.sequence_parallel = sequence_parallel
        if sequence_parallel:
            self.weight.sequence_parallel = True
            self.bias.sequence_parallel = True
        with torch.no_grad():
            generator = torch.Generator(device=self.weight.device).manual_seed(seed)
            self.weight.normal_(mean=0.0, std=init_std, generator=generator)
            self.bias.zero_()

    def forward(self, input_, weight=None, runtime_gather_output=None):
        """Called as ``output_layer(hidden, weight=..., runtime_gather_output=...)``.

        ``weight`` (the shared embedding, if any) is ignored. Returns
        ``(values, None)`` with values of shape ``[s, b, 1]`` in fp32.
        """
        values = super().forward(input_).float()
        if self.sequence_parallel:
            values = tensor_parallel.gather_from_sequence_parallel_region(
                values, tensor_parallel_output_grad=False
            )
        return values, None
