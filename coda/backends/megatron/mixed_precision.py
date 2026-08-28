"""Mixed precision utilities for the Megatron backend."""

import logging

import torch
from megatron.core.transformer.module import Float16Module, float16_to_fp32, fp32_to_float16
from megatron.core.transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)


class KeepFP32Module(Float16Module):
    """
    A module wrapper that keeps specified parameters in FP32 during mixed-precision training.

    Extends Megatron's Float16Module to selectively preserve FP32 precision for
    designated parameters. For each matched parameter, its weight is cast to FP32
    and forward hooks are registered on the owning submodule:
      - pre_hook:  casts submodule inputs from fp16/bf16 to fp32 so the forward
                   computation runs in full precision.
      - post_hook: casts submodule outputs back to fp16/bf16 (can be skipped to
                   keep fp32 output).

    Typical use case: keeping precision-sensitive layers (e.g. lm_head) in FP32
    during fp16/bf16 training to improve numerical stability.

    Args:
        keep_fp32_weights: A mapping of parameter names to keep in FP32.
            Keys are substrings matched against parameter names (fuzzy match).
            Values indicate whether the layer's output should also remain FP32.
            Example: {"output_layer": True} keeps both weights and outputs in FP32
            for any parameter whose name contains "output_layer".
        config: Megatron TransformerConfig containing mixed-precision settings.
        module: The original model module to be wrapped.
    """

    def __init__(self, keep_fp32_weights: dict[str, bool] | None, config: TransformerConfig, module: torch.nn.Module):
        super().__init__(config, module)
        if not keep_fp32_weights:
            return

        # Collect the set of submodules whose parameters need to stay in fp32.
        # key = submodule, value = whether the submodule should also keep fp32 output
        fp32_submodules: dict[torch.nn.Module, bool] = {}

        for name, param in module.named_parameters():
            for keep_fp32_name, keep_fp32_output in keep_fp32_weights.items():
                if keep_fp32_name in name:
                    origin_dtype = param.dtype
                    param.data = param.data.to(dtype=torch.float32)
                    # Find the direct parent module that owns this parameter
                    owner = self._find_param_owner(module, name)
                    if owner is None:
                        raise ValueError(f"can not find owner for parameter: {name}, keep_fp32_name: {keep_fp32_name}")
                    owner_name = type(owner).__name__
                    if  owner in fp32_submodules and fp32_submodules[owner] != keep_fp32_output:
                        raise ValueError(f"conflict keep fp32 setting on {owner_name}, "
                                         f"name: {name}, keep_fp32_name: {keep_fp32_name}, "
                                         f"previous: {fp32_submodules[owner]}, new: {keep_fp32_output}")
                    fp32_submodules[owner] = keep_fp32_output
                    logger.info(f"keep fp32 weight param precision for {name}, origin dtype: {origin_dtype}, "
                                f"owner: {owner_name}, keep_fp32_output: {keep_fp32_output}")
                    break

        # Register hooks on each submodule that owns fp32 weights
        for submodule, keep_fp32_output in fp32_submodules.items():
            # pre-hook: cast inputs to fp32 so the computation runs in fp32
            submodule.register_forward_pre_hook(self._make_pre_hook())
            # post-hook: cast outputs back to low precision unless fp32 output is requested
            if not keep_fp32_output:
                submodule.register_forward_hook(self._make_post_hook())

    @staticmethod
    def _find_param_owner(root: torch.nn.Module, param_name: str) -> torch.nn.Module | None:
        """Given a dotted parameter name (e.g. 'decoder.output_layer.weight'),
        return the direct parent module that owns the parameter."""
        parts = param_name.split(".")
        # The last part is the actual parameter attr, everything before is the module path
        module = root
        for part in parts[:-1]:
            module = getattr(module, part, None)
            if module is None:
                return None
        return module

    def _make_pre_hook(self):
        """Return a forward_pre_hook that casts all float inputs to fp32."""
        def hook(_module, input):
            return float16_to_fp32(input)
        return hook

    def _make_post_hook(self):
        """Return a forward_hook that casts outputs back to low precision (fp16/bf16)."""
        def hook(_module, _input, output):
            return fp32_to_float16(output, self.float16_convertor)
        return hook
