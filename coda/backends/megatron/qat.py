"""QAT against a rollout that serves the checkpoint's own block-scaled FP8 / MXFP4 format.

The quantization format of every tensor is read from the training checkpoint itself
(``trainer.hf_model_path``; the rollout boots from the same directory), see
:func:`coda.utils.quant_formats.read_ckpt_quant`. Megatron params are matched to checkpoint tensors
through the bridge's conversion tasks, so nothing here is model specific; per-model rollout
numerics that cannot be read from the checkpoint live in a small :class:`QATAdapter`.

* Export (always, independent of QAT): :meth:`QuantPlan.encode` turns the bridge's bf16 export
  back into the checkpoint format (weight + ``.scale``) for every tensor the checkpoint stores
  quantized. A bf16 checkpoint yields an empty plan and plain bf16 export.
* Weight QAT -- "snap in place": after every optimizer step (and after any weight load) the bf16
  params of the quantized tensors are overwritten with ``dequant(quant(W))``. The fp32 master
  keeps accumulating un-quantized updates and the forward sees Q(W): an exact straight-through
  estimator with no forward-time cost and no interference with TE's fused wgrad accumulation
  (param objects are never swapped). Snapped weights are a fixed point of the quantizer, so the
  rollout receives exactly the training weights.
* Activation QAT: forward pre-hooks fake-quantize the inputs of the linears the rollout runs as
  W8A8 (dynamic per-token, group = scale tile width, ue8m0) with SGLang's own kernel;
  straight-through backward. Temporary escape hatch: ``CODA_QAT_DISABLE_ACT=1`` (trainer env)
  keeps weight QAT only, since activation QAT currently raises k3 on DeepSeek-V4.1.

Snapping a local (TP/EP) shard equals snapping the full checkpoint tensor only when the shard is a
contiguous, tile-aligned slice of it; :func:`_local_layout_ok` checks this per param and raises
otherwise (e.g. head-interleaved QKV), which then needs an adapter.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Iterator, Sequence

import torch
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    ReplicatedMapping,
    RowParallelMapping,
)

from coda.utils.quant_formats import CkptQuant, QuantScheme, read_ckpt_quant

logger = logging.getLogger(__name__)

# Temporary: set to "1" in trainer.env_vars to keep weight QAT but skip the activation fake-quant.
DISABLE_ACT_ENV = "CODA_QAT_DISABLE_ACT"

# Mappings that place a contiguous dim-0 / dim-1 slice (or the whole tensor, or a dim-0 concat of
# such slices for GatedMLP) of each HF tensor into the local param.
_SLICE_MAPPINGS = (AutoMapping, ColumnParallelMapping, RowParallelMapping, ReplicatedMapping, GatedMLPMapping)
_NON_SLICE_MAPPINGS = (FusedExpertMapping, FusedGatedExpertMapping)


# ---------------------------------------------------------------- adapters

class QATAdapter:
    """Rollout numerics that are not recorded in the checkpoint. Default: dense FP8 linears run
    W8A8 with dynamic activation quant, MXFP4 experts run W4A16."""

    def act_quant(self, hf_name: str, scheme: QuantScheme) -> bool:
        """Whether the rollout quantizes the input of the GEMM that uses ``hf_name``."""
        return scheme.kind == "fp8_block"

    def validate(self, config) -> None:
        """Reject rollout settings whose numerics this QAT does not model."""


class DeepseekV41Adapter(QATAdapter):
    """SGLang DeepSeek-V4.1: wo_a is dequantized to bf16 at load (bf16 activations); routed MXFP4
    experts must run W4A16; shared experts stay dense FP8 linears."""

    def act_quant(self, hf_name: str, scheme: QuantScheme) -> bool:
        return super().act_quant(hf_name, scheme) and not hf_name.endswith(".attn.wo_a.weight")

    def validate(self, config) -> None:
        rollout = config.rollout
        if rollout.backend != "sglang":
            raise NotImplementedError(f"DeepSeek-V4.1 QAT is only verified for the SGLang rollout, got {rollout.backend}")
        args, env = rollout.sglang_args, rollout.get("env_vars") or {}
        problems = []
        if args.get("flashinfer_mxfp4_moe_precision") != "bf16":
            problems.append("sglang_args.flashinfer_mxfp4_moe_precision must be bf16 (W4A16 experts; W4A8 is not modeled)")
        if str(env.get("SGLANG_OPT_FP8_WO_A_GEMM")) != "0":
            problems.append("env SGLANG_OPT_FP8_WO_A_GEMM must be 0 (FP8 wo_a GEMM quantizes its input; not modeled)")
        if not args.get("disable_shared_experts_fusion"):
            problems.append("sglang_args.disable_shared_experts_fusion must be true (shared experts as dense FP8 linears)")
        if problems:
            raise ValueError("DeepSeek-V4.1 QAT: " + "; ".join(problems))


_ADAPTERS: dict[str, QATAdapter] = {"DeepseekV41ForCausalLM": DeepseekV41Adapter()}


def get_adapter(architectures: Sequence[str] | None) -> QATAdapter:
    for arch in architectures or ():
        if arch in _ADAPTERS:
            return _ADAPTERS[arch]
    return QATAdapter()


# ---------------------------------------------------------------- activation fake-quant

class _FakeQuantActFP8(torch.autograd.Function):
    """SGLang per-token group ue8m0 FP8 quant -> dequant; straight-through backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: int) -> torch.Tensor:
        from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        if x2.shape[0] == 0:
            return x.clone()
        q, s = sglang_per_token_group_quant_fp8(x2, group, scale_ue8m0=True)
        out = q.float().view(x2.shape[0], -1, group) * s.view(x2.shape[0], -1, 1)
        return out.view(shape).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None


def _make_act_pre_hook(group: int):
    def hook(module, args):
        if not args or not torch.is_tensor(args[0]):
            return None
        return (_FakeQuantActFP8.apply(args[0], group), *args[1:])

    return hook


# ---------------------------------------------------------------- plan

def hf_names(mapping) -> list[str]:
    hf = mapping.hf_param
    return [hf] if isinstance(hf, str) else list(hf.values())


def _local_layout_ok(mapping, n_parts: int, local: torch.Size, full: tuple[int, int], block) -> bool:
    """True iff the local param is ``n_parts`` dim-0-stacked, tile-aligned contiguous slices of
    ``full``-shaped checkpoint tensors, so snapping it equals snapping the full tensors."""
    if isinstance(mapping, _NON_SLICE_MAPPINGS) or not isinstance(mapping, _SLICE_MAPPINGS) or len(local) != 2:
        return False
    (o, i), (O, I), (br, bc) = local, full, block
    if o % n_parts:
        return False
    o //= n_parts
    if (o, i) == (O, I):
        return True
    if i == I and O % o == 0:  # dim-0 shard
        return o % br == 0
    if o == O and I % i == 0:  # dim-1 shard
        return i % bc == 0
    return False


class QuantPlan:
    """Quantized-tensor bookkeeping for one Megatron model (see module docstring).

    ``quantized`` -- the checkpoint stores some trained tensors quantized, i.e. the rollout serves
    that format and :meth:`encode` must be applied to every weight sync.
    """

    def __init__(self, bridge, model_chunks: Sequence[torch.nn.Module], hf_model_path: str, config):
        self.qat = bool(config.trainer.qat)
        self.ckpt: dict[str, CkptQuant] = read_ckpt_quant(hf_model_path) if os.path.isdir(hf_model_path) else {}
        self.snap_params: list[tuple[str, torch.nn.Parameter, QuantScheme]] = []
        self.act_modules: list[tuple[str, torch.nn.Module, int]] = []
        self.quantized = False
        if not self.ckpt:
            if self.qat:
                raise ValueError(f"trainer.qat needs a quantized checkpoint, {hf_model_path} has none")
            return
        adapter = get_adapter(getattr(bridge.hf_pretrained.config, "architectures", None))
        if self.qat:
            adapter.validate(config)
        own = {id(p) for chunk in model_chunks for p in chunk.parameters()}
        hooked: set[int] = set()
        for task in bridge.get_conversion_tasks(list(model_chunks)):
            if task is None:
                continue
            names = hf_names(task.mapping)
            entries = [self.ckpt.get(n) for n in names]
            if not any(entries):
                continue
            self.quantized = True
            if task.param_weight is None or not self.qat:  # other PP stage / export-only
                continue
            schemes = {e.scheme if e else None for e in entries}
            shapes = {e.shape if e else None for e in entries}
            if len(schemes) != 1 or len(shapes) != 1:
                raise NotImplementedError(f"[QAT] {task.param_name}: mixed formats {dict(zip(names, entries))}")
            scheme, full = entries[0].scheme, entries[0].shape
            param = task.param_weight
            if id(param) not in own:
                raise RuntimeError(f"[QAT] {task.param_name}: bridge task does not hold the live model param")
            if not _local_layout_ok(task.mapping, len(names), param.shape, full, scheme.block):
                raise NotImplementedError(
                    f"[QAT] {task.param_name} ({type(task.mapping).__name__}, local {tuple(param.shape)}) is not a "
                    f"tile-aligned slice of {names} {full} with {scheme}; add a QATAdapter for this model"
                )
            self.snap_params.append((task.param_name, param, scheme))
            module = task.megatron_module
            is_module_weight = task.param_name.rsplit(".", 1)[-1] == "weight"
            if is_module_weight and adapter.act_quant(names[0], scheme) and id(module) not in hooked:
                if hasattr(module, "layer_norm_weight"):
                    raise NotImplementedError(f"[QAT] {task.param_name}: input of a fused norm+linear is pre-norm")
                hooked.add(id(module))
                self.act_modules.append((task.param_name[: -len(".weight")], module, scheme.block[1]))
        if self.qat and not self.quantized:
            raise ValueError(f"trainer.qat: no trained tensor of {hf_model_path} is stored quantized")

    # -- weight QAT
    @torch.no_grad()
    def snap_(self) -> int:
        """Snap every quantized param in place onto its checkpoint grid. Returns #params."""
        for _, p, scheme in self.snap_params:
            scheme.snap_(p.data)
        return len(self.snap_params)

    @torch.no_grad()
    def count_off_grid(self) -> int:
        """Debug check: number of quantized params that are NOT on their grid."""
        bad = 0
        for name, p, scheme in self.snap_params:
            if not torch.equal(scheme.snap_(p.data.clone()), p.data):
                bad += 1
                logger.warning(f"[QAT] param off-grid: {name}")
        return bad

    # -- activation QAT
    def register_activation_hooks(self) -> int:
        """Hook the activation fake-quant. Returns #hooks; 0 when disabled via ``DISABLE_ACT_ENV``."""
        if os.environ.get(DISABLE_ACT_ENV) == "1":
            logger.info(f"[QAT] activation QAT disabled by {DISABLE_ACT_ENV}=1, weight QAT only")
            return 0
        for _, module, group in self.act_modules:
            module.register_forward_pre_hook(_make_act_pre_hook(group))
        return len(self.act_modules)

    # -- export
    def encode(self, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        """bf16 HF tensors -> checkpoint format: quantized weights are followed by their scale."""
        for name, t in named_tensors:
            entry = self.ckpt.get(name)
            if entry is None:
                yield name, t
                continue
            q, s = entry.scheme.quant(t)
            yield name, q
            yield entry.scale_name, s
