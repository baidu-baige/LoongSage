"""Monkey-patches for Megatron-Core.

All patches are applied at module import time. Import this module before
any Megatron objects that depend on these patches are instantiated.
"""

import importlib
import sys
from contextlib import contextmanager

import torch
from megatron.core import tensor_parallel
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer


def _patch_distributed_optimizer_init():
    """Fix stale model_param_group_index_map when mixing fp32/bf16 params.

    Root cause (Megatron-LM issue #2777):
      _build_optimizer_group_ranges builds model_param_group_index_map in gbuf_ranges
      traversal order (bf16 params first, then fp32 params).  But
      _build_model_and_main_param_groups sets orig_group["params"] as
      [*shard_fp32_params, *shard_fp32_from_float16_params], putting fp32 shards first.
      When all params share the same dtype the two orders are identical, but mixing
      fp32 params (e.g. via KeepFP32Module) causes the map indices to diverge, which
      makes checkpoint save read the wrong optimizer state and hit an assertion error.

    Fix: After the original __init__ completes, rebuild model_param_group_index_map
    so that the ordering matches the actual optimizer.param_groups layout
    (fp32 first, then fp16/bf16).

    Ref: https://github.com/ahmadki/Megatron-LM/commit/3f593b76425eef46cc844f5cdffa5faa5edfa845
    """
    _original_init = DistributedOptimizer.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)

        for group_index, group_range in enumerate(self.opt_group_ranges):
            param_order = 0
            for model_param in group_range["params"]:
                if model_param.type() == 'torch.cuda.FloatTensor':
                    self.model_param_group_index_map[model_param] = (group_index, param_order)
                    param_order += 1
            for model_param in group_range["params"]:
                if model_param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                    self.model_param_group_index_map[model_param] = (group_index, param_order)
                    param_order += 1

    DistributedOptimizer.__init__ = _patched_init


def _patch_fp32_shard_leaf():
    """Fix "can't optimize a non-leaf Tensor" when fp32 params meet cpu offload.

    Root cause:
      ``_build_model_and_main_param_groups`` builds the optimizer's param shard
      for fp32 model params as ``model_param.view(-1)[start:end]`` *without*
      ``.detach()`` (unlike the bf16 branch, which detaches first).  Since the
      fp32 model param has ``requires_grad=True``, this sliced view is a
      non-leaf tensor.  Normally this is harmless: ``DistributedOptimizer``
      assigns ``optimizer.param_groups`` directly and never validates leaf-ness.

      But with ``optimizer_cpu_offload``, the inner optimizer is rebuilt as a
      ``HybridDeviceOptimizer`` via ``add_param_group`` -> torch's leaf check
      (``optimizer.py``: "can't optimize a non-leaf Tensor"), which rejects the
      non-leaf fp32 shard.  This surfaces whenever fp32 and bf16 params are
      mixed, e.g. ``use_fp32_lm_head`` keeping ``output_layer`` in fp32.

    Fix: after the original method builds the groups, replace each non-leaf
      fp32 shard with ``shard.detach()`` (a leaf, requires_grad=False, sharing
      the same storage so in-place optimizer updates still write back to the
      model param).  The same tensor object lives in both ``shard_fp32_groups``
      and the inner optimizer's ``orig_group["params"]``, so both references are
      swapped in lock-step to keep downstream grad/param copies aligned.  This
      mirrors the bf16 master shard, which is already a requires_grad=False leaf.
    """
    _original_build = DistributedOptimizer._build_model_and_main_param_groups.__func__

    def _patched_build(cls, gbuf_ranges, param_gbuf_map, opt_group_ranges, config):
        groups = _original_build(cls, gbuf_ranges, param_gbuf_map, opt_group_ranges, config)
        shard_fp32_groups = groups[3]

        for group_range, shard_group in zip(opt_group_ranges, shard_fp32_groups):
            orig_params = group_range["orig_group"]["params"]
            for i, shard in enumerate(shard_group):
                if shard.is_leaf:
                    continue
                leaf = shard.detach()
                tensor_parallel.copy_tensor_model_parallel_attributes(leaf, shard)
                if hasattr(shard, "shared"):
                    leaf.shared = shard.shared
                shard_group[i] = leaf
                for j, p in enumerate(orig_params):
                    if p is shard:
                        orig_params[j] = leaf
                        break

        return groups

    DistributedOptimizer._build_model_and_main_param_groups = classmethod(_patched_build)


def _patch_hdo_update_fp32_params():
    """Fix KeyError in HybridDeviceOptimizer when a model param is already fp32.

    Root cause:
      ``HybridDeviceOptimizer._update_fp32_params_by_new_state`` iterates over every
      param in ``self.state`` and does a *direct* lookup
      ``fp32_param = self.param_to_fp32_param[param]``.  But ``param_to_fp32_param``
      is only populated for params whose dtype is NOT fp32 (see
      ``_get_sub_optimizer_param_groups``: the fp32-master clone is created only under
      ``param.dtype != torch.float32``).  A param that is *already* fp32 — e.g. the
      ``output_layer.weight`` kept in fp32 by ``KeepFP32Module`` when
      ``use_fp32_lm_head`` is set — has no entry, so this raises ``KeyError``.

      This fires on checkpoint load: ``DistributedOptimizer.sharded_state_dict(
      is_loading=True)`` -> ``load_state_dict(state_dict())`` -> torch post-load hook
      -> ``_sync_hdo_state_to_sub_optimizers`` -> ``_update_fp32_params_by_new_state``.
      It only hits the last pipeline stage (the stage that owns ``output_layer``),
      so those ranks raise while the other ranks proceed into the next collective,
      deadlocking the whole job (a Ray actor-method exception is swallowed into the
      ObjectRef and never surfaced, so it looks like a silent hang).

    Fix: use ``self.param_to_fp32_param.get(param, param)`` — for an already-fp32
      param the master copy IS the param itself, so we copy the new master state
      back into it.  This mirrors the ``.get(param, param)`` pattern used everywhere
      else in ``hybrid_optimizer.py`` (pre/post_load_state_dict_hook,
      _move_new_state_to_right_device); only this one call site used direct indexing.
    """
    def _patched_update_fp32_params_by_new_state(self):
        if not self.param_update_in_fp32:
            return
        for param, v in self.state.items():
            fp32_param = self.param_to_fp32_param.get(param, param)
            fp32_param.data.copy_(v["master_param"])

    HybridDeviceOptimizer._update_fp32_params_by_new_state = (
        _patched_update_fp32_params_by_new_state
    )


def _patch_dsa_cudnn_default_stream():
    """Fix cuDNN DSA silently dropping half the rows of its fp32 score matrix.

    Symptom:
      ``cudnn.DSA.indexer_forward_wrapper`` / ``dense_indexer_score_recompute_wrapper``
      / ``dense_attn_score_recompute_wrapper`` return score matrices where whole
      Q rows are still ``-inf``.  No error is raised.  The first call after
      ``cute.compile`` is always correct; the drop starts at the second call, the
      lost fraction grows with the buffer size, and the counts jitter between runs.
      Downstream this shows up as inflated ``approx_kl`` / ``is_approx_k3_kl`` and
      nonzero ``nan_ratio``.

    Root cause:
      ``cudnn/deepseek_sparse_attention/utils/runtime.py::resolve_stream(None)``
      returns ``CUstream(0)``, because 0 is the handle PyTorch reports for its
      default stream.  The CuTe kernel is then launched on raw handle 0 (the
      legacy default stream), but every torch-side helper in the interface runs
      inside ``torch_stream_context(CUstream(0))`` ==
      ``torch.cuda.stream(torch.cuda.ExternalStream(0))``.

      ``ExternalStream(0)`` does NOT round-trip to the default stream: PyTorch
      treats ``stream_ptr=0`` as "no external stream given" and hands back a
      stream from its pool -- a *different* one on every call.  So the interface's
      ``out.fill_(float("-inf"))`` prefill runs concurrently with the kernel that
      is supposed to fill ``out``, and the prefill's tail wipes out part of the
      kernel's writes.

      Corroboration: the backward wrappers pass ``backend_stream = None`` on SM90
      (``indexer_backward/api.py``), which short-circuits the context manager --
      and those are exactly the wrappers that do not exhibit the failure.

    Fix: make ``torch_stream_context`` a no-op when the handle is 0, so the
      prefill stays on the caller's default stream and is correctly ordered
      before the kernel.  One rebind of ``utils.runtime.torch_stream_context`` is
      enough as long as it happens before the first DSA call: the interface
      modules that use it are imported lazily inside ``execute()`` and pick up
      the patched function by name.  The ``sys.modules`` sweep is only a safety
      net for the case where this module is imported late.

    Without the patch, indexer_forward and dense indexer score calls corrupt their
    output, as do THD packed multi-segment calls. The patch is also slightly
    faster, because the unpatched path allocates a fresh pool stream on every call.
    """
    try:
        from cudnn.deepseek_sparse_attention.utils import runtime as dsa_runtime
    except ImportError:
        # cuDNN DSA kernels are only present for DeepSeek-V4 style models.
        return

    original = dsa_runtime.torch_stream_context
    if getattr(original, "_coda_default_stream_safe", False):
        return

    @contextmanager
    def _safe_torch_stream_context(current_stream=None):
        if current_stream is None or int(current_stream) == 0:
            # handle 0 == torch's default stream; wrapping it in an
            # ExternalStream would silently switch to a fresh pool stream.
            yield
            return
        with torch.cuda.stream(torch.cuda.ExternalStream(int(current_stream))):
            yield

    _safe_torch_stream_context._coda_default_stream_safe = True
    dsa_runtime.torch_stream_context = _safe_torch_stream_context

    for name, module in list(sys.modules.items()):
        if not name.startswith("cudnn.deepseek_sparse_attention") or module is None:
            continue
        for attr in ("torch_stream_context", "_torch_stream_context"):
            if getattr(module, attr, None) is original:
                setattr(module, attr, _safe_torch_stream_context)


def _patch_csa_cute_launch_cache():
    """Stop the CSA CuTe launch cache from growing once per microbatch shape.

    Symptom:
      Unbounded per-step host growth with `use_dynamic_batch_size` +
      `apply_dsa_kernel_fusion`. `_COMPILED_LAUNCH_CACHE` gains tens of entries
      per step and keeps accelerating, along with a matching climb in live
      `cutlass.base_dsl.dsl` function objects and in ~1 MiB anonymous mappings
      (which also walks towards `vm.max_map_count`). Each entry pins a compiled
      CuTe artifact plus its loaded module, so nothing is ever returned to the OS.

    Root cause:
      `csa_cp_layout_kernels._run_compiled_launch` keys the cache on the *full*
      shape and stride of every tensor argument::

          key = (launch_fn.__name__,
                 tuple((t.dtype, tuple(t.shape), tuple(t.stride())) for t in tensor_args),
                 tuple((i, scalar_args[i]) for i in static_arg_indices))

      but the artifact it caches is not specialised on the extents:

      1. every tensor goes through `mark_layout_dynamic(leading_dim=ndim-1)`,
         which per CuTe's own docs "marks the layout as dynamic while setting the
         stride at leading_dim to 1" — extents stay dynamic;
      2. the data-dependent scalars are not in the key at all. For
         `_compressor_input_compact_fwd_launch`, `static_arg_indices=(3, 4, 5, 7)`
         selects only `ratio/d_comp/d_window/row_width` (config constants);
         `n_seq/global_start/l_local/compact_len` are passed as runtime
         `cutlass.Int32`;
      3. the last dim's extent is already pinned via the static `row_width`, so
         the one shape component that varies in the key — the leading dim
         (17992/18016/18048...) — is exactly the dynamic one.

      So the key is strictly finer than the specialisation, and every new token
      count recompiles a kernel that was already usable.

    Fix: swap the module-level dict for one that normalises the key to
      `(dtype, ndim, contiguity_pattern)` per tensor. Entries already compiled are
      re-inserted through the new mapping so nothing is lost or double-compiled.
      Keys then collapse to kernels x dtype/layout x static scalars, which is
      bounded and saturates within a step or two.

    Effect: driving `CompressorInputCompact.apply` over a sequence of varying
    `l_local` values compiles once instead of once per shape, with every output
    bitwise-identical to the per-shape-compiled baseline. Each avoided
    `cute.compile` is a few tenths of a second of CPU.
    """
    module = None
    for path in (
        "megatron.core.transformer.experimental_attention_variant.csa_cp_layout_kernels",
        # Same code, moved after an upstream CSA restructuring.
        "megatron.core.transformer.experimental_attention_variant.csa_utils.cp_layout_kernels",
    ):
        try:
            module = importlib.import_module(path)
            break
        except Exception:  # pylint: disable=broad-except
            # Absent (non-DSv4 build) or unimportable here; a patch must never
            # be the reason startup fails.
            continue
    if module is None:
        return

    cache = getattr(module, "_COMPILED_LAUNCH_CACHE", None)
    if not isinstance(cache, dict) or isinstance(cache, _ShapeAgnosticLaunchCache):
        return

    replacement = _ShapeAgnosticLaunchCache()
    for key, compiled in cache.items():
        replacement[key] = compiled
    module._COMPILED_LAUNCH_CACHE = replacement


class _ShapeAgnosticLaunchCache(dict):
    """CSA CuTe launch cache keyed by what the compilation actually specialises on.

    ``_run_compiled_launch`` reads and writes through ``.get(key)`` / ``[key] = v``
    only, so intercepting those two is enough to re-key the cache without copying
    any of the dispatch logic. ``__getitem__`` / ``__contains__`` are overridden
    too so the class stays a correct mapping if upstream starts using them.

    The normalisation drops the tensor **extents** and keeps the contiguity
    *pattern*, so genuinely different layouts remain separate:
    ``(17992,4096) s(4096,1)`` and ``(18048,4096) s(4096,1)`` both map to
    ``(True, True)`` and share an entry, while a padded ``s(4104,1)`` gives
    ``(True, False)`` and a transposed ``s(1,4096)`` gives ``(False, False)``.
    """

    @staticmethod
    def _layout_class(shape, stride):
        """Extent-free description of what ``mark_layout_dynamic`` keeps static."""
        expected = 1
        pattern = []
        for size, step in zip(reversed(shape), reversed(stride)):
            pattern.append(step == expected)
            expected *= size
        return tuple(pattern)

    @classmethod
    def _normalize(cls, key):
        try:
            launch_name, tensor_specs, static_scalars = key
            return (
                launch_name,
                tuple(
                    (dtype, len(shape), cls._layout_class(shape, stride))
                    for dtype, shape, stride in tensor_specs
                ),
                static_scalars,
            )
        except (TypeError, ValueError):
            # Upstream changed the key layout: fall back to the raw key so the
            # worst case is today's behaviour, never a broken kernel dispatch.
            return key

    def get(self, key, default=None):
        return super().get(self._normalize(key), default)

    def __getitem__(self, key):
        return super().__getitem__(self._normalize(key))

    def __setitem__(self, key, value):
        super().__setitem__(self._normalize(key), value)

    def __contains__(self, key):
        return super().__contains__(self._normalize(key))


_patch_distributed_optimizer_init()
_patch_fp32_shard_leaf()
_patch_hdo_update_fp32_params()
_patch_dsa_cudnn_default_stream()
_patch_csa_cute_launch_cache()

