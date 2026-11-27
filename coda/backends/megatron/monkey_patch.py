"""Monkey-patches for Megatron-Core and its kernel dependencies.

All patches are applied at module import time. Import this module before
any Megatron objects that depend on these patches are instantiated.
"""

import importlib

import torch
from megatron.core import tensor_parallel
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer


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


def _patch_fused_adam_fp32_master_remainder():
    """Fix int16-vs-fp32 ``master_param`` mismatch when store_param_remainders meets fp32 params.

    Root cause:
      TE ``FusedAdam.initialize_state(param, store_param_remainders)`` forwards the
      raw ``store_param_remainders`` flag straight to the ``master_param`` buffer
      allocation, so it creates an **int16** remainder buffer for *every* param when
      the flag is on -- regardless of the param's dtype.  But the state accessors
      ``get_unscaled_state`` / ``set_scaled_state`` only treat ``master_param`` as an
      int16 remainder when ``param.dtype == torch.bfloat16``; for any other dtype they
      assert the state is fp32 (``fused_adam.py``: ``assert ... == torch.float32``).

      The two paths disagree for fp32 params.  ``chunked_optimizer_state_offload``
      (and ``optimizer_cpu_offload``) force ``use_precision_aware_optimizer`` ->
      ``store_param_remainders``, while ``use_fp32_lm_head`` keeps ``output_layer`` in
      fp32 (via ``keep_fp32_weights``).  That fp32 param then gets an int16
      ``master_param`` at init.

      It only bites on **resume**: ``optimizer.sharded_state_dict(is_loading=True)``
      (checkpoint.py) runs Megatron's ``init_state_fn`` ->
      ``FusedAdam.initialize_state`` to build the load template, then ``state_dict()``
      -> ``get_unscaled_state`` asserts the fp32 param's ``master_param`` is fp32 and
      crashes.  Saving never calls ``initialize_state`` (the state is already trained
      fp32), which is why the assertion fires only when loading a checkpoint.

    Fix: gate ``store_param_remainders`` on ``param.dtype == torch.bfloat16`` inside
      ``initialize_state`` -- exactly the condition the accessors use -- so fp32 params
      get a real fp32 ``master_param`` while bf16 params keep the int16 remainder trick.
    """
    try:
        from transformer_engine.pytorch.optimizers.fused_adam import FusedAdam
    except ImportError:
        # TE not available (non-Megatron build); nothing to patch.
        return

    _original_initialize_state = FusedAdam.initialize_state
    if getattr(_original_initialize_state, "_coda_fp32_remainder_safe", False):
        return

    def _patched_initialize_state(self, param, store_param_remainders):
        # Only bf16 params use the int16 remainder representation; matching the
        # gating in get_unscaled_state / set_scaled_state keeps init consistent
        # with (de)serialization so fp32 params get an fp32 master_param.
        effective = store_param_remainders and param.dtype == torch.bfloat16
        return _original_initialize_state(self, param, effective)

    _patched_initialize_state._coda_fp32_remainder_safe = True
    FusedAdam.initialize_state = _patched_initialize_state


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


def _patch_fa4_causal_unbounded_window():
    """Stop FA4 from resolving Transformer Engine's causal window to an empty local window.

    Symptom (B200, Qwen3.8-27B, head_dim 256, flash-attn-4 >= 4.0.0b30): the first
    training backward asserts "SM100 backward with head_dim=256 does not support
    local attention". The forward before it already returns all zeros.

    Root cause:
      TE encodes causal as ``window_size=(-1, 0)`` (-1 = no left limit). FA4's
      ``_resolve_causal_local_window`` only treats -1 as unbounded when *both*
      sides are negative (since Dao-AILab/flash-attention#2490), so ``(-1, 0)``
      becomes local attention with a left window of -1. Token i may then attend
      only to keys in ``[i + 1, i]``, which is empty, so every output row is 0.
      Builds without the dedicated hd256 backward kernel (b39+) do not assert and
      train on the zeros silently. Tracked upstream in
      Dao-AILab/flash-attention#3084.

    Fix: under ``causal=True``, map a negative left window to ``None`` before FA4
    resolves it. This matches flash-attn-4 <= 4.0.0b29, which also resolved a
    causal ``(-1, 0)`` to plain causal. Sliding windows ``(>= 0, 0)`` and
    non-causal windows are passed through unchanged.
    """
    try:
        module = importlib.import_module("flash_attn.cute.interface")
    except Exception:  # pylint: disable=broad-except
        # FA4 not installed; a patch must never be the reason startup fails.
        return
    original = getattr(module, "_resolve_causal_local_window", None)
    if original is None or getattr(original, "_coda_causal_unbounded_window", False):
        return

    def _patched_resolve(causal, window_size_left, window_size_right, mask_mod=None):
        if causal and window_size_left is not None and window_size_left < 0:
            window_size_left = None
        return original(causal, window_size_left, window_size_right, mask_mod)

    _patched_resolve._coda_causal_unbounded_window = True
    module._resolve_causal_local_window = _patched_resolve


def _patch_ernie55_drop_mtp():
    """Build ERNIE 5.5 as the decoder only (LoongSage does not train MTP).

    Replaces ``docker/patch/ernie55/megatron-bridge.patch``. Two coordinated edits that
    upstream Megatron-Bridge does not expose as config:

    * ``Ernie55Config`` drops its MTP depths (truncate ``compress_ratios`` to
      ``num_hidden_layers``, zero ``num_nextn_predict_layers`` / ``mtp_shared_last_layer``),
      so ``get_model_config`` builds the decoder only. Applied as an instance fix-up after
      the original ``__init__``, which only stores these fields and derives nothing from
      them, so the post-init overwrite is equivalent to editing the body.
    * ``check_ernie55_checkpoint_keys`` ignores the MTP tensors a released checkpoint still
      stores, which the decoder-only mapping would otherwise reject. The in-module caller
      resolves the name at call time, so reassigning the module attribute suffices.
    """
    from megatron.bridge.models.ernie import configuration_ernie55, ernie55_mappings

    ernie55_config_cls = configuration_ernie55.Ernie55Config
    if not getattr(ernie55_config_cls.__init__, "_coda_drop_mtp", False):
        _original_init = ernie55_config_cls.__init__

        def _patched_init(self, *args, **kwargs):
            _original_init(self, *args, **kwargs)
            if self.num_nextn_predict_layers and self.compress_ratios is not None:
                self.compress_ratios = list(self.compress_ratios)[: self.num_hidden_layers]
            self.num_nextn_predict_layers = 0
            self.mtp_shared_last_layer = False

        _patched_init._coda_drop_mtp = True
        ernie55_config_cls.__init__ = _patched_init

    _original_check = ernie55_mappings.check_ernie55_checkpoint_keys
    if not getattr(_original_check, "_coda_drop_mtp", False):
        _hf_get = ernie55_mappings._hf_get

        def _patched_check(hf_config, hf_keys):
            # The Preview checkpoints store DSpark draft weights under ``model.mtp.`` even with
            # ``num_nextn_predict_layers: 0``; the decoder-only model never reads them.
            hf_keys = {k for k in hf_keys if not k.startswith("model.mtp.")}
            if not int(_hf_get(hf_config, "num_nextn_predict_layers", 0) or 0):
                num_hidden_layers = int(_hf_get(hf_config, "num_hidden_layers", 0))
                hf_keys = {
                    k
                    for k in hf_keys
                    if not (k.startswith("model.layers.") and int(k.split(".")[2]) >= num_hidden_layers)
                }
            return _original_check(hf_config, hf_keys)

        _patched_check._coda_drop_mtp = True
        ernie55_mappings.check_ernie55_checkpoint_keys = _patched_check


def _patch_ernie55_quantized_checkpoint():
    """Load and export an ERNIE 5.5 checkpoint stored in FP8 / MXFP4 (ERNIE-5.5-Flash).

    Upstream ``Ernie55Bridge`` only converts bf16 checkpoints. The quantized release stores every
    dense linear as e4m3 with an fp32 power-of-two ``X_scale_inv`` per 128x128 tile (the last
    tile may be partial: ``kv_a_proj_with_mqa`` has 576 rows), and every routed expert as packed
    e2m1 with an e8m0 scale per 32 inputs. Three edits, all no-ops on a bf16 checkpoint:

    * ``check_ernie55_checkpoint_keys`` ignores the ``*_scale_inv`` companions, which no mapping
      reads (the import hook below consumes them).
    * ``maybe_modify_loaded_hf_weight`` dequantizes to bf16 on import, exactly (an e4m3 / e2m1
      value times a power of two is a bf16 value). An fp32 scale that is not a power of two is
      rejected: the export below could not reproduce it.
    * ``maybe_modify_converted_hf_weight`` re-quantizes the export into the checkpoint's own
      format (weight, then ``X_scale_inv`` in the stored dtype), with the QAT quantizer
      (:mod:`coda.utils.quant_formats`). Snapped weights are its fixed points, so the rollout
      gets exactly the training weights. The bridge's own quantizer is not usable here: for an
      fp32 scale it uses ``amax / 448``, not a power of two. Formats come from the safetensors
      headers, read once; a weight sync reads no checkpoint data.
    """
    import functools

    from megatron.bridge.models.ernie import ernie55_mappings
    from megatron.bridge.models.ernie.ernie55_bridge import Ernie55Bridge

    from coda.utils.hf_ckpt import SAFETENSORS_DTYPE, read_source_layout
    from coda.utils.quant_formats import ckpt_quant_config, e8m0_to_float, infer_scheme, scan_ckpt_quant

    _original_check = ernie55_mappings.check_ernie55_checkpoint_keys
    if not getattr(_original_check, "_coda_quantized", False):

        def _patched_check(hf_config, hf_keys):
            hf_keys = set(hf_keys)
            scales = {k for k in hf_keys if k.endswith("_scale_inv") and k[: -len("_scale_inv")] in hf_keys}
            return _original_check(hf_config, hf_keys - scales)

        _patched_check._coda_quantized = True
        ernie55_mappings.check_ernie55_checkpoint_keys = _patched_check

    if getattr(Ernie55Bridge.maybe_modify_loaded_hf_weight, "_coda_quantized", False):
        return

    def _quant_config(bridge) -> dict:
        qcfg = getattr(bridge.hf_config, "quantization_config", None) or {}
        return qcfg if isinstance(qcfg, dict) else qcfg.to_dict()

    def _loaded(self, hf_param, hf_state_dict):
        if isinstance(hf_param, dict):
            return {k: _loaded(self, v, hf_state_dict) for k, v in hf_param.items()}
        weight = hf_state_dict[hf_param]
        if weight.dtype not in (torch.float8_e4m3fn, torch.int8, torch.uint8):
            return weight
        scale_name = hf_param + "_scale_inv"
        if scale_name not in hf_state_dict:
            raise RuntimeError(f"ERNIE 5.5: {hf_param} is stored as {weight.dtype} without {scale_name}")
        scale = hf_state_dict[scale_name]
        scheme = infer_scheme(
            SAFETENSORS_DTYPE[weight.dtype], tuple(weight.shape), SAFETENSORS_DTYPE[scale.dtype], tuple(scale.shape),
            _quant_config(self),
        )
        if isinstance(scheme, str):
            raise RuntimeError(f"ERNIE 5.5: cannot dequantize {hf_param}: {scheme}")
        if scale.dtype == torch.float32 and not bool((torch.frexp(scale)[0] == 0.5).all()):
            raise RuntimeError(f"ERNIE 5.5: {scale_name} has non-power-of-two scales")
        return scheme.dequant(weight, scale, torch.bfloat16)

    @functools.lru_cache(maxsize=None)
    def _export_layout(ckpt_dir: str):
        """weight name -> (QuantScheme, scale name, stored scale shape, stored scale dtype)."""
        quantized, unsupported = scan_ckpt_quant(ckpt_dir)
        if not quantized and not unsupported:
            return {}, {}
        src = read_source_layout(ckpt_dir)
        torch_dtype = {v: k for k, v in SAFETENSORS_DTYPE.items()}
        layout = {
            name: (e.scheme, e.scale_name, tuple(src[e.scale_name].shape), torch_dtype[src[e.scale_name].dtype])
            for name, e in quantized.items()
        }
        return layout, unsupported

    def _checkpoint_dir(self, hf_state_dict) -> str | None:
        path = getattr(getattr(hf_state_dict, "source", None), "path", None)
        if path is None:
            path = getattr(self.hf_config, "_name_or_path", None) or getattr(self.hf_config, "name_or_path", None)
        return str(path) if path else None

    def _converted(self, task, converted_weights_dict, hf_state_dict):
        if task is not None and task.weight_dtype is not None:
            return converted_weights_dict
        ckpt_dir = _checkpoint_dir(self, hf_state_dict)
        if not ckpt_dir:
            return converted_weights_dict
        layout, unsupported = _export_layout(ckpt_dir)
        if not layout and not unsupported:
            return converted_weights_dict
        result: dict[str, torch.Tensor] = {}
        for name, weight in converted_weights_dict.items():
            if name in unsupported:
                raise RuntimeError(f"ERNIE 5.5: cannot re-quantize {name}: {unsupported[name].reason}")
            entry = layout.get(name)
            if entry is None:
                result[name] = weight
                continue
            scheme, scale_name, scale_shape, scale_dtype = entry
            q, scale = scheme.quant(weight)
            if scale_dtype == torch.float32:
                scale = e8m0_to_float(scale)
            if tuple(scale.shape) != scale_shape or scale.dtype != scale_dtype:
                raise RuntimeError(
                    f"ERNIE 5.5: re-quantized {scale_name} is {scale.dtype} {tuple(scale.shape)}, "
                    f"the checkpoint stores {scale_dtype} {scale_shape}"
                )
            result[name] = q
            result[scale_name] = scale
        return result

    _loaded._coda_quantized = True
    Ernie55Bridge.maybe_modify_loaded_hf_weight = _loaded
    Ernie55Bridge.maybe_modify_converted_hf_weight = _converted


def _patch_auto_attention_backend_inherits():
    """Let ``attention_backend=auto`` sub-models inherit the process-wide TE backend.

    Symptom (Qwen3.8-27B, ``attention_backend: flash``): model construction asserts
    "NVTE_FUSED_ATTN is set to 1, but attention_backend='flash' ... require 0".

    Root cause:
      ``set_attention_backend`` (called by every ``LanguageModule`` / ``VisionModule``
      constructor) writes ``NVTE_{FLASH,FUSED,UNFUSED}_ATTN`` and, since Megatron-LM
      e6f591f73 (#7362), asserts that every model built in the process agrees. Sub-model
      configs that a bridge builds from scratch (e.g. the Qwen3-VL vision encoder from
      ``get_vision_model_config``) never see the user's ``attention_backend`` and keep the
      default ``auto``, which writes all three flags as 1. Whichever of ``auto`` and the
      explicit backend is built second trips the assert.

    Fix: when a config is ``auto`` with no pinned ``flash_attention_version``, leave the
      env untouched instead of writing it. TE already defaults all three flags to 1 when
      unset, so on a clean env this is the same as ``auto``; after an explicit backend
      has been set, the ``auto`` sub-model runs on that backend. Explicit backends still
      go through the original check, so a real conflict between two explicit backends
      still fails. ``language_module`` / ``vision_module`` import the function by name,
      so their module attributes are rebound too.
    """
    from megatron.core.models.common.language_module import language_module
    from megatron.core.models.common.vision_module import vision_module
    from megatron.core.transformer import utils as transformer_utils
    from megatron.core.transformer.enums import AttnBackend

    original = transformer_utils.set_attention_backend
    if getattr(original, "_coda_auto_inherits", False):
        return

    def _patched_set_attention_backend(config):
        backend = config.attention_backend
        if (
            backend in (AttnBackend.auto, "auto")
            and getattr(config, "flash_attention_version", None) is None
            and not getattr(config, "batch_invariant_mode", False)
        ):
            return
        original(config)

    _patched_set_attention_backend._coda_auto_inherits = True
    for module in (transformer_utils, language_module, vision_module):
        module.set_attention_backend = _patched_set_attention_backend


def _patch_dsv4_builder_pipeline_layout():
    """Auto-set the DeepSeek-V4 pipeline layout at PP > 1 on the builder path.

    Symptom: building DeepSeek-V4 with ``bridge.get_model_config()`` / ``bridge.get_model()``
    at PP > 1 asserts "pipeline_model_parallel_layout must be set when using hash MoE layers
    with pipeline parallelism (PP > 1)".

    Root cause:
      The hash-routed MoE layers must sit on the embedding stage, which needs an explicit
      layout. The provider path sets one in ``ModelProvider.apply_overrides_and_finalize``
      (gated on ``experimental_attention_variant == "dsv4_hybrid"``); the builder path's
      ``finalize`` does not.

    Fix: run ``set_deepseek_v4_pipeline_model_parallel_layout`` before
      ``DeepSeekV4ModelConfig.finalize``, which ``get_model`` calls after the yaml overrides
      are applied, so the PP size is final. Scoped to the class rather than the attention
      variant: ERNIE 5.5 is also ``dsv4_hybrid``, but its pattern has dense layers, so the
      DSv4 helper rejects it; ``Ernie55ModelConfig.finalize`` splits its own stages. A layout
      set by the user is never overridden.
    """
    from megatron.bridge.models.deepseek.deepseek_v4_bridge import set_deepseek_v4_pipeline_model_parallel_layout
    from megatron.bridge.models.deepseek.deepseek_v4_config import DeepSeekV4ModelConfig

    _original_finalize = DeepSeekV4ModelConfig.finalize
    if getattr(_original_finalize, "_coda_dsv4_pp_layout", False):
        return

    def _patched_finalize(self):
        transformer = self.transformer
        if (transformer.pipeline_model_parallel_size or 1) > 1 and transformer.pipeline_model_parallel_layout is None:
            set_deepseek_v4_pipeline_model_parallel_layout(self)
        _original_finalize(self)

    _patched_finalize._coda_dsv4_pp_layout = True
    DeepSeekV4ModelConfig.finalize = _patched_finalize


_patch_distributed_optimizer_init()
_patch_fp32_shard_leaf()
_patch_fused_adam_fp32_master_remainder()
_patch_csa_cute_launch_cache()
_patch_fa4_causal_unbounded_window()
_patch_ernie55_drop_mtp()
_patch_ernie55_quantized_checkpoint()
_patch_auto_attention_backend_inherits()
_patch_dsv4_builder_pipeline_layout()

