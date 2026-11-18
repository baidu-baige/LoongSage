"""Validation and cross-key linkage for the `multimodal` config block.

`validate_multimodal_config` runs once on the driver, after `normalize_data_sources`, so every
data source already carries its `dataset.multimodal_keys`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from coda.utils.hf_config import DEEPSEEK_V41_MODEL_TYPE, read_hf_config, read_hf_model_type
from coda.utils.types import ModalityKind

logger = logging.getLogger(__name__)

# Modalities whose processing logic is implemented. VIDEO / AUDIO only have reserved interfaces.
SUPPORTED_MODALITIES = (ModalityKind.IMAGE,)


def multimodal_enabled(config) -> bool:
    """Return True when `multimodal.enable` is set; tolerate configs without the block."""
    multimodal = getattr(config, "multimodal", None)
    return bool(multimodal is not None and getattr(multimodal, "enable", False))


def validate_multimodal_config(config: DictConfig) -> DictConfig:
    """Validate the `multimodal` block and link it to `rollout.sglang_args`.

    When `multimodal.enable` is true, `rollout.sglang_args.enable_multimodal` is set to true;
    an explicit false there is a configuration error rather than being silently overridden.
    When it is false, no data source may declare `dataset.multimodal_keys` or `dataset.max_image_tokens`.
    When it is true, a data source that sets `dataset.max_image_tokens` must declare an `image` column
    in `dataset.multimodal_keys`.

    Raises:
        ValueError: On any inconsistent setting.
    """
    if not multimodal_enabled(config):
        for idx, ds in enumerate(config.data_sources):
            if ds.dataset.get("multimodal_keys"):
                raise ValueError(
                    f"data_sources[{idx}].dataset.multimodal_keys is set but multimodal.enable is false"
                )
            if ds.dataset.get("max_image_tokens") is not None:
                raise ValueError(
                    f"data_sources[{idx}].dataset.max_image_tokens is set but multimodal.enable is false"
                )
        return config

    multimodal = config.multimodal
    modalities = _parse_modalities(multimodal.get("modalities"))

    hf_path = config.get("hf_model_path")
    model_type = read_hf_model_type(hf_path)
    # DeepSeek-V4.1 plans each image from the checkpoint's `vision_*` config. Other models resize images
    # into the bounds of their own processor config; fail at startup if the checkpoint declares none.
    if ModalityKind.IMAGE in modalities and hf_path and model_type != DEEPSEEK_V41_MODEL_TYPE:
        read_image_pixel_bounds(hf_path)

    for idx, ds in enumerate(config.data_sources):
        keys = ds.dataset.get("multimodal_keys")
        _validate_multimodal_keys(keys, modalities, f"data_sources[{idx}]")
        if ds.dataset.get("max_image_tokens") is not None and not (keys and keys.get(ModalityKind.IMAGE.value)):
            raise ValueError(
                f"data_sources[{idx}].dataset.max_image_tokens is set but dataset.multimodal_keys "
                "declares no 'image' column"
            )

    _link_sglang_enable_multimodal(config)
    if model_type == DEEPSEEK_V41_MODEL_TYPE:
        _validate_dsv41_multimodal(config)
    return config


def _validate_dsv41_multimodal(config: DictConfig) -> None:
    """DeepSeek-V4.1 image-modality constraints that would otherwise fail late or silently.

    Raises:
        ValueError: On the first violated constraint.
    """
    sglang_args = config.rollout.sglang_args

    if not read_hf_config(config.get("hf_model_path")).get("vision_config"):
        raise ValueError(
            "multimodal.enable=true but the DeepSeek-V4.1 checkpoint has no vision_config in "
            f"{config.get('hf_model_path')}/config.json"
        )

    if rollout_model_override_args(config).get("vision_n_layers") == 0:
        raise ValueError(
            "multimodal.enable=true on DeepSeek-V4.1 needs the vision tower: remove "
            'vision_n_layers=0 from rollout.sglang_args.json_model_override_args'
        )

    for key in ("pp_size", "pipeline_parallel_size", "attn_cp_size"):
        if int(sglang_args.get(key) or 1) != 1:
            raise ValueError(f"DeepSeek-V4.1 vision requires rollout.sglang_args.{key}=1, got {sglang_args.get(key)}")
    a2a = sglang_args.get("moe_a2a_backend")
    if a2a not in (None, "none", "megamoe"):
        raise ValueError(
            f"DeepSeek-V4.1 vision supports moe_a2a_backend none or megamoe only, got {a2a!r} "
            "(rollout.sglang_args.moe_a2a_backend)"
        )

    backend = sglang_args.get("image_processor_backend")
    if backend != "pil":
        raise ValueError(
            "DeepSeek-V4.1 vision needs rollout.sglang_args.image_processor_backend='pil' so rollout and "
            f"training patchify identically, got {backend!r}"
        )

    model_cfg = (config.get("megatron") or {}).get("model") or {}
    if model_cfg.get("moe_router_fusion"):
        raise ValueError("DeepSeek-V4.1 vision needs megatron.model.moe_router_fusion=false")
    if model_cfg.get("freeze_vision_model") is False:
        raise ValueError(
            "DeepSeek-V4.1 vision trains with a frozen vision tower so that the trainer and the rollout engine "
            "see the same image features; remove megatron.model.freeze_vision_model=false"
        )


def drop_dsv41_vision_if_text_only(hf_config, enable: bool) -> None:
    """Run a DeepSeek-V4.1 checkpoint as a text-only model when `multimodal.enable` is false.

    The bridge builds the vision tower whenever `hf_config.vision_config` is set, so clear it here.
    `validate_multimodal_config` already rejects `enable=true` without a `vision_config`.
    """
    if not enable and getattr(hf_config, "model_type", None) == DEEPSEEK_V41_MODEL_TYPE:
        hf_config.vision_config = None


def rollout_model_override_args(config) -> dict:
    """`rollout.sglang_args.json_model_override_args` as a plain dict (empty when unset).

    Accepts the JSON-string form used in recipes as well as a YAML mapping.
    """
    rollout = getattr(config, "rollout", None)
    sglang_args = getattr(rollout, "sglang_args", None) if rollout is not None else None
    raw = sglang_args.get("json_model_override_args") if sglang_args is not None else None
    if not raw:
        return {}
    if isinstance(raw, DictConfig):
        return dict(OmegaConf.to_container(raw, resolve=True))
    return json.loads(raw) if isinstance(raw, str) else dict(raw)


def validate_vlm_parallel_config(config: DictConfig, model_cfg: DictConfig, hf_path: str | None) -> None:
    """Check the parallel settings of a Megatron model section against the image-modality constraints.

    Args:
        config: Full config; only `multimodal.enable` is read.
        model_cfg: `config.megatron.model` (student) or `config.opd.model` (teacher).
        hf_path: HF checkpoint directory used to detect the model family; detection is skipped
            when its `config.json` is not readable.

    Raises:
        ValueError: CP > 1 with `cp_partition_mode == "contiguous"` on a non-DeepSeek-V4.1 model, CP > 1 with
            a mode other than `contiguous` on DeepSeek-V4.1, or Qwen2.5-VL with CP > 1.
    """
    if not multimodal_enabled(config):
        return

    cp_size = int(model_cfg.get("context_parallel_size") or 1)
    model_type = read_hf_model_type(hf_path)

    if cp_size > 1:
        cp_mode = model_cfg.get("cp_partition_mode", "zigzag")
        if model_type == DEEPSEEK_V41_MODEL_TYPE:
            # The V4.1 vision tower picks each CP rank's image tokens out of one contiguous chunk of the
            # packed buffer; a zigzag rank holds two chunks per trajectory, which that logic does not handle.
            if cp_mode != "contiguous":
                raise ValueError(
                    "DeepSeek-V4.1 with multimodal.enable=true and context_parallel_size > 1 only supports "
                    f"cp_partition_mode='contiguous', got {cp_mode!r}"
                )
        elif cp_mode == "contiguous":
            raise ValueError(
                "multimodal.enable=true with context_parallel_size > 1 only supports "
                "cp_partition_mode='zigzag'; 'contiguous' is DSv4-only"
            )
        if model_type == "qwen2_5_vl":
            raise ValueError(
                "Qwen2.5-VL does not support context_parallel_size > 1 in Megatron-Bridge; "
                "set context_parallel_size=1 or use a Qwen3-VL model"
            )

    if model_type == "qwen3_vl_moe":
        for key in ("freeze_language_model", "freeze_vision_model"):
            if model_cfg.get(key) is None:
                logger.warning(
                    "Qwen3-VL-MoE defaults %s=True in Megatron-Bridge; set %s explicitly to train that part",
                    key, key,
                )


def _parse_modalities(raw) -> list[ModalityKind]:
    if not raw:
        raise ValueError("multimodal.modalities must be a non-empty list when multimodal.enable is true")
    modalities = []
    for name in raw:
        try:
            kind = ModalityKind(name)
        except ValueError as e:
            raise ValueError(
                f"multimodal.modalities has unknown modality {name!r}; "
                f"known: {[m.value for m in ModalityKind]}"
            ) from e
        if kind not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"multimodal.modalities: {kind.value!r} is reserved but not implemented yet; "
                f"supported: {[m.value for m in SUPPORTED_MODALITIES]}"
            )
        modalities.append(kind)
    return modalities


def read_image_pixel_bounds(hf_path: str) -> tuple[int, int]:
    """`(min_pixels, max_pixels)` the Router resizes images into, taken from the model's own processor config.

    These are `size.shortest_edge` and `size.longest_edge` of `<hf_path>/preprocessor_config.json`.

    Raises:
        ValueError: The file is unreadable or does not declare both edges.
    """
    cfg_file = Path(str(hf_path)) / "preprocessor_config.json"
    try:
        size = json.loads(cfg_file.read_text()).get("size") or {}
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read the image pixel bounds from {cfg_file}: {e}") from e
    shortest, longest = size.get("shortest_edge"), size.get("longest_edge")
    if shortest is None or longest is None:
        raise ValueError(f"{cfg_file} must declare size.shortest_edge and size.longest_edge, got size={size}")
    return int(shortest), int(longest)


def _validate_multimodal_keys(keys, modalities: list[ModalityKind], where: str) -> None:
    if not keys:
        return
    for modality_name, column in keys.items():
        try:
            kind = ModalityKind(modality_name)
        except ValueError as e:
            raise ValueError(f"{where}.dataset.multimodal_keys has unknown modality {modality_name!r}") from e
        if kind not in modalities:
            raise ValueError(
                f"{where}.dataset.multimodal_keys declares {modality_name!r}, "
                f"which is not in multimodal.modalities"
            )
        if not isinstance(column, str) or not column:
            raise ValueError(f"{where}.dataset.multimodal_keys[{modality_name!r}] must be a column name")


def _link_sglang_enable_multimodal(config: DictConfig) -> None:
    sglang_args = config.rollout.sglang_args
    current = sglang_args.get("enable_multimodal")
    if current is False:
        raise ValueError(
            "multimodal.enable is true but rollout.sglang_args.enable_multimodal is explicitly false; "
            "remove it or set it to true"
        )
    if current is None:
        OmegaConf.update(config, "rollout.sglang_args.enable_multimodal", True, force_add=True)
        logger.info("multimodal.enable=true: set rollout.sglang_args.enable_multimodal=true")
