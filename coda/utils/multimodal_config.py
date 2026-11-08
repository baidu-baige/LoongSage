"""Validation and cross-key linkage for the `multimodal` config block.

`validate_multimodal_config` runs once on the driver, after `normalize_data_sources`, so every
data source already carries its `dataset.multimodal_keys`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

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
    When it is false, no data source may declare `dataset.multimodal_keys`.

    Raises:
        ValueError: On any inconsistent setting.
    """
    if not multimodal_enabled(config):
        for idx, ds in enumerate(config.data_sources):
            if ds.dataset.get("multimodal_keys"):
                raise ValueError(
                    f"data_sources[{idx}].dataset.multimodal_keys is set but multimodal.enable is false"
                )
        return config

    multimodal = config.multimodal
    modalities = _parse_modalities(multimodal.get("modalities"))

    image = multimodal.get("image")
    if ModalityKind.IMAGE in modalities:
        _validate_image_config(image)

    for idx, ds in enumerate(config.data_sources):
        _validate_multimodal_keys(ds.dataset.get("multimodal_keys"), modalities, f"data_sources[{idx}]")

    _link_sglang_enable_multimodal(config)
    return config


def read_hf_model_type(hf_path: str | None) -> str | None:
    """Return `model_type` from `<hf_path>/config.json`; None when the file is unavailable."""
    if not hf_path:
        return None
    cfg_file = Path(str(hf_path)) / "config.json"
    if not cfg_file.is_file():
        return None
    try:
        return json.loads(cfg_file.read_text()).get("model_type")
    except (OSError, ValueError):
        return None


def validate_vlm_parallel_config(config: DictConfig, model_cfg: DictConfig, hf_path: str | None) -> None:
    """Check the parallel settings of a Megatron model section against the image-modality constraints.

    Args:
        config: Full config; only `multimodal.enable` is read.
        model_cfg: `config.megatron.model` (student) or `config.opd.model` (teacher).
        hf_path: HF checkpoint directory used to detect the model family; detection is skipped
            when its `config.json` is not readable.

    Raises:
        ValueError: `cp_partition_mode == "contiguous"` with CP > 1, or Qwen2.5-VL with CP > 1.
    """
    if not multimodal_enabled(config):
        return

    cp_size = int(model_cfg.get("context_parallel_size") or 1)
    model_type = read_hf_model_type(hf_path)

    if cp_size > 1:
        if model_cfg.get("cp_partition_mode", "zigzag") == "contiguous":
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


def validate_pixel_bounds(min_pixels: int, max_pixels: int, shortest_edge: int, longest_edge: int) -> None:
    """Check the pre-resize bounds against the processor's own `shortest_edge` / `longest_edge`.

    The processor values are only known once the processor is loaded, so this is called by the
    Router side at processor-load time.

    Raises:
        ValueError: If `min_pixels` is below `shortest_edge` (the processor would upscale an
            already aligned image again) or `max_pixels` is above `longest_edge`.
    """
    if min_pixels < shortest_edge:
        raise ValueError(
            f"multimodal.image.min_pixels ({min_pixels}) must be >= processor shortest_edge ({shortest_edge})"
        )
    if max_pixels > longest_edge:
        raise ValueError(
            f"multimodal.image.max_pixels ({max_pixels}) must be <= processor longest_edge ({longest_edge})"
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


def _validate_image_config(image) -> None:
    if image is None:
        raise ValueError("multimodal.image is required when 'image' is in multimodal.modalities")
    min_pixels, max_pixels = int(image.min_pixels), int(image.max_pixels)
    if min_pixels <= 0 or max_pixels <= 0:
        raise ValueError("multimodal.image.min_pixels and max_pixels must be > 0")
    if min_pixels > max_pixels:
        raise ValueError(
            f"multimodal.image.min_pixels ({min_pixels}) must be <= max_pixels ({max_pixels})"
        )


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
