"""Validation and cross-key linkage for the `multimodal` config block.

`validate_multimodal_config` runs once on the driver, after `normalize_data_sources`, so every
data source already carries its `dataset.multimodal_keys`.
"""

from __future__ import annotations

from omegaconf import DictConfig

from coda.utils.hf_config import read_hf_model_type
from coda.utils.types import ModalityKind

# Modalities whose processing logic is implemented. VIDEO / AUDIO only have reserved interfaces.
SUPPORTED_MODALITIES = (ModalityKind.IMAGE,)


def validate_multimodal_config(config: DictConfig) -> DictConfig:
    """Validate the `multimodal` block.

    When `multimodal.enable` is false, no data source may declare `dataset.multimodal_keys` or
    `dataset.max_image_tokens`.
    When it is true, a data source that sets `dataset.max_image_tokens` must declare an `image` column
    in `dataset.multimodal_keys`.
    The engine sets SGLang's `enable_multimodal` at launch; nothing is written to the config here.

    Raises:
        ValueError: On any inconsistent setting.
    """
    if not config.multimodal.enable:
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

    for idx, ds in enumerate(config.data_sources):
        keys = ds.dataset.get("multimodal_keys")
        _validate_multimodal_keys(keys, modalities, f"data_sources[{idx}]")
        if ds.dataset.get("max_image_tokens") is not None and not (keys and keys.get(ModalityKind.IMAGE.value)):
            raise ValueError(
                f"data_sources[{idx}].dataset.max_image_tokens is set but dataset.multimodal_keys "
                "declares no 'image' column"
            )

    return config


def validate_vlm_parallel_config(config: DictConfig, model_cfg: DictConfig, hf_path: str | None) -> None:
    """Reject parallel settings that the image modality does not support.

    Args:
        config: Full config; only `multimodal.enable` is read.
        model_cfg: `config.megatron.model` (student) or `config.opd.model` (teacher).
        hf_path: HF checkpoint directory used to detect the model family; detection is skipped
            when its `config.json` is not readable.

    Raises:
        ValueError: Qwen2.5-VL with CP > 1.
    """
    if not config.multimodal.enable:
        return

    cp_size = int(model_cfg.get("context_parallel_size") or 1)
    model_type = read_hf_model_type(hf_path)

    if cp_size > 1 and model_type == "qwen2_5_vl":
        raise ValueError(
            "Qwen2.5-VL does not support context_parallel_size > 1 in Megatron-Bridge; "
            "set context_parallel_size=1 or use a Qwen3-VL model"
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
