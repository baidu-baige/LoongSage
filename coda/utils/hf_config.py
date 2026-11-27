"""Facts read from the JSON configs of a HuggingFace checkpoint (`hf_model_path`).

`config.json` gives the model type and architecture; `preprocessor_config.json` gives the image pixel bounds.
"""

from __future__ import annotations

import json
from pathlib import Path

DEEPSEEK_V4_MODEL_TYPE = "deepseek_v4"
DEEPSEEK_V41_MODEL_TYPE = "deepseek_v41"


def read_hf_config(hf_path: str | None) -> dict:
    """Return the parsed `<hf_path>/config.json`; an empty dict when the file is unavailable."""
    if not hf_path:
        return {}
    cfg_file = Path(str(hf_path)) / "config.json"
    if not cfg_file.is_file():
        return {}
    try:
        return json.loads(cfg_file.read_text())
    except (OSError, ValueError):
        return {}


def read_hf_model_type(hf_path: str | None) -> str | None:
    """Return `model_type` from `<hf_path>/config.json`; None when the file is unavailable."""
    return read_hf_config(hf_path).get("model_type")


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
