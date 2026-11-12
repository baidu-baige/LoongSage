"""Facts read from the `config.json` of a HuggingFace checkpoint (`hf_model_path`)."""

from __future__ import annotations

import json
from pathlib import Path

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
