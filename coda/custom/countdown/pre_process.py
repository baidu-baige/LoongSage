"""Countdown pre-processor: reshape raw ``Jiayi-Pan/Countdown-Tasks-3to4`` rows (``nums``/``target``)."""

from __future__ import annotations

from coda.data_factory import register_data_pre_processor

_FORMAT_INSTRUCTION = (
    "\nUse every number exactly once with only + - * / and parentheses. "
    "Give the final expression inside answer tags, e.g. <answer>(1 + 2) * 8</answer>."
)


def _to_int_list(value) -> list[int]:
    """Coerce a nums field (list / numpy array / tuple) into a list of ints."""
    if value is None:
        return []
    if hasattr(value, "tolist"):  # numpy.ndarray from parquet
        value = value.tolist()
    return [int(n) for n in value]


@register_data_pre_processor("countdown")
def countdown_pre_process(data: dict, prompt_key: str) -> dict:
    """Build a prompt from ``nums``/``target`` and attach the label dict the countdown reward reads."""
    if isinstance(data.get("label"), dict) and data.get(prompt_key):
        return data

    nums = _to_int_list(data.get("nums"))
    target_raw = data.get("target")
    if not nums or target_raw is None:
        raise ValueError("countdown_pre_process: 'nums' and 'target' are required in the dataset record")
    target = int(target_raw)

    data[prompt_key] = f"Numbers: {nums}\nTarget: {target}" + _FORMAT_INSTRUCTION
    data["label"] = {"target": target, "nums": nums}
    return data


# Columns this pre-processor reads, so `Dataset` can restrict the parquet scan to them.
countdown_pre_process.source_columns = ("nums", "target")
