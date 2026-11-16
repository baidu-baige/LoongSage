"""Dataset utilities for loading and preprocessing RL training prompts.

Supports reading prompt datasets from `.jsonl` and `.parquet` files — or from a directory holding
such files, e.g. a `hf download --local-dir` snapshot — with optional row slicing and
sequence-length filtering.  Raw records can additionally be rewritten by a pluggable
pre-processor (see `coda.data_factory.data_pre_processor`) before messages are built.

Public API:
    `Dataset`: Iterable dataset class used by the training loop.

Internal helpers:
    `read_file`:                Generator that yields raw dicts from a dataset file or directory.
    `_list_dataset_files`:      Lists the dataset files below a directory in sorted order.
    `_read_one_file`:           Generator that yields raw dicts from a single dataset file.
    `_parse_generalized_path`:  Parses an optional `@[start:stop]` slice suffix from a file path string.
    `_build_messages`:          Converts a raw data dict into a list of message dicts.
"""

import itertools
import json
import logging
import os
import random
import re

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from coda.agentflow.multimodal import estimate_data_uri_tokens
from coda.agentflow.trajectory_store import Trajectory
from coda.utils.types import MODALITY_PLACEHOLDER, ModalityKind

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)

# Rows per parquet batch when the dataset carries images: rows hold data URIs, so pyarrow's default batch
# (65536 rows) could be many GB. Text-only datasets keep the pyarrow default.
_MULTIMODAL_PARQUET_BATCH_ROWS = 32


def read_file(path, columns=None, batch_size=None):
    """Yield raw dicts from a `.jsonl`/`.parquet` dataset file, or from a directory of them.

    Supports an optional row-slice suffix in the path (see `_parse_generalized_path`), e.g.
        read_file("data.jsonl@[100:200]")  # rows 100–199

    When *path* is a directory (e.g. the `--local-dir` of a `hf download`), every `.jsonl` and
    `.parquet` file below it is read in sorted path order and the rows are concatenated, so a
    sharded dataset such as `data/train-00000-of-00008.parquet` can be used as-is. Hidden
    directories (`.cache`, `.git`, ...) are skipped. A row slice applies to the concatenated
    stream. Note that all splits present in the directory are read, so pass a single file when a
    directory holds more than the split you want to train on.

    Invalid JSON lines in `.jsonl` files are logged and skipped rather than raising an exception.

    Args:
        path:    Path to a `.jsonl`|`.parquet` file or to a directory containing such files, optionally
                 followed by `@[start:stop]` to limit which rows are yielded.
        columns: Optional collection of column names to restrict `.parquet` reads to; names absent from
                 a file are ignored, and `None` (or no name matching the file) reads every column.
                 `.jsonl` files are always read in full.
        batch_size: Optional cap on the rows per pyarrow batch of `.parquet` reads, to bound memory when rows
                 are large (e.g. embed image data URIs). `None` keeps pyarrow's default; `.jsonl` ignores it.

    Yields:
        dict: One parsed record per row.

    Raises:
        FileNotFoundError:  If *path* (after stripping the slice suffix) does not exist on disk.
        ImportError:        If a file is `.parquet` and `pyarrow` is not installed.
        ValueError:         If a file extension is neither `.jsonl` nor `.parquet`, or if *path* is a
                            directory containing no such file.
    """
    path, row_slice = _parse_generalized_path(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if os.path.isdir(path):
        files = _list_dataset_files(path)
        if not files:
            raise ValueError(f"No .jsonl or .parquet file found under directory: {path}")
        logger.info("read_file dir=%s files=%s", path, files)
        reader = itertools.chain.from_iterable(_read_one_file(p, columns, batch_size) for p in files)
    else:
        reader = _read_one_file(path, columns, batch_size)

    if row_slice is not None:
        logger.info("read_file path=%s applying slice row_slice=%s", path, row_slice)
        reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)

    yield from reader


def _list_dataset_files(root):
    """Return sorted `.jsonl`/`.parquet` paths below *root*, skipping hidden directories."""
    matched = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        matched.extend(
            os.path.join(dirpath, name)
            for name in filenames
            if name.endswith((".jsonl", ".parquet"))
        )

    return sorted(matched)


def _read_one_file(path, columns=None, batch_size=None):
    """Yield raw dicts from a single `.jsonl` or `.parquet` file, optionally reading only *columns*.

    *batch_size* caps the rows per pyarrow batch of a `.parquet` read (`None` = pyarrow default).
    """
    if path.endswith(".jsonl"):

        def jsonl_reader(p):
            with open(p, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning("JSON decode error at line %d: %s", line_num, e)
                        continue

        reader = jsonl_reader(path)

    elif path.endswith(".parquet"):
        if pq is None:
            raise ImportError("pyarrow is required for parquet support")

        def parquet_reader(p):
            pf = pq.ParquetFile(p)
            # Unknown names make pyarrow raise, and an empty projection would drop every row, so
            # intersect with the file schema and fall back to reading all columns.
            projected = [c for c in columns if c in pf.schema_arrow.names] if columns else None
            logger.info("read_file path=%s projected_columns=%s", p, projected)

            # pyarrow's default batch (65536 rows) is fine for text; image rows carry data URIs and need a small one.
            batch_kwargs = {"batch_size": batch_size} if batch_size else {}
            for batch in pf.iter_batches(columns=projected or None, **batch_kwargs):
                yield from batch.to_pylist()

        reader = parquet_reader(path)

    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    yield from reader


def _parse_generalized_path(s: str):
    """Parse an optional `@[start:stop]` slice suffix from a path string.

    Examples::
        _parse_generalized_path("data.jsonl")           # -> ("data.jsonl", None)
        _parse_generalized_path("data.jsonl@[10:50]")   # -> ("data.jsonl", slice(10, 50))
        _parse_generalized_path("data.jsonl@[:]")       # -> ("data.jsonl", slice(None, None))

    Args:
        s: Raw path string, possibly including a `@[start:stop]` suffix.

    Returns:
        Tuple of `(real_path, row_slice)` where *row_slice* is a `slice` object when a suffix
        was found, or `None` otherwise.
    """
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None


def _build_messages(data: dict, prompt_key: str, multimodal_keys: dict | None = None):
    """Build a prompt from a raw dataset record.

    Converts the value at *prompt_key* into a list of message dicts.  Plain strings are wrapped in a
    single `{"role": "user", "content": ...}` message; values that are already a list are returned as-is.

    When *multimodal_keys* is given, every modality placeholder (e.g. `<image>`) in a string `content`
    is replaced by an OpenAI-style content part, so that `content` becomes a list of
    `{"type": "text", ...}` and `{"type": "image_url", "image_url": {"url": <data URI>}}` parts.

    Args:
        data:            Raw dataset record (one row from the dataset file).
        prompt_key:      Key in *data* whose value is the prompt (string or list of message dicts).
        multimodal_keys: Optional `{modality: column}` mapping, e.g. `{"image": "images"}`.  The column
                         holds one data URI per placeholder in the prompt text.

    Returns:
        A `list[dict]` of message dicts, or `None` if *prompt_key* is absent from *data*.

    Raises:
        ValueError: If the placeholder count does not match the number of items in the column, or an
                    item is not a data URI.
    """
    prompt = data.get(prompt_key)

    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    elif hasattr(prompt, "tolist"):
        # numpy.ndarray from parquet — convert to plain list of dicts
        prompt = [dict(m) for m in prompt]

    if multimodal_keys and isinstance(prompt, list):
        prompt = _expand_multimodal_placeholders(prompt, data, multimodal_keys)

    return prompt


def _load_placeholder_items(multimodal_keys: dict, data: dict) -> dict:
    """Map each configured placeholder to `(column, items)`; every item must be a data URI."""
    items_by_placeholder = {}
    for modality_name, column in multimodal_keys.items():
        kind = ModalityKind(modality_name)
        if kind is not ModalityKind.IMAGE:
            raise ValueError(f"multimodal_keys: modality {modality_name!r} is reserved but not implemented yet")
        raw_items = data.get(column)
        items = list(raw_items) if raw_items is not None else []
        for item in items:
            if not isinstance(item, str) or not item.startswith("data:"):
                raise ValueError(f"{column!r} items must be data URIs ('data:...'), got {str(item)[:64]!r}")
        items_by_placeholder[MODALITY_PLACEHOLDER[kind]] = (column, items)
    return items_by_placeholder


def _reject_unconfigured_placeholders(messages: list, configured: dict) -> None:
    """Raise if a string content uses a modality placeholder that has no entry in `multimodal_keys`."""
    unconfigured = [p for p in MODALITY_PLACEHOLDER.values() if p not in configured]
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            for placeholder in unconfigured:
                if placeholder in content:
                    raise ValueError(
                        f"prompt contains {placeholder!r} modality placeholder, but this modality is not supported yet"
                    )


def _expand_multimodal_placeholders(messages: list, data: dict, multimodal_keys: dict) -> list:
    """Replace modality placeholders in string contents with OpenAI-style content parts.

    Only string `content` is handled; messages whose content is already a list are passed through.
    """
    items_by_placeholder = _load_placeholder_items(multimodal_keys, data)
    _reject_unconfigured_placeholders(messages, items_by_placeholder)

    # Split every string content once; None marks non-string content, which is passed through untouched.
    pattern = "(" + "|".join(re.escape(p) for p in items_by_placeholder) + ")"
    segments_per_message = [
        re.split(pattern, message["content"]) if isinstance(message.get("content"), str) else None
        for message in messages
    ]

    placeholder_count = {p: 0 for p in items_by_placeholder}
    for segments in segments_per_message:
        for segment in segments or ():
            if segment in placeholder_count:
                placeholder_count[segment] += 1

    for placeholder, (column, items) in items_by_placeholder.items():
        if placeholder_count[placeholder] != len(items):
            raise ValueError(
                f"{placeholder!r} placeholder count ({placeholder_count[placeholder]}) does not match "
                f"the number of items in column {column!r} ({len(items)})"
            )

    if not any(placeholder_count.values()):
        return messages

    # Counts match the item lists, so each placeholder consumes its items in order.
    item_iters = {p: iter(items) for p, (_, items) in items_by_placeholder.items()}
    expanded = []
    for message, segments in zip(messages, segments_per_message):
        if segments is None:
            expanded.append(message)
            continue
        parts = [
            {"type": "image_url", "image_url": {"url": next(item_iters[segment])}}
            if segment in item_iters
            else {"type": "text", "text": segment}
            for segment in segments
            if segment
        ]
        expanded.append({**message, "content": parts})
    return expanded


class Dataset:
    """Iterable dataset for RL rollout prompts.

    Reads prompts from a `.jsonl`|`.parquet` file (or a directory of them), applies optional preprocessing
    (length filtering and a pluggable pre-processor), and exposes the resulting
    `Trajectory` objects via `__getitem__` / `__len__`.

    Attributes:
        origin_prompts: Full list of `Trajectory` objects after loading and optional length
                        filtering.  This list is never re-ordered.
        prompts:        Current view of prompts used by `__getitem__`.  Starts identical to
                        *origin_prompts* and is shuffled in-place by `shuffle`.
        epoch_id:       Epoch index of the last shuffle; `-1` before any shuffle.
        seed:           Base random seed used for reproducible shuffling.
    """

    def __init__(
        self,
        path,
        max_length,
        *,
        prompt_key="text",
        label_key=None,
        metadata_key="metadata",
        seed=42,
        data_pre_processor=None,
        multimodal_keys=None,
        max_image_tokens=None,
        image_pixel_bounds=None,
    ):
        """Load and preprocess the dataset from *path*.

        Args:
            path:         Path to the dataset file (`.jsonl` or `.parquet`) or to a directory containing such files,
                          optionally with a row-slice suffix (e.g. `data.jsonl@[0:1000]`).
            max_length:   Maximum number of characters allowed per prompt.  Samples that exceed this limit are dropped.
                          Pass `None` to disable filtering.
            prompt_key:   Key in each dataset record that holds the prompt text or message list. Defaults to `"text"`.
            label_key:    Optional key for the ground-truth label stored in `Trajectory.label`.
            metadata_key: Key for an optional metadata dict merged into `Trajectory.metadata`. Defaults to `metadata`.
            seed:         Random seed for `shuffle`.  Defaults to `42`.
            data_pre_processor: Optional callable `(data: dict, prompt_key: str) -> dict` applied to every raw record
                          before messages are built.  Resolved from `DATA_PRE_PROCESSOR_REGISTRY` by the caller
                          (see `coda.data_factory.data_pre_processor`).  `None` disables pre-processing.
                          A pre-processor that sets a `source_columns` attribute declares the raw columns it
                          reads; parquet reads are then restricted to those plus the key columns above.  When
                          the attribute is absent the pre-processor may read anything, so no projection is done.
            multimodal_keys: Optional `{modality: column}` mapping (e.g. `{"image": "images"}`); the column holds
                          one data URI per placeholder in the prompt text.  `None` = text-only dataset.
            max_image_tokens: Optional cap on the estimated vision tokens summed over all images of one prompt.
                          Samples above it are dropped.  Independent of *max_length*, which only counts text
                          characters.  Needs *image_pixel_bounds*.  `None` disables the check.
            image_pixel_bounds: `(min_pixels, max_pixels)` the router resizes images into (see `multimodal.image`);
                          used to estimate image tokens.  The estimate assumes a Qwen3-VL token size of 32x32 pixels.
        """
        if max_image_tokens is not None and image_pixel_bounds is None:
            raise ValueError("max_image_tokens requires image_pixel_bounds=(min_pixels, max_pixels)")
        multimodal_keys = dict(multimodal_keys) if multimodal_keys else None
        columns = {key for key in (prompt_key, label_key, metadata_key) if key}
        if multimodal_keys:
            columns |= set(multimodal_keys.values())
        if data_pre_processor is not None:
            source_columns = getattr(data_pre_processor, "source_columns", None)
            columns = columns | set(source_columns) if source_columns else None

        origin_prompts = []
        for data in read_file(
            path, columns=columns, batch_size=_MULTIMODAL_PARQUET_BATCH_ROWS if multimodal_keys else None
        ):
            if data_pre_processor is not None:
                data = data_pre_processor(data, prompt_key)

            prompt = _build_messages(data, prompt_key, multimodal_keys)

            metadata = data.get(metadata_key) or {}

            raw_label = data.get(label_key) if label_key is not None else None
            if raw_label is not None and not isinstance(raw_label, dict):
                raw_label = {"value": raw_label}

            origin_prompts.append(
                Trajectory(
                    trajectory_id="",
                    prompt_id="",
                    prompt=prompt,
                    label=raw_label,
                    metadata=metadata,
                )
            )

        if max_length is not None:
            self.origin_prompts = self.filter_long_prompt(origin_prompts, max_length)
        else:
            self.origin_prompts = origin_prompts

        if max_image_tokens is not None:
            self.origin_prompts = self.filter_large_images(self.origin_prompts, max_image_tokens, image_pixel_bounds)

        self.epoch_id = -1
        self.seed = seed
        self.prompts = self.origin_prompts

    def shuffle(self, new_epoch_id):
        """Shuffle prompts for a new epoch using a deterministic seed.

        Shuffling is skipped when *new_epoch_id* equals the current `epoch_id`, making repeated calls with the same
        epoch idempotent.

        Args:
            new_epoch_id: Integer epoch identifier.  Combined with `seed` to produce a reproducible permutation.
        """
        if self.epoch_id == new_epoch_id:
            return

        rng = random.Random(self.seed + new_epoch_id)
        permutation = list(range(len(self.prompts)))
        rng.shuffle(permutation)
        self.prompts = [self.origin_prompts[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def filter_long_prompt(
        self, origin_prompts: list[Trajectory], max_length: int | None
    ) -> list[Trajectory]:
        """Remove prompts that exceed *max_length* characters.

        For string prompts, length is measured directly via `len(prompt)`.  For list-of-message-dict prompts, length is
        the sum of all `"content"` string lengths across all messages; for list-valued content, only the `"text"` parts
        are counted (image parts are not).

        Args:
            origin_prompts: List of `Trajectory` objects to filter.
            max_length:     Maximum allowed character count (inclusive).  When `None` the list is returned as-is.

        Returns:
            Filtered list of `Trajectory` objects whose prompt length does not exceed *max_length*.
        """
        if max_length is None:
            return origin_prompts

        if not origin_prompts:
            return origin_prompts

        def _prompt_len(prompt) -> int:
            if isinstance(prompt, str):
                return len(prompt)
            # list of message dicts — sum up all text content
            total = 0
            for msg in prompt:
                content = msg.get("content", "")
                if isinstance(content, str):
                    total += len(content)
                elif isinstance(content, list):
                    # Content parts (e.g. after multimodal placeholder expansion): count text parts only.
                    total += sum(
                        len(part.get("text") or "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
            return total

        filtered = [
            p for p in origin_prompts if _prompt_len(p.prompt) <= max_length
        ]

        logger.info(
            f"Filtered {len(origin_prompts) - len(filtered)} prompts longer than max_length={max_length}."
        )

        return filtered

    @staticmethod
    def filter_large_images(
        origin_prompts: list[Trajectory], max_image_tokens: int, image_pixel_bounds: tuple[int, int]
    ) -> list[Trajectory]:
        """Remove prompts whose images add up to more than *max_image_tokens* estimated vision tokens.

        Only `image_url` parts of list-form message content are counted; only image headers are parsed.

        Args:
            origin_prompts:     List of `Trajectory` objects to filter.
            max_image_tokens:   Maximum allowed estimated image tokens per prompt (inclusive).
            image_pixel_bounds: `(min_pixels, max_pixels)` used for the token estimate.

        Returns:
            Filtered list of `Trajectory` objects.
        """
        min_pixels, max_pixels = image_pixel_bounds

        def _image_tokens(prompt) -> int:
            if isinstance(prompt, str):
                return 0
            total = 0
            for msg in prompt:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        value = part.get("image_url")
                        url = value.get("url") if isinstance(value, dict) else value
                        total += estimate_data_uri_tokens(url, min_pixels, max_pixels)
            return total

        filtered = [p for p in origin_prompts if _image_tokens(p.prompt) <= max_image_tokens]
        logger.info(
            f"Filtered {len(origin_prompts) - len(filtered)} prompts with more than "
            f"max_image_tokens={max_image_tokens} estimated image tokens."
        )
        return filtered

    def __getitem__(self, idx):
        """Return the prompt at position *idx* in the current (possibly shuffled) view.

        Args:
            idx: Integer index into `self.prompts`.

        Returns:
            `Trajectory` at the given index.
        """
        return self.prompts[idx]

    def __len__(self):
        """Return the number of prompts in the current view.

        Returns:
            Integer count of prompts in `self.prompts`.
        """
        return len(self.prompts)
