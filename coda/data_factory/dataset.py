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

from coda.agentflow.trajectory_store import Trajectory

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)


def read_file(path, columns=None):
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
        reader = itertools.chain.from_iterable(_read_one_file(p, columns) for p in files)
    else:
        reader = _read_one_file(path, columns)

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


def _read_one_file(path, columns=None):
    """Yield raw dicts from a single `.jsonl` or `.parquet` file, optionally reading only *columns*."""
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

            for batch in pf.iter_batches(columns=projected or None):
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


def _build_messages(data: dict, prompt_key: str):
    """Build a prompt from a raw dataset record.

    Converts the value at *prompt_key* into a list of message dicts.  Plain strings are wrapped in a
    single `{"role": "user", "content": ...}` message; values that are already a list are returned as-is.

    Args:
        data:       Raw dataset record (one row from the dataset file).
        prompt_key: Key in *data* whose value is the prompt (string or list of message dicts).

    Returns:
        A `list[dict]` of message dicts, or `None` if *prompt_key* is absent from *data*.
    """
    prompt = data.get(prompt_key)

    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    elif hasattr(prompt, "tolist"):
        # numpy.ndarray from parquet — convert to plain list of dicts
        prompt = [dict(m) for m in prompt]

    # TODO: support multi modal

    return prompt


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
        """
        columns = {key for key in (prompt_key, label_key, metadata_key) if key}
        if data_pre_processor is not None:
            source_columns = getattr(data_pre_processor, "source_columns", None)
            columns = columns | set(source_columns) if source_columns else None

        origin_prompts = []
        for data in read_file(path, columns=columns):
            if data_pre_processor is not None:
                data = data_pre_processor(data, prompt_key)

            prompt = _build_messages(data, prompt_key)

            metadata = data.get(metadata_key) or {}

            raw_label = data.get(label_key) if label_key is not None else None
            if raw_label is not None and not isinstance(raw_label, dict):
                raw_label = {"value": raw_label}

            logger.debug("data: %s\n, prompt_key: %s\n, label_key: %s\n, prompt: %s\n, raw_label: %s",
                         data, prompt_key, label_key, prompt, raw_label)

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
        the sum of all `"content"` string lengths across all messages.

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
            return total

        filtered = [
            p for p in origin_prompts if _prompt_len(p.prompt) <= max_length
        ]

        logger.info(
            f"Filtered {len(origin_prompts) - len(filtered)} prompts longer than max_length={max_length}."
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
