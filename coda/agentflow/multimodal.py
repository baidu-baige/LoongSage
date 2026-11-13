"""Router-side multimodal (image) helpers.

Free of sglang / router imports to avoid import cycles. PIL, transformers and qwen_vl_utils are imported lazily.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from typing import Any

logger = logging.getLogger(__name__)

# Roles whose `image_url` parts the chat template renders. Images in other roles would be silently
# dropped by the template, so they are rejected up front.
IMAGE_ALLOWED_ROLES = ("user", "tool")

# Image url form used after `replace_images_with_hashes`; carries the content hash only.
HASH_URL_PREFIX = "sha256:"

# Default byte cap (LRU) of the processor-result cache; 0 disables caching.
DEFAULT_IMAGE_CACHE_MAX_BYTES = 4 << 30


class MultimodalRequestError(ValueError):
    """Invalid multimodal request content; the router maps it to HTTP 400."""


def parse_data_uri(ref: str) -> bytes:
    """Decode a `data:<mime>;base64,<payload>` URI into raw bytes."""
    if not isinstance(ref, str) or not ref.startswith("data:"):
        raise MultimodalRequestError("only data URI images are supported")
    header, sep, payload = ref.partition(",")
    if not sep or ";base64" not in header:
        raise MultimodalRequestError("image data URI must be base64 encoded")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MultimodalRequestError(f"invalid base64 image payload: {exc}") from exc


def image_sha256(ref: str) -> str:
    """sha256 hex digest of the decoded image bytes of a data URI."""
    return hashlib.sha256(parse_data_uri(ref)).hexdigest()


class ImageHashCache(dict):
    """Per-request memo: image data URI -> `image_sha256`, computed on first use. Single-thread only."""

    def __missing__(self, ref: str) -> str:
        digest = self[ref] = image_sha256(ref)
        return digest


def estimate_image_tokens(width: int, height: int, min_pixels: int, max_pixels: int, factor: int = 32) -> int:
    """Estimate the vision tokens of one image without running the HF processor.

    Mirrors the `smart_resize` of Qwen-VL processors: each side is rounded to a multiple of *factor*
    (32 for Qwen3-VL, 28 for Qwen2.5-VL), then the area is clamped to [*min_pixels*, *max_pixels*].
    One token covers `factor * factor` pixels.
    """
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar * w_bar // (factor * factor)


def estimate_data_uri_tokens(ref: str, min_pixels: int, max_pixels: int) -> int:
    """Estimated vision tokens of a data URI image; only the image header is parsed, not the pixels."""
    import io

    from PIL import Image

    with Image.open(io.BytesIO(parse_data_uri(ref))) as image:
        width, height = image.size
    return estimate_image_tokens(width, height, min_pixels, max_pixels)


def _image_url_of(part: Any) -> str | None:
    """Return the url of an `image_url` content part, or None if not one."""
    if not isinstance(part, dict) or part.get("type") != "image_url":
        return None
    value = part.get("image_url")
    url = value.get("url") if isinstance(value, dict) else value
    if not isinstance(url, str):
        raise MultimodalRequestError("image_url part is missing a url")
    return url


def extract_image_refs(messages: list[dict[str, Any]], hashes: ImageHashCache | None = None) -> list[str]:
    """Ordered image data URIs found in `messages`, each validated by decoding.

    Pass `hashes` to memoize the digests during that decode.

    Raises `MultimodalRequestError` (HTTP 400) when an image url is not a base64 data URI (e.g. http(s)),
    or an image sits in a role outside `IMAGE_ALLOWED_ROLES`, which the chat template would silently drop.
    """
    refs: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            url = _image_url_of(part)
            if url is None:
                continue
            role = message.get("role")
            if role not in IMAGE_ALLOWED_ROLES:
                raise MultimodalRequestError(
                    f"images are only supported in {'/'.join(IMAGE_ALLOWED_ROLES)} messages, got role={role!r}"
                )
            if hashes is None:
                parse_data_uri(url)  # validate only
            else:
                _ = hashes[url]  # validate and memoize the digest
            refs.append(url)
    return refs


def has_images(messages: list[dict[str, Any]]) -> bool:
    """True if any message carries an `image_url` part."""
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(_image_url_of(p) is not None for p in content):
            return True
    return False


def replace_images_with_hashes(
    messages: list[dict[str, Any]], hashes: ImageHashCache | None = None
) -> list[dict[str, Any]]:
    """Copy of `messages` with image urls replaced by `sha256:<hex>`, keeping base64 out of the stored history.

    Messages without images are returned as-is. Pass `hashes` to share digests with other calls on the request.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list) or not any(_image_url_of(p) is not None for p in content):
            out.append(message)
            continue
        new_parts = []
        for part in content:
            url = _image_url_of(part)
            if url is None or url.startswith(HASH_URL_PREFIX):
                new_parts.append(part)
            else:
                digest = image_sha256(url) if hashes is None else hashes[url]
                new_parts.append({"type": "image_url", "image_url": {"url": f"{HASH_URL_PREFIX}{digest}"}})
        new_message = copy.copy(message)
        new_message["content"] = new_parts
        out.append(new_message)
    return out


# ---------------------------------------------------------------------------
# Pad-span utilities
# ---------------------------------------------------------------------------


def find_pad_spans(ids: list[int], pad_id: int) -> list[tuple[int, int]]:
    """Maximal `[start, end)` spans of consecutive `pad_id` tokens, one per image.

    E.g. with `pad_id=99`, `[10, 11, 99, 99, 99, 12, 99, 99, 13]` gives `[(2, 5), (6, 8)]`.
    Images with no separator token between them merge into one span.
    """
    spans: list[tuple[int, int]] = []
    n = len(ids)
    i = 0
    while True:
        # `list.index` skips the non-pad stretches at C speed.
        try:
            start = ids.index(pad_id, i)
        except ValueError:
            return spans
        end = start + 1
        while end < n and ids[end] == pad_id:
            end += 1
        spans.append((start, end))
        i = end


def fold_pad_spans(ids: list[int], pad_id: int) -> list[int]:
    """Collapse each pad span into one pad token (the SGLang-side form); inverse of `expand_pad_spans`.

    E.g. with `pad_id=99`, `[10, 11, 99, 99, 99, 12, 99, 99, 13]` gives `[10, 11, 99, 12, 99, 13]`.
    """
    out: list[int] = []
    prev_pad = False
    for tok in ids:
        is_pad = tok == pad_id
        if not (is_pad and prev_pad):
            out.append(tok)
        prev_pad = is_pad
    return out


def expand_pad_spans(folded_ids: list[int], pad_id: int, counts: list[int]) -> tuple[list[int], list[tuple[int, int]]]:
    """Expand each single pad of `folded_ids` to `counts[i]` pads.

    Returns `(expanded_ids, spans)`, where `spans[i]` is the `[start, end)` range of image i.
    Raises `MultimodalRequestError` if the pad spans do not match `counts` one-to-one or a folded span
    is longer than one token.
    """
    folded_spans = find_pad_spans(folded_ids, pad_id)
    if len(folded_spans) != len(counts):
        raise MultimodalRequestError(
            f"image placeholder count mismatch: {len(folded_spans)} in prompt vs {len(counts)} images"
        )
    expanded: list[int] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for (start, end), count in zip(folded_spans, counts):
        if end - start != 1:
            raise MultimodalRequestError("adjacent image placeholders without separator are not supported")
        if count <= 0:
            raise MultimodalRequestError(f"image produced a non-positive token count: {count}")
        expanded.extend(folded_ids[cursor:start])
        spans.append((len(expanded), len(expanded) + count))
        expanded.extend([pad_id] * count)
        cursor = end
    expanded.extend(folded_ids[cursor:])
    return expanded, spans


# ---------------------------------------------------------------------------
# Processed image + cache
# ---------------------------------------------------------------------------


@dataclass
class ProcessedImage:
    """One preprocessed image.

    `tensors` holds the CPU tensors of the image (Qwen: `pixel_values`, `image_grid_thw`; V4.1: `pixel_values`,
    `image_grid_hw`). `png_data_uri` is the data URI sent to SGLang as `image_data`: for Qwen the lossless PNG
    of the pre-resized image, for V4.1 the original data URI (SGLang resizes it itself).
    """
    image_hash: str
    png_data_uri: str
    tensors: dict[str, Any]
    num_tokens: int
    nbytes: int = field(default=0)


def _tensor_nbytes(t: Any) -> int:
    nbytes = getattr(t, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    return int(t.element_size() * t.nelement())


def processed_image_nbytes(png_data_uri: str, tensors: dict[str, Any]) -> int:
    return len(png_data_uri) + sum(_tensor_nbytes(t) for t in tensors.values())


class ImageCache:
    """Thread-safe byte-capped LRU of `ProcessedImage`, keyed by `(image_hash, min_pixels, max_pixels)`.

    Entries are shared by reference with trajectories, so eviction only drops the cache's reference.
    """

    def __init__(self, max_bytes: int):
        import threading

        self._max_bytes = int(max_bytes)
        self._entries: OrderedDict[tuple, ProcessedImage] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    @property
    def total_bytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: tuple) -> ProcessedImage | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    def put(self, key: tuple, entry: ProcessedImage) -> None:
        if entry.nbytes <= 0:
            entry.nbytes = processed_image_nbytes(entry.png_data_uri, entry.tensors)
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            if entry.nbytes > self._max_bytes:
                return
            self._entries[key] = entry
            self._bytes += entry.nbytes
            while self._bytes > self._max_bytes and self._entries:
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= evicted.nbytes


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class _ImagePipeline:
    """Cache, in-flight dedup and thread-pool plumbing shared by the image processors.

    Subclasses set `cache`, `_executor` and `_inflight`, and implement `_cache_key` and `_process_uncached`.
    """

    cache: ImageCache
    _executor: Any
    _inflight: dict[str, Any]

    def _cache_key(self, image_hash: str) -> tuple:
        raise NotImplementedError

    def _process_uncached(self, ref: str, image_hash: str) -> ProcessedImage:
        raise NotImplementedError

    def process_image(self, ref: str, image_hash: str | None = None) -> ProcessedImage:
        """Preprocess one data-URI image synchronously (cache aware).

        Pass `image_hash` (`image_sha256(ref)`) if already known, to skip decoding the image again.
        """
        if image_hash is None:
            image_hash = image_sha256(ref)
        key = self._cache_key(image_hash)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        try:
            entry = self._process_uncached(ref, image_hash)
        except (MultimodalRequestError, ImportError, MemoryError):
            raise  # server-side faults must not be reported to the client as a 400
        except Exception as exc:
            logger.warning("image processing failed: %s", exc, exc_info=True)
            raise MultimodalRequestError(f"failed to process image: {exc}") from exc
        self.cache.put(key, entry)
        return entry

    def _forget_inflight(self, ref: str, fut: Any) -> None:
        if self._inflight.get(ref) is fut:
            del self._inflight[ref]

    async def prepare(self, refs: list[str], hashes: ImageHashCache | None = None) -> list[ProcessedImage]:
        """Preprocess `refs` concurrently on the processor's own thread pool.

        Call from the event loop thread only (the `hashes` memo is read here). Concurrent calls for the
        same data URI, e.g. the n samples of one prompt, share one in-flight job.
        """
        import asyncio

        if not refs:
            return []
        loop = asyncio.get_running_loop()
        futures: dict[str, asyncio.Future] = {}
        for ref in refs:
            if ref in futures:
                continue
            fut = self._inflight.get(ref)
            if fut is None:
                digest = None if hashes is None else hashes[ref]
                fut = loop.run_in_executor(self._executor, self.process_image, ref, digest)
                self._inflight[ref] = fut
                fut.add_done_callback(partial(self._forget_inflight, ref))
            futures[ref] = fut
        # shield: one cancelled request must not cancel the job other requests are waiting on.
        results = await asyncio.gather(*(asyncio.shield(f) for f in futures.values()))
        by_ref = dict(zip(futures, results))
        return [by_ref[r] for r in refs]


class MultimodalProcessor(_ImagePipeline):
    """Pre-resizes images and runs the HF image processor (`hf_processor`, e.g. Qwen3-VL's) on the Router side."""

    def __init__(
        self,
        hf_processor: Any,
        tokenizer: Any,
        *,
        min_pixels: int,
        max_pixels: int,
        cache_max_bytes: int = DEFAULT_IMAGE_CACHE_MAX_BYTES,
        num_workers: int = 4,
    ):
        from concurrent.futures import ThreadPoolExecutor

        from coda.utils.multimodal_config import validate_pixel_bounds

        image_processor = hf_processor.image_processor
        size = getattr(image_processor, "size", None) or {}
        shortest, longest = size.get("shortest_edge"), size.get("longest_edge")
        if shortest is not None and longest is not None:
            validate_pixel_bounds(int(min_pixels), int(max_pixels), int(shortest), int(longest))

        self.hf_processor = hf_processor
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.patch_size = int(image_processor.patch_size)
        self.merge_size = int(image_processor.merge_size)

        pad_id = getattr(hf_processor, "image_token_id", None)
        if pad_id is None:
            pad_id = tokenizer.convert_tokens_to_ids(hf_processor.image_token)
        self.image_pad_id = int(pad_id)

        self.cache = ImageCache(cache_max_bytes)
        self._executor = ThreadPoolExecutor(max_workers=max(1, num_workers), thread_name_prefix="mm-proc")
        # data URI -> in-flight `process_image` future; only touched from the event loop thread.
        self._inflight: dict[str, Any] = {}

    def _cache_key(self, image_hash: str) -> tuple:
        return (image_hash, self.min_pixels, self.max_pixels)

    def _process_uncached(self, ref: str, image_hash: str) -> ProcessedImage:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ref, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels}
                ],
            }
        ]
        images = process_vision_info(messages, image_patch_size=self.patch_size)[0]
        if not images:
            raise MultimodalRequestError("image could not be decoded")
        image = images[0]

        out = self.hf_processor.image_processor(images=[image], return_tensors="pt")
        pixel_values, grid_thw = out["pixel_values"], out["image_grid_thw"]
        num_tokens = int(grid_thw[0].prod()) // (self.merge_size**2)

        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        png_data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        tensors = {"pixel_values": pixel_values, "image_grid_thw": grid_thw}
        return ProcessedImage(
            image_hash=image_hash,
            png_data_uri=png_data_uri,
            tensors=tensors,
            num_tokens=num_tokens,
            nbytes=processed_image_nbytes(png_data_uri, tensors),
        )

    @staticmethod
    def _load_pils(images: list[ProcessedImage]) -> list[Any]:
        """Decode the PNG data URIs of `images` back to RGB PIL images."""
        import io

        from PIL import Image

        pils = []
        for image in images:
            with Image.open(io.BytesIO(parse_data_uri(image.png_data_uri))) as handle:
                pils.append(handle.convert("RGB"))
        return pils

    def expand_with_processor(
        self, folded_ids: list[int], images: list[ProcessedImage]
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """Expanded ids of `folded_ids` as produced by the HF processor itself (used only by `verify_expansion`).

        `folded_ids` holds one image pad per entry of `images`. The ids are decoded to text and run through
        `hf_processor` with the pre-resized images. Returns `(expanded_ids, spans)`.

        Raises:
            MultimodalRequestError: The processor output does not fold back to `folded_ids`, or its pad
                layout or image grids disagree with the cached ones.
        """
        if not images:
            return expand_pad_spans(folded_ids, self.image_pad_id, [])

        import torch

        pils = self._load_pils(images)
        text = self.hf_processor.tokenizer.decode(
            folded_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        out = self.hf_processor(text=[text], images=pils, return_tensors="pt")

        expanded = out["input_ids"][0].tolist()
        if fold_pad_spans(expanded, self.image_pad_id) != folded_ids:
            raise MultimodalRequestError(
                "processor-expanded ids do not fold back to the tokenizer ids "
                f"({len(expanded)} expanded vs {len(folded_ids)} folded tokens)"
            )

        expected_grid = torch.cat([image.tensors["image_grid_thw"] for image in images], dim=0)
        if not torch.equal(out["image_grid_thw"], expected_grid):
            raise MultimodalRequestError("processor image_grid_thw differs from the cached image grids")

        spans = find_pad_spans(expanded, self.image_pad_id)
        if len(spans) != len(images):
            raise MultimodalRequestError(
                f"image placeholder count mismatch: {len(spans)} in processor output vs {len(images)} images"
            )
        return expanded, spans

    def verify_expansion(self) -> None:
        """Startup self-check that `expand_pad_spans` matches the HF processor's own expansion.

        Runs the processor once on synthetic images (min and max pixel bounds, plus a wide aspect ratio)
        and requires both ways of expanding to give identical ids and spans.

        Raises:
            RuntimeError: The two disagree, i.e. this model's pad layout is not what the Router assumes.
        """
        import io
        import math

        from PIL import Image

        wide_h = max(1, math.isqrt(math.isqrt(self.min_pixels * self.max_pixels) // 3))
        sizes = [
            (math.isqrt(self.min_pixels),) * 2,
            (math.isqrt(self.max_pixels),) * 2,
            (3 * wide_h, wide_h),
        ]
        images = []
        for i, (width, height) in enumerate(sizes):
            buf = io.BytesIO()
            Image.new("RGB", (max(1, width), max(1, height)), (40 * i + 20, 90, 200 - 40 * i)).save(buf, format="PNG")
            ref = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
            images.append(self._process_uncached(ref, image_sha256(ref)))  # bypass the cache: probes stay out of it

        tokenizer = self.hf_processor.tokenizer
        words = [tokenizer.encode(w, add_special_tokens=False) for w in ("look", "then", "compare", "end")]
        folded: list[int] = [*words[0]]
        for word_ids in words[1:]:
            folded += [self.image_pad_id, *word_ids]  # one pad per image, each followed by text

        try:
            actual = self.expand_with_processor(folded, images)
        except MultimodalRequestError as exc:
            raise RuntimeError(f"multimodal startup check failed: HF processor rejected the probe: {exc}") from exc
        expected = expand_pad_spans(folded, self.image_pad_id, [image.num_tokens for image in images])
        if actual != expected:
            raise RuntimeError(
                "multimodal startup check failed: HF processor expansion differs from num_tokens-based expansion "
                f"(spans {actual[1]} vs {expected[1]}, {len(actual[0])} vs {len(expected[0])} tokens); "
                "this model's pad layout is unsupported"
            )


# ---------------------------------------------------------------------------
# DeepSeek-V4.1 image processor
# ---------------------------------------------------------------------------

_DSV41_VISION_KEYS = (
    "vision_patch_size",
    "vision_downsample_ratio",
    "vision_max_n_token",
    "vision_min_pixels",
    "vision_max_wh_ratio",
)


def load_dsv41_vision_args(model_path: str, overrides: dict[str, Any] | None = None) -> Any:
    """Vision preprocessing args of a V4.1 checkpoint, in the attribute form SGLang's image helpers expect.

    `overrides` is the rollout's `json_model_override_args`; its `vision_*` entries win over `config.json`,
    so the Router plans images exactly like the SGLang server does.
    """
    import json
    import os
    from types import SimpleNamespace

    from sglang.srt.configs.deepseek_v41 import normalize_deepseek_v41_config

    with open(os.path.join(model_path, "config.json")) as f:
        values = normalize_deepseek_v41_config(json.load(f))
    for key, value in (overrides or {}).items():
        if key.startswith("vision_"):
            values[key] = value
    missing = [k for k in _DSV41_VISION_KEYS if k not in values and k != "vision_max_wh_ratio"]
    if missing:
        raise ValueError(f"{model_path}/config.json has no vision config (missing {missing})")
    values.setdefault("vision_max_wh_ratio", None)
    return SimpleNamespace(**{k: values[k] for k in _DSV41_VISION_KEYS})


def dsv41_image_token_estimator(vision_args: Any):
    """Returns `ref -> token count`, reading only the image header (no pixel decode)."""

    def estimate(ref: str) -> int:
        import io

        from PIL import Image
        from sglang.srt.multimodal.deepseek_v41_image_processing import num_image_tokens, plan_image_grid

        with Image.open(io.BytesIO(parse_data_uri(ref))) as image:
            width, height = image.size
        n_llm_h, n_llm_w, _, _ = plan_image_grid(width, height, vision_args)
        return num_image_tokens(n_llm_h, n_llm_w)

    return estimate


class Dsv41ImageProcessor(_ImagePipeline):
    """Router-side image preprocessing for DeepSeek-V4.1, using SGLang's own patchify code.

    Stands alone (not a `MultimodalProcessor`): there is no HF processor, and the image resize plan is
    fixed by the checkpoint's `vision_*` config. `ProcessedImage.png_data_uri` is the original data URI
    (SGLang decodes and patchifies it again); `tensors` holds `pixel_values` (bf16 `[N, 3, p, p]`) and
    `image_grid_hw` (`[[n_vit_h, n_vit_w]]`).
    """

    def __init__(
        self,
        vision_args: Any,
        image_token_id: int,
        *,
        cache_max_bytes: int = DEFAULT_IMAGE_CACHE_MAX_BYTES,
        num_workers: int = 4,
    ):
        from concurrent.futures import ThreadPoolExecutor

        self.vision_args = vision_args
        self.image_pad_id = int(image_token_id)
        self.patch_size = int(vision_args.vision_patch_size)
        self.merge_size = int(vision_args.vision_downsample_ratio)
        self.min_pixels = int(vision_args.vision_min_pixels)
        self.max_n_token = int(vision_args.vision_max_n_token)

        self.cache = ImageCache(cache_max_bytes)
        self._executor = ThreadPoolExecutor(max_workers=max(1, num_workers), thread_name_prefix="mm-proc")
        self._inflight: dict[str, Any] = {}

    def _cache_key(self, image_hash: str) -> tuple:
        return (image_hash,)

    def _process_uncached(self, ref: str, image_hash: str) -> ProcessedImage:
        import io

        import torch
        from PIL import Image
        from sglang.srt.multimodal.deepseek_v41_image_processing import num_image_tokens, patchify_image, to_rgb

        with Image.open(io.BytesIO(parse_data_uri(ref))) as handle:
            image = to_rgb(handle)
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = patchify_image(image, self.vision_args)
        tensors = {
            "pixel_values": patches,
            "image_grid_hw": torch.tensor([[n_vit_h, n_vit_w]], dtype=torch.int64),
        }
        return ProcessedImage(
            image_hash=image_hash,
            png_data_uri=ref,
            tensors=tensors,
            num_tokens=num_image_tokens(n_llm_h, n_llm_w),
            nbytes=processed_image_nbytes(ref, tensors),
        )
