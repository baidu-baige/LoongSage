"""Unit tests for coda.agentflow.multimodal (pure helpers, cache and processor init)."""

import base64
from types import SimpleNamespace

import pytest
import torch

from coda.agentflow.multimodal import (
    ImageCache,
    ImageHashCache,
    MultimodalProcessor,
    MultimodalRequestError,
    ProcessedImage,
    expand_pad_spans,
    extract_image_refs,
    find_pad_spans,
    fold_pad_spans,
    has_images,
    image_sha256,
    parse_data_uri,
    replace_images_with_hashes,
)

PAD = 9


def _uri(payload: bytes = b"abc") -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode()


def _img_msg(role: str, *uris: str, text: str = "hi") -> dict:
    parts = [{"type": "text", "text": text}]
    parts += [{"type": "image_url", "image_url": {"url": u}} for u in uris]
    return {"role": role, "content": parts}


class TestDataUri:
    def test_roundtrip_and_hash(self):
        assert parse_data_uri(_uri(b"xyz")) == b"xyz"
        assert image_sha256(_uri(b"xyz")) == image_sha256(_uri(b"xyz"))
        assert image_sha256(_uri(b"xyz")) != image_sha256(_uri(b"xyw"))

    @pytest.mark.parametrize("ref", ["/tmp/a.png", "http://x/a.png", "data:image/png,abc", "data:image/png;base64,@@@"])
    def test_rejects_non_data_or_bad(self, ref):
        with pytest.raises(MultimodalRequestError):
            parse_data_uri(ref)


class TestExtract:
    def test_ordered_refs_user_and_tool(self):
        a, b = _uri(b"a"), _uri(b"b")
        msgs = [{"role": "system", "content": "s"}, _img_msg("user", a), _img_msg("tool", b)]
        assert extract_image_refs(msgs) == [a, b]

    @pytest.mark.parametrize("role", ["system", "assistant"])
    def test_rejects_disallowed_roles(self, role):
        with pytest.raises(MultimodalRequestError):
            extract_image_refs([_img_msg(role, _uri())])

    def test_rejects_non_data_ref(self):
        with pytest.raises(MultimodalRequestError):
            extract_image_refs([_img_msg("user", "http://x/a.png")])

    def test_string_content_ignored(self):
        assert extract_image_refs([{"role": "user", "content": "plain"}]) == []
        assert not has_images([{"role": "user", "content": "plain"}])
        assert has_images([_img_msg("user", _uri())])


class TestReplaceWithHashes:
    def test_replaces_and_does_not_mutate(self):
        uri = _uri(b"big" * 100)
        msgs = [{"role": "user", "content": "plain"}, _img_msg("user", uri)]
        out = replace_images_with_hashes(msgs)
        assert out[0] is msgs[0]
        assert out[1]["content"][1]["image_url"]["url"] == f"sha256:{image_sha256(uri)}"
        assert msgs[1]["content"][1]["image_url"]["url"] == uri

    def test_same_image_same_form(self):
        a = replace_images_with_hashes([_img_msg("user", _uri(b"q"))])
        b = replace_images_with_hashes([_img_msg("user", _uri(b"q"))])
        assert a == b


class TestImageHashCache:
    def test_hashes_each_ref_once(self, monkeypatch):
        calls = []
        real = image_sha256
        monkeypatch.setattr("coda.agentflow.multimodal.image_sha256", lambda ref: calls.append(ref) or real(ref))
        hashes = ImageHashCache()
        ref = _uri(b"once")

        assert hashes[ref] == real(ref) and hashes[ref] == real(ref)
        assert calls == [ref]

    def test_bad_ref_raises_and_is_not_cached(self):
        hashes = ImageHashCache()
        with pytest.raises(MultimodalRequestError):
            hashes["http://x/a.png"]
        assert len(hashes) == 0

    def test_helpers_share_one_memo(self, monkeypatch):
        calls = []
        real = image_sha256
        monkeypatch.setattr("coda.agentflow.multimodal.image_sha256", lambda ref: calls.append(ref) or real(ref))
        uri = _uri(b"shared")
        msgs = [_img_msg("user", uri)]
        hashes = ImageHashCache()

        extract_image_refs(msgs, hashes)
        out = replace_images_with_hashes(msgs, hashes)
        extract_image_refs(msgs, hashes)

        assert calls == [uri]
        assert out[0]["content"][1]["image_url"]["url"] == f"sha256:{real(uri)}"


class TestPadRuns:
    def test_find_and_fold(self):
        ids = [1, PAD, PAD, PAD, 2, 3, PAD, PAD, 4]
        assert find_pad_spans(ids, PAD) == [(1, 4), (6, 8)]
        assert fold_pad_spans(ids, PAD) == [1, PAD, 2, 3, PAD, 4]
        assert find_pad_spans([], PAD) == []
        assert find_pad_spans([PAD], PAD) == [(0, 1)]

    def test_expand_roundtrip(self):
        folded = [1, PAD, 2, 3, PAD, 4]
        expanded, spans = expand_pad_spans(folded, PAD, [3, 2])
        assert expanded == [1, PAD, PAD, PAD, 2, 3, PAD, PAD, 4]
        assert spans == [(1, 4), (6, 8)]
        assert fold_pad_spans(expanded, PAD) == folded
        assert find_pad_spans(expanded, PAD) == spans

    def test_expand_no_images(self):
        assert expand_pad_spans([1, 2], PAD, []) == ([1, 2], [])

    def test_expand_count_mismatch(self):
        with pytest.raises(MultimodalRequestError):
            expand_pad_spans([1, PAD, 2], PAD, [1, 1])
        with pytest.raises(MultimodalRequestError):
            expand_pad_spans([1, 2], PAD, [1])

    def test_expand_rejects_adjacent_and_nonpositive(self):
        with pytest.raises(MultimodalRequestError):
            expand_pad_spans([1, PAD, PAD, 2], PAD, [2])
        with pytest.raises(MultimodalRequestError):
            expand_pad_spans([1, PAD, 2], PAD, [0])


def _entry(h: str, size: int) -> ProcessedImage:
    return ProcessedImage(image_hash=h, png_data_uri="x" * size, tensors={"t": torch.zeros(0)}, num_tokens=1)


class TestImageCache:
    def test_lru_eviction_by_bytes(self):
        cache = ImageCache(max_bytes=100)
        cache.put("a", _entry("a", 40))
        cache.put("b", _entry("b", 40))
        assert cache.get("a") is not None  # refresh a; b is now LRU
        cache.put("c", _entry("c", 40))
        assert cache.get("b") is None
        assert cache.get("a") is not None and cache.get("c") is not None
        assert cache.total_bytes == 80 and len(cache) == 2

    def test_oversized_entry_not_cached(self):
        cache = ImageCache(max_bytes=10)
        cache.put("a", _entry("a", 50))
        assert len(cache) == 0 and cache.total_bytes == 0

    def test_replace_same_key_updates_bytes(self):
        cache = ImageCache(max_bytes=100)
        cache.put("a", _entry("a", 40))
        cache.put("a", _entry("a", 10))
        assert cache.total_bytes == 10 and len(cache) == 1


class _FakeImageProcessor:
    def __init__(self, shortest=65536, longest=16777216):
        self.size = {"shortest_edge": shortest, "longest_edge": longest}
        self.patch_size = 16
        self.merge_size = 2


class TestProcessorInit:
    def test_pad_id_from_processor(self):
        hf = SimpleNamespace(image_processor=_FakeImageProcessor(), image_token_id=151655)
        proc = MultimodalProcessor(hf, None, min_pixels=65536, max_pixels=1048576, cache_max_bytes=0)
        assert (proc.image_pad_id, proc.patch_size, proc.merge_size) == (151655, 16, 2)

    def test_pad_id_from_tokenizer_fallback(self):
        hf = SimpleNamespace(image_processor=_FakeImageProcessor(), image_token="<|image_pad|>")
        tok = SimpleNamespace(convert_tokens_to_ids=lambda t: 7 if t == "<|image_pad|>" else -1)
        proc = MultimodalProcessor(hf, tok, min_pixels=65536, max_pixels=1048576, cache_max_bytes=0)
        assert proc.image_pad_id == 7

    def test_bounds_validated_against_processor(self):
        hf = SimpleNamespace(image_processor=_FakeImageProcessor(), image_token_id=1)
        with pytest.raises(ValueError):
            MultimodalProcessor(hf, None, min_pixels=1024, max_pixels=1048576, cache_max_bytes=0)
        with pytest.raises(ValueError):
            MultimodalProcessor(hf, None, min_pixels=65536, max_pixels=1 << 30, cache_max_bytes=0)


class _FakeHfProcessor:
    """Mimics HF processor expansion: each image pad becomes ``counts[i]`` pads (tokens are ints)."""

    def __init__(self, counts, grids=None):
        self.image_processor = _FakeImageProcessor()
        self.image_token_id = PAD
        self.counts = counts
        self.grids = grids
        self.tokenizer = SimpleNamespace(decode=lambda ids, **kw: ids)  # "text" is the id list itself
        self.calls = []

    def __call__(self, text, images, return_tensors):
        self.calls.append((text, images))
        expanded, _ = expand_pad_spans(text[0], PAD, self.counts)
        grids = self.grids if self.grids is not None else [[1, 2, 2 * c] for c in self.counts]
        return {"input_ids": torch.tensor([expanded]), "image_grid_thw": torch.tensor(grids)}


def _processed(n: int) -> ProcessedImage:
    grid = torch.tensor([[1, 2, 2 * n]])
    return ProcessedImage("h", "png", {"pixel_values": torch.zeros(n, 3), "image_grid_thw": grid}, n)


def _processor(hf, monkeypatch) -> MultimodalProcessor:
    monkeypatch.setattr(MultimodalProcessor, "_load_pils", staticmethod(lambda images: ["pil"] * len(images)))
    return MultimodalProcessor(hf, None, min_pixels=65536, max_pixels=1048576, cache_max_bytes=0)


class TestExpandWithProcessor:
    def test_ids_come_from_processor(self, monkeypatch):
        hf = _FakeHfProcessor([3, 2])
        proc = _processor(hf, monkeypatch)
        folded = [1, 50, PAD, 51, 7, 50, PAD, 51, 5]

        expanded, spans = proc.expand_with_processor(folded, [_processed(3), _processed(2)])

        assert expanded == [1, 50, PAD, PAD, PAD, 51, 7, 50, PAD, PAD, 51, 5]
        assert spans == [(2, 5), (8, 10)]
        assert hf.calls[0][0] == [folded] and len(hf.calls[0][1]) == 2

    def test_no_images_skips_processor(self, monkeypatch):
        hf = _FakeHfProcessor([])
        proc = _processor(hf, monkeypatch)

        assert proc.expand_with_processor([1, 2, 3], []) == ([1, 2, 3], [])
        assert hf.calls == []

    def test_no_images_but_pad_in_ids_raises(self, monkeypatch):
        proc = _processor(_FakeHfProcessor([]), monkeypatch)
        with pytest.raises(MultimodalRequestError):
            proc.expand_with_processor([1, PAD, 2], [])

    def test_fold_mismatch_raises(self, monkeypatch):
        hf = _FakeHfProcessor([2])
        hf.tokenizer = SimpleNamespace(decode=lambda ids, **kw: [*ids, 99])  # round trip drifts
        proc = _processor(hf, monkeypatch)
        with pytest.raises(MultimodalRequestError, match="fold back"):
            proc.expand_with_processor([1, PAD, 2], [_processed(2)])

    def test_grid_mismatch_raises(self, monkeypatch):
        hf = _FakeHfProcessor([2], grids=[[1, 2, 8]])
        proc = _processor(hf, monkeypatch)
        with pytest.raises(MultimodalRequestError, match="image_grid_thw"):
            proc.expand_with_processor([1, PAD, 2], [_processed(2)])

class TestVerifyExpansion:
    """Startup check that the HF processor expands pads exactly like `expand_pad_spans`."""

    def _proc(self, monkeypatch, counts, grids=None, tokens=(2, 3, 4)):
        hf = _FakeHfProcessor(counts, grids=grids)
        hf.tokenizer = SimpleNamespace(
            decode=lambda ids, **kw: ids,
            encode=lambda word, add_special_tokens=False: [10 + len(word)],
        )
        proc = _processor(hf, monkeypatch)
        produced = iter(tokens)
        monkeypatch.setattr(proc, "_process_uncached", lambda ref, image_hash: _processed(next(produced)))
        return hf, proc

    def test_passes_when_processor_agrees(self, monkeypatch):
        hf, proc = self._proc(monkeypatch, [2, 3, 4])
        proc.verify_expansion()
        assert len(hf.calls) == 1 and len(hf.calls[0][1]) == 3  # one probe with three images
        assert len(proc.cache) == 0  # probes bypass the image cache

    def test_fails_when_expansion_differs(self, monkeypatch):
        # grids still match the cached ones, but the processor emits more pads than num_tokens says
        _, proc = self._proc(monkeypatch, [2, 3, 5], grids=[[1, 2, 4], [1, 2, 6], [1, 2, 8]])
        with pytest.raises(RuntimeError, match="differs"):
            proc.verify_expansion()

    def test_fails_when_processor_rejects_probe(self, monkeypatch):
        _, proc = self._proc(monkeypatch, [2, 3, 4], grids=[[1, 2, 4], [1, 2, 6], [1, 2, 99]])
        with pytest.raises(RuntimeError, match="rejected"):
            proc.verify_expansion()

    def test_probe_sizes_cover_bounds(self, monkeypatch):
        _, proc = self._proc(monkeypatch, [2, 3, 4])
        sizes = []

        def capture(ref, image_hash):
            import io

            from PIL import Image

            sizes.append(Image.open(io.BytesIO(parse_data_uri(ref))).size)
            return _processed(len(sizes) + 1)

        monkeypatch.setattr(proc, "_process_uncached", capture)
        proc.verify_expansion()
        assert sizes[0] == (256, 256) and sizes[1] == (1024, 1024)  # sqrt of min / max pixels
        assert sizes[2][0] == 3 * sizes[2][1]  # wide aspect ratio


class TestPrepareInflightDedup:
    def _proc(self, monkeypatch):
        proc = _processor(_FakeHfProcessor([]), monkeypatch)
        return proc

    def test_concurrent_identical_refs_run_once(self, monkeypatch):
        import asyncio
        import threading

        proc = self._proc(monkeypatch)
        gate = threading.Event()
        calls = []

        def slow_process(ref, image_hash=None):
            calls.append(ref)
            gate.wait(5)
            return _processed(2)

        monkeypatch.setattr(proc, "process_image", slow_process)

        async def run():
            tasks = [asyncio.ensure_future(proc.prepare(["a", "b", "a"])) for _ in range(8)]
            await asyncio.sleep(0.05)  # every request has joined the in-flight jobs
            gate.set()
            return await asyncio.gather(*tasks)

        results = asyncio.run(run())
        assert sorted(calls) == ["a", "b"]
        assert all(len(r) == 3 and r[0] is r[2] for r in results)
        assert proc._inflight == {}

    def test_failure_is_shared_and_not_sticky(self, monkeypatch):
        import asyncio

        proc = self._proc(monkeypatch)
        attempts = []

        def flaky(ref, image_hash=None):
            attempts.append(ref)
            if len(attempts) == 1:
                raise MultimodalRequestError("boom")
            return _processed(2)

        monkeypatch.setattr(proc, "process_image", flaky)

        async def run():
            with pytest.raises(MultimodalRequestError):
                await proc.prepare(["a"])
            return await proc.prepare(["a"])

        assert len(asyncio.run(run())) == 1
        assert len(attempts) == 2
        assert proc._inflight == {}

    def test_cancelled_waiter_does_not_cancel_shared_job(self, monkeypatch):
        import asyncio
        import threading

        proc = self._proc(monkeypatch)
        gate = threading.Event()

        def slow_process(ref, image_hash=None):
            gate.wait(5)
            return _processed(2)

        monkeypatch.setattr(proc, "process_image", slow_process)

        async def run():
            first = asyncio.ensure_future(proc.prepare(["a"]))
            second = asyncio.ensure_future(proc.prepare(["a"]))
            await asyncio.sleep(0.05)
            first.cancel()
            gate.set()
            return await second

        assert len(asyncio.run(run())) == 1


class TestProcessImageErrors:
    def _proc(self, monkeypatch, exc):
        proc = _processor(_FakeHfProcessor([]), monkeypatch)

        def boom(ref, image_hash):
            raise exc

        monkeypatch.setattr(proc, "_process_uncached", boom)
        return proc

    @pytest.mark.parametrize("exc", [ValueError("bad"), OSError("cannot identify image"), RuntimeError("x")])
    def test_client_side_failures_become_request_errors(self, monkeypatch, exc):
        proc = self._proc(monkeypatch, exc)
        with pytest.raises(MultimodalRequestError, match="failed to process image"):
            proc.process_image(_uri(b"x"))

    @pytest.mark.parametrize("exc", [MemoryError(), ImportError("qwen_vl_utils")])
    def test_server_side_failures_propagate_unchanged(self, monkeypatch, exc):
        proc = self._proc(monkeypatch, exc)
        with pytest.raises(type(exc)) as info:
            proc.process_image(_uri(b"x"))
        assert not isinstance(info.value, MultimodalRequestError)


class TestProcessImageWithHash:
    def test_cache_hit_with_hash_does_not_decode(self, monkeypatch):
        proc = _processor(_FakeHfProcessor([]), monkeypatch)
        proc.cache = ImageCache(max_bytes=1 << 20)
        entry = _processed(2)
        proc.cache.put(("h", proc.min_pixels, proc.max_pixels), entry)

        def boom(ref):
            raise AssertionError("must not hash or decode when the hash is supplied")

        monkeypatch.setattr("coda.agentflow.multimodal.image_sha256", boom)
        assert proc.process_image("ignored", "h") is entry

    def test_prepare_passes_memoised_hashes_to_workers(self, monkeypatch):
        import asyncio

        proc = _processor(_FakeHfProcessor([]), monkeypatch)
        seen = []
        monkeypatch.setattr(proc, "process_image", lambda ref, image_hash=None: seen.append((ref, image_hash)) or _processed(2))
        ref = _uri(b"p")
        hashes = ImageHashCache()

        asyncio.run(proc.prepare([ref, ref], hashes))

        assert seen == [(ref, image_sha256(ref))]


class TestBuildMultimodalProcessorDeps:
    def test_disabled_does_not_require_qwen_vl_utils(self, monkeypatch):
        import sys

        from coda.agentflow.tokenizer_manager import build_multimodal_processor

        monkeypatch.setitem(sys.modules, "qwen_vl_utils", None)  # makes `import qwen_vl_utils` raise ImportError
        cfg = SimpleNamespace(enable=False)
        assert build_multimodal_processor(cfg, "/unused", tokenizer=None, num_workers=1) is None

    def test_enabled_without_qwen_vl_utils_fails_at_startup(self, monkeypatch):
        import sys

        from coda.agentflow.tokenizer_manager import build_multimodal_processor

        monkeypatch.setitem(sys.modules, "qwen_vl_utils", None)
        cfg = SimpleNamespace(enable=True, image=SimpleNamespace(min_pixels=1, max_pixels=2))
        with pytest.raises(ImportError, match="qwen-vl-utils"):
            build_multimodal_processor(cfg, "/unused", tokenizer=None, num_workers=1)
