"""Unit tests for multimodal prompt building in coda.data_factory.dataset."""

import json
import os
import tempfile
import unittest

from coda.data_factory.dataset import Dataset, _build_messages

IMG_A = "data:image/png;base64,AAAA"
IMG_B = "data:image/png;base64,BBBB"
KEYS = {"image": "images"}


class TestBuildMessagesMultimodal(unittest.TestCase):
    def test_single_placeholder_expands_to_parts(self):
        data = {"text": "<image>What is this?", "images": [IMG_A]}
        prompt = _build_messages(data, "text", KEYS)
        self.assertEqual(
            prompt,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": IMG_A}},
                        {"type": "text", "text": "What is this?"},
                    ],
                }
            ],
        )

    def test_multiple_placeholders_keep_order(self):
        data = {"text": "a<image>b<image>c", "images": [IMG_A, IMG_B]}
        content = _build_messages(data, "text", KEYS)[0]["content"]
        self.assertEqual(
            content,
            [
                {"type": "text", "text": "a"},
                {"type": "image_url", "image_url": {"url": IMG_A}},
                {"type": "text", "text": "b"},
                {"type": "image_url", "image_url": {"url": IMG_B}},
                {"type": "text", "text": "c"},
            ],
        )

    def test_message_list_input_expands_each_string_content(self):
        data = {
            "text": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "<image>q"},
            ],
            "images": [IMG_A],
        }
        prompt = _build_messages(data, "text", KEYS)
        self.assertEqual(prompt[0], {"role": "system", "content": [{"type": "text", "text": "sys"}]})
        self.assertEqual(prompt[1]["content"][0], {"type": "image_url", "image_url": {"url": IMG_A}})

    def test_too_few_images_raises(self):
        with self.assertRaises(ValueError):
            _build_messages({"text": "<image><image>", "images": [IMG_A]}, "text", KEYS)

    def test_too_many_images_raises(self):
        with self.assertRaises(ValueError):
            _build_messages({"text": "<image>", "images": [IMG_A, IMG_B]}, "text", KEYS)

    def test_placeholder_without_image_column_raises(self):
        with self.assertRaises(ValueError):
            _build_messages({"text": "<image>hi"}, "text", KEYS)

    def test_non_data_uri_raises(self):
        for ref in ("/local/path.png", "https://example.com/a.png"):
            with self.assertRaises(ValueError):
                _build_messages({"text": "<image>", "images": [ref]}, "text", KEYS)

    def test_reserved_modality_raises(self):
        for name in ("video", "audio"):
            with self.assertRaises(ValueError):
                _build_messages({"text": "hi", name + "s": []}, "text", {name: name + "s"})

    def test_unknown_modality_raises(self):
        with self.assertRaises(ValueError):
            _build_messages({"text": "hi", "imgs": []}, "text", {"img": "imgs"})

    def test_undeclared_placeholder_raises(self):
        for text in ("<video>only", "<image>a<audio>", "<video>x"):
            data = {"text": text, "images": [IMG_A] if "<image>" in text else []}
            with self.assertRaises(ValueError):
                _build_messages(data, "text", KEYS)

    def test_text_only_row_is_unchanged(self):
        prompt = _build_messages({"text": "plain question"}, "text", KEYS)
        self.assertEqual(prompt, [{"role": "user", "content": "plain question"}])

    def test_no_multimodal_keys_keeps_placeholder_text(self):
        prompt = _build_messages({"text": "<image>hi", "images": [IMG_A]}, "text")
        self.assertEqual(prompt, [{"role": "user", "content": "<image>hi"}])

    def test_list_content_passes_through(self):
        parts = [{"type": "text", "text": "x"}]
        data = {"text": [{"role": "user", "content": parts}], "images": []}
        self.assertEqual(_build_messages(data, "text", KEYS), [{"role": "user", "content": parts}])

    def test_does_not_mutate_input(self):
        messages = [{"role": "user", "content": "<image>q"}]
        _build_messages({"text": messages, "images": [IMG_A]}, "text", KEYS)
        self.assertEqual(messages, [{"role": "user", "content": "<image>q"}])


class TestDatasetMultimodal(unittest.TestCase):
    def _write(self, records):
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        path = os.path.join(tmp, "data.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return path

    def test_dataset_builds_parts_prompt(self):
        path = self._write([{"q": "<image>what?", "images": [IMG_A], "a": "x"}])
        ds = Dataset(path, None, prompt_key="q", label_key="a", multimodal_keys=KEYS)
        self.assertEqual(len(ds), 1)
        content = ds[0].prompt[0]["content"]
        self.assertEqual(content[0], {"type": "image_url", "image_url": {"url": IMG_A}})

    def test_dataset_without_keys_is_text_only(self):
        path = self._write([{"q": "<image>what?", "images": [IMG_A], "a": "x"}])
        ds = Dataset(path, None, prompt_key="q", label_key="a")
        self.assertEqual(ds[0].prompt, [{"role": "user", "content": "<image>what?"}])

    def test_image_column_is_added_to_projected_columns(self):
        path = self._write([{"q": "<image>q", "images": [IMG_A]}])
        seen = {}

        def fake_read_file(p, columns=None, batch_size=None):
            seen["columns"] = columns
            seen["batch_size"] = batch_size
            yield {"q": "<image>q", "images": [IMG_A]}

        from unittest.mock import patch

        with patch("coda.data_factory.dataset.read_file", fake_read_file):
            Dataset(path, None, prompt_key="q", multimodal_keys=KEYS)
        self.assertIn("images", seen["columns"])
        self.assertEqual(seen["batch_size"], 32)

    def test_text_only_dataset_keeps_default_parquet_batch(self):
        path = self._write([{"q": "text", "a": "x"}])
        seen = {}

        def fake_read_file(p, columns=None, batch_size=None):
            seen["batch_size"] = batch_size
            yield {"q": "text", "a": "x"}

        from unittest.mock import patch

        with patch("coda.data_factory.dataset.read_file", fake_read_file):
            Dataset(path, None, prompt_key="q", label_key="a")
        self.assertIsNone(seen["batch_size"])

    def test_long_prompt_filter_counts_text_parts_only(self):
        path = self._write(
            [
                {"q": "<image>" + "x" * 50, "images": [IMG_A]},
                {"q": "<image>" + "x" * 5, "images": [IMG_A]},
            ]
        )
        ds = Dataset(path, 10, prompt_key="q", multimodal_keys=KEYS)
        # The 50-char text part exceeds the limit; the image url itself is not counted, so the short row is kept.
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0].prompt[0]["content"][1], {"type": "text", "text": "x" * 5})


def _png_uri(width, height):
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class TestEstimateImageTokens(unittest.TestCase):
    def test_exact_multiple_of_factor(self):
        from coda.agentflow.multimodal import estimate_image_tokens

        self.assertEqual(estimate_image_tokens(1024, 1024, 65536, 16777216), 1024)

    def test_scaled_down_to_max_pixels(self):
        from coda.agentflow.multimodal import estimate_image_tokens

        self.assertEqual(estimate_image_tokens(4096, 4096, 65536, 16777216), 16384)
        self.assertEqual(estimate_image_tokens(8192, 8192, 65536, 16777216), 16384)

    def test_scaled_up_to_min_pixels(self):
        from coda.agentflow.multimodal import estimate_image_tokens

        self.assertEqual(estimate_image_tokens(100, 100, 65536, 16777216), 64)


class TestDatasetMaxImageTokens(unittest.TestCase):
    BOUNDS = (65536, 16777216)

    def _write(self, records):
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        path = os.path.join(tmp, "data.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return path

    def _records(self):
        small, big = _png_uri(512, 512), _png_uri(2048, 2048)  # 256 and 4096 tokens
        return [
            {"q": "<image><image>a", "images": [small, small]},  # 512
            {"q": "<image>b", "images": [big]},  # 4096
            {"q": "<image><image>c", "images": [big, small]},  # 4352
        ]

    def test_drops_samples_above_cap(self):
        path = self._write(self._records())
        ds = Dataset(
            path, None, prompt_key="q", multimodal_keys=KEYS, max_image_tokens=4096, image_pixel_bounds=self.BOUNDS
        )
        self.assertEqual([len(p.prompt[0]["content"]) for p in ds.prompts], [3, 2])

    def test_disabled_keeps_everything(self):
        path = self._write(self._records())
        ds = Dataset(path, None, prompt_key="q", multimodal_keys=KEYS)
        self.assertEqual(len(ds), 3)

    def test_text_only_prompts_are_kept(self):
        path = self._write([{"q": "no image", "images": []}])
        ds = Dataset(path, None, prompt_key="q", max_image_tokens=1, image_pixel_bounds=self.BOUNDS)
        self.assertEqual(len(ds), 1)

    def test_requires_pixel_bounds(self):
        path = self._write(self._records())
        with self.assertRaises(ValueError):
            Dataset(path, None, prompt_key="q", multimodal_keys=KEYS, max_image_tokens=100)


if __name__ == "__main__":
    unittest.main()
