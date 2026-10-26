"""Unit tests for the geo3k data pre-processor."""

import base64
import unittest

from coda.data_factory.data_pre_processor import geo3k_pre_process

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 8
JPEG = b"\xff\xd8\xff" + b"0" * 8


def _b64(raw):
    return base64.b64encode(raw).decode("ascii")


class TestGeo3kPreProcess(unittest.TestCase):
    def test_hf_image_struct_to_data_uri(self):
        out = geo3k_pre_process({"problem": "<image>Find x.", "images": [{"bytes": PNG, "path": None}]}, "problem")
        self.assertEqual(out["images"], [f"data:image/png;base64,{_b64(PNG)}"])

    def test_jpeg_mime(self):
        out = geo3k_pre_process({"problem": "<image>Find x.", "images": [{"bytes": JPEG}]}, "problem")
        self.assertTrue(out["images"][0].startswith("data:image/jpeg;base64,"))

    def test_data_uri_passthrough(self):
        uri = f"data:image/png;base64,{_b64(PNG)}"
        out = geo3k_pre_process({"problem": "<image>Find x.", "images": [uri]}, "problem")
        self.assertEqual(out["images"], [uri])

    def test_prepends_missing_placeholder(self):
        out = geo3k_pre_process({"problem": "Find x.", "images": [{"bytes": PNG}]}, "problem")
        self.assertTrue(out["problem"].startswith("<image>Find x."))

    def test_keeps_existing_placeholder(self):
        out = geo3k_pre_process({"problem": "<image>Find x.", "images": [{"bytes": PNG}]}, "problem")
        self.assertEqual(out["problem"].count("<image>"), 1)

    def test_appends_boxed_instruction_once(self):
        out = geo3k_pre_process({"problem": "<image>Find x.", "images": [{"bytes": PNG}]}, "problem")
        self.assertIn("\\boxed{}", out["problem"])
        again = geo3k_pre_process(out, "problem")
        self.assertEqual(again["problem"], out["problem"])

    def test_unsupported_image_item_raises(self):
        with self.assertRaises(ValueError):
            geo3k_pre_process({"problem": "<image>x", "images": ["http://example.com/a.png"]}, "problem")


if __name__ == "__main__":
    unittest.main()
