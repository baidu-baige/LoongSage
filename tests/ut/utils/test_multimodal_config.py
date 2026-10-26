"""Unit tests for coda.utils.multimodal_config."""

import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from coda.utils.multimodal_config import (
    multimodal_enabled,
    validate_multimodal_config,
    validate_pixel_bounds,
    validate_vlm_parallel_config,
)


def _config(*, enable=True, modalities=("image",), image=None, keys=None, sglang_args=None):
    image = {"min_pixels": 65536, "max_pixels": 1048576} if image is None else image
    return OmegaConf.create(
        {
            "multimodal": {"enable": enable, "modalities": list(modalities), "image": image},
            "data_sources": [{"dataset": {"multimodal_keys": keys}}],
            "rollout": {"sglang_args": dict(sglang_args or {})},
        }
    )


class TestMultimodalEnabled(unittest.TestCase):
    def test_missing_block(self):
        self.assertFalse(multimodal_enabled(OmegaConf.create({})))

    def test_flags(self):
        self.assertTrue(multimodal_enabled(_config(enable=True)))
        self.assertFalse(multimodal_enabled(_config(enable=False)))


class TestValidateDisabled(unittest.TestCase):
    def test_disabled_without_keys_ok(self):
        cfg = _config(enable=False)
        self.assertIs(validate_multimodal_config(cfg), cfg)
        self.assertNotIn("enable_multimodal", cfg.rollout.sglang_args)

    def test_disabled_with_keys_raises(self):
        with self.assertRaisesRegex(ValueError, "multimodal.enable is false"):
            validate_multimodal_config(_config(enable=False, keys={"image": "images"}))


class TestValidateEnabled(unittest.TestCase):
    def test_links_sglang_enable_multimodal(self):
        cfg = _config(keys={"image": "images"})
        validate_multimodal_config(cfg)
        self.assertTrue(cfg.rollout.sglang_args.enable_multimodal)

    def test_explicit_true_is_kept(self):
        cfg = _config(sglang_args={"enable_multimodal": True})
        validate_multimodal_config(cfg)
        self.assertTrue(cfg.rollout.sglang_args.enable_multimodal)

    def test_explicit_false_raises(self):
        with self.assertRaisesRegex(ValueError, "explicitly false"):
            validate_multimodal_config(_config(sglang_args={"enable_multimodal": False}))

    def test_empty_modalities_raises(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            validate_multimodal_config(_config(modalities=()))

    def test_unknown_modality_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown modality"):
            validate_multimodal_config(_config(modalities=("hologram",)))

    def test_reserved_modality_raises(self):
        for name in ("video", "audio"):
            with self.assertRaisesRegex(ValueError, "not implemented"):
                validate_multimodal_config(_config(modalities=(name,)))

    def test_image_block_required(self):
        cfg = _config()
        cfg.multimodal.image = None
        with self.assertRaisesRegex(ValueError, "multimodal.image is required"):
            validate_multimodal_config(cfg)

    def test_pixel_values_validated(self):
        bad = [
            {"min_pixels": 0, "max_pixels": 100},
            {"min_pixels": 200, "max_pixels": 100},
        ]
        for image in bad:
            with self.assertRaises(ValueError):
                validate_multimodal_config(_config(image=image))

    def test_keys_must_name_known_declared_modality(self):
        with self.assertRaisesRegex(ValueError, "unknown modality"):
            validate_multimodal_config(_config(keys={"hologram": "col"}))
        with self.assertRaisesRegex(ValueError, "not in multimodal.modalities"):
            validate_multimodal_config(_config(keys={"video": "col"}))

    def test_keys_column_must_be_non_empty_string(self):
        with self.assertRaisesRegex(ValueError, "must be a column name"):
            validate_multimodal_config(_config(keys={"image": ""}))


class TestPixelBounds(unittest.TestCase):
    def test_ok(self):
        validate_pixel_bounds(65536, 1048576, 65536, 16777216)

    def test_min_below_shortest_edge(self):
        with self.assertRaisesRegex(ValueError, "shortest_edge"):
            validate_pixel_bounds(1000, 1048576, 65536, 16777216)

    def test_max_above_longest_edge(self):
        with self.assertRaisesRegex(ValueError, "longest_edge"):
            validate_pixel_bounds(65536, 20000000, 65536, 16777216)


class TestVlmParallelConfig(unittest.TestCase):
    def _hf_dir(self, model_type):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        if model_type is not None:
            Path(tmp.name, "config.json").write_text(json.dumps({"model_type": model_type}))
        return tmp.name

    def test_disabled_skips_all_checks(self):
        cfg = _config(enable=False)
        model = OmegaConf.create({"context_parallel_size": 2, "cp_partition_mode": "contiguous"})
        validate_vlm_parallel_config(cfg, model, self._hf_dir("qwen2_5_vl"))

    def test_contiguous_cp_rejected(self):
        model = OmegaConf.create({"context_parallel_size": 2, "cp_partition_mode": "contiguous"})
        with self.assertRaisesRegex(ValueError, "zigzag"):
            validate_vlm_parallel_config(_config(), model, self._hf_dir("qwen3_vl"))

    def test_contiguous_with_cp1_allowed(self):
        model = OmegaConf.create({"context_parallel_size": 1, "cp_partition_mode": "contiguous"})
        validate_vlm_parallel_config(_config(), model, self._hf_dir("qwen3_vl"))

    def test_qwen25_vl_cp_rejected(self):
        model = OmegaConf.create({"context_parallel_size": 2, "cp_partition_mode": "zigzag"})
        with self.assertRaisesRegex(ValueError, "Qwen2.5-VL"):
            validate_vlm_parallel_config(_config(), model, self._hf_dir("qwen2_5_vl"))

    def test_qwen25_vl_cp1_allowed(self):
        model = OmegaConf.create({"context_parallel_size": 1})
        validate_vlm_parallel_config(_config(), model, self._hf_dir("qwen2_5_vl"))

    def test_qwen3_vl_zigzag_cp_allowed(self):
        model = OmegaConf.create({"context_parallel_size": 2, "cp_partition_mode": "zigzag"})
        validate_vlm_parallel_config(_config(), model, self._hf_dir("qwen3_vl"))

    def test_unreadable_hf_path_skips_family_check(self):
        model = OmegaConf.create({"context_parallel_size": 2, "cp_partition_mode": "zigzag"})
        validate_vlm_parallel_config(_config(), model, self._hf_dir(None))
        validate_vlm_parallel_config(_config(), model, None)

    def test_qwen3_vl_moe_warns_when_freeze_unset(self):
        model = OmegaConf.create({"context_parallel_size": 1})
        with self.assertLogs("coda.utils.multimodal_config", level="WARNING") as cm:
            validate_vlm_parallel_config(_config(), model, self._hf_dir("qwen3_vl_moe"))
        self.assertEqual(len(cm.records), 2)

    def test_qwen3_vl_moe_no_warning_when_freeze_explicit(self):
        model = OmegaConf.create(
            {"context_parallel_size": 1, "freeze_language_model": False, "freeze_vision_model": True}
        )
        with self.assertNoLogs("coda.utils.multimodal_config", level="WARNING"):
            validate_vlm_parallel_config(_config(), model, self._hf_dir("qwen3_vl_moe"))


if __name__ == "__main__":
    unittest.main()
