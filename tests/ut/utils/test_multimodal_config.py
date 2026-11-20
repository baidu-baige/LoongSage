"""Unit tests for coda.utils.multimodal_config."""

import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from coda.utils.hf_config import read_image_pixel_bounds
from coda.utils.multimodal_config import (
    validate_multimodal_config,
    validate_vlm_parallel_config,
)


def _config(*, enable=True, modalities=("image",), keys=None, sglang_args=None):
    return OmegaConf.create(
        {
            "multimodal": {"enable": enable, "modalities": list(modalities)},
            "data_sources": [{"dataset": {"multimodal_keys": keys}}],
            "rollout": {"sglang_args": dict(sglang_args or {})},
        }
    )


class TestValidateDisabled(unittest.TestCase):
    def test_disabled_without_keys_ok(self):
        cfg = _config(enable=False)
        self.assertIs(validate_multimodal_config(cfg), cfg)

    def test_disabled_with_keys_raises(self):
        with self.assertRaisesRegex(ValueError, "multimodal.enable is false"):
            validate_multimodal_config(_config(enable=False, keys={"image": "images"}))

    def test_disabled_with_max_image_tokens_raises(self):
        cfg = _config(enable=False)
        cfg.data_sources[0].dataset.max_image_tokens = 15000
        with self.assertRaisesRegex(ValueError, "max_image_tokens is set but multimodal.enable is false"):
            validate_multimodal_config(cfg)

    def test_disabled_with_null_max_image_tokens_ok(self):
        cfg = _config(enable=False)
        cfg.data_sources[0].dataset.max_image_tokens = None
        self.assertIs(validate_multimodal_config(cfg), cfg)


class TestValidateEnabled(unittest.TestCase):
    def test_explicit_true_ok(self):
        cfg = _config(sglang_args={"enable_multimodal": True})
        self.assertIs(validate_multimodal_config(cfg), cfg)

    def test_validate_does_not_write_sglang_args(self):
        cfg = _config()
        validate_multimodal_config(cfg)
        self.assertNotIn("enable_multimodal", cfg.rollout.sglang_args)

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

    def test_keys_must_name_known_declared_modality(self):
        with self.assertRaisesRegex(ValueError, "unknown modality"):
            validate_multimodal_config(_config(keys={"hologram": "col"}))
        with self.assertRaisesRegex(ValueError, "not in multimodal.modalities"):
            validate_multimodal_config(_config(keys={"video": "col"}))

    def test_keys_column_must_be_non_empty_string(self):
        with self.assertRaisesRegex(ValueError, "must be a column name"):
            validate_multimodal_config(_config(keys={"image": ""}))

    def test_max_image_tokens_needs_image_column(self):
        for keys in (None, {}):
            cfg = _config(keys=keys)
            cfg.data_sources[0].dataset.max_image_tokens = 15000
            with self.assertRaisesRegex(ValueError, "declares no 'image' column"):
                validate_multimodal_config(cfg)

    def test_max_image_tokens_with_image_column_ok(self):
        cfg = _config(keys={"image": "images"})
        cfg.data_sources[0].dataset.max_image_tokens = 15000
        validate_multimodal_config(cfg)


class TestImagePixelBoundsFromModel(unittest.TestCase):
    """Image pixel bounds come from the model's own `preprocessor_config.json`."""

    @staticmethod
    def _cfg(tmp, *, model_type="qwen3_vl", size=None):
        (Path(tmp) / "config.json").write_text(json.dumps({"model_type": model_type}))
        if size is not None:
            (Path(tmp) / "preprocessor_config.json").write_text(json.dumps({"size": size}))
        cfg = _config()
        cfg.hf_model_path = tmp
        return cfg

    def test_reads_size_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._cfg(tmp, size={"shortest_edge": 65536, "longest_edge": 16777216})
            self.assertEqual(read_image_pixel_bounds(tmp), (65536, 16777216))

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "preprocessor_config.json"):
                read_image_pixel_bounds(tmp)

    def test_missing_edge_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._cfg(tmp, size={"shortest_edge": 65536})
            with self.assertRaisesRegex(ValueError, "size.longest_edge"):
                read_image_pixel_bounds(tmp)


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

    def test_contiguous_with_cp1_allowed(self):
        model = OmegaConf.create({"context_parallel_size": 1, "cp_partition_mode": "contiguous"})
        validate_vlm_parallel_config(_config(), model, self._hf_dir("qwen3_vl"))

    def test_dsv41_contiguous_cp_allowed(self):
        model = OmegaConf.create({"context_parallel_size": 2, "cp_partition_mode": "contiguous"})
        validate_vlm_parallel_config(_config(), model, self._hf_dir("deepseek_v41"))

    def test_dsv41_cp1_allowed(self):
        model = OmegaConf.create({"context_parallel_size": 1, "cp_partition_mode": "zigzag"})
        validate_vlm_parallel_config(_config(), model, self._hf_dir("deepseek_v41"))

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


if __name__ == "__main__":
    unittest.main()
