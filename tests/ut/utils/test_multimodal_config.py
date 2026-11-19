"""Unit tests for coda.utils.multimodal_config."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from coda.utils.hf_config import read_image_pixel_bounds
from coda.utils.multimodal_config import (
    drop_dsv41_vision_if_text_only,
    multimodal_enabled,
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

    def test_startup_validation_passes_with_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            validate_multimodal_config(self._cfg(tmp, size={"shortest_edge": 65536, "longest_edge": 16777216}))

    def test_startup_validation_fails_without_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "preprocessor_config.json"):
                validate_multimodal_config(self._cfg(tmp))

    def test_dsv41_does_not_need_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp, model_type="deepseek_v41")
            cfg.rollout.sglang_args = {"enable_multimodal": True}
            try:
                validate_multimodal_config(cfg)
            except ValueError as e:
                self.assertNotIn("preprocessor_config", str(e))


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

    def test_dsv41_contiguous_cp_allowed(self):
        model = OmegaConf.create({"context_parallel_size": 2, "cp_partition_mode": "contiguous"})
        validate_vlm_parallel_config(_config(), model, self._hf_dir("deepseek_v41"))

    def test_dsv41_zigzag_cp_rejected(self):
        model = OmegaConf.create({"context_parallel_size": 2, "cp_partition_mode": "zigzag"})
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_vlm_parallel_config(_config(), model, self._hf_dir("deepseek_v41"))

    def test_dsv41_default_mode_cp_rejected(self):
        model = OmegaConf.create({"context_parallel_size": 2})
        with self.assertRaisesRegex(ValueError, "contiguous"):
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


class TestDropDsv41VisionIfTextOnly(unittest.TestCase):
    def test_disabled_clears_vision_config(self):
        hf_config = SimpleNamespace(model_type="deepseek_v41", vision_config={"patch_size": 14})
        drop_dsv41_vision_if_text_only(hf_config, False)
        self.assertIsNone(hf_config.vision_config)

    def test_enabled_keeps_vision_config(self):
        hf_config = SimpleNamespace(model_type="deepseek_v41", vision_config={"patch_size": 14})
        drop_dsv41_vision_if_text_only(hf_config, True)
        self.assertEqual(hf_config.vision_config, {"patch_size": 14})

    def test_other_models_untouched(self):
        hf_config = SimpleNamespace(model_type="qwen3_vl", vision_config={"depth": 27})
        drop_dsv41_vision_if_text_only(hf_config, False)
        self.assertEqual(hf_config.vision_config, {"depth": 27})


if __name__ == "__main__":
    unittest.main()


class TestValidateDsv41(unittest.TestCase):
    def _cfg(self, tmp, *, sglang_args=None, model=None, vision_config=None):
        hf_config = {"model_type": "deepseek_v41"}
        hf_config["vision_config"] = {"patch_size": 14} if vision_config is None else vision_config
        (Path(tmp) / "config.json").write_text(json.dumps(hf_config))
        cfg = _config(
            keys={"image": "images"},
            sglang_args={"image_processor_backend": "pil", **(sglang_args or {})},
        )
        cfg.hf_model_path = tmp
        cfg.megatron = {"model": dict(model or {})}
        return cfg

    def _raises(self, pattern, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, pattern):
                validate_multimodal_config(self._cfg(tmp, **kwargs))

    def test_valid_config_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            validate_multimodal_config(self._cfg(tmp, sglang_args={"moe_a2a_backend": "none"}))

    def test_missing_vision_config_raises(self):
        self._raises("no vision_config", vision_config={})

    def test_vision_disabled_override_raises(self):
        self._raises("vision_n_layers", sglang_args={"json_model_override_args": '{"vision_n_layers": 0}'})

    def test_other_vision_override_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            validate_multimodal_config(
                self._cfg(tmp, sglang_args={"json_model_override_args": '{"vision_max_n_token": 512}'})
            )

    def test_rollout_parallelism_raises(self):
        self._raises("pp_size", sglang_args={"pp_size": 2})
        self._raises("attn_cp_size", sglang_args={"attn_cp_size": 2})
        self._raises("moe_a2a_backend", sglang_args={"moe_a2a_backend": "deepep"})

    def test_image_processor_backend_must_be_pil(self):
        self._raises("image_processor_backend", sglang_args={"image_processor_backend": "auto"})

    def test_router_fusion_raises(self):
        self._raises("moe_router_fusion", model={"moe_router_fusion": True})

    def test_frozen_vision_false_raises(self):
        self._raises("freeze_vision_model", model={"freeze_vision_model": False})

    def test_frozen_vision_true_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            validate_multimodal_config(self._cfg(tmp, model={"freeze_vision_model": True}))

    def test_non_dsv41_model_is_not_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
            (Path(tmp) / "preprocessor_config.json").write_text(
                json.dumps({"size": {"shortest_edge": 65536, "longest_edge": 16777216}})
            )
            cfg = _config(keys={"image": "images"}, sglang_args={"pp_size": 2})
            cfg.hf_model_path = tmp
            validate_multimodal_config(cfg)
