"""CPU tests for the DeepSeek-V4.1 vision bridge pieces (mappings, provider choice, ViT, image merge).

Skipped when the patched Megatron-Bridge is not installed. The GPU model is covered by the toy-model runs.
"""

import glob
import json
import re
import struct
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.bridge")
bridge_mod = pytest.importorskip("megatron.bridge.models.deepseek.deepseek_v41_bridge")
vit_mod = pytest.importorskip("megatron.bridge.models.deepseek.deepseek_v41_vit")
vl_model_mod = pytest.importorskip("megatron.bridge.models.deepseek.deepseek_v41_vl_model")
vl_provider_mod = pytest.importorskip("megatron.bridge.models.deepseek.deepseek_v41_vl_provider")

TOY_CKPT = "/mnt/cfs-ainf/workspace/coda-data/models/DeepSeek-V4.1-Flash-1017-8l16e"
VISION = {"hidden_size": 1024}


def _bridge(vision_config):
    bridge = bridge_mod.DeepSeekV41Bridge()
    bridge.hf_config = SimpleNamespace(vision_config=vision_config)
    return bridge


def _hf_patterns(registry):
    """Compiled (regex, hf pattern) pairs for every HF-side name of the mappings."""
    pairs = []
    for m in registry.get_all_mappings():
        for p in m.hf_param.values() if isinstance(m.hf_param, dict) else [m.hf_param]:
            regex = "^" + re.escape(p).replace(r"\*\*", ".*").replace(r"\*", r"[^.]+") + "$"
            pairs.append((re.compile(regex), p))
    return pairs


def _safetensors_names(ckpt):
    names = []
    for path in sorted(glob.glob(f"{ckpt}/*.safetensors")):
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            names += [k for k in json.loads(f.read(n)) if k != "__metadata__"]
    return names


class TestMappings:
    def test_text_only_names_have_no_prefix(self):
        names = [m.megatron_param for m in _bridge(None).mapping_registry().get_all_mappings()]
        assert not any(n.startswith(("language_model.", "vision.", "aligner.")) for n in names)

    def test_vision_prefixes_the_language_model(self):
        mappings = _bridge(VISION).mapping_registry().get_all_mappings()
        owned = ("language_model.", "vision.", "aligner.", "image_")
        assert all(m.megatron_param.startswith(owned) for m in mappings)
        alpha = [m for m in mappings if isinstance(m, bridge_mod._HCAlphaMapping)]
        assert alpha
        assert all(m._megatron_post.startswith("language_model.") for m in alpha)
        assert all(m._megatron_res.startswith("language_model.") for m in alpha)

    def test_hf_names_stay_flat(self):
        registry = _bridge(VISION).mapping_registry()
        lookup = registry.megatron_to_hf_lookup
        assert lookup("language_model.embedding.word_embeddings.weight").hf_param == "embed.weight"
        assert lookup("language_model.decoder.layers.3.mlp.router.expert_bias").hf_param == "layers.3.ffn.gate.bias"
        assert lookup("language_model.decoder.layers.3.mlp.router.expert_bias_vl").hf_param == "layers.3.ffn.gate.bias_vl"
        assert lookup("vision.blocks.2.attn.wqkv.weight").hf_param == "vision.blocks.2.attn.wqkv.weight"
        assert lookup("image_newline").hf_param == "image_newline"

    def test_text_only_has_no_bias_vl(self):
        assert _bridge(None).mapping_registry().megatron_to_hf_lookup("decoder.layers.3.mlp.router.expert_bias_vl") is None

    @pytest.mark.skipif(not glob.glob(f"{TOY_CKPT}/*.safetensors"), reason="toy checkpoint not available")
    def test_every_checkpoint_name_is_mapped(self):
        pairs = _hf_patterns(_bridge(VISION).mapping_registry())
        extra_ok = re.compile(r"^layers\.\d+\.engram\.embed\.weight$")  # host table, loaded by the Engram module
        for name in _safetensors_names(TOY_CKPT):
            if name.endswith(".scale") or extra_ok.match(name):
                continue
            hits = [p for rx, p in pairs if rx.match(name)]
            assert hits, f"{name} has no mapping"
            if name.startswith(("vision.", "aligner.", "image_")) or name.endswith(".bias_vl"):
                assert len(hits) == 1, f"{name} is consumed by {hits}"

    @pytest.mark.skipif(not glob.glob(f"{TOY_CKPT}/*.safetensors"), reason="toy checkpoint not available")
    def test_text_only_leaves_vision_names_unmapped(self):
        pairs = _hf_patterns(_bridge(None).mapping_registry())
        assert not any(rx.match("vision.norm.weight") or rx.match("layers.0.ffn.gate.bias_vl") for rx, _ in pairs)


class TestProviderChoice:
    @pytest.fixture
    def pretrained(self):
        path = f"{TOY_CKPT}/config.json"
        if not glob.glob(path):
            pytest.skip("toy checkpoint not available")
        from megatron.bridge.models.deepseek.deepseek_v41_config import DeepseekV41Config

        with open(path) as f:
            return SimpleNamespace(config=DeepseekV41Config(**json.load(f)), name_or_path=TOY_CKPT)

    def test_vision_config_selects_the_vl_provider(self, pretrained):
        provider = bridge_mod.DeepSeekV41Bridge().provider_bridge(pretrained)
        assert type(provider) is vl_provider_mod.DeepSeekV41VLModelProvider
        assert provider.image_token_id == 129264
        assert provider.freeze_vision_model is True
        assert provider.scatter_embedding_sequence_parallel is False
        assert provider.vision_config["downsample_ratio"] == 3

    def test_dropping_vision_config_gives_the_text_provider(self, pretrained):
        pretrained.config.vision_config = None
        provider = bridge_mod.DeepSeekV41Bridge().provider_bridge(pretrained)
        assert type(provider) is bridge_mod.MLAModelProvider

    def test_bridge_instance_switches_between_calls(self, pretrained):
        bridge = bridge_mod.DeepSeekV41Bridge()
        assert type(bridge.provider_bridge(pretrained)) is vl_provider_mod.DeepSeekV41VLModelProvider
        pretrained.config.vision_config = None
        assert type(bridge.provider_bridge(pretrained)) is bridge_mod.MLAModelProvider

    def test_vision_config_without_image_token_id_raises(self, pretrained):
        del pretrained.config.image_token_id
        with pytest.raises(ValueError, match="image_token_id"):
            bridge_mod.DeepSeekV41Bridge().provider_bridge(pretrained)


def _tiny_vit_cfg():
    return {
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 24,
        "patch_size": 2,
        "rope_theta": 10000.0,
        "downsample_ratio": 3,
    }


class TestVit:
    def test_parameter_names_match_the_checkpoint(self):
        names = set(vit_mod.ViT(_tiny_vit_cfg()).state_dict())
        assert {
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
            "blocks.1.attn.wqkv.weight",
            "blocks.1.attn.wqkv.bias",
            "blocks.1.attn.wo.weight",
            "blocks.1.attn.wo.bias",
            "blocks.1.mlp.w1.weight",
            "blocks.1.mlp.w2.weight",
            "blocks.1.norm1.weight",
            "blocks.1.norm2.weight",
            "norm.weight",
        } == names - {n for n in names if n.startswith("blocks.0.")}

    def test_output_shape(self):
        vit = vit_mod.ViT(_tiny_vit_cfg())
        out = vit(torch.randn(6 * 9, 3 * 2 * 2), 6, 9)
        assert out.shape == (54, 16)

    def test_matches_sglang_helpers(self):
        ref = pytest.importorskip("sglang.srt.models.deepseek_v41_vit")
        cos, sin = vit_mod.get_vision_cos_sin(5, 7, 8, 10000.0)
        ref_cos, ref_sin = ref.get_vision_cos_sin(5, 7, 8, 10000.0)
        assert torch.equal(cos, ref_cos) and torch.equal(sin, ref_sin)
        x = torch.randn(35, 2, 16, dtype=torch.bfloat16)
        assert torch.equal(vit_mod.apply_rotary(x, cos, sin), ref.apply_rotary(x, cos, sin))
        aligner = vit_mod.Aligner(8, 12, 3)
        ref_aligner = ref.Aligner(SimpleNamespace(vision_dim=8, dim=12, vision_downsample_ratio=3))
        ref_aligner.load_state_dict(aligner.state_dict())
        feats = torch.randn(7 * 8, 8)
        assert torch.equal(aligner(feats, 7, 8), ref_aligner(feats, 7, 8))


class TestImageMerge:
    def test_token_types_layout(self):
        assert vit_mod.image_token_types(2, 3).tolist() == [0, 1, 1, 1, 2, 1, 1, 1, 2, 3]

    def test_spans_hold_markers_and_features(self):
        cfg = _tiny_vit_cfg()
        stub = SimpleNamespace(
            vision=vit_mod.ViT(cfg),
            aligner=vit_mod.Aligner(16, 8, 3),
            downsample_ratio=3,
            image_start=torch.full((8,), 1.0),
            image_newline=torch.full((8,), 2.0),
            image_end=torch.full((8,), 3.0),
        )
        pixels = torch.randn(6 * 6 + 3 * 7, 12)
        spans = vl_model_mod.DeepSeekV41VLModel._image_spans(stub, pixels, torch.tensor([[6, 6], [3, 7]]))
        assert [s.shape[0] for s in spans] == [2 * 3 + 2, 1 * 4 + 2]
        first = spans[0]  # 2x2 LLM grid: start, f f nl, f f nl, end
        assert first[0].eq(1.0).all() and first[-1].eq(3.0).all()
        assert first[3].eq(2.0).all() and first[6].eq(2.0).all()
        assert not first[1].eq(2.0).all()

    def test_pixel_count_mismatch_raises(self):
        stub = SimpleNamespace(
            vision=vit_mod.ViT(_tiny_vit_cfg()),
            aligner=vit_mod.Aligner(16, 8, 3),
            downsample_ratio=3,
            image_start=torch.zeros(8),
            image_newline=torch.zeros(8),
            image_end=torch.zeros(8),
        )
        with pytest.raises(ValueError, match="covers"):
            vl_model_mod.DeepSeekV41VLModel._image_spans(stub, torch.randn(10, 12), torch.tensor([[3, 3]]))

    def test_merge_replaces_image_rows_in_pad_order(self):
        ids = torch.tensor([[1, 9, 9, 2], [9, 3, 9, 9]])
        embeds = torch.zeros(2, 4, 2)
        spans = [torch.full((2, 2), 1.0), torch.full((3, 2), 2.0)]
        out = vl_model_mod.merge_image_spans(embeds, ids, 9, spans)
        assert out[0, :, 0].tolist() == [0.0, 1.0, 1.0, 0.0]
        assert out[1, :, 0].tolist() == [2.0, 0.0, 2.0, 2.0]

    def test_merge_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="image tokens"):
            vl_model_mod.merge_image_spans(torch.zeros(1, 3, 2), torch.tensor([[9, 9, 1]]), 9, [torch.ones(1, 2)])

    def test_text_only_batch_is_unchanged(self):
        embeds = torch.randn(1, 3, 2)
        assert torch.equal(vl_model_mod.merge_image_spans(embeds, torch.tensor([[1, 2, 3]]), 9, []), embeds)
