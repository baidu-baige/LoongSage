"""Unit tests for coda/backends/megatron/teacher_lm_head.py.

Covers the teacher ``lm_head`` weight-source resolution for
``opd.teachers[].dist_ckpt_path``:
  * ``_load`` dedups per resolved *source*, not per ``hf_path``, so two teachers
    sharing an HF config but pointing at different checkpoints each get their own
    head instead of silently collapsing into one.
  * a set-but-unusable ``dist_ckpt_path`` raises rather than falling back to HF.
  * ``_build`` routes to the dist-checkpoint reader vs the HF safetensors reader.
  * The candidate-key order for the checkpoint read: Megatron ties
    ``output_layer.weight`` to the word embedding by default and then stores only
    the embedding key, so both keys must be tried, in that order.

CPU-only: ``mpu`` and ``torch.cuda.current_device`` are patched, so neither a
process group nor a CUDA device is needed.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

# teacher_lm_head imports Megatron-Core (and, through checkpoint.py, TE) at module
# scope, so skip the module rather than fail collection where it is not installed.
pytest.importorskip("megatron", reason="Megatron-Core is not installed")

import coda.backends.megatron.teacher_lm_head as tlm
from coda.backends.megatron.teacher_lm_head import TeacherLMHeads

HIDDEN, VOCAB = 3, 8


def make_dist_ckpt(tmp_path, name):
    """Create a dir that ``resolve_dist_ckpt_dir`` accepts, and return its path."""
    d = tmp_path / name / "train_step_5" / "dist_ckpt"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text('{"sharded_backend": "torch_dist"}')
    return str(d)


def make_config(teachers, bf16=True, fp16=False, use_fp32_lm_head=False):
    return OmegaConf.create({
        "opd": {"teachers": teachers, "model": {"bf16": bf16, "fp16": fp16}},
        "trainer": {"use_fp32_lm_head": use_fp32_lm_head},
    })


@pytest.fixture
def cpu_tp1():
    """Single-rank TP on CPU: no process group, no CUDA device."""
    mpu_stub = MagicMock()
    mpu_stub.get_tensor_model_parallel_rank.return_value = 0
    mpu_stub.get_tensor_model_parallel_world_size.return_value = 1
    with patch.object(tlm, "mpu", mpu_stub), \
         patch("torch.cuda.current_device", return_value="cpu"):
        yield


class TestCandidateKeys:
    def test_key_order(self):
        """Order is load-bearing: tied checkpoints only hold the embedding key."""
        assert tlm._CKPT_LM_HEAD_KEYS == (
            "output_layer.weight",
            "embedding.word_embeddings.weight",
        )


class TestLoadDedup:
    """``_load`` keys its cache on the resolved weight source."""

    def test_same_hf_path_no_ckpt_shares_one_head(self):
        cfg = make_config([
            {"name": "a", "hf_path": "/hf/model"},
            {"name": "b", "hf_path": "/hf/model"},
        ])
        inst = TeacherLMHeads(cfg)
        with patch.object(TeacherLMHeads, "_build", side_effect=lambda *a, **k: MagicMock()) as build:
            inst._load()
        assert build.call_count == 1
        assert list(inst._by_source) == ["/hf/model"]
        assert inst._by_idx[0] is inst._by_idx[1]

    def test_same_hf_path_different_ckpt_gets_two_heads(self, tmp_path):
        dist_a = make_dist_ckpt(tmp_path, "ckpt_a")
        dist_b = make_dist_ckpt(tmp_path, "ckpt_b")
        cfg = make_config([
            {"name": "a", "hf_path": "/hf/model", "dist_ckpt_path": dist_a},
            {"name": "b", "hf_path": "/hf/model", "dist_ckpt_path": dist_b},
        ])
        inst = TeacherLMHeads(cfg)
        with patch.object(TeacherLMHeads, "_build", side_effect=lambda *a, **k: MagicMock()) as build:
            inst._load()
        assert build.call_count == 2
        assert sorted(inst._by_source) == sorted([dist_a, dist_b])
        assert inst._by_idx[0] is not inst._by_idx[1]

    def test_from_ckpt_flag_per_teacher(self, tmp_path):
        """A ckpt teacher and an HF-only teacher coexist, each with its own flag."""
        dist = make_dist_ckpt(tmp_path, "ckpt_a")
        cfg = make_config([
            {"name": "a", "hf_path": "/hf/model", "dist_ckpt_path": dist},
            {"name": "b", "hf_path": "/hf/other"},
        ])
        inst = TeacherLMHeads(cfg)
        with patch.object(TeacherLMHeads, "_build", side_effect=lambda *a, **k: MagicMock()) as build:
            inst._load()
        assert [c.args[0] for c in build.call_args_list] == [dist, "/hf/other"]
        assert [c.kwargs["from_ckpt"] for c in build.call_args_list] == [True, False]

    def test_unset_dist_ckpt_path_falls_back_to_hf(self):
        """An explicit null dist_ckpt_path is treated as unset, not as a path."""
        cfg = make_config([{"name": "a", "hf_path": "/hf/model", "dist_ckpt_path": None}])
        inst = TeacherLMHeads(cfg)
        with patch.object(TeacherLMHeads, "_build", side_effect=lambda *a, **k: MagicMock()) as build:
            inst._load()
        assert build.call_args.args[0] == "/hf/model"
        assert build.call_args.kwargs["from_ckpt"] is False

    def test_unusable_dist_ckpt_path_raises(self, tmp_path):
        """No silent HF fallback: that would distil from the wrong model in silence.

        This process (the train worker's last PP stage) used to disagree with the
        teacher worker, which already raised for the same config.
        """
        cfg = make_config([
            {"name": "a", "hf_path": "/hf/model", "dist_ckpt_path": str(tmp_path / "nope")},
        ])
        inst = TeacherLMHeads(cfg)
        with patch.object(TeacherLMHeads, "_build") as build:
            with pytest.raises(ValueError, match=r"opd\.teachers\[0\]\.dist_ckpt_path"):
                inst._load()
        build.assert_not_called()


class TestBuildRouting:
    """``_build`` reads from the dist checkpoint or the HF dir, never both."""

    def test_from_ckpt_reads_checkpoint(self, cpu_tp1):
        weight = torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)
        inst = TeacherLMHeads(make_config([]))
        with patch.object(tlm, "load_tensor_from_checkpoint",
                          return_value=("output_layer.weight", weight)) as from_ckpt, \
             patch.object(tlm, "_load_lm_head_weight") as from_hf:
            lm_head = inst._build("/some/dist_ckpt", from_ckpt=True)
        from_hf.assert_not_called()
        from_ckpt.assert_called_once_with("/some/dist_ckpt", tlm._CKPT_LM_HEAD_KEYS)
        assert lm_head.weight.shape == (VOCAB, HIDDEN)
        torch.testing.assert_close(lm_head.weight.detach(), weight.to(torch.bfloat16))

    def test_hf_path_reads_safetensors(self, cpu_tp1):
        weight = torch.zeros(VOCAB, HIDDEN)
        inst = TeacherLMHeads(make_config([]))
        with patch.object(tlm, "load_tensor_from_checkpoint") as from_ckpt, \
             patch.object(tlm, "_load_lm_head_weight", return_value=weight) as from_hf:
            lm_head = inst._build("/hf/model", from_ckpt=False)
        from_ckpt.assert_not_called()
        from_hf.assert_called_once_with("/hf/model")
        assert lm_head.weight.shape == (VOCAB, HIDDEN)

    def test_tp_slices_vocab_dim(self):
        """TP rank 1 of 2 takes the second half of the vocab dim."""
        weight = torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)
        mpu_stub = MagicMock()
        mpu_stub.get_tensor_model_parallel_rank.return_value = 1
        mpu_stub.get_tensor_model_parallel_world_size.return_value = 2
        inst = TeacherLMHeads(make_config([]))
        with patch.object(tlm, "mpu", mpu_stub), \
             patch("torch.cuda.current_device", return_value="cpu"), \
             patch.object(tlm, "load_tensor_from_checkpoint",
                          return_value=("output_layer.weight", weight)):
            lm_head = inst._build("/some/dist_ckpt", from_ckpt=True)
        assert lm_head.weight.shape == (VOCAB // 2, HIDDEN)
        torch.testing.assert_close(
            lm_head.weight.detach(), weight[VOCAB // 2:].to(torch.bfloat16)
        )


class TestDtype:
    """The lm_head dtype is stated by config, never inherited from the source.

    A dist checkpoint written by a run with ``use_fp32_lm_head`` on stores
    ``output_layer.weight`` in fp32; feeding that to ``F.linear`` alongside bf16
    teacher hidden states raises "expected mat1 and mat2 to have the same dtype".
    """

    def test_fp32_source_is_cast_to_bf16(self, cpu_tp1):
        weight = torch.zeros(VOCAB, HIDDEN, dtype=torch.float32)
        inst = TeacherLMHeads(make_config([], bf16=True))
        with patch.object(tlm, "load_tensor_from_checkpoint",
                          return_value=("output_layer.weight", weight)):
            lm_head = inst._build("/some/dist_ckpt", from_ckpt=True)
        assert lm_head.weight.dtype == torch.bfloat16

    def test_use_fp32_lm_head_keeps_fp32(self, cpu_tp1):
        weight = torch.zeros(VOCAB, HIDDEN, dtype=torch.bfloat16)
        inst = TeacherLMHeads(make_config([], bf16=True, use_fp32_lm_head=True))
        with patch.object(tlm, "load_tensor_from_checkpoint",
                          return_value=("output_layer.weight", weight)):
            lm_head = inst._build("/some/dist_ckpt", from_ckpt=True)
        assert lm_head.weight.dtype == torch.float32

    def test_fp16_teacher(self, cpu_tp1):
        weight = torch.zeros(VOCAB, HIDDEN, dtype=torch.float32)
        inst = TeacherLMHeads(make_config([], bf16=False, fp16=True))
        with patch.object(tlm, "_load_lm_head_weight", return_value=weight):
            lm_head = inst._build("/hf/model", from_ckpt=False)
        assert lm_head.weight.dtype == torch.float16

    def test_fp32_teacher(self, cpu_tp1):
        weight = torch.zeros(VOCAB, HIDDEN, dtype=torch.bfloat16)
        inst = TeacherLMHeads(make_config([], bf16=False, fp16=False))
        with patch.object(tlm, "_load_lm_head_weight", return_value=weight):
            lm_head = inst._build("/hf/model", from_ckpt=False)
        assert lm_head.weight.dtype == torch.float32
