"""CPU tests for the DeepSeek-V4.1 image-token handling in the patched Megatron-Core.

Covers the router's image mask / per-token bias selection and the Engram n-gram hashing, using the pure
functions only (the router and Engram modules need CUDA and a checkpoint to construct).
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")
router = pytest.importorskip("megatron.core.transformer.moe.router")
engram = pytest.importorskip("megatron.core.transformer.experimental_attention_variant.dsv41.engram")
if not hasattr(router, "image_token_mask"):
    pytest.skip("Megatron-Core is not patched with DeepSeek-V4.1 image routing", allow_module_level=True)

IMG = 9


class TestImageTokenMask:
    def test_s_major_order(self):
        # [b=2, s=4] ids; the router flattens hidden states [s, b] s-major
        ids = torch.tensor([[IMG, 1, 1, 1], [1, IMG, 1, 1]])
        mask = router.image_token_mask(ids, IMG, 8)
        assert mask.tolist() == [True, False, False, True, False, False, False, False]

    def test_sequence_parallel_takes_local_shard(self):
        ids = torch.tensor([[1, IMG, IMG, 2, IMG, 3, 4, IMG]])
        shards = [
            router.image_token_mask(ids, IMG, 4, tp_size=2, tp_rank=r, sequence_parallel=True).tolist()
            for r in range(2)
        ]
        assert shards == [[False, True, True, False], [True, False, False, True]]

    def test_tp_without_sequence_parallel_keeps_full_sequence(self):
        ids = torch.tensor([[1, IMG, 2, 3]])
        assert router.image_token_mask(ids, IMG, 4, tp_size=2, tp_rank=1).tolist() == [False, True, False, False]

    def test_token_count_mismatch_raises(self):
        with pytest.raises(AssertionError, match="router sees"):
            router.image_token_mask(torch.tensor([[1, IMG, 2, 3]]), IMG, 3)

    def test_indivisible_sequence_raises(self):
        with pytest.raises(AssertionError, match="not divisible"):
            router.image_token_mask(torch.tensor([[1, 2, 3]]), IMG, 1, tp_size=2, sequence_parallel=True)


class TestSelectExpertBias:
    def test_image_rows_use_vl_bias(self):
        text, vl = torch.tensor([0.0, 1.0, 2.0]), torch.tensor([10.0, 11.0, 12.0])
        bias = router.select_expert_bias(text, vl, torch.tensor([True, False, True]))
        assert bias.tolist() == [[10.0, 11.0, 12.0], [0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]

    def test_keeps_float32(self):
        bias = router.select_expert_bias(torch.zeros(4), torch.ones(4), torch.tensor([True, False]))
        assert bias.dtype == torch.float32

    def test_selection_changes_topk(self):
        # Same scores: the text bias picks expert 0, the VL bias picks expert 2.
        scores = torch.tensor([[0.5, 0.4, 0.3], [0.5, 0.4, 0.3]])
        text, vl = torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])
        bias = router.select_expert_bias(text, vl, torch.tensor([False, True]))
        assert (scores + bias).argmax(dim=1).tolist() == [0, 2]


def _hasher(image_token_id, ngram=3):
    """A stand-in with the attributes `hash_ids` reads; token ids are their own compressed ids."""
    vocab = 16
    return SimpleNamespace(
        token_map=torch.arange(vocab),
        pad_id=0,
        max_ngram_size=ngram,
        multipliers=torch.tensor([3, 5, 7][:ngram]),
        primes=[101, 103][: ngram - 1],
        offsets=torch.zeros(ngram - 1, dtype=torch.long),
        image_token_id=image_token_id,
    )


def _hash(ids, image_token_id):
    return engram.DeepSeekV41Engram.hash_ids(_hasher(image_token_id), torch.tensor([ids]))


class TestEngramHashIds:
    def test_text_only_path_ignores_image_id(self):
        # No image token in the input: enabling image handling changes nothing.
        ids = [1, 2, 3, 4, 5]
        assert torch.equal(_hash(ids, None), _hash(ids, IMG))

    def test_image_token_blocks_history_behind_it(self):
        # Token 4 follows an image token, so its n-gram must equal the one after plain PAD (0) history.
        with_image = _hash([1, 2, IMG, 4], IMG)
        with_pad = _hash([0, 0, 0, 4], None)
        assert torch.equal(with_image[0, 3], with_pad[0, 3])

    def test_history_before_the_image_token_is_unaffected(self):
        base = _hash([1, 2, 3, 4], IMG)
        with_image_later = _hash([1, 2, 3, IMG], IMG)
        assert torch.equal(base[0, :3], with_image_later[0, :3])

    def test_without_image_handling_history_leaks_through(self):
        # Guards the test above: without the id, the n-gram after the image token differs from the PAD one.
        leaked = _hash([1, 2, IMG, 4], None)
        with_pad = _hash([0, 0, 0, 4], None)
        assert not torch.equal(leaked[0, 3], with_pad[0, 3])
