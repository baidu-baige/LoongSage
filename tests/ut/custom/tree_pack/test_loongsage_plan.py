"""TreePacker.plan through get_data_iterator: node and logit-row budgets."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import coda.backends.megatron.data as data_mod  # noqa: E402

from coda.custom.tree_pack.core.batch import chain_branch_points  # noqa: E402
from coda.custom.tree_pack.packer import TreePacker  # noqa: E402


@pytest.fixture(autouse=True)
def single_rank(monkeypatch):
    mpu = MagicMock()
    mpu.get_data_parallel_world_size.return_value = 1
    mpu.get_context_parallel_world_size.return_value = 1
    mpu.get_virtual_pipeline_model_parallel_world_size.return_value = None
    for f in ("get_data_parallel_rank", "get_tensor_model_parallel_rank",
              "get_pipeline_model_parallel_rank", "get_context_parallel_rank"):
        getattr(mpu, f).return_value = 0
    monkeypatch.setattr(data_mod, "mpu", mpu)
    with patch.object(data_mod.dist, "all_reduce"):
        yield


def _caterpillar_rollout(n_traj: int, n_rows: int, spine0: int, grow: int, thorn: int) -> dict:
    """Trajectories whose consecutive rows share a growing prefix (distinct values per traj)."""
    tokens, tid = [], []
    for t in range(n_traj):
        base = 100_000 * (t + 1)
        spine = list(range(base, base + spine0))
        for k in range(n_rows):
            spine = spine + list(range(base + 50_000 + k * grow, base + 50_000 + (k + 1) * grow))
            th = list(range(base + 90_000 + k * thorn, base + 90_000 + (k + 1) * thorn))
            tokens.append(torch.tensor(spine + th))
            tid.append(t)
    lengths = [int(x.numel()) for x in tokens]
    return {"tokens": tokens, "loss_masks": [torch.ones(n) for n in lengths],
            "total_lengths": lengths, "response_lengths": [thorn] * len(lengths),
            "dp_local_traj_idx": tid}


def _iter(budget, mbs, data, cut="min_cost", tree_cap=None, lm_cap=None, lm_rows=True, tree=True):
    trainer = OmegaConf.create({"use_dynamic_batch_size": True, "mini_batch_size": mbs,
                                "max_tokens_per_gpu": budget, "packer": "tree", "packer_options": {
                                    "cut": cut, "max_nodes_per_gpu": tree_cap,
                                    "max_lm_rows_per_gpu": lm_cap, "lm_head_rows": lm_rows}})
    packer = TreePacker(OmegaConf.create({"trainer": trainer})) if tree else None
    return data_mod.get_data_iterator(trainer, MagicMock(), data, packer=packer)


def _nodes(data, mb):
    toks = [data["tokens"][r] for r in mb]
    bp = chain_branch_points(toks, [data["dp_local_traj_idx"][r] for r in mb])
    return sum(int(t.numel()) - b for t, b in zip(toks, bp))


def _all_rows(idx):
    return sorted(r for mb in idx for r in mb if r >= 0)


def test_node_budget_needs_fewer_micro_batches_than_rows():
    data = _caterpillar_rollout(4, n_rows=6, spine0=400, grow=100, thorn=50)
    _, rows_mbs = _iter(2000, 4, data, tree=False)
    it, tree_mbs = _iter(2000, 4, data)
    assert tree_mbs[0] < rows_mbs[0]
    idx = it[0].micro_batch_indices
    assert _all_rows(idx) == list(range(len(data["tokens"])))
    for mb in idx:     # rows sorted, so a trajectory's rows stay in order inside each mb
        assert mb == sorted(mb)


def test_groups_respect_node_budget():
    data = _caterpillar_rollout(2, n_rows=8, spine0=600, grow=200, thorn=100)
    it, _ = _iter(1800, 2, data)
    for mb in it[0].micro_batch_indices:
        assert _nodes(data, mb) <= 1800 or len(mb) == 1, mb


def test_min_cost_cut_never_more_nodes_than_greedy():
    data = _caterpillar_rollout(3, n_rows=10, spine0=300, grow=250, thorn=120)
    out = {}
    for cut in ("greedy", "min_cost"):
        it, _ = _iter(2500, 3, data, cut)
        idx = it[0].micro_batch_indices
        assert _all_rows(idx) == list(range(len(data["tokens"])))
        out[cut] = sum(_nodes(data, mb) for mb in idx)
    assert out["min_cost"] <= out["greedy"]


def test_node_budget_overrides_token_budget():
    data = _caterpillar_rollout(2, n_rows=8, spine0=600, grow=200, thorn=100)
    _, small = _iter(1800, 2, data)
    _, large = _iter(1800, 2, data, tree_cap=10**6)
    assert large[0] < small[0]


def _lm_rows(data, idx):
    return [sum(data["response_lengths"][r] for r in mb if r >= 0) for mb in idx]


def test_logit_row_budget_caps_micro_batches():
    """Large node budget, small logit-row budget: every micro-batch (and every tree) stays
    within the logit rows, at the cost of more micro-batches."""
    data = _caterpillar_rollout(2, n_rows=8, spine0=600, grow=200, thorn=100)
    it_free, free = _iter(1800, 2, data, tree_cap=10**6, lm_cap=10**6)
    it, capped = _iter(1800, 2, data, tree_cap=10**6, lm_cap=300)
    idx = it[0].micro_batch_indices
    assert _all_rows(idx) == list(range(len(data["tokens"])))
    assert max(_lm_rows(data, idx)) <= 300 < max(_lm_rows(data, it_free[0].micro_batch_indices))
    assert capped[0] > free[0]


def test_logit_row_budget_defaults_to_token_budget_and_is_off_without_lm_rows():
    data = _caterpillar_rollout(2, n_rows=8, spine0=600, grow=200, thorn=100)
    it, _ = _iter(250, 2, data, tree_cap=10**6)
    _, off = _iter(250, 2, data, tree_cap=10**6, lm_rows=False)
    assert max(_lm_rows(data, it[0].micro_batch_indices)) <= 250
    assert off[0] == 1          # nodes alone fit one micro-batch per mini-batch
