"""CPU tests for tree_pack/core/loss_index.py and tree_pack/core/cost.py."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from coda.custom.tree_pack.core.batch import chain_branch_points, plan_trees
from coda.custom.tree_pack.core.cost import TREE_LEVEL_COST, chain_cost
from coda.custom.tree_pack.core.loss_index import build_loss_index, restrict_to_lm_rows

from .tree_test_utils import caterpillar as _caterpillar


def _index(seed, pad=7):
    rows, _, resp = _caterpillar(seed, n_rows=5, spine0=60, grow=(5, 20), thorn=(3, 12))
    toks = [torch.tensor(r) for r in rows]
    lens = [len(r) for r in rows]
    seqs = plan_trees(lens, chain_branch_points(toks, [0] * len(toks)), resp)
    total = sum(ts.num_nodes for ts in seqs) + pad
    return toks, resp, build_loss_index(seqs, toks, resp, total)


def _targets(idx):
    return torch.cat([idx.node_targets, idx.extra_targets])[idx.logp_src]


@pytest.mark.parametrize("seed", range(4))
def test_loss_index_targets_are_each_rows_response(seed):
    toks, resp, idx = _index(seed)
    want = torch.cat([t[len(t) - n:] for t, n in zip(toks, resp)])
    assert torch.equal(_targets(idx), want)
    assert idx.resp_lens == resp


@pytest.mark.parametrize("seed", range(4))
def test_restrict_to_lm_rows_keeps_the_targets(seed):
    _, _, idx = _index(seed)
    sub = restrict_to_lm_rows(idx)
    assert torch.equal(_targets(sub), _targets(idx))
    assert torch.equal(sub.lm_rows[sub.logit_nodes], idx.logit_nodes)
    with pytest.raises(ValueError):
        restrict_to_lm_rows(sub)


def test_response_length_out_of_range_rejected():
    toks = [torch.tensor([1, 2, 3])]
    seqs = plan_trees([3], [0], [3])
    with pytest.raises(ValueError):
        build_loss_index(seqs, toks, [3], 3)


def test_chain_cost_counts_shared_prefix_once():
    spine = np.arange(100)
    rows = [np.concatenate([spine[:100 - 10 * (2 - k)], np.full(5, 1000 + k)]) for k in range(3)]
    nodes = 100 + 3 * 5                      # spine once + every response
    cost = chain_cost(rows)
    assert nodes <= cost - TREE_LEVEL_COST * 4 < sum(len(r) for r in rows)


def test_chain_cost_restarts_when_rows_share_nothing():
    rows = [np.arange(10), np.arange(100, 110)]
    assert chain_cost(rows) == pytest.approx(chain_cost(rows[:1]) + chain_cost(rows[1:]) - TREE_LEVEL_COST)
    assert chain_cost([]) == TREE_LEVEL_COST
