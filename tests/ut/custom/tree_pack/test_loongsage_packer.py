"""TreePacker: config gating, options, DP loads, VPP chunk cache."""
from __future__ import annotations

import pytest
from omegaconf import OmegaConf  # noqa: E402

from coda.agentflow.trajectory_store import Segment, Trajectory, TrajectoryGroup  # noqa: E402
from coda.data_factory.data_processor import split_traj_group_by_dp  # noqa: E402
from coda.utils.packing import build_packer, get_packer_class  # noqa: E402

from coda.custom.tree_pack.packer import TreePacker, _ChunkCache  # noqa: E402


def _cfg(options=None, **model):
    m = {"context_parallel_size": 1, "attention_backend": "flash"} | model
    return OmegaConf.create({
        "megatron": {"model": m}, "opd": {"enable": False},
        "trainer": {"packer": "tree", "packer_options": options or {},
                    "use_dynamic_batch_size": True, "max_tokens_per_gpu": 1000},
    })


def test_custom_module_registers_tree():
    assert get_packer_class("tree") is TreePacker
    assert type(build_packer(_cfg())) is TreePacker


def test_supported_config():
    assert TreePacker.unsupported_reason(_cfg()) is None


def test_unsupported_configs_fall_back_to_rows():
    assert "context_parallel" in TreePacker.unsupported_reason(_cfg(context_parallel_size=2))
    assert "attention_backend" in TreePacker.unsupported_reason(_cfg(attention_backend="auto"))
    assert "fusion" in TreePacker.unsupported_reason(_cfg(gdn_pre_gated_delta_rule_fusion=True))
    cfg = _cfg()
    cfg.opd.enable = True
    assert "OPD" in TreePacker.unsupported_reason(cfg)
    cfg = _cfg()
    cfg.trainer.use_dynamic_batch_size = False
    assert "dynamic" in TreePacker.unsupported_reason(cfg)
    cfg = _cfg()
    cfg.multimodal = {"enable": True}
    assert "multimodal" in TreePacker.unsupported_reason(cfg)
    assert type(build_packer(cfg)) is not TreePacker


def test_options():
    assert TreePacker(_cfg()).options["cut"] == "min_cost"
    assert TreePacker(_cfg()).dp_loads is not None
    assert TreePacker(_cfg({"dp_balance": False})).dp_loads is None
    with pytest.raises(ValueError, match="cut"):
        TreePacker(_cfg({"cut": "bogus"}))
    with pytest.raises(ValueError, match="tree_cut"):
        TreePacker(_cfg({"tree_cut": "greedy"}))      # old PR-era key name


def test_pg_policy_keeps_rows():
    packer = TreePacker(_cfg())
    assert packer.applies(pg_policy=None) and not packer.applies(pg_policy=object())


def test_chunk_cache_builds_once_per_micro_batch():
    """VPP: every model chunk asks for the same micro-batch; the tree is built once, the
    first caller is flagged (stats), and the entry is dropped after the last chunk."""
    built = []
    cache = _ChunkCache(n_chunks=2)
    order = [1, 2, 1, 3, 2, 3]            # interleaved schedule: chunks reach mbs out of order
    out = [cache.get(k, lambda k=k: built.append(k) or f"tmb{k}") for k in order]
    assert built == [1, 2, 3]
    assert [f for _, f in out] == [True, True, False, True, False, False]
    assert [t for t, _ in out] == ["tmb1", "tmb2", "tmb1", "tmb3", "tmb2", "tmb3"]
    assert cache._entries == {}
    single = _ChunkCache(n_chunks=1)
    assert single.get(5, lambda: "a") == ("a", True) and single._entries == {}


# ── DP balance by tree cost ──────────────────────────────────────────────────

def _chain_group(pid: str, prompt: int, turns: list[int], resp: int) -> TrajectoryGroup:
    """One trajectory whose Segment rows chain by shared prefix: row k = prompt + the first
    k+1 turns + its own response (a caterpillar, like multi-turn agent rows)."""
    stream, segs, spine = [], [], list(range(10**6, 10**6 + prompt))
    nxt = 2 * 10**6
    for k, t in enumerate(turns):
        spine = spine + list(range(nxt, nxt + t)); nxt += t
        row = spine + list(range(nxt, nxt + resp)); nxt += resp
        segs.append(Segment(token_start=len(stream), token_end=len(stream) + len(row),
                            logprob_start=0, logprob_end=resp, segment_id=k))
        stream += row
    t = Trajectory(trajectory_id=f"{pid}_t0", prompt_id=pid)
    t.tokens, t.segments = stream, segs
    return TrajectoryGroup(prompt_id=pid, trajectories=[t])


def test_tree_load_counts_shared_prefix_once():
    g = _chain_group("p", prompt=100, turns=[10, 10, 10], resp=5)
    rows = [s.token_end - s.token_start for s in g.trajectories[0].segments]
    nodes = 100 + 30 + 3 * 5            # spine once + every response
    assert sum(rows) > nodes
    (c,) = TreePacker(_cfg()).dp_loads([g])
    assert nodes <= c < sum(rows) + 10**5


def test_balances_tree_cost_where_row_split_does_not():
    # Equal Segment counts and row tokens that the row split balances by pairing the two
    # deep chains (prompts 1000 + 5000) against the two flat ones (3000 + 3000). Deep rows
    # share most tokens, flat rows share none, so that pairing is badly off in tree cost.
    deep = [_chain_group(f"d{i}", prompt=p, turns=[50] * 8, resp=50)
            for i, p in enumerate((1000, 5000))]
    flat = []
    for i in range(2):
        g = _chain_group(f"f{i}", prompt=3000, turns=[50] * 8, resp=50)
        tr = g.trajectories[0]                  # break sharing: distinct tokens per row
        tr.tokens = list(range(len(tr.tokens)))
        flat.append(g)
    groups = deep + flat
    loads = TreePacker(_cfg()).dp_loads

    def spread(parts):
        per = [sum(loads(p)) for p in parts]
        return max(per) / min(per)

    row = split_traj_group_by_dp(groups, dp_size=2, num_mini_batch=1)
    tree = split_traj_group_by_dp(groups, dp_size=2, num_mini_batch=1, load_fn=loads)
    assert sorted(g.prompt_id for p in tree for g in p) == sorted(g.prompt_id for g in groups)
    assert [len(p) for p in tree] == [2, 2]
    assert spread(row) > 1.3 and spread(tree) < 1.15


def test_untrainable_segments_ignored():
    g = _chain_group("p", prompt=100, turns=[10, 10], resp=5)
    loads = TreePacker(_cfg()).dp_loads
    base = loads([g])
    g.trajectories[0].segments.append(Segment(token_start=0, token_end=0, trainable=False))
    assert loads([g]) == base
