"""Unit tests for routing-replay setup in coda/backends/megatron/model.py.

Real ``RouterReplay`` objects are used; ``TransformerLayer`` is subclassed without
running its heavy ``__init__`` so no GPU or process group is required.

Regression focus: ``RouterReplay.global_router_replay_instances`` can contain orphans
(Qwen3-VL's text model builds a decoder, then replaces it). Targets must land on the
routers that actually live in the model chunk, not on list positions.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction  # noqa: E402
from megatron.core.transformer.transformer_layer import TransformerLayer  # noqa: E402

from coda.backends.megatron import model as model_mod  # noqa: E402

NUM_LAYERS = 8
TOPK = 2


class _FakeMoELayer(torch.nn.Module):
    def __init__(self, with_replay: bool = True):
        super().__init__()
        self.router = torch.nn.Module()
        self.router.router_replay = RouterReplay() if with_replay else None


class _FakeLayer(TransformerLayer):
    """TransformerLayer stand-in exposing only what collect_replay_routers reads."""

    def __init__(self, layer_number: int, *, is_mtp: bool = False, mlp: torch.nn.Module | None = None):
        torch.nn.Module.__init__(self)
        self.layer_number = layer_number
        self.is_mtp_layer = is_mtp
        self.mlp = mlp if mlp is not None else _FakeMoELayer()


class _Chunk(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


@pytest.fixture(autouse=True)
def isolated_global_instances():
    saved = list(RouterReplay.global_router_replay_instances)
    RouterReplay.global_router_replay_instances.clear()
    yield
    RouterReplay.global_router_replay_instances[:] = saved


def test_collect_maps_layer_number_to_zero_based_id_with_pp_offset():
    # Second PP stage of an 8-layer model: global layer numbers 5..8.
    chunk = _Chunk([_FakeLayer(n) for n in range(5, 9)])

    routers = model_mod.collect_replay_routers(chunk)

    assert sorted(routers) == [4, 5, 6, 7]
    assert routers[4] is chunk.layers[0].mlp.router.router_replay


def test_collect_skips_mtp_dense_and_router_less_layers():
    live = _FakeLayer(1)
    mtp = _FakeLayer(1, is_mtp=True)  # MTP numbering restarts at 1 and collides with decoder layer 1
    dense = _FakeLayer(2, mlp=torch.nn.Linear(2, 2))  # no router at all
    no_replay = _FakeLayer(3, mlp=_FakeMoELayer(with_replay=False))
    chunk = _Chunk([live, mtp, dense, no_replay])

    routers = model_mod.collect_replay_routers(chunk)

    assert routers == {0: live.mlp.router.router_replay}


def test_orphan_instances_do_not_steal_targets():
    # Orphans are created first (the discarded decoder), then the live decoder.
    orphans = [_FakeMoELayer() for _ in range(NUM_LAYERS)]
    chunk = _Chunk([_FakeLayer(n) for n in range(1, NUM_LAYERS + 1)])
    assert len(RouterReplay.global_router_replay_instances) == 2 * NUM_LAYERS

    packed = torch.arange(5 * NUM_LAYERS * TOPK).reshape(5, NUM_LAYERS, TOPK)
    batch = {"rollout_routed_experts": [torch.zeros(5, NUM_LAYERS * TOPK)]}
    cfg = SimpleNamespace(num_layers=NUM_LAYERS, moe_router_topk=TOPK)

    with patch.object(model_mod, "get_model_config", return_value=cfg), \
         patch.object(model_mod, "prepare_routing_replay_data", return_value=packed):
        model_mod.setup_routing_replay(None, batch, chunk)

    for layer_id, layer in enumerate(chunk.layers):
        replay = layer.mlp.router.router_replay
        assert torch.equal(replay.target_topk_idx, packed[:, layer_id])
        assert replay.router_replay_action == RouterReplayAction.REPLAY_FORWARD
    assert all(o.router.router_replay.target_topk_idx is None for o in orphans)


def test_setup_fails_loudly_when_no_router_found():
    chunk = _Chunk([_FakeLayer(1, mlp=torch.nn.Linear(2, 2))])
    batch = {"rollout_routed_experts": [torch.zeros(5, NUM_LAYERS * TOPK)]}
    cfg = SimpleNamespace(num_layers=NUM_LAYERS, moe_router_topk=TOPK)

    with patch.object(model_mod, "get_model_config", return_value=cfg), \
         patch.object(model_mod, "prepare_routing_replay_data", return_value=torch.zeros(5, NUM_LAYERS, TOPK)):
        with pytest.raises(AssertionError, match="no live RouterReplay"):
            model_mod.setup_routing_replay(None, batch, chunk)
