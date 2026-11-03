"""Unit tests for create_reward_fn's __call__ / group_rm validation."""
import pytest

from coda.reward import REWARD_REGISTRY, create_reward_fn
from coda.reward.base import RewardFunction
from coda.reward.reward import Reward


class _CallOnly(RewardFunction):
    def __call__(self, messages, label, context, **kwargs):
        return Reward(final_reward=1.0)


class _GroupOnly(RewardFunction):
    async def group_rm(self, messages, label, contexts, **kwargs):
        return [Reward(final_reward=1.0) for _ in messages]


class _SyncGroup(RewardFunction):
    def group_rm(self, messages, label, contexts, **kwargs):
        return [Reward(final_reward=1.0) for _ in messages]


_CLASSES = {
    "_test_call_only": _CallOnly,
    "_test_group_only": _GroupOnly,
    "_test_sync_group": _SyncGroup,
}


@pytest.fixture(autouse=True, scope="module")
def _registered():
    for name, cls in _CLASSES.items():
        REWARD_REGISTRY.register(name)(cls)
    yield
    for name in _CLASSES:
        REWARD_REGISTRY._registry.pop(name, None)


def test_call_only_without_group_rm():
    assert isinstance(create_reward_fn({"name": "_test_call_only"}), _CallOnly)


def test_group_only_with_group_rm():
    reward_fn = create_reward_fn({"name": "_test_group_only", "group_rm": True})
    assert isinstance(reward_fn, _GroupOnly)
    assert reward_fn([], None, {}).final_reward == 0.0  # placeholder __call__


def test_group_rm_enabled_but_not_implemented():
    with pytest.raises(ValueError, match="does not implement group_rm"):
        create_reward_fn({"name": "_test_call_only", "group_rm": True})


def test_call_missing_when_group_rm_disabled():
    with pytest.raises(ValueError, match="does not implement __call__"):
        create_reward_fn({"name": "_test_group_only"})


def test_sync_group_rm_rejected():
    with pytest.raises(ValueError, match="must be an async def"):
        create_reward_fn({"name": "_test_sync_group", "group_rm": True})
