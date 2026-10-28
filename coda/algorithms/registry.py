"""Generic registry pattern for extensible components.

Provides decorator-based registration for policy-loss functions, advantage
estimators, and KL-policy classes (OPD).

Usage example::

    @register_policy_loss("grpo")
    def grpo_loss(config, old_log_prob, log_prob, advantages, loss_masks, **kwargs):
        ...

    loss_fn = get_policy_loss("grpo")
"""

from __future__ import annotations

from coda.utils.registry import Registry


# ── Policy-loss registry ────────────────────────────────────────────────
_POLICY_LOSS_REGISTRY = Registry("policy_loss")
register_policy_loss = _POLICY_LOSS_REGISTRY.register
get_policy_loss = _POLICY_LOSS_REGISTRY.get


# ── Advantage-estimator registry ────────────────────────────────────────
# Estimators are called as ``fn(config.algorithm, rollout_data)`` and return
# ``(advantages, returns)``: per-Segment tensors of shape ``(response_length,)``.
# ``returns`` are the critic's value targets; critic-free estimators return
# their advantages again.
_ADVANTAGE_REGISTRY = Registry("advantage")
register_advantage = _ADVANTAGE_REGISTRY.register
get_advantage = _ADVANTAGE_REGISTRY.get


# ── KL-policy registry ──────────────────────────────────────────────────
# OPD (Online Policy Distillation) KL methods register their
# :class:`~coda.algorithms.kl_policy.KLPolicy` subclass here so users can plug
# in custom divergences without touching the backend dispatch.
_KL_POLICY_REGISTRY = Registry("kl_policy")
register_kl_policy = _KL_POLICY_REGISTRY.register
get_kl_policy_cls = _KL_POLICY_REGISTRY.get

