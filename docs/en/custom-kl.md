# Custom KL Algorithm Development Guide

This document explains how to add a KL / divergence algorithm for on-policy distillation (OPD). The KL extension is likewise **"registry + config-driven"**: inherit the corresponding base class, add a decorator, and it can be referenced via `opd.pg_kl_method` / `opd.gkd_kl_method`. For the built-in `k1`/`k2`/`k3`, `topk_*` and `full_*` methods and their applicable scenarios, see [On-Policy Distillation](./on-policy-distillation.md).

## 1. Registering a KL Method

The KL math layer is decoupled from the backend: TP/CP communication is concentrated in `coda.backends.megatron.kl_ctx`, and policy classes only write the divergence formula. Adding a new KL method only requires inheriting the corresponding base class and registering it:

```python
from coda.algorithms.kl_policy import TopkVocabPolicy
from coda.algorithms.registry import register_kl_policy

@register_kl_policy("my_kl")
class MyKL(TopkVocabPolicy):        # or LogProbKLPolicy / FullVocabPolicy
    def compute_kl(self, config, ctx):
        ...                        # returns (per_token_kl, extra_metrics)
```

Placing the module under [coda/custom/](../../coda/custom/) is enough for it to be discovered and registered automatically (see [Custom Extensions](./custom-extensions.md)), after which `my_kl` can be referenced in the configuration.

## 2. The Three Base Classes

Pick the base class according to the teacher data you need:

- `LogProbKLPolicy`: requires the teacher's log-prob (the k1/k2/k3 family).
- `TopkVocabPolicy`: requires the teacher's top-k log-probs and indices.
- `FullVocabPolicy`: declares `need_teacher_logits()`, and obtains the reconstructed full-vocabulary logits via `ctx.teacher_logits()`.

## 3. Config Reference

After registration, it can be referenced by name in `opd.pg_kl_method` / `opd.gkd_kl_method`:

```yaml
opd:
  enable: true
  gkd_ratio: 1
  gkd_kl_method: my_kl      # ← the @register_kl_policy registered name
  topk: 64                  # needed by the top-k family
```

Note that PG and GKD use the KL differently — PG uses only its **value** (which must be signed and unbiased), GKD only its **gradient** — so a custom method should be explicit about which side it applies to; see the KL methods section of [On-Policy Distillation](./on-policy-distillation.md).
