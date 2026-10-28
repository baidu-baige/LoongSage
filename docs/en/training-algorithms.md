# Training Algorithms

The framework supports the **GRPO-like** (critic-free) algorithm family, where advantages come from normalizing rewards within a prompt group, and **PPO**, where a co-located critic estimates per-token advantages with GAE (see §1.6). The whole algorithm pipeline is split into several independent components — advantage estimation, policy loss, off-policy protection (IS correction, M2PO i.e. Second-Moment Trust Policy Optimization, OPSM i.e. Off-Policy Sequence Masking), regularization terms (entropy / ref KL), and loss aggregation. All switches are centralized in the `algorithm` section of the configuration, where advantage and policy loss support registering custom implementations — see the [Custom RL Algorithm Development Guide](./custom-algorithm.md).

The execution order of the components within a mini-batch is: (optional) recompute `old_log_probs` → (optional) ref model forward → (optional) M2PO masking → (optional) IS weight computation → advantage computation → actor forward to obtain `log_probs` → (optional) OPSM masking → policy loss → (optional) multiply by OPSM / IS weights → aggregation, and (optional) add the entropy and ref KL terms on top.

## 1. Built-in Algorithms

### 1.1. Advantage Estimation

`algorithm.advantage_estimator` provides the built-in `grpo` (default) and `gae` (PPO, see §1.6). `grpo` normalizes the scalar reward of each trajectory into an advantage and broadcasts it to every response token of that trajectory. The normalization method is controlled by `algorithm.advantage_norm_mode`:

| Value | Meaning |
| --- | --- |
| `group_zscore` (default) | z-score within the same prompt group: `(x - mean) / std` |
| `group_mean` | mean subtraction within the group: `x - mean` |
| `none` | use the raw reward directly |

The `group_*` modes require the number of trajectories in each prompt group to be identical (i.e. `num_trajectories_per_prompt`). `advantage_norm_mode` only applies to `grpo`; `gae` ignores it.

Normalization over the whole step is a separate, estimator-independent step: with `algorithm.use_masked_whiten: true`, the advantages returned by any estimator are whitened, `(A - mean) / std`, with the mean / unbiased std taken over all response tokens (`raw_loss_masks`) of the step, all-reduced over the DP group.

Every estimator reports `train/adv_zero_ratio`: the fraction of Segment rows whose advantage is all zero, taken after the optional whitening.

### 1.2. Policy Loss

`algorithm.policy_loss` provides two built-in GRPO-like clipped surrogate losses:

| Value | Importance Ratio | Description |
| --- | --- | --- |
| `grpo` (default) | token-level `exp(logπ - logπ_old)` | Asymmetric clipping (`clip_ratio_low` / `clip_ratio_high`, default 0.2 / 0.28, i.e. DAPO's clip-higher); additionally applies dual-clip for negative advantages (`clip_ratio_c`, default 10.0) |
| `gspo` | sequence-level `exp(mean(Δlogp))` | Sequence-level ratio with token-level gradient routing, see [GSPO](https://arxiv.org/pdf/2507.18071); only uses `clip_ratio_low` / `clip_ratio_high` |
| `ppo` | token-level, same as `grpo` | The same clipped surrogate (PPO and GRPO differ only in the advantage); the standard PPO setting is `clip_ratio_low = clip_ratio_high = 0.2`, `clip_ratio_c = 3.0` |

The behavior policy `logπ_old` is by default recomputed by the training engine at the beginning of every step; setting `trainer.use_rollout_log_probs: true` switches to the `rollout_log_probs` returned by the inference engine (in which case IS correction and M2PO cannot be enabled, see the next section).

### 1.3. Off-Policy Protection

The following mechanisms all target train-inference inconsistency (the deviation between `π_old` and `π_rollout`) as well as the off-policy data introduced by async / partial rollout, and can be enabled in combination; however, IS correction, M2PO, and `trainer.use_rollout_log_probs: true` cannot be used at the same time:

| Mechanism | Configuration | Principle | Constraint |
| --- | --- | --- | --- |
| IS correction | `algorithm.is_correction.enable` | Weights the per-token loss by `π_old / π_rollout`, clipping or masking when the weight goes out of bounds | Mutually exclusive with `trainer.use_rollout_log_probs: true` |
| M2PO | `algorithm.m2po.enable` | Masks the tokens with the largest `(log(π_old/π_rollout))²` across the whole batch until the average second moment of the remaining tokens drops below `threshold` (default 0.04) | Same as above |
| OPSM | `algorithm.opsm.enable` | When a sequence satisfies `advantage < 0` and its sequence-level KL exceeds `delta` (default 0.1), the gradient of that sequence is discarded (the denominator stays unchanged) | None |

The behavior of IS correction is composed of two dimensions:

- `level`: weight granularity, `token` (per-token ratio), `sequence` (probability ratio of the whole sequence), or `geometric` (geometric mean of the token ratios);
- `action`: out-of-bound handling, `clip` clamps the weight into `[lower_bound, upper_bound]`, `mask` additionally removes the out-of-bound tokens (or the whole sequence) from both the numerator and the denominator of the loss.

When not in pure-GKD and `old_log_probs` has been computed on the training side, the framework reports the train-inference mismatch metric `train/is_approx_k3_kl`; `train/is_clip_ratio` and `train/is_nan_ratio` are reported only when IS correction is enabled.

### 1.4. Regularization Terms

- `algorithm.entropy_coef` (default 0.0): when non-zero, `entropy_coef × entropy` is subtracted from the loss.
- `algorithm.ref_kl.enable`: adds the reference model KL penalty `coef × KL(π_θ ‖ π_ref)`. `kl_type` supports `k1|k2|k3`; when `use_unbiased_kl: true`, the per-token KL is multiplied by the importance ratio (the DeepSeek-V3.2 approach); when `update_interval > 0`, the reference model is refreshed with the current actor every N steps. `ref_dist_ckpt_path` (one concrete `train_step_N/dist_ckpt` dir) or `ref_hf_model_path` must be provided when enabled.

### 1.5. Configuration Example

```yaml
algorithm:
  advantage_estimator: grpo
  advantage_norm_mode: group_zscore
  policy_loss: grpo
  loss_agg_mode: token-mean
  clip_ratio_low: 0.2
  clip_ratio_high: 0.28
  is_correction:
    enable: true
    action: clip
    level: token
    lower_bound: 0.5
    upper_bound: 2.0
  opsm:
    enable: true
    delta: 0.1
```

### 1.6. PPO (GAE + critic)

Set `critic.enable: true` together with `algorithm.advantage_estimator: gae` (and usually `algorithm.policy_loss: ppo`). The critic is a second Megatron model with the actor's parallelism and its own optimizer, scheduler and checkpoint (see [Configuration Reference §7.3](./config-reference.md)); it shares the actor's GPUs and runs first in every step:

1. A forward-only pass of the pre-update critic gives `old_values`. The value head (`hidden -> 1`, replacing the LM head) is read at the same positions as the log-probs: the value at position `t-1` belongs to response token `t`.
2. GAE over the rollout's `token_rewards` with `algorithm.ppo.gamma` / `algorithm.ppo.lam`, tokens with mask 0 are not steps, the bootstrap value and the running advantage are carried across them. `returns = advantages + old_values`; then the optional `use_masked_whiten` step whitens the advantages (not the returns).
3. Clipped value loss `0.5 * max((V - R)^2, (clip(V, V_old ± ppo.value_clip) - R)^2)`, aggregated with `loss_agg_mode`; metrics are reported under `critic/`.
4. The advantages are handed to the actor, which uses them as they are (GAE and whitening run once per step).

KL to the reference model is only applied as a loss term (`ref_kl`). Steps `<= critic.critic_warmup` train only the critic. Current limitations: `num_trajectories_per_prompt == 1`, one Segment per trajectory (no context compression / subagent), and no OPD.

## 2. Unbiased Loss Aggregation

`algorithm.loss_agg_mode` determines how the per-token loss is aggregated within a mini-batch:

- `token-mean` (default): the mean over all valid tokens in the mini-batch, with every token equally weighted;
- `seq-mean-token-mean`: first take the token mean within each trajectory, then take the mean over trajectories, with every sequence equally weighted.

The framework adopts the newer Megatron per-token loss aggregation protocol (forcing `calculate_per_token_loss=True`): what the loss function returns to Megatron is the **loss sum and the denominator** (token count or sequence count) of the current micro-batch, rather than the mean within the micro-batch. During the backward phase the gradients of each micro-batch are accumulated directly, and `finalize_model_grads` sums the denominator across all micro-batches and DP ranks, then applies a one-time scaling to the gradients:

```text
g = Σ(gradients of all valid tokens in the minibatch) / Σ(number of all valid tokens in the minibatch)
```

Therefore the aggregation within a mini-batch is **unbiased**: the gradient is independent of how micro-batches are split (including the variable-length splitting of `use_dynamic_batch_size`) and of the sample length distribution across DP ranks. In contrast, the traditional approach of "first take the mean within each micro-batch, then average over micro-batches" gives greater weight to the tokens in short micro-batches, introducing a length-dependent bias in RL training with variable-length sequences. The same holds for `seq-mean-token-mean`, except that the global denominator becomes the sequence count.

The monitoring metrics follow the same convention: the loss function reports summed values, and the framework uniformly divides by the global denominator after the DP all-reduce, so `pg_loss`, `approx_kl`, etc. in the logs are all global means.
