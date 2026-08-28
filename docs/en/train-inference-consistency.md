# Train-Inference Consistency

Train-inference inconsistency means that the same trajectory yields different token probabilities under the inference engine (SGLang) and the training engine (Megatron), i.e. `π_rollout ≠ π_train`. The sources of deviation fall into four categories. First, the token sequence used on the training side may itself differ from the one actually generated on the inference side — if a trajectory is returned as text and then re-encoded, the non-invertible tokenizer round-trip causes misaligned token boundaries, and at that point the two sides are no longer comparing the same sequence. Second, the numerical precision of the two sides may not be unified, for example when the inference side enables quantization or a low-precision KV cache while the training side runs bf16. Third, even if the precision is exactly the same, the operator implementations of the two engines are not identical — differences in kernel selection, parallel sharding, and floating-point accumulation order all introduce tiny numerical perturbations. Fourth, these perturbations are amplified by the model architecture — for MoE models, the router's top-k is a discrete decision, so a tiny difference in logits can flip the expert selection and take a completely different compute path, making MoE the worst-hit area for inconsistency. In addition, async and partial rollout scenarios stack weight version differences on top of this. Accumulated deviation invalidates the on-policy assumption and distorts importance ratios, and in severe cases leads to training collapse.

The framework addresses this at three levels:

1. **Source alignment**: TITO whole-chain token passing, R3 routing replay, and FP32 LM head directly narrow the gap between `π_train` and `π_rollout`;
2. **Algorithm-level correction**: IS correction / M2PO / OPSM weight or mask out the residual deviation;
3. **Continuous monitoring**: metrics such as `train/is_approx_k3_kl` quantify the deviation between the two engines.

## 1. TITO (Token-In-Token-Out)

The most upstream consistency issue is not probability, but **whether the two sides obtain the same token sequence**.

If trajectories are collected as text, the training side has to re-encode them, and the tokenizer's encode/decode is not strictly invertible (whitespace merging, special tokens being shown or hidden, multi-byte characters being split at token boundaries, and so on), so the re-encoded result may be misaligned with the sequence the engine actually generated; once misaligned, logprobs no longer correspond to tokens, training computes the probability of a different sequence, and in multi-turn settings the error accumulates turn by turn.

By default LoongSage passes only tokens along the whole chain, with no switch required: the Router middleware rewrites the agent's OpenAI chat request into SGLang's native `/generate` and issues it with `input_ids`, always setting `return_logprob=true` (for the DeepSeek-V4 family it additionally forces `skip_special_tokens=false`); on the response side, token ids and logprobs are taken directly from `output_token_logprobs`; multi-turn continuation only incrementally encodes the newly added messages, and the already generated part is never re-encoded; rewrites such as think trimming and length truncation are also performed on the token sequence, with the mask and logprobs sliced in sync.

As a result, the token sequences used by training and inference are exactly the same, and token ↔ logprob ↔ mask are strictly aligned; this is also the precondition for the R3 routing tensors in the next section to be replayed token by token. The cost is that AgentFlow needs to hold a tokenizer consistent with the inference side (provided uniformly by TokenizerManager), while the agent side still faces only the standard OpenAI interface and does not need to be aware of tokens.

## 2. Rollout Routing Replay (R3)

R3 ([arXiv:2510.11370](https://arxiv.org/abs/2510.11370)) targets MoE models: it records the experts actually selected by each MoE layer when the inference engine generates each token, and the training forward pass directly replays the same routing. This eliminates expert selection inconsistency; the replay only fixes the top-k expert selection, and the router weights still participate in gradient computation. It is recommended to enable this by default for RL training of MoE models.

It is enabled with `trainer.use_rollout_routing_replay: true`, after which the whole chain is wired up automatically:

1. SGLang is started with `enable_return_routed_experts`, and every generation returns per-token, per-MoE-layer expert indices along with the response;
2. The Router middleware incrementally collects them turn by turn and concatenates them into the routing tensor for the entire trajectory — operations that rewrite the token sequence, such as multi-turn dialogue, partial rollout continuation, and think block trimming, all maintain this tensor in sync;
3. The training side forces `megatron.model.moe_enable_routing_replay=true`: in the forward-only pass that recomputes `old_log_probs` and in the training forward/backward, the routing tensor is aligned according to CP/TP sharding and then written into the Megatron `RouterReplay` instance of each MoE layer, replacing the router's top-k output.

Notes:

- It is only meaningful for MoE models, and requires SGLang support for returning routed experts (>= 0.5.14);
- If a DP shard contains a trajectory that lacks routing records (for example when recovering from old data), the routing tensor for that whole shard is not passed to the training side (the number of omitted entries is logged at info level), and a training step with R3 enabled then fails because the replay data is unavailable; so make sure every trajectory used for training carries routing records when R3 is on;
- It is compatible with partial rollout: a trajectory continued across weight versions replays the real routing from when each segment was generated; see [bcp_h20_1node.yaml](../../conf/qwen3_30b_a3b/bcp_h20_1node.yaml).

## 3. FP32 LM head

`trainer.use_fp32_lm_head: true` keeps the weights and computation of the output layer (LM head) in FP32 (equivalent to configuring `output_layer` in `megatron.keep_fp32_weights`). Under bf16, the rounding error of the logits enters log_probs directly through softmax, and may become a major source of train-inference probability deviation for dense models; computing the LM head in FP32 can reduce this part of the error, at the cost of extra memory and computation for the output layer.

## 4. Algorithm-Level Correction

Source alignment cannot fully eliminate the deviation (differences in precision and operator implementation remain), and it does not cover the weight version differences introduced by async / partial rollout; the residual part is handled at the algorithm level, see Section 1.3 of [Training Algorithms](./training-algorithms.md):

- **IS correction**: weights the per-token loss by `π_old / π_rollout`, explicitly folding the residual deviation into the gradient, and clipping or masking when the weight goes out of bounds;
- **M2PO / OPSM**: instead of weighting, they mask out the tokens / sequences with the worst deviation, preventing extreme samples from dominating the update.

Another route is `trainer.use_rollout_log_probs: true`: use `π_rollout` directly as the behavior policy in the policy loss, letting the ratio `π_θ / π_rollout` implicitly absorb the train-inference difference and saving the recomputation of `old_log_probs`; this switch is mutually exclusive with IS correction and M2PO.

## 5. Monitoring Metrics

When not in pure-GKD and when the training side has computed `old_log_probs`, the framework always reports the following metrics; among them the clip/nan metrics only appear when IS correction is enabled:

| Metric | Meaning | Interpretation |
| --- | --- | --- |
| `train/is_approx_k3_kl` | The k3 KL estimate between `π_old` and `π_rollout` | A continuous increase means the train-inference deviation is worsening; check the R3 / precision settings, or strengthen the algorithm-level correction |
| `train/is_clip_ratio`, `train/is_nan_ratio` | The fraction of tokens where the IS weight goes out of bounds and the log-ratio is clamped | A high value means the bounds are too tight or the deviation is too large |
| `rollout/partial_ratio`, `rollout/partial_span_max` | The fraction of trajectories spanning weight versions and the maximum span | The degree of off-policyness in async / partial scenarios, see [Fully Async Mode](./fully-async-mode.md) |

In practice, for MoE models the recommended baseline is R3 + IS correction.
