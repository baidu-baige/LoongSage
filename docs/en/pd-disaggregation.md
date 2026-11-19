# PD Disaggregation

PD disaggregation (Prefill-Decode Disaggregation) splits one generation into two phases served by different SGLang engines: the prefill engine computes the prompt's KV cache and samples the first token, and the decode engine receives that KV cache over RDMA and keeps generating token by token. Each kind of engine owns its own GPUs and can use its own parallelism and replica count, so a long prompt's prefill no longer interrupts in-progress decoding. This suits agent workloads with long prompts and multi-turn contexts that keep growing.

The default configuration enables only the `regular` group, where each engine handles both prefill and decode with no split. With PD disaggregation enabled, the agent-facing interface is unchanged: agents still call the Router's chat completion endpoint, and the Router handles the split, the pairing and the merge of results.

## 1. Enabling It

Turn off `regular` and give GPUs to both `prefill` and `decode`. The example below places one TP4 prefill and one TP4 decode on a single 8-GPU node:

```yaml
rollout:
  num_gpus_per_node: 8
  sglang_replicas:
    regular:
      num_nodes: 0
    prefill:
      num_replicas: 1
      num_gpus_per_replica: 4
    decode:
      num_replicas: 1
      num_gpus_per_replica: 4
```

When both groups need whole nodes, `num_nodes` alone is enough, e.g. `prefill.num_nodes: 1` and `decode.num_nodes: 2`.

## 2. Key Parameters

| Parameter | Default | Description | Constraints / advice |
| --- | --- | --- | --- |
| `rollout.sglang_replicas.regular.num_nodes` | `1` | Nodes allotted to the regular group. | Must be `0` under PD, otherwise startup fails. |
| `rollout.sglang_replicas.{prefill,decode}.num_nodes` | `0` | Nodes allotted to each group; the GPU budget is `num_nodes × rollout.num_gpus_per_node`. | Both groups need replicas; enabling only one fails at startup. |
| `rollout.sglang_replicas.{prefill,decode}.num_replicas` | `null` | Explicit replica count; overrides `num_nodes` when set. | Required when a group needs fewer GPUs than one whole node. |
| `rollout.sglang_replicas.{prefill,decode}.num_gpus_per_replica` | `8` | GPUs per replica; the two groups may differ. | Must equal the group's `tp_size × pp_size`, otherwise weight-sync ranks are misaligned. |
| `rollout.sglang_replicas.{prefill,decode}.sglang_args` | `{disable_cuda_graph: false}` for decode | Overrides `rollout.sglang_args` for that group only. | CUDA graph is on for decode by default. |
| `rollout.sglang_args.disaggregation_ib_device` | unset | RDMA NICs used to transfer the KV cache. | Resolved automatically when unset, see Section 4. |
| `agentflow.router.rollout_worker_load_threshold` | `{regular: 32, prefill: 32, decode: 16}` | Per-worker-type in-flight threshold of one slot; above it the router drops stickiness and picks the least-loaded slot. | Must contain both a `prefill` and a `decode` key. The two pools differ in parallelism, so one engine carries a different number of requests; tune them separately. |
| `agentflow.router.max_connections` | `512` | Concurrency cap of Router forwarding. | A prefill/decode pair takes one permit but opens two connections. |

## 3. How It Works

### 3.1. Request Flow

1. The Router middleware rewrites the agent's request into an SGLang `/generate` body.
2. The Router picks one slot per worker type, i.e. one prefill slot and one decode slot. All turns of one trajectory attempt stick to the same pair of slots to reuse the prefix cache; once a slot's in-flight count exceeds its type's threshold, the Router picks the least-loaded slot instead.
3. The Router draws a random `bootstrap_room` and injects the same `bootstrap_host`, `bootstrap_port` and `bootstrap_room` into both bodies, with host and port taken from the chosen prefill engine. The two halves pair up on these three fields. When prefill runs DP attention, decode also gets `disagg_prefill_dp_rank` to locate the DP group that holds the KV cache.
4. Both requests are sent concurrently. The Router waits for prefill first; if prefill fails, it stops waiting for decode and tells decode to abort by `rid`.
5. The agent gets decode's response. Prefill's response is used only when R3 is on, to fill in the expert routing of the prompt range.

### 3.2. Engine Startup and Ports

Prefill and decode engines are laid out back to back on the rollout GPUs, so one node may host both kinds. Ports are allocated per node, and each worker type scans from its own starting port (regular 2000, prefill 4000, decode 6000). The prefill bootstrap port is `port + 2`.

### 3.3. Weight Sync

Engines of both pools are updated in the same weight update, and sampling resumes only after all of them finish, so there is no window where prefill and decode run different weight versions. RolloutManager assigns each engine's starting rank in the weight channel by accumulating `num_gpus_per_replica` engine by engine, so groups with different parallelism still tile the ranks correctly.

### 3.4. Offload Under Colocate

The KV transfer is an RDMA write by prefill into memory that decode registered. When colocate mode offloads, prefill is released first, and decode only after prefill finishes, so decode never deregisters its memory while prefill is still writing. Onload needs no staging.

## 4. RDMA NICs and GID

The KV cache is transferred over RDMA. NIC names are machine-specific, so they are resolved on the node each engine runs on, in this order:

1. An explicit `sglang_args.disaggregation_ib_device`;
2. `NCCL_IB_HCA` in `/etc/nccl.conf`;
3. The `NCCL_IB_HCA` environment variable;
4. When none of these is available, selection is left to the transfer backend's auto-discovery, which may pick NICs outside the RDMA fabric; a warning is logged.

The leading `=` and the `:port` suffixes in `NCCL_IB_HCA` are stripped; an exclusion list starting with `^` cannot be turned into a device list and is skipped.

mooncake reads the GID index from the environment only. Before starting SGLang, the engine exports `NCCL_IB_GID_INDEX` from `/etc/nccl.conf` into its environment; if `NCCL_IB_GID_INDEX` or `MC_GID_INDEX` is already set, it is left alone. This step does not depend on the transfer backend and has no effect on backends that do not read the variable.

## 5. R3 Routing Replay

With `trainer.use_rollout_routing_replay` on, the expert routing of one record under PD is stitched from both halves: prefill returns the prompt range `[start_len, prompt_len)`, and decode returns only the generated range `[prompt_len, seqlen - 1)`. The Router injects `routed_experts_start_len` into the decode request so decode does not transfer the prompt range again.

The KV transfer moves cache contents only, so decode has no routing record for the prompt range, which must therefore come from prefill. R3 under PD requires SGLang 0.5.23 with `docker/patch/sglang-0.5.23.patch` applied.

`trainer.routed_experts_dtype` applies to both halves; under `auto`, each half is decoded with the dtype it reports in `meta_info.routed_experts_dtype`.

## 6. Notes

- **Incompatible with the Rust server**: with `rollout.env_vars.SGLANG_RUST_SERVER` true, SGLang requires the bootstrap port to equal the HTTP port, while the framework always uses `port + 2`, so PD disaggregation fails right at startup.
- **Prefill failure**: when prefill returns a non-2xx status or the request raises, the Router tells decode to abort and returns 502 `Prefill worker failed` to the agent. A decode request error (including timeouts) also returns 502.
- **Removing a prefill engine**: an engine calls `exclude_worker` when it shuts down. Excluding a prefill also aborts the decode half of every session pinned to it; otherwise decode would keep waiting for the KV cache until SGLang's `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` (300 seconds by default).
- **Aborting one attempt**: `abort_session` aborts decode first, then prefill; if either fails it returns 503 and keeps the session's binding.
- **Tune the load thresholds separately**: a request holds one prefill slot and one decode slot at the same time and releases both only when the whole request finishes, so the difference in in-flight counts comes from the ratio of slots in the two pools; with 1P:3D, for example, one prefill slot carries about 3 times the requests of one decode slot. The benefit of stickiness also differs. In multi-turn agent workloads the context grows every turn, and sticking to the same prefill reuses its prefix cache and skips most of the prompt computation. Decode has no radix cache by default (`disaggregation_decode_enable_radix_cache: false`) and receives the full KV cache from prefill every turn, so stickiness gains almost nothing and its threshold can be lower for better balance. With that switch on, decode first matches the prefix locally and prefill transfers only the missed part, so sticking to the same decode avoids re-transferring the prefix, and the decode threshold should be raised accordingly.
