# ResourceScheduler

ResourceScheduler is LoongSage's GPU orchestration layer, implemented under `coda/resource_scheduler/`. It requests all GPUs for the job once at training startup and places training workers, teacher workers, and rollout engines onto deterministic physical GPUs. Its goals are:

- **Request once, allocate centrally**: create a single Ray placement group holding all GPUs of the job, from which every actor is carved out afterwards, avoiding the fragmentation and contention caused by each module requesting resources from Ray independently;
- **Deterministic physical placement**: sort bundles globally by `(node IP, GPU ID)` so that "the N-th bundle" maps to the same physical GPU on every run — the prerequisite for CUDA IPC weight transfer in colocate mode;
- **One code path for both topologies**: support colocated training/rollout (sharing GPUs) and disaggregated training/rollout (each owning its GPUs) through a difference in cursor strategy alone;
- **In-place rebuild after failures**: a dead engine can be relaunched on its original bundle index and thus return to the same GPUs.

ResourceScheduler is not responsible for in-process parallel topology (TP/PP/EP are decided by Megatron and SGLang themselves), nor for actor lifecycle management (owned by TrainManager / TeacherManager / RolloutManager). It answers exactly one question: "which GPU should this actor run on."

## 1. Key Parameters

The scheduler introduces no configuration of its own; the total GPU count is derived entirely from the scale parameters of each module:

| Parameter | Description |
| --- | --- |
| `colocate` | Whether training and rollout are colocated. Determines both how the GPU total is computed and the cursor strategy. |
| `trainer.num_nodes` × `trainer.num_gpus_per_node` | Number of training workers; each worker owns one GPU. |
| `rollout.sglang_replicas[*].num_nodes` × `rollout.num_gpus_per_node` | GPUs per replica group; counted only in disaggregated mode. |
| `opd.teacher_nodes` × `opd.teacher_gpus_per_node` | Teacher model GPUs; counted only when `opd.enable=true` and in disaggregated mode. |

The total is computed per mode:

- **Colocate mode**: `num_gpus = trainer.num_nodes × trainer.num_gpus_per_node`. Rollout engines and teacher workers share the same GPUs as training workers and are not counted separately.
- **Disaggregated mode**: `num_gpus = training GPUs + rollout GPUs + teacher GPUs`, each owning its own share. This mode asserts `rollout.backend == "sglang"`.

## 2. Placement Group Creation and Ordering

`create_placement_group` proceeds in three steps:

1. Request `num_gpus` bundles of `{"CPU": 1, "GPU": 1}` with the `PACK` strategy and block on `ray.get(pg.ready())` until all bundles are ready. `PACK` tries to squeeze bundles onto as few nodes as possible to reduce cross-machine traffic, but it is best-effort rather than `STRICT_PACK`, so it will still spread across nodes when resources are tight.
2. Launch a short-lived `Probe` actor on each bundle to read which node IP and which physical GPU ID the bundle landed on, then `ray.kill` it immediately. A failed probe is retried `MAX_PROBE_RETRIES` (3 by default) times; if it still fails, startup aborts with an error.
3. Sort the probe results by `(_ip_sort_key(ip), gpu_id)` to produce `reorder_bundle_list`.

Step 3 is the crux of this module. The order in which Ray assigns bundles is not deterministic: `placement_group_bundle_index=0` may land on any GPU of any node. After sorting, the index into `reorder_bundle_list` becomes a stable global GPU numbering — ascending by GPU ID within a node, and by **numeric** IP order across nodes (`_ip_sort_key` compares IPv4 as a tuple of four integers, so lexicographic sorting cannot put `10.0.0.9` after `10.0.0.84`; non-IPv4 addresses fall back to raw string comparison). Downstream code can therefore assume that "bundles with adjacent indices are most likely on the same machine with consecutive GPU IDs," which is what makes NVLink affinity and CUDA IPC meaningful.

Each element of `reorder_bundle_list` looks like `{"pg", "p_idx", "ip", "gpu_id"}`, where `p_idx` is Ray's original bundle index and `gpu_id` is the physical GPU ID (needed by SGLang engines to set `base_gpu_id`, see [`replica_group.py`](../../coda/backends/replica_group.py)).

## 3. Allocation Strategy: Cursors

`schedule` carves bundles out of `reorder_bundle_list` in cursor order. The cursor key depends on the mode:

| Mode | Cursor key | Effect |
| --- | --- | --- |
| Colocate (`colocate=true`) | One cursor per role, keyed by actor class name | Every role starts allocating from index 0, so training worker `i` and rollout engine `i` land on the same GPU |
| Disaggregated (`colocate=false`) | A single shared `_global` cursor | All roles carve up the GPU pool sequentially without overlap |

In disaggregated mode the allocation order follows the construction order of the managers in [`trainer.py`](../../coda/controller/trainer.py): **training workers → teacher workers → rollout engines**. Training therefore occupies the low indices, teachers follow, and rollout engines take the remaining high bundles.

Allocation granularity per role:

- Training workers: one `schedule(cls)` call per rank with `num_bundles=1`, `world_size` times in total ([`train_manager.py`](../../coda/controller/train_manager.py)).
- Teacher workers: one bundle per (group, rank) pair ([`teacher_manager.py`](../../coda/controller/teacher_manager.py)).
- Rollout engines: each engine requests `min(num_gpus_per_replica, rollout.num_gpus_per_node)` bundles ([`replica_group.py`](../../coda/backends/replica_group.py)). A multi-node engine is split into one actor per node, each occupying that node's GPUs.

Note that when `num_bundles > 1`, the scheduler advances the cursor across all of them but uses only the **first** bundle as the placement target for `PlacementGroupSchedulingStrategy`. The remaining bundles are effectively *reserved* — the actual multi-GPU occupancy happens inside the engine process (SGLang spreads its TP ranks itself starting from `base_gpu_id`), not through Ray's per-bundle constraints.

## 4. Return Value and Resource Declarations

```python
prepared_actor, bundle_index = scheduler.schedule(ray_actor_cls, num_bundles=1, recover_bundle_index=-1)
handle = prepared_actor.remote(...)   # the caller instantiates it
```

`schedule` only binds the placement strategy; it does not create the actor. It returns the configured `ActorClass` plus the index of the primary bundle in `reorder_bundle_list`. Callers must retain this index to support failure recovery.

One easily misread detail: `prepared_actor` is forced to `num_cpus=0.1, num_gpus=0.1`. These fractional values do not reflect real usage — GPU exclusivity is already guaranteed by the bundle partitioning, and declaring small values merely lets Ray's scheduler admit the actor while preventing actors on the same bundle from blocking each other on paper accounting. Consequently, **the GPU utilization numbers shown in the Ray dashboard do not reflect actual usage**.

## 5. Failure Recovery

When `recover_bundle_index >= 0` is passed, the scheduler bypasses the cursor logic and places the actor back on the bundle at that index. Rollout engines take this path during `start_engines(recover=True)` using the recorded `engine_bundle_indices[i]` ([`replica_group.py`](../../coda/backends/replica_group.py)), so a rebuilt engine returns to its original physical GPUs and the weight-transfer topology (same-GPU IPC / cross-GPU NCCL) stays unchanged.

## 6. Probe: IP and Port Discovery

`Probe` is a single-use actor declared with `num_gpus=1` that serves two kinds of discovery:

| Method | Purpose |
| --- | --- |
| `get_ip_and_gpu_id` | Discover the physical location of each bundle while building the placement group |
| `get_free_port` | Find `port_num` **consecutive** free ports on a given node; the starting port is picked randomly within `[15000, 50000]` by default, with up to 100 attempts |

`get_gloo_master_address` combines the two and returns `(ip, port)` for TransferMesh's gloo ring initialization. It always requests 3 consecutive ports (required to build the TransferMesh groups) and returns only the first. The call site is [`trainer.py`](../../coda/controller/trainer.py), after the rollout engines have been created.

There is a known timing constraint here: `get_free_port` has a 60-second timeout, and `Probe` needs GPU resources to be scheduled at all. If the target bundle's GPU accounting is already saturated by previously scheduled workers, the Probe stays pending until it times out. The current implementation mitigates this by lowering the request to `num_gpus=0.1` via `options`, and the timeout error message spells out this root cause explicitly.

## 7. Known Limitations

- **Bundle shape is fixed** at `{"CPU": 1, "GPU": 1}`; there is no way to grant more CPUs to a specific role. CPU-heavy work (such as sandboxes) is not managed by this scheduler.
- **Colocate mode sizes the pool from the training side only**. If the rollout or teacher configuration asks for more GPUs than training provides, allocation fails with `No available bundles to allocate for role ...` once the cursor runs past the end.
- **The cursor key is the actor class name**, so in colocate mode two distinct roles reusing the same class share one cursor instead of each starting from 0.
- **No runtime scaling**: the placement group is created once during `__init__`, and afterwards actors can only be rebuilt within the existing bundles.

## Further Reading

- [TransferMesh Weight Transfer](transfer-mesh.md) — relies on the bundle ordering to choose IPC / NCCL paths
