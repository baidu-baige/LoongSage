# ResourceScheduler

ResourceScheduler 是 LoongSage 的 GPU 编排层，实现于 `coda/resource_scheduler/`。它在训练启动时一次性申请整个作业的 GPU，并把训练 Worker、教师 Worker、rollout 引擎放置到确定的物理 GPU 上。它的目标是：

- **一次申请，集中分配**：创建单个 Ray placement group 持有作业全部 GPU，后续所有 actor 都从这个池子里切分，避免各模块各自向 Ray 申请资源导致的碎片与竞争；
- **物理位置可确定**：把 bundle 按 `(节点 IP, GPU ID)` 全局排序，使「第 N 个 bundle」在任何一次运行中都对应同一张物理卡，这是共置模式下 CUDA IPC 权重传输能成立的前提；
- **共置与分离两种拓扑统一**：通过游标（cursor）策略的差异，用同一套代码同时支持训推共置（同卡复用）与训推分离（独占各自 GPU）；
- **支持故障恢复原地重建**：引擎挂掉后可以按原 bundle 索引重新拉起，回到同一张卡。

ResourceScheduler 不负责进程内的并行拓扑（TP/PP/EP 由 Megatron 与 SGLang 自己决定），也不负责 actor 的生命周期管理（由 TrainManager / TeacherManager / RolloutManager 负责）。它只回答一个问题：「这个 actor 应该跑在哪张卡上」。

## 1. 关键配置

调度器不引入自己的配置项，GPU 总量完全由各模块的规模参数推导：

| 参数 | 说明 |
| --- | --- |
| `colocate` | 是否训推共置。决定 GPU 总量的计算方式与游标策略。 |
| `trainer.num_nodes` × `trainer.num_gpus_per_node` | 训练 Worker 数量，每个 Worker 独占一张卡。 |
| `rollout.sglang_replicas[*].num_nodes` × `rollout.num_gpus_per_node` | 各 replica 组的 GPU 数，仅在分离模式下计入总量。 |
| `opd.teacher_nodes` × `opd.teacher_gpus_per_node` | 教师模型 GPU 数，仅在 `opd.enable=true` 且分离模式下计入总量。 |

GPU 总量按模式计算：

- **共置模式**：`num_gpus = trainer.num_nodes × trainer.num_gpus_per_node`。rollout 引擎和教师 Worker 与训练 Worker 共享同一批卡，不额外计入。
- **分离模式**：`num_gpus = 训练 GPU + rollout GPU + 教师 GPU`，三者各自独占。分离模式下断言 `rollout.backend == "sglang"`。

## 2. Placement group 的创建与排序

`create_placement_group` 分三步：

1. 用 `PACK` 策略申请 `num_gpus` 个 `{"CPU": 1, "GPU": 1}` 的 bundle，并 `ray.get(pg.ready())` 阻塞等待全部就绪。`PACK` 会尽量把 bundle 挤在少数节点上以减少跨机通信，但它是 best-effort 而非 `STRICT_PACK`，资源不足时仍会跨节点铺开。
2. 逐个 bundle 拉起一个临时 `Probe` actor，读取该 bundle 落在哪个节点 IP 与哪个物理 GPU ID，读完立即 `ray.kill`。单个 bundle 探测失败会重试 `MAX_PROBE_RETRIES`（默认 3）次，仍失败则直接抛错终止启动。
3. 把探测结果按 `(_ip_sort_key(ip), gpu_id)` 排序，得到 `reorder_bundle_list`。

第 3 步是整个模块的关键。Ray 分配 bundle 的顺序是不确定的：`placement_group_bundle_index=0` 可能落在任意节点的任意卡上。排序之后，`reorder_bundle_list` 的下标就成了一个稳定的全局 GPU 编号——同一节点内按 GPU ID 递增，节点之间按 IP **按段数值**排列（`_ip_sort_key` 把 IPv4 拆成四段整数比较，避免字典序把 `10.0.0.9` 排到 `10.0.0.84` 之后；非 IPv4 地址回退为原字符串比较）。下游因此可以假设「索引相邻的 bundle 大概率在同一台机器上、且 GPU ID 连续」，NVLink 亲和性与 CUDA IPC 才有意义。

`reorder_bundle_list` 每个元素形如 `{"pg", "p_idx", "ip", "gpu_id"}`，其中 `p_idx` 是 Ray 原始的 bundle 下标，`gpu_id` 是物理 GPU ID（SGLang 引擎需要它来设置 `base_gpu_id`，见 [`replica_group.py`](../../coda/backends/replica_group.py)）。

## 3. 分配策略：游标

`schedule` 按游标顺序从 `reorder_bundle_list` 切分 bundle。游标的键由模式决定：

| 模式 | 游标键 | 效果 |
| --- | --- | --- |
| 共置（`colocate=true`） | 每个角色一个游标，键为 actor 类名 | 每种角色都从索引 0 开始分配，因此训练 Worker `i` 与 rollout 引擎 `i` 落在同一张卡上 |
| 分离（`colocate=false`） | 全局共享一个游标 `_global` | 所有角色顺序瓜分卡池，互不重叠 |

分离模式下的分配顺序由 [`trainer.py`](../../coda/controller/trainer.py) 里各 Manager 的构造顺序决定：**训练 Worker → 教师 Worker → rollout 引擎**。也就是说训练占据低位索引，教师紧随其后，rollout 引擎拿走剩余的高位 bundle。

各角色的申请粒度：

- 训练 Worker：每个 rank 一次 `schedule(cls)`，`num_bundles=1`，共 `world_size` 次（[`train_manager.py`](../../coda/controller/train_manager.py)）。
- 教师 Worker：按 group × rank 逐个申请，每次一个 bundle（[`teacher_manager.py`](../../coda/controller/teacher_manager.py)）。
- rollout 引擎：每个引擎申请 `min(num_gpus_per_replica, rollout.num_gpus_per_node)` 个 bundle（[`replica_group.py`](../../coda/backends/replica_group.py)）。多机引擎按节点拆成多个 actor，每个 actor 占一个节点的卡。

注意 `num_bundles > 1` 时，调度器会连续推进游标占用这些 bundle，但只用**首个** bundle 作为 `PlacementGroupSchedulingStrategy` 的落点。其余 bundle 相当于被「预留」——真正的多卡占用由引擎进程内部（SGLang 自己按 `base_gpu_id` 铺开 TP）完成，而不是由 Ray 按 bundle 约束。

## 4. schedule() 的返回值与资源声明

```python
prepared_actor, bundle_index = scheduler.schedule(ray_actor_cls, num_bundles=1, recover_bundle_index=-1)
handle = prepared_actor.remote(...)   # 调用方自己实例化
```

`schedule` 只做「绑定放置策略」，不创建 actor，返回配置好的 `ActorClass` 与首个 bundle 在 `reorder_bundle_list` 中的索引。调用方需要保存这个索引以便故障恢复。

一个容易误解的细节：`prepared_actor` 被强制设置为 `num_cpus=0.1, num_gpus=0.1`。这里的小数值不代表真实用量——GPU 的独占性已经由 bundle 划分保证，声明小值只是为了让 Ray 的调度器放行，同时避免同一 bundle 内 actor 之间因资源账面不足而互相阻塞。因此 **Ray dashboard 上看到的 GPU 占用数字不反映真实使用情况**。

## 5. 故障恢复

传入 `recover_bundle_index >= 0` 时，调度器跳过游标逻辑，直接把 actor 放回指定索引的 bundle。rollout 引擎在 `start_engines(recover=True)` 时用记录下来的 `engine_bundle_indices[i]` 走这条路径（[`replica_group.py`](../../coda/backends/replica_group.py)），保证重建后的引擎回到原来的物理卡上，权重传输的拓扑关系（同卡 IPC / 跨卡 NCCL）不发生变化。

## 6. Probe：IP 与端口探测

`Probe` 是一个 `num_gpus=1` 的一次性 actor，承担两类探测任务：

| 方法 | 用途 |
| --- | --- |
| `get_ip_and_gpu_id` | 建 placement group 时探测每个 bundle 的物理位置 |
| `get_free_port` | 在指定节点上找 `port_num` 个**连续**空闲端口，随机选起始端口，范围默认 `[15000, 50000]`，最多尝试 100 次 |

`get_gloo_master_address` 组合两者，返回 `(ip, port)` 供 TransferMesh 的 gloo 环初始化使用；它固定申请 3 个连续端口（TransferMesh 建组需要），只返回第一个。调用点在 [`trainer.py`](../../coda/controller/trainer.py)，位于 rollout 引擎创建之后。

这里有一个已知的时序约束：`get_free_port` 带 60 秒超时，且 `Probe` 需要 GPU 资源才能调度起来。如果目标 bundle 的 GPU 已被先前调度的 Worker 占满账面资源，Probe 会一直 pending 直到超时。当前实现通过在 `options` 里降到 `num_gpus=0.1` 来缓解，并在超时错误信息里明确提示了这一根因。

## 7. 已知限制

- **bundle 规格固定**为 `{"CPU": 1, "GPU": 1}`，无法为需要更多 CPU 的角色单独放宽；CPU 密集型工作（如沙箱）不通过本调度器管理。
- **共置模式的 GPU 总量只按训练侧计算**，若 rollout 或教师侧配置的 GPU 数超过训练侧，会在分配时因游标越界抛出 `No available bundles to allocate for role ...`。
- **游标键是 actor 的类名**，共置模式下若两个不同角色复用同一个类，它们会共享游标而非各自从 0 开始。
- **不支持运行中扩缩容**：placement group 在 `__init__` 阶段一次性创建，之后只能在既有 bundle 内重建 actor。

## 延伸阅读

- [TransferMesh](transfer-mesh.md) — 依赖 bundle 排序结果决定 IPC / NCCL 路径
