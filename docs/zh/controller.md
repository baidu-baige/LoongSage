# Single Controller

`coda.controller` 是 LoongSage 的控制面，负责组织 rollout、Teacher 推理、训练、checkpoint 和权重更新的执行顺序。Controller 不直接执行模型前向或参数更新，而是管理相关 Worker 的生命周期，并协调其他模块提供的数据、资源和传输能力。

本页分三部分：[总览](#总览) 描述包边界与内部架构，[核心组件](#核心组件) 逐个介绍 `Trainer` 与各 Manager，[执行流程](#执行流程) 描述各运行模式下一个 step 是怎么被编排的。
启动方式和运行模式的选择见 [运行指南](run-guide.md)。

## 总览

### 包职责

- 解析并校验 Controller 相关配置。
- 按运行模式创建并持有需要的 Manager：`TrainManager` 总是创建，`TeacherManager` 仅
  `opd.enable=true` 时创建，`RolloutManager` 在 `run_mode=train-only` 下不创建。
- 调度数据采样、轨迹处理和训练 batch 切分。
- 在同步训练中周期性评测，并上报评测指标。
- 组织同步训练、全异步训练、仅采样和仅训练流程。
- 协调 checkpoint、权重版本及训练到推理的权重更新。
- 处理共置场景下不同 Worker 的 offload 和 onload。

Controller 不负责实现 Agent、数据源、GPU 资源调度、模型后端或权重传输协议；这些能力由其他 LoongSage 包提供。

### 包内文件

`coda.controller` 当前核心源文件：

| 文件 | 核心对象 | 定位 |
| --- | --- | --- |
| `trainer.py` | `Trainer`、`Mode` | 顶层入口和训练流程编排 |
| `rollout_sampler.py` | `RolloutSampler`、`SlidingWindowStrategy` | 采样协调、评测协同和全异步滑动窗口策略 |
| `rollout_manager.py` | `RolloutManager` | Rollout Worker 生命周期管理 |
| `train_manager.py` | `TrainManager` | Train Worker 生命周期管理 |
| `teacher_manager.py` | `TeacherManager` | OPD Teacher Worker 生命周期管理 |

### Controller 内部架构

#### 组件层次

![LoongSage Controller 架构图](../_static/image/controller-architecture.svg)

图中的箭头统一表示调用方向或数据流向：

- Manager 之间的横向连线是训练循环中的协作，轨迹组从 `RolloutSampler` 派发，经 `TeacherManager` 附加 Teacher 前向结果后汇入 `TrainManager`。
- Manager 到下方 Worker 池的纵向 `manage` 连线是生命周期管理，即控制面作用于执行面的入口。
- `Train Workers` 到 `Rollout Workers` 的 `update weight` 连线是权重数据面，权重经 `TransferMeshChannel` 从训练 Worker 直达 Rollout Worker.

Worker 内部实现没有在图中展开。

#### 对象持有关系

`Trainer` 是唯一的顶层编排器，其初始化过程建立以下持有关系：

```text
Trainer
|-- scheduler          -> ResourceScheduler（跨包）
|-- datasources[]      -> RolloutDataSourceWithBuffer（跨包）
|-- train_manager      -> TrainManager
|-- teacher_manager    -> TeacherManager | None
|-- rollout_manager    -> RolloutManager
`-- rollout_sampler    -> RolloutSampler
                         |-- agentflow（跨包）
                         |-- traj_queue（跨包，由 AgentFlow 持有）
                         |-- data_filter（跨包）
                         |-- pipeline_buf（跨包，仅全异步）
                         `-- _strategy（Controller 包内，仅全异步）
```

这里的"持有"指 Controller 生命周期内的对象引用，而非类继承关系。三个 Manager 之间没有直接依赖，由 `Trainer` 按执行阶段协调。

各组件的详细职责、内部协调流程与跨包依赖见[核心组件](#核心组件)。

## 核心组件

本页详细介绍定义在 `coda.controller` 包中的核心类，包括其编排职责、采样协调和 Worker 管理的设计细节，并在最后一节列出它们依赖的跨包组件。

### Trainer

源码：[coda/controller/trainer.py](../../coda/controller/trainer.py)

`Trainer` 是 LoongSage 的顶层编排器，也是 Hydra 的模块入口。它不实现具体的 rollout 或训练算法，而是负责把各个阶段按配置组织起来。

#### 初始化职责

`Trainer.__init__` 按固定顺序完成四件事：

- **配置校验**：校验 `colocate`、`fully_async` 和数据源的组合约束。
- **依赖构建**：初始化 tracking，创建 `ResourceScheduler`、各数据源和 `AgentFlow`。
- **Manager 装配**：`TrainManager` → 可选 `TeacherManager`（`opd.enable=true`）→
  `RolloutManager` + `RolloutSampler`（`run_mode=train-only` 时二者都为 `None`）。
- **权重通道引导**：取得 gloo master 地址和端口，供后续构造 `ChannelMeta`。

#### 训练职责

`Trainer` 是每个训练 step 的编排入口，按 `run_mode` 依次驱动 rollout、可选 Teacher、训练和权重更新阶段，并负责 `rollout-only` 与 `train-only` 模式的磁盘数据交接，以及恢复时的 step 和数据源状态。

各阶段在 step 内部的执行顺序本页不展开，见[执行流程](#执行流程)。

### RolloutManager

源码：[coda/controller/rollout_manager.py](../../coda/controller/rollout_manager.py)

`RolloutManager` 管理由一个或多个推理 replica group 组成的 rollout 侧资源。每个 replica group 中的 Rollout Worker 数量和 GPU 配置由 rollout 配置决定。

#### 主要职责

三个 Worker 侧 Manager 共用一套职责分类：**Actor 拓扑构建**、**Worker 调用代理**、**显存生命周期**、**权重同步**、**故障处理**、**前向计算**。某个 Manager 不具备的分类直接省略。

- **Actor 拓扑构建**：按 `rollout.sglang_replicas` 的 `regular` / `prefill` / `decode` 构建 `ReplicaGroup` 和 `SglangEngine` Ray Actor。
- **Worker 调用代理**：把控制调用转发到全部 Rollout Worker 并汇总结果。
- **显存生命周期**：offload 释放全部显存，onload 支持 weights 与 KV cache 分阶段恢复。
- **权重同步**：作为接收侧，为每个 Rollout Worker 下发带 `engine_id` 和 `weight_version` 的 `ChannelMeta`。
- **故障处理**：通过 `RolloutHealthMonitor` 探活，重建失联的 Rollout Worker，并告知 `Trainer` 是否需要重建传输通道。

#### 组件关系

```text
RolloutManager
    |
    +-- replica_groups: list[ReplicaGroup]
    |       `-- all_engines: SglangEngine x N（Ray Actor）
    |
    `-- _health_monitors: list[RolloutHealthMonitor]
            （仅 rollout.use_fault_tolerance=true 时创建，每个 ReplicaGroup 一个）
```

`Router` 不属于 `ReplicaGroup`，也不由 `RolloutManager` 创建。Router 定义在 `coda.agentflow` 中、由 `AgentFlow` 持有；`ReplicaGroup` 只是把 `agentflow.router` 的 ip 和端口透传给每个 Rollout Worker，由 Worker 启动后自行向 Router 注册。

RolloutManager 不负责生成 prompt，也不负责决定一个 step 需要多少条轨迹；这些策略由 `RolloutSampler` 和数据源负责。

### TrainManager

源码：[coda/controller/train_manager.py](../../coda/controller/train_manager.py)

`TrainManager` 管理训练侧的 Ray Actor 池。当前实现根据 `trainer.backend` 创建 `MegatronTrainWorker`，并为每个 Worker 分配 GPU 资源。

#### 主要职责

- **Actor 拓扑构建**：按训练 world size 为每个 rank 创建一个 `MegatronTrainWorker` Actor。
- **Worker 调用代理**：把 init、train、save_model、update_weights 等调用转发到全部 rank，返回句柄由 `Trainer` 决定何时等待。
- **显存生命周期**：共置模式下在 GPU 与 CPU 之间搬运参数与优化器状态。
- **权重同步**：作为发送侧，接收 `ChannelMeta` 并把权重写入 `TransferMeshChannel`。

训练并行度由 Megatron 配置和 LoongSage trainer 配置共同决定。Controller 只负责 Worker 池的生命周期，不替代 Megatron 对 TP、PP、CP、EP、DP 等并行策略的实现。当前对非 `megatron` backend 会直接报错。

### TeacherManager

源码：[coda/controller/teacher_manager.py](../../coda/controller/teacher_manager.py)

`TeacherManager` 是 OPD（Online Policy Distillation）场景下的可选控制器。只有 `opd.enable=true` 时，`Trainer` 才会创建它。

#### 主要职责

- **Actor 拓扑构建**：按 Teacher 并行度把 `opd.teachers` 切分到多个 group，每个 group 一组 `MegatronTeacherWorker` Actor。
- **Worker 调用代理**：把 init、onload、offload、compute_teacher 转发到全部 Teacher Actor。
- **显存生命周期**：共置模式下与训练、rollout 错峰占用显存。
- **前向计算**：`compute_teacher()` 对 `Trainer` 传入的 DP shard 做 Teacher 前向，把结果引用挂到对应 shard 上。

`compute_teacher()` 不产生新的 batch 对象，而是就地扩充 Trainer 已经切分好的 DP shard 引用；因此启用与不启用 OPD 的差别只在于每个 shard 是否携带 `teacher_worker_ref`。

TeacherManager 与 TrainManager 使用相似的资源管理方式，但二者生命周期和配置语义不同。Teacher 不是普通训练 Worker 的别名，也不应在未启用 OPD 时出现在主训练链路中。

### RolloutSampler

源码：[coda/controller/rollout_sampler.py](../../coda/controller/rollout_sampler.py)

`RolloutSampler` 连接数据源、AgentFlow 和 Trainer。它负责把“本 step 要采哪些数据”转换为“可被训练消费的 TrajectoryGroup 列表”。

#### 主要职责

`RolloutSampler` 不管理 Worker，按数据流水线阶段划分职责：

- **数据源消费**：从各 `RolloutDataSourceWithBuffer` 取出本 step 的 prompt group，并决定补采样数量。
- **任务派发**：为每个 prompt group 创建一个 `AgentFlow.generate()` 任务。
- **轨迹聚合**：从 AgentFlow 的 `TrajQueue` 消费按 prompt 聚合完成的轨迹组。
- **准入过滤**：用 `DataFilter` 判定轨迹组是否进入训练，并汇总采样指标。
- **评测协同**：评测轮次与训练 rollout 同批派发评测集，把评测轨迹从训练 batch 中分流出来，并聚合上报 `eval/*` 指标。
- **线程与背压控制**：全异步模式下运行 rollout 与 collector 线程，按 `PipelineBuffer` 水位控制派发额度，并维护 pause / resume / stop 状态。

`RolloutSampler` 不负责分布式训练 batch 的最终 DP 切分；该步骤由 `Trainer` 调用 data processor 完成。

#### 采样协调

同步和全异步模式共用数据处理逻辑，调度方式不同：

```text
同步模式
Trainer -> RolloutSampler.__call__()
        -> dynamic_rollout()
        -> 返回 TrajectoryGroup[]

全异步模式
Trainer 主线程                    rollout 线程                 collector 线程
     |                                 |                            |
     |                         rollout_loop()                       |
     |                                 |                            |
     |                    SlidingWindowStrategy                     |
     |                    计算可派发 group 数                         |
     |                                 |                            |
     |                           AgentFlow tasks                    |
     |                                 |                            |
     |                                 `------> TrajQueue ----------+
     |                                                              |
     `---------------- async_get() <--- PipelineBuffer <--- 完整轨迹组
```

全异步调度策略由 `create_strategy()` 根据 `fully_async.sliding_window` 创建，并由 `_ThreadSafeStrategy` 串行化跨线程调用：

- `no-window`：按照最大在途容量补充任务。
- `window-gated`：使用窗口门控限制派发。
- `windowed-fifo`：按窗口和 FIFO 语义管理序列。

`RolloutSampler` 同时维护 pause、resume、stop、cleanup 和 fatal error 状态，使后台线程的异常和停止信号能够传递到训练主线程。

### 跨包协作组件

以下组件不是 Controller 包的组成部分，本目录只描述 Controller 与它们的协作关系：

| 组件 | 所属模块 | Controller 中的协作关系 |
| --- | --- | --- |
| `ResourceScheduler` | `coda.resource_scheduler` | 创建 Ray placement group，并为各 Manager 的 Actor 分配 GPU bundle |
| `RolloutDataSourceWithBuffer` | `coda.data_factory` | 提供 prompt、cursor 和 unused-prompt buffer 状态 |
| `AgentFlow` | `coda.agentflow` | 执行 Agent、持有 Router 与 `TrajQueue`、驱动 SGLang 推理并产出轨迹 |
| `DataFilter` | `coda.data_factory` | 由 `RolloutSampler` 用于过滤轨迹组 |
| `PipelineBuffer` | `coda.utils` | 连接全异步 rollout 生产线程和训练消费线程 |
| `eval_utils` | `coda.utils` | 由 `RolloutSampler` 在评测轮次调用，聚合评测轨迹并产出 `eval/*` 指标 |
| `ReplicaGroup` / `SglangEngine` | `coda.backends` | 承载 SGLang rollout replica 和 Rollout Worker Ray Actor |
| `MegatronTrainWorker` / `MegatronTeacherWorker` | `coda.backends` | 执行训练与 Teacher 前向 |
| `RolloutHealthMonitor` | `coda.utils` | 由 `RolloutManager` 用于 Rollout Worker 健康检查 |
| `TransferMeshChannel` | `coda.transfer_mesh` | 在 Train Worker 与 Rollout Worker 之间传输权重 |

这些组件的详细设计和 API 分别维护在对应模块的文档目录中。

## 执行流程

### 启动流程

Controller 的入口是 `coda.controller.trainer`。启动后，`Trainer` 先完成配置和资源初始化，再进入指定运行模式。

### 完整同步流程

`run_mode=default` 且未启用全异步时，一个训练 step 的逻辑顺序如下：

```text
1. 从数据源获取 prompt
2. RolloutSampler 调用 AgentFlow 采样训练与评测 prompt
3. TrajQueue 聚合完整 TrajectoryGroup
4. DataFilter 筛选轨迹
5. Trainer 按 DP 切分 batch
6. 可选 Teacher / OPD 计算
7. TrainManager 调用 MegatronTrainWorker
8. 保存 checkpoint 和指标
9. 更新 SGLang rollout 权重及 weight_version
10. 进入下一个 step
```

### 周期性评测流程

评测没有独立的运行模式，它在同步 rollout 内部完成。每个 step 先判定本轮是否为评测轮次：

```text
                              step
                                |
                                v
        interval > 0 且（step % interval == 0 或 step == 1 或 step == total_steps）?
                 |                                        |
                no                                       yes
                 |                                        |
                 v                                        v
          只派发训练 prompt                  训练 prompt + 评测集整集一起派发
                                                          |
                                                          v
                                       评测轨迹按 is_eval 分流，聚合为 eval 指标
```

评测轨迹沿用父训练数据源的 `ds_index`，只通过 `is_eval` 和轨迹 id 中的 `_eval` 段与训练轨迹区分。

评测轨迹组不进入训练链路：不过滤、不计入 quota 与 refill、不计入 `rollout/*` 指标；清理阶段残留的评测轨迹会被丢弃，不会回灌进数据源 buffer 被当作训练数据。

#### 当前不支持的场景

- `fully_async.enable=true` 时没有周期性评测。全异步模式的 rollout 在后台线程中执行，不经过 `dynamic_rollout`。

### 全异步流程

当 `fully_async.enable=true` 时，`Trainer` 会启动后台 rollout 线程。rollout 和训练不再严格按一个完整 step 交替，而是通过 `PipelineBuffer` 形成流水线：

```text
后台 rollout 线程                         训练主线程
       |                                     |
       v                                     |
读取数据源 -> AgentFlow -> TrajQueue           |
       |                                     |
       v                                     |
收集完整轨迹 -> PipelineBuffer --------------+--> async_get()
                                             |
                                             v
                                  DP 切分 -> 训练 batch
                                             |
                                             v
                                      Megatron training
```

Sampler 在后台线程中维护以下状态：

- `pause`：停止继续派发新的 rollout，并根据 partial rollout 配置清理在途任务。
- `resume`：重新打开运行门，继续产生 rollout。
- `stop`：通知生产线程退出，并将终止状态传递给训练循环。
- fatal error：后台线程发生不可恢复异常时，交由训练主循环处理。

全异步模式存在比同步模式更严格的配置约束。完整的参数和约束清单见 [全异步模式](fully-async-mode.md)；运行时以 `Trainer._validate_config()` 和当前配置文件为准。

### Rollout-only

`run_mode=rollout-only` 只执行采样和轨迹预处理，不创建训练更新链路的最终消费步骤：

```text
数据源 -> RolloutSampler -> 过滤 -> DP 切分 -> 磁盘 batch
```

保存前，Trainer 会为轨迹写入当前 `weight_version`，并按照训练所需的 DP 结构组织数据。该模式适用于预先生成 rollout 数据，之后使用 `train-only` 复用。

Rollout-only 不代表不需要训练 backend 的配置。数据切分和 batch 形态仍然依赖 trainer 的并行度、mini batch 等配置。

### Train-only

`run_mode=train-only` 从磁盘加载已经保存的 batch，不再执行 rollout：

```text
磁盘 batch
    -> 恢复数据源 / step
    -> 加载并按 DP 放入 Ray
    -> TrainManager
    -> MegatronTrainWorker
    -> checkpoint / 指标
```

Train-only 不创建 `RolloutManager` 和 `RolloutSampler`，因此整个流程里没有权重传输通道，也没有一次
`_update_weights` 调用；`_post_train_weight_version()` 只把 `weight_version` 计数往前推，供 checkpoint
记录使用。它要求磁盘数据的结构与当前训练配置兼容，包括 DP 切分、mini batch 和轨迹字段。

### 轨迹到训练 batch

Trainer 使用 data processor 将 sampler 返回的 `TrajectoryGroup` 转换为训练输入：

```text
TrajectoryGroup 列表
        |
        +--> 添加当前 weight_version
        +--> 计算 total trajectories
        +--> 计算 mini-batch 数量
        +--> split_traj_group_by_dp(...)
        `--> put_dp_shards_to_ray(...)
                         |
                         v
                   Ray ObjectRef 列表
```

在完整训练模式中，Ray ObjectRef 被传给训练 Worker；在 rollout-only 模式中，DP shard 被保存到磁盘并返回空列表；在 train-only 模式中，磁盘 batch 直接恢复为 Ray 引用。

启用 OPD 时，`TeacherManager.compute_teacher()` 在这一步之后被调用，它不重建 batch，而是把 Teacher 前向结果的 ObjectRef 追加到每个 DP shard 的 `teacher_worker_ref` 字段上。

### 权重更新流程

`_update_weights()` 只在 `run_mode=default` 和 `rollout-only` 下被调用（`train-only` 没有 rollout 侧，
不走这个流程）。Trainer 用同一个 `ChannelMeta` 同时向训练侧和 rollout 侧发起更新，并等待两侧全部完成：

```text
Trainer._update_weights(meta, weight_version)
  |
  +--> TrainManager.async_update_weights(meta)      -> 训练 Worker 导出权重
  |
  `--> RolloutManager.async_update_weights(meta, v) -> 每个 Rollout Worker 一份 ChannelMeta
                     |
                     v
        权重经 TransferMeshChannel 传输到 Rollout Worker
                     |
                     v
        Rollout Worker 侧 weight_version = v
```

### 恢复和 checkpoint

恢复流程的目标不仅是加载模型，还要恢复数据消费位置：

1. 读取训练 checkpoint 和 step。
2. 恢复各数据源 cursor 以及 replay/buffer 状态。
3. 初始化当前 `weight_version`。
4. 重新建立 Worker 和权重传输通道。
5. 让 rollout 与训练使用一致的模型版本继续执行。

相关实现位于 [trainer.py](../../coda/controller/trainer.py) 和 [checkpoint_utils.py](../../coda/utils/checkpoint_utils.py)。

### 运行模式对照

各运行模式的配置参数和使用方法见 [配置参数参考 - 顶层运行参数](config-reference.md#1-顶层运行参数)。下表从执行视角补充权重行为差异：

| 配置 | Rollout | 训练 | 权重行为 | 周期性评测 |
| --- | --- | --- | --- | --- |
| `run_mode=default` | 按 step 执行 | 执行 | 每次训练后同步新权重到 Rollout Worker | 支持 |
| `run_mode=default` + `fully_async.enable=true` | 后台执行 | 执行 | 训练后的权重按异步流程更新 Rollout Worker | 不支持 |
| `run_mode=rollout-only` | 执行 | 不执行 | 只在启动时同步一次初始权重，训练中不产生新权重 | 支持，但权重始终为初始权重 |
| `run_mode=train-only` | 不执行 | 执行 | 无权重传输通道，也不做权重更新；训练产生的新权重只落盘 | 不执行（无 rollout 阶段） |
