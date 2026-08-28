# TransferMesh

TransferMesh 是 LoongSage 中训练进程与推理进程之间的权重同步组件，实现于 `coda/transfer_mesh/`。每个训练 step 结束后，训练侧需要将最新参数同步至 SGLang 推理引擎，供下一轮 rollout 使用。TransferMesh 的目标是：

- 自动感知发送方（trainer worker）和接收方（rollout engine 内的 TP worker）在物理拓扑上的位置关系，同卡使用 CUDA IPC 零拷贝，跨卡使用 NCCL 广播；
- 通过 GPU 侧的桶聚合（阈值由 `buffer_size_bytes` 控制，默认 1 GiB，可按需调整），将大量小张量聚合到少量通信轮次；
- 与 `torch.distributed` 已有的 NCCL / Gloo 组解耦，不依赖默认 process group，可在 Ray Actor 内独立初始化和销毁；
- 与 SGLang 的 `update_weights_from_channel` HTTP 接口对接，实现在线热替换权重。

TransferMesh 不负责 Megatron 的 TP/PP 分片聚合（该部分交给 `megatron.bridge.export_hf_weights`），也不负责 SGLang 内部将接收到的张量写入其模型权重的具体过程（该部分由 SGLang 引擎实现）。TransferMesh 只保证"对于发送方提供的张量，接收方能获得形状与名称完全一致的完整张量"。

## 1. 关键参数

TransferMesh 通道通过 [`ChannelMeta`](../../coda/utils/channel_helper.py) 创建，运行时由 [`Trainer._init_channel_meta`](../../coda/controller/trainer.py) 统一构造。`ChannelMeta` 在代码中显式区分了 **必填字段（Required）** 与 **可选字段（Optional，均带默认值）**，业务代码只需要提供必填字段即可拉起通道。

### 1.1. 必填字段（Required）

以下三个字段没有默认值，构造 `ChannelMeta` 时必须显式提供；三者用于确定通道的 rendezvous 点、总参与者数量以及发送/接收 rank 的分段。

| 字段 | 取值 | 说明 |
| --- | --- | --- |
| `master_addr` | 由 `scheduler.get_gloo_master_address()` 提供 | rank 0 用于 Gloo TCPStore 集合的 IP。 |
| `world_size` | `train_world_size + rollout_world_size` | 参与本通道的所有 rank 数，等于训练 GPU 数与 rollout GPU 数之和。 |
| `train_world_size` | trainer GPU 数 | 训练侧 world size；接收方 rank 会加上该值以避免与发送方 rank 冲突（详见 `_compute_rank`）。 |

### 1.2. 可选字段（Optional，均带默认值）

以下字段在 `ChannelMeta` 中都提供了默认值，仅在需要偏离默认行为时才显式覆盖。

| 字段 | 默认值 | 说明 |
| --- | ---: | --- |
| `engine_id` | `0` | 多 replica 场景下用于区分不同 SGLang engine，参与 rank 计算：`engine_id * dist.get_world_size() + dist.get_rank()`（此处 `dist.get_world_size()` 为训练 / 推理进程各自的 dist world size，与本表中的通道级 `world_size` 不同；接收方额外 `+ train_world_size`）。 |
| `recreate` | `False` | 是否强制关闭并重建 process-local channel。Rollout 侧检测到新 engine（`num_new_engines > 0`）时会置为 `True`；启用 `rollout.use_fault_tolerance` 后，`_init_channel_meta` 会先调用 `recover_faulty_engines()`，故障恢复通过增加 `num_new_engines` 走到同一分支。 |
| `buffer_size_bytes` | `1 GiB` | 发送侧的桶 flush 阈值。默认值定义于 [`channel.py:104`](../../coda/transfer_mesh/channel.py)；累计字节数 ≥ 该值时触发 flush，单张量超阈值时会按 `max(阈值, tensor_bytes)` 分配桶，因此实际桶大小并非固定 1 GiB。 |
| `timeout` | `300s` | Gloo/NCCL 集合操作的超时。 |
| `gloo_port` | `29500` | Gloo TCPStore 端口。TransferMesh 内部还会使用 `port+1`（NCCL 数据组，NCCL 后端）和 `port+2`（NCCL 元数据组，Gloo 后端，通过 `broadcast_object_list` 传输控制帧）。 |
| `local_ip` | `None` | 本地 IP；未指定时按 `HOST_IP` 环境变量 → Ray 的顺序自动解析；若都失败则抛 `RuntimeError`。 |

`gpu_id`、`rank` 与 `addr` 由 [`channel_helper.py`](../../coda/utils/channel_helper.py) 在进程内自动解析，无需业务代码填写。发送方与接收方仅提供 `Role.SENDER` / `Role.RECEIVER` 即可。

## 2. 原理

下图突出 TransferMesh 作为核心通道的定位：中央实线橙框为 TransferMesh 本身，内部按 **① 发送侧（Send Path，`send_channel.send` + GPU 桶缓冲 + `_flush_bucket`）→ ② 传输通道（Transport，Gloo 主组 / NCCL 数据组 / NCCL 元数据组 + `partition_receivers` 拓扑分区）→ ③ 接收侧（Recv Path，`get_iterator` 拆桶 + `payload.view.clone()` 独立副本）** 三个阶段组织；两侧虚线灰框为 Worker 侧调用入口——发送端（trainer worker）通过 `bridge.export_hf_weights` 逐张量将权重递交给 TransferMesh，接收端（SGLang TP worker）通过 `update_weights_from_channel` 消费权重流并写入模型；`partition_receivers` 下方的两个徽章分别对应同卡 IPC 零拷贝路径与跨卡 NCCL 广播路径。

![TransferMesh 架构图](../_static/image/transfer-mesh-architecture.svg)

### 2.1. 组件

TransferMesh 由三个模块组成：

| 文件 | 核心对象 | 定位 |
| --- | --- | --- |
| [`topology.py`](../../coda/transfer_mesh/topology.py) | `Role`、`RankInfo`、`partition_receivers` | 拓扑发现和 IPC/NCCL 分区 |
| [`protocol.py`](../../coda/transfer_mesh/protocol.py) | `TensorSpec`、`MetaFrame`、`str_to_dtype` | 桶级元数据协议 |
| [`channel.py`](../../coda/transfer_mesh/channel.py) | `TransferMeshChannel`、`create_channel` | 通道生命周期、桶聚合、发送/接收路径 |

`TransferMeshChannel` 由 `create_channel()` 构造，在构造函数返回后自动调用 `init_groups()`，一次性完成 Gloo 组建立、拓扑收集、NCCL 组建立三步；[`channel_helper.py`](../../coda/utils/channel_helper.py) 并在每个进程内保存一个 module-level `_channel` 单例，供 `create_sender_channel(meta)` / `create_receiver_channel(meta)` 复用。

### 2.2. 通信拓扑

`init_groups` 通过 `torch.distributed.distributed_c10d` 的私有 API（由 [`_c10d_compat.py`](../../coda/transfer_mesh/_c10d_compat.py) 抽象）**独立创建 process group**，不依赖默认 process group。这样即使 Megatron 或 SGLang 已经用 NCCL 初始化了默认组，TransferMesh 仍可以在同一进程内并存自己的 Gloo/NCCL 组。

拓扑发现只需一轮 `all_gather_object`：每个 rank 广播一份 `RankInfo(rank, gpu_id, ip, role)`，随后所有 rank 都能看到完整的部署视图。`partition_receivers` 按 `(ip, gpu_id)` 对每个接收方查表：

- 命中一个同位置的发送方 → 加入该发送方的 `ipc_assignments`（同卡 IPC）；
- 未命中 → 归入 `nccl_receivers`（由 `src_rank` 通过 NCCL 广播）。

若两个发送方共享相同的 `(ip, gpu_id)`，`partition_receivers` 会抛出 `ValueError`（见 [`topology.py:66-70`](../../coda/transfer_mesh/topology.py)），用于在拓扑构造阶段尽早暴露配置错误。此外，进程侧的 `gpu_id` 并非直接使用 `torch.cuda.current_device()`，而是由 [`channel_helper.py`](../../coda/utils/channel_helper.py) 中的 `_get_physical_gpu_id()` 通过解析 `CUDA_VISIBLE_DEVICES`（以 `torch.cuda.current_device()` 为下标取出对应的物理 id；该变量为空、解析失败或下标越界时回退为 `torch.cuda.current_device()` 返回的逻辑设备索引）还原为物理 GPU 索引；这一步是 IPC 路径能够正确匹配的前提，因为 Ray Actor 内可见的本地 device index 始终为 0，与物理拓扑无关。

只有存在跨卡接收方时才会额外创建两个组（各自使用独立的 TCPStore 端口）：

- `_nccl_data_group`（NCCL 后端）：承载张量 payload；
- `_nccl_meta_group`（Gloo 后端）：以 `broadcast_object_list` 广播 `MetaFrame` 控制帧。

同卡 IPC 不需要额外 group，直接在主 Gloo 组上进行点对点的 handle 传递。

### 2.3. 发送侧：桶聚合与流式转换

训练侧的发送逻辑集中在 [`megatron_train_worker.py:554-560`](../../coda/backends/megatron/megatron_train_worker.py)：

```python
send_channel = create_sender_channel(channel_meta)
for name, tensor in self.bridge.export_hf_weights(self.model):
    send_channel.send((name, tensor))
send_channel.send(None, flush=True)
```

- `bridge.export_hf_weights` 是 `megatron.bridge.AutoBridge` 返回的 **generator**，负责在发送前将 Megatron 的 TP/PP 分片通过 all-gather 等操作还原为标准 HuggingFace 完整格式，逐张量 yield，使用后即释放。发送与聚合逻辑因此对训练侧的并行策略透明。
- `send()` 内部维护一个桶 `self._buffer`（一维 uint8 扁平缓冲，按字节存储；dtype 由每张量的 `TensorSpec.dtype` 单独携带），行为如下：
  - 若剩余字节不足以放下下一张量，先 flush 再重新分配 `max(buffer_size_bytes, tensor_bytes)` 大小的桶；
  - 每张量通过 `flat_tensor = tensor.flatten().contiguous().view(torch.uint8)` 一次 D2D 拷贝进入桶，并追加一条 `TensorSpec(name, shape, offset, dtype)`（`offset` 为字节偏移）；非连续张量的 `flatten()` 会先分配一块连续副本，构成一次额外 D2D 拷贝；
  - 当累计字节数 ≥ `buffer_size_bytes`（默认 1 GiB）时调用 `_flush_bucket()` 发送整桶；
  - 调用 `send(None)` 或 `send(..., flush=True)` 会立即触发 `_flush_bucket(is_end=True)`，用于向接收方广播"流结束"标记，并作为发送流的正常终止路径。

**桶是每个发送进程独立分配**（而非全局共享），且发送后 `self._buffer = None`，下一次 `send()` 会按 `max(buffer_size_bytes, tensor_bytes)` 重新分配一块新的 GPU 缓冲，配合 `torch.cuda.ipc_collect()` 回收共享内存。

`_flush_bucket()` 的调度顺序（元数据先于 payload 广播，便于跨卡接收方按 `payload_numel` 预分配 buffer）：

1. 生成 `MetaFrame(is_end, tensor_specs, payload_numel)`（`payload_numel` 为桶内累计字节数，`payload` 本身是 uint8，因此元素数即字节数）；
2. 有跨卡接收方且当前 rank 是 `src_rank` → 通过 `_nccl_meta_group.broadcast_object_list` 广播元数据；
3. 有同卡接收方 → 使用 IPC 路径（下一节）；
4. 有跨卡接收方且当前 rank 是 `src_rank` → 在 `_nccl_data_group` 上 `broadcast(payload)`；
5. 清空桶并 `ipc_collect()`。

### 2.4. 同卡 IPC 路径（零拷贝）

`_send_ipc` 只发送 IPC handle 和桶级元数据，不传输张量本体：

```python
ipc_handle = tensor.untyped_storage()._share_cuda_()
handle_data = {"ipc_handle": ipc_handle, "size": ..., "dtype": ..., "shape": ..., "meta": meta}
handle_bytes = pickle.dumps(handle_data)
payload = torch.tensor(list(handle_bytes), dtype=torch.uint8)
header = torch.tensor([len(handle_bytes)], dtype=torch.int64)
for recv_rank in receiver_ranks:
    dist.send(header, dst=recv_rank, group=self._gloo_group)
    dist.send(payload, dst=recv_rank, group=self._gloo_group)
```

- `_share_cuda_()` 底层调用 CUDA `cudaIpcGetMemHandle`，将发送端的 GPU 显存注册为可跨进程共享，返回一个 tuple 形式的 handle；随后 `handle_data` 中还打包了描述整桶布局的 `MetaFrame`（包含所有 `TensorSpec`）；
- 通过主 Gloo 组分两步发送：先发 8 字节 header 表示 payload 长度，再发送序列化后的 handle payload。CPU 侧传输的是 handle 和桶元数据（相比 GB 级的桶张量本体可忽略），且不涉及跨卡 D2D 拷贝。

接收侧 `_recv_ipc`：

```python
storage = UntypedStorage._new_shared_cuda(*ipc_handle)
tensor = torch.empty(size, dtype=dtype, device=self._local_device)
tensor.set_(storage)
```

打开共享显存前，若 handle tuple 的 device 字段与接收方本地 device index 不一致，则替换为接收方本地 device index，避免 Ray Actor 里 `CUDA_VISIBLE_DEVICES` 单卡可见导致的偏移。绑定完成后，接收方读到的即为发送方桶所在的物理显存。

### 2.5. 跨卡 NCCL 路径

`_send_nccl` 将当前桶通过 `dist.broadcast` 广播到 `_nccl_data_group`；跨卡接收方在 `_recv_nccl` 中按 `MetaFrame` 声明的 `payload_numel` 分配一块临时 buffer 承接。桶不一定被填满（例如单张量大小刚好触发 flush，或最后一桶提前收尾），因此每次 flush 的实际 payload 大小随之变化，通常小于 `buffer_size_bytes`。

元数据在 `_nccl_meta_group` 上以 `broadcast_object_list` 单独广播（实际传输的是 `MetaFrame.serialize()` 得到的 pickle bytes，包装在 list 中）；这样即使 payload 为 0（仅 `is_end` 标记），接收方也能正确收到"流结束"通知。

### 2.6. 接收侧：拆桶与独立副本

接收侧的核心逻辑在 `get_iterator()`：

```python
for spec in meta.tensor_specs:
    tensor_dtype = str_to_dtype(spec.dtype)
    num_bytes = spec.numel() * tensor_dtype.itemsize
    raw_bytes = payload[spec.offset:spec.offset + num_bytes]
    tensor = raw_bytes.view(tensor_dtype).view(spec.shape).clone()
    yield (spec.name, tensor)
```

`payload` 是 uint8 原始字节缓冲，`spec.offset` 与 `num_bytes` 均以字节为单位；先按 `spec.dtype` `view` 回原始 dtype，再 `view` 回原始 shape，最后 `.clone()` 是关键的一次 D2D 拷贝，将切片从共享桶（IPC 情况下的发送方显存 / NCCL 情况下的临时接收 buffer）复制到接收方独立的显存，产生一份独立副本。这是为了让下游可以放心持有这份张量：发送方会在下一次 `send()` 时释放并重建桶，共享内存本身随时会被覆盖或回收。

`get_iterator()` 也支持 `yield_buckets=True`：将整桶的 `(payload, meta)` 一次性交给调用方，由调用方自行按 `TensorSpec` 拆分，适合追求更细粒度调度的接收端（例如 SGLang 内部可以在接收过程中同步分派给 TP worker）。

### 2.7. 生命周期

> **重点：meta 由 trainer 统一生成，两侧引擎无需感知内容。** meta 的解耦体现在两个层级：
>
> - **通道级 `ChannelMeta`**：由 [`Trainer._init_channel_meta`](../../coda/controller/trainer.py) 集中构造后，作为不透明对象透传给发送方（trainer worker）和接收方（SGLang engine）。两侧只需整体传给 `create_sender_channel(meta)` / `create_receiver_channel(meta)`，无需读取或理解其中的任何字段——`world_size`、`buffer_size_bytes`、`gloo_port` 等均由 TransferMesh 内部消费。
> - **桶级 `MetaFrame` / `TensorSpec`**：由发送方在 `_flush_bucket` 时自动填充，由接收方 `get_iterator` 自动按协议解析。trainer worker 与 SGLang engine 从不构造、读取或修改 `MetaFrame` 字段，仅通过 `send((name, tensor))` / `get_iterator()` 交互。
>
> 因此，**权重传输流程与训练引擎（Megatron 等）、推理引擎（SGLang 等）完全解耦**——更换任一侧的实现只要遵循 `ChannelMeta` 协议接入通道 API，即可复用 TransferMesh，无需在引擎侧感知任何 meta 语义。

- **创建/复用**：`create_sender_channel(meta)` / `create_receiver_channel(meta)` 是幂等的进程级入口。`meta.recreate=True` 时会先关闭已有通道再创建新的；`recreate=False` 且已有通道时直接复用。
- **发送流终止**：发送方调用 `send(None, flush=True)`，触发一次 `is_end=True` 的 flush；接收方 iterator 见到 `meta.is_end` 后终止循环。
- **销毁**：`close()` 依次销毁 NCCL 数据/元数据组和 Gloo 主组，并清空 `_ipc_assignments`、`_nccl_receivers`、`_buffer`、`_buffer_offset`、`_pending_specs` 等内部状态；发送方额外再调一次 `ipc_collect()` 回收共享内存（接收方不涉及）。

## 3. 顶层调用链

一次完整的权重传输由 [`Trainer._update_weights`](../../coda/controller/trainer.py) 编排：

```text
Trainer._update_weights(meta, weight_version)
  ├── TrainManager.async_update_weights(meta)
  │     └── 每个 train worker (Megatron) : update_weights(meta)
  │           ├── create_sender_channel(meta)
  │           ├── for (name, tensor) in bridge.export_hf_weights(self.model):
  │           │       send_channel.send((name, tensor))
  │           ├── send_channel.send(None, flush=True)
  │           └── 若 `self.offloaded` 为 True（说明之前调用过 `offload()`）：`_offload_model(move_params=True, move_grads=False)`，将先前保留在 GPU 上的训练参数下移至 CPU
  └── RolloutManager.async_update_weights(meta, weight_version)
        └── 每个 SGLang engine : update_weights_from_channel(meta_dict)
              ├── 若尚未 flush（`_cache_flushed=False`）：_flush_cache()  # 清空旧 KV / prefix cache
              └── 调用 SGLang `update_weights_from_channel` HTTP 接口   # SGLang 内部通过 TransferMesh 接收权重流并覆盖旧权重
```

两侧的 `async_update_weights` 都返回 Ray 的 `ObjectRef` 列表，`_update_weights` 通过 `ray.get(all_refs)` 同时等待训练侧和推理侧全部完成，保证发送方与接收方在同一次通道生命周期内匹配。

`_init_channel_meta` 每次都会重新计算 `world_size` 并决定是否设置 `recreate`：只要 rollout 端有新的 engine 加入（例如 fault tolerance 恢复了故障 engine），就会重新从 scheduler 获取 Gloo master 地址与端口，并强制两侧 recreate 通道。

## 4. 与 colocate / 全异步模式的关系

TransferMesh 本身不区分模式，但两种模式下的调度上下文不同，反映到 [`Trainer.train_loop`](../../coda/controller/trainer.py) 中：

- **colocate 模式**（rollout 与 trainer 复用同一批 GPU）：因为单卡显存无法同时容纳两套完整模型状态，`update_weights` 前后需要在 GPU 上进行分阶段调度：
  1. `train_manager.offload()`：将训练侧梯度和优化器状态卸载至 CPU，**但保留参数在 GPU**（[`megatron_train_worker.py:586`](../../coda/backends/megatron/megatron_train_worker.py) 中 `move_params=False`，注释 "keep params in gpu for latter update weights"）；
  2. `rollout_manager.onload_weights()`：将 SGLang 的权重容器载回 GPU（`tags=[GPU_MEMORY_TYPE_WEIGHTS]`），KV cache 继续留在 CPU；
  3. `_update_weights(...)`：通过 TransferMesh 完成权重传输，此时同卡的发送方（trainer worker）和接收方（SGLang TP worker）会自动选择 IPC 零拷贝路径；
  4. 传输结束后，若之前调用过 `offload()`（`self.offloaded=True`），`update_weights` 内部再调 `_offload_model(move_params=True, move_grads=False)` 将训练参数完全卸载至 CPU；
  5. `rollout_manager.onload_kv()`：将 KV cache 载回 GPU，进入下一 step 的 rollout。

- **全异步模式**（rollout 与 trainer 分卡常驻）：不涉及 `offload_train` / `onload_rollout_weights` / `onload_rollout_kv` 阶段，两侧模型都常驻各自 GPU。step 边界的 `pause()` 完成后，直接 `_init_channel_meta()` + `_update_weights()`。此时通常不存在 `(ip, gpu_id)` 完全匹配的发送/接收对，TransferMesh 会自动落到跨卡 NCCL 广播路径。

无论哪种模式，`SglangEngine.update_weights_from_channel` 都会在真正加载新权重前先 `_flush_cache()`，清空旧权重对应的 prefix / KV cache，以避免新旧混用；colocate 模式下的 `release_memory_occupation` 已经 flush 过，会通过 `_cache_flushed` 标记跳过一次冗余 flush。

## 5. 快速上手

### 5.1. 基本用法

**发送端（Sender）**

```python
import torch
from coda.transfer_mesh import create_channel, Role

channel = create_channel(
    master_addr="127.0.0.1",    # Rank 0 用于集合协商的地址
    addr="127.0.0.1",           # 本地 IP 地址
    gpu_id=0,                   # 本地 GPU 设备编号
    world_size=2,               # 参与本通道的所有 rank 数
    rank=0,                     # 本进程的 rank
    role=Role.SENDER,           # Role.SENDER 或 Role.RECEIVER
    src_rank=0,                 # 主发送方 rank
    buffer_size_bytes=1024 * 1024 * 1024,  # 桶大小（默认 1 GiB）
)

# 逐张量发送（累积到桶中，桶满时自动 flush）
for name, tensor in model_weights:
    channel.send((name, tensor))

# flush=True：flush 剩余桶并广播 end-of-stream 标记
channel.send(None, flush=True)
channel.close()
```

**接收端（Receiver）**

```python
channel = create_channel(
    master_addr="127.0.0.1",
    addr="127.0.0.1",
    gpu_id=0,
    world_size=2,
    rank=1,
    role=Role.RECEIVER,
    src_rank=0,
)

# 以 iterator 方式接收张量
for name, tensor in channel.get_iterator():
    model.load_single_weight(name, tensor)
channel.close()
```

### 5.2. 多 GPU Colocated 场景

当发送方与接收方 colocate 在同一批 GPU 上时，TransferMesh 会自动走 IPC 零拷贝路径：

```
拓扑（world_size=8）：
  GPU 0: Rank 0 (sender)  ──IPC──>  Rank 4 (receiver)
  GPU 1: Rank 1 (sender)  ──IPC──>  Rank 5 (receiver)
  GPU 2: Rank 2 (sender)  ──IPC──>  Rank 6 (receiver)
  GPU 3: Rank 3 (sender)  ──IPC──>  Rank 7 (receiver)
```

```python
# 每个发送方进程（rank 0-3）
channel = create_channel(
    master_addr="10.0.0.1",
    addr="10.0.0.1",
    gpu_id=rank,           # 与对应接收方的 gpu_id 相同
    world_size=8,
    rank=rank,
    role=Role.SENDER,
    src_rank=0,
)

# 每个接收方进程（rank 4-7）
channel = create_channel(
    master_addr="10.0.0.1",
    addr="10.0.0.1",
    gpu_id=rank - 4,       # 匹配对应发送方
    world_size=8,
    rank=rank,
    role=Role.RECEIVER,
    src_rank=0,
)
```

## 6. 使用限制

- **`src_rank` 目前固定为 0**。跨卡 NCCL 广播全部由 rank 0 发起，rank 0 若成为瓶颈，需要考虑替换或做进一步分片。
- **同卡 IPC 要求发送方与接收方物理 GPU 相同**（`(ip, gpu_id)` 完全匹配）。分卡部署或 replica 拓扑变化会自动落到 NCCL 路径，无需业务感知，但要保证两侧 `world_size`、`train_world_size` 计算一致。
- **`world_size` 依赖精确的 GPU 总数计算**：`train_world_size = trainer.num_gpus_per_node * trainer.num_nodes`，`rollout_world_size` 按各 replica 的 `num_nodes * rollout.num_gpus_per_node` 累加，任何一侧配置错误都会导致 all_gather 阶段死锁。
- **依赖 `torch.distributed.distributed_c10d` 私有 API**：`_new_process_group_helper`、`_world.pg_group_ranks` 等符号不受 PyTorch 稳定 API 保证；[`_c10d_compat.py`](../../coda/transfer_mesh/_c10d_compat.py) 已经将符号加载集中并给出带 `torch.__version__` 的清晰错误信息，PyTorch 大版本升级时优先关注这一层的适配。
- **桶大小 1 GiB 属于经验值**：过小会增加通信轮次，过大会推高峰值显存与 flush 抖动。可以在 `ChannelMeta.buffer_size_bytes` 中按需调整，但要确认接收侧显存足以承接（NCCL 路径下接收方也需要临时分配对应大小的接收 buffer）。

## 7. 源码入口

- [`TransferMeshChannel`](../../coda/transfer_mesh/channel.py) — 通道核心实现
- [`partition_receivers`](../../coda/transfer_mesh/topology.py) — 拓扑分区
- [`MetaFrame` / `TensorSpec`](../../coda/transfer_mesh/protocol.py) — 桶元数据协议
- [`ChannelMeta` / `create_sender_channel` / `create_receiver_channel`](../../coda/utils/channel_helper.py) — 进程内通道生命周期
- [`Trainer._init_channel_meta` / `Trainer._update_weights`](../../coda/controller/trainer.py) — 顶层编排
- [`MegatronTrainWorker.update_weights`](../../coda/backends/megatron/megatron_train_worker.py) — 训练侧发送
- [`SglangEngine.update_weights_from_channel`](../../coda/backends/sglang/engine.py) — 推理侧接收
