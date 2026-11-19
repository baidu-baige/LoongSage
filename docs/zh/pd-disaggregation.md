# PD 分离

PD 分离（Prefill-Decode Disaggregation）把一次生成拆成两个阶段，分别交给不同的 SGLang 引擎：prefill 引擎计算 prompt 的 KV cache 并采样首个 token，decode 引擎通过 RDMA 接收这份 KV cache 后继续逐 token 生成。两类引擎各自占用 GPU，可以配置不同的并行度和 replica 数，避免长 prompt 的 prefill 打断正在进行的 decode，适合 prompt 长、多轮上下文不断增长的 agent 场景。

默认配置只启用 `regular` 一组引擎，即每个引擎同时负责 prefill 与 decode，不做拆分。开启 PD 分离后，智能体侧接口不变，仍通过 Router 的 chat completion 接口访问；拆分、配对和结果合并都由 Router 完成。

## 1.开启方式

把 `regular` 关掉，同时为 `prefill` 与 `decode` 分配 GPU。下例在一个 8 卡节点上放 1 个 TP4 的 prefill 和 1 个 TP4 的 decode：

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

两组需要的 GPU 都是整节点时，也可以只用 `num_nodes` 分配，例如 `prefill.num_nodes: 1`、`decode.num_nodes: 2`。

## 2.关键参数

| 参数 | 默认值 | 说明 | 约束或建议 |
| --- | --- | --- | --- |
| `rollout.sglang_replicas.regular.num_nodes` | `1` | regular 组的节点数。 | PD 分离时必须为 `0`，否则启动时报错。 |
| `rollout.sglang_replicas.{prefill,decode}.num_nodes` | `0` | 两组各自的节点数，GPU 预算为 `num_nodes × rollout.num_gpus_per_node`。 | 两组都要有 replica，只开一组会在启动时报错。 |
| `rollout.sglang_replicas.{prefill,decode}.num_replicas` | `null` | 显式指定 replica 数，设置后覆盖 `num_nodes`。 | 某组需要的 GPU 少于一个整节点时必须使用。 |
| `rollout.sglang_replicas.{prefill,decode}.num_gpus_per_replica` | `8` | 单个 replica 的 GPU 数，两组可以不同。 | 必须等于该组 `tp_size × pp_size`，否则权重同步的 rank 会错位。 |
| `rollout.sglang_replicas.{prefill,decode}.sglang_args` | decode 为 `{disable_cuda_graph: false}` | 覆盖 `rollout.sglang_args`，只作用于该组。 | decode 默认开启 CUDA graph。 |
| `rollout.sglang_args.disaggregation_ib_device` | 未设置 | 传输 KV cache 使用的 RDMA 网卡。 | 未设置时自动解析，见第 4 节。 |
| `agentflow.router.rollout_worker_load_threshold` | `{regular: 32, prefill: 32, decode: 16}` | 每类 worker 的单 slot 在途请求阈值，超过后放弃粘滞、改选最闲的 slot。 | 必须包含 `prefill` 和 `decode` 两个键。两池并行度不同，单个引擎能承载的请求数也不同，需分别调整。 |
| `agentflow.router.max_connections` | `512` | Router 转发的并发上限。 | 一对 prefill/decode 只占一个名额，但会建立两条连接。 |

## 3.原理

### 3.1.请求流程

1. 智能体的请求经 Router 中间件改写为 SGLang `/generate` 请求体。
2. Router 为每类 worker 各选一个 slot，即一个 prefill slot 和一个 decode slot。同一 trajectory attempt 的多轮请求固定到同一对 slot（sticky），以复用 prefix cache；某个 slot 的在途请求超过该类阈值后，改选最闲的 slot。
3. Router 生成一个随机的 `bootstrap_room`，向两份请求体注入相同的 `bootstrap_host`、`bootstrap_port`、`bootstrap_room`，host 和 port 取自选中的 prefill 引擎。两端靠这三个字段配对。prefill 开启 DP attention 时，decode 还会收到 `disagg_prefill_dp_rank`，用来定位 KV cache 所在的 DP 组。
4. 两份请求并发发出。Router 先等 prefill 的结果；prefill 失败就不再等 decode，并按 `rid` 通知 decode 中止。
5. 返回给智能体的是 decode 的响应。prefill 的响应只在开启 R3 时用于补全 prompt 段的专家路由。

### 3.2.引擎启动与端口

prefill 与 decode 的引擎依次排布在 rollout GPU 上，同一节点可能同时有两类引擎。端口按节点分配，不同类型从不同的起点扫描（regular 2000、prefill 4000、decode 6000），prefill 的 bootstrap 端口为 `port + 2`。

### 3.3.权重同步

两个池的引擎在同一次权重更新中同步，全部完成后才恢复采样，不会出现 prefill 与 decode 权重版本不一致的窗口。每个引擎在权重通道中的起始 rank 由 RolloutManager 按 `num_gpus_per_replica` 依次累加分配，因此两组并行度不同也能正确排布。

### 3.4.colocate 下的 offload

KV cache 传输是 prefill 向 decode 注册的显存做 RDMA 写入。colocate 模式 offload 时，先释放 prefill，等它完成后再释放 decode，避免 decode 先注销显存、而 prefill 仍在写入。onload 不需要分阶段。

## 4.RDMA 网卡与 GID

KV cache 通过 RDMA 传输。网卡名与机器相关，所以在每个引擎所在的节点上分别解析，优先级如下：

1. `sglang_args.disaggregation_ib_device` 显式配置；
2. `/etc/nccl.conf` 中的 `NCCL_IB_HCA`；
3. 环境变量 `NCCL_IB_HCA`；
4. 以上都没有时交给传输后端自动发现，此时可能选到不在 RDMA 网络里的网卡，日志会给出 warning。

`NCCL_IB_HCA` 中的 `=` 前缀和 `:port` 后缀会被去掉；以 `^` 开头的排除列表无法转成设备列表，会被跳过。

mooncake 只从环境变量读取 GID index。引擎启动 SGLang 前，会把 `/etc/nccl.conf` 中的 `NCCL_IB_GID_INDEX` 导出到环境变量；如果环境里已经设置了 `NCCL_IB_GID_INDEX` 或 `MC_GID_INDEX`，则不覆盖。这一步不区分传输后端，对不读该变量的后端没有影响。

## 5.R3 路由回放

开启 `trainer.use_rollout_routing_replay` 时，PD 下一条记录的专家路由由两端拼接而成：prefill 返回 prompt 段 `[start_len, prompt_len)`，decode 只返回生成段 `[prompt_len, seqlen - 1)`。Router 会给 decode 请求注入 `routed_experts_start_len`，避免 decode 重复传输 prompt 段。

KV cache 传输只搬运缓存内容，decode 侧没有 prompt 段的路由记录，所以 prompt 段必须来自 prefill。PD 下开启 R3 需要使用 SGLang 0.5.23，并打上 `docker/patch/sglang-0.5.23.patch`。

`trainer.routed_experts_dtype` 对两端同样生效，`auto` 时各端按自己在 `meta_info.routed_experts_dtype` 中报告的类型解码。

## 6.注意事项

- **与 Rust server 不兼容**：`rollout.env_vars.SGLANG_RUST_SERVER` 为真时，SGLang 要求 bootstrap 端口与 HTTP 端口相同，而框架固定使用 `port + 2`，因此 PD 分离下会在启动时直接报错。
- **prefill 失败**：prefill 返回非 2xx 或请求异常时，Router 通知 decode 中止，并向智能体返回 502 `Prefill worker failed`。decode 请求异常（包括超时）同样返回 502。
- **摘除 prefill 引擎**：引擎关闭时会调用 `exclude_worker`。摘除 prefill 时，所有绑定到它的会话的 decode 端会被一并中止，否则 decode 会一直等待 KV cache，直到 SGLang 的 `SGLANG_DISAGGREGATION_WAITING_TIMEOUT`（默认 300 秒）。
- **中止单个 attempt**：`abort_session` 会先中止 decode、再中止 prefill；任一端失败都返回 503，并保留该会话的绑定。
- **负载阈值需分别调**：一个请求会同时占用一个 prefill slot 和一个 decode slot，直到整个请求结束才一起释放，所以两端的在途数差异来自两池的 slot 数量比例，例如 1P:3D 时单个 prefill slot 承载的请求约是 decode 的 3 倍。粘滞的收益也不同：多轮 agent 场景下上下文逐轮增长，粘到同一个 prefill 能复用 prefix cache、省掉大部分 prompt 计算。decode 默认不开 radix cache（`disaggregation_decode_enable_radix_cache: false`），每轮都要从 prefill 收完整的 KV cache，粘滞几乎没有收益，阈值可以设得更低，让负载更均衡；开启后 decode 会先在本地匹配前缀，prefill 只传输未命中的部分，此时粘到同一个 decode 可以省掉前缀的重复传输，decode 阈值也应相应调高。
