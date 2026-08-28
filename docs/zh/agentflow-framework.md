# AgentFlow

AgentFlow 是 LoongSage 强化学习框架中负责 **rollout（采样）编排** 的核心模块。它把"训练侧下发的一批 prompt"转化为"可直接用于训练的 trajectory（含 token、logprob、loss mask 等对齐信息）"，并在这个过程中管理推理请求路由、多轮 agent 执行、代码沙箱、失败重试与中止等全部细节。

一句话概括它的定位：**AgentFlow 是 agent / 推理引擎 / 训练器三者之间的中间层**，对上接收训练控制器下发的 `Trajectory`，对下驱动 SGLang 推理后端与 agent，最终产出与训练严格对齐的采样序列。

其中最关键的一条设计约束是：**与 LLM 的实际交互是 token in / token out，走 SGLang 原生 `/generate` 接口，而不是 OpenAI `/v1/chat/completions`**。agent 侧仍然写标准 OpenAI 代码，Router 在中间做双向改写——请求方向把 `messages` 用统一的 tokenizer 转成 `input_ids` 再发给 `/generate`，响应方向把 `/generate` 返回的 `output_token_logprobs` 还原成 OpenAI 响应交给 agent。这样"采样时喂进模型的 token 序列"与"训练时回放的 token 序列"是同一份数据，中间不经过任何文本二次 tokenize。

## 1. 整体架构

AgentFlow 由一个编排器（`AgentFlow`）和若干协作组件组成。编排器内部持有 4 个协作组件，其中 Router 运行在后台线程并挂载中间件链：

![AgentFlow 整体架构](../_static/image/agentflow-architecture.svg)

关键点：

- **Router 是训练对齐存在的核心原因**：agent 用标准 OpenAI 接口请求 LLM，Router 拦截后把请求改写为 SGLang 原生 `/generate`，以 `input_ids` 送入推理（token in）、以 `token_ids + logprob` 取回结果（token out），并借助中间件把逐 token 的 logprob、weight version 等写入 `TrajectoryStore`，保证"采样序列"与"训练可回放序列"严格一致。
- **TrajectoryStore 是采样期间的唯一数据真源**，被 `AgentFlow` 与 Router 的 `ParserMiddleware` 共享。
- **一条 trajectory 对应一个 Agent 实例**（多轮模式）；单轮模式不创建 agent，直接由 AgentFlow 调用 Router。

## 2. 核心组件

### 2.1 AgentFlow 编排器

入口类是 [agent_flow.py](../../coda/agentflow/agent_flow.py) 中的 `AgentFlow`。它在 `__init__` 里完成三件初始化：

1. `_init_per_datasource()` —— 按 `data_sources` 逐个数据源初始化 agent 类与 reward 函数。每个数据源可以有独立的 agent、reward、token 配置。是否多轮由 `get_rollout_mode()` 判断：配置了 `agent.name` 即为 `multi_turn`，否则为 `single_turn`。
2. `_init_tokenizer()` —— 通过 `create_tokenizer_manager()` 构建 `TokenizerManager`。
3. `_init_router()` —— 构建并在**后台守护线程**中启动 `Router`；`middleware_kwargs` 由 `vars(self)` 中所有非下划线属性组成，从而把 `trajectory_store`、`tokenizer_manager`、`accumulate_reasoning`、`r3_enabled` 等一并注入到中间件构造上下文。

对外的主接口是 `generate(trajectories)`（`agent_flow.py:218`）：并发地为每条 trajectory 建 asyncio task，`asyncio.gather` 汇总，返回与输入顺序一致的 `Reward` 列表。已经是终态（`COMPLETED`/`FAILED`）的输入直接走 `_reemit_existing()` 重新入队（partial rollout resume 场景）；输入状态若为 `GENERATING` 则直接抛错（不允许重复提交在途 trajectory）。首次调用还会把默认 asyncio executor 扩为 `ThreadPoolExecutor(max_workers=8192)`，避免大量 `asyncio.to_thread`（如并发建 pod）挤占推理线程。

另一个重要接口是 `abort()`（`agent_flow.py:268`），用于训练 step 边界中止在途 rollout，步骤严格有序：
1. `POST /abort_all_workers` 中止所有 worker 上的 LLM 请求；
2. cancel 所有活跃 asyncio task；
3. 等待 trajectory 入队的 future 完成；
4. `POST /wait_inflight_requests_complete` 等待 Router 中间件把落盘写完（避免脏写）；
5. 把状态不在 `{COMPLETED, ABORTED}` 中的 trajectory 标记为 `ABORTED` 并 emit（注意：处于 `FAILED` 的 attempt 也会被重标为 `ABORTED` 重发）。

### 2.2 Router 与中间件体系

`Router`（[router/router.py](../../coda/agentflow/router/router.py)）是夹在 agent 与 SGLang worker 之间的反向代理，具备 **sticky（会话粘滞）+ least-loaded（最小负载）** 路由能力。

暴露的 HTTP 接口分两类：

| 类别 | 方法 | 路径 | 作用 |
| --- | --- | --- | --- |
| Worker 管理 | POST | `/add_worker` | 注册一个 SGLang worker |
| | GET | `/list_workers` | 列出活跃/隔离 worker 及负载 |
| | PUT | `/exclude_worker` / `/include_worker` | 隔离 / 恢复 worker |
| Session 管理 | DELETE | `/release_session/{trajectory_id}/{attempt_id}` | 释放该 attempt 的 sticky 映射 |
| | POST | `/abort_session/{trajectory_id}/{attempt_id}` | 中止失败 attempt 绑定 worker 上的请求 |
| | POST | `/abort_all_workers` | 中止所有 worker 并清空 sticky 状态 |
| | POST | `/wait_inflight_requests_complete` | 等待中间件在途落盘完成 |
| 采样代理 | POST | `/{trajectory_id}/{attempt_id}/v1/chat/completions` | 接收 OpenAI 请求 → 改写为 token 请求 → 转发 SGLang `/generate` |

**Session = 一次 trajectory attempt**，标识符是 `request_id = build_request_id(trajectory_id, attempt_id)`（形如 `trajectory_id#attempt_id`）。URL 前缀 `/{trajectory_id}/{attempt_id}` 让 Router 能对同一 attempt 的多轮/重试请求做 sticky 路由（打到同一 worker，复用 KV cache），并支持精确 abort。

**中间件体系**（名字→类的映射表位于 `router/__init__.py`）设计成可插拔：名字查表 + 配置驱动 + 有序链。`resolve_middleware_chain()` 把 YAML 配置规范化为有序 spec，`_setup_middlewares()` 以 `reversed` 顺序 `add_middleware`，保证配置中靠前的中间件成为最外层、最先执行。目前内置唯一中间件是 `ParserMiddleware`（名字 `"parser"`）。

### 2.3 ParserMiddleware 与 TrajectoryParser

这两者分工明确：

- **ParserMiddleware（传输层，[parser_middleware.py](../../coda/agentflow/router/parser_middleware.py)）**：负责路由匹配、per-trajectory 并发锁（串行化 litellm 重试等并发请求）、**OpenAI `/v1/chat/completions` ↔ SGLang `/generate` 的双向改写**、response token 预算控制、in-flight 请求跟踪。
- **TrajectoryParser（领域/训练层，[parser.py](../../coda/agentflow/router/parser.py)）**：负责 tokenize、trajectory 构建与持久化、reasoning / tool_call 解析、token/logprob/loss_mask 对齐。其 `reasoning_parser` / `tool_call_parser` 可通过中间件 `params` 配置，按模型族切换解析器（并非写死）。

#### token in / token out：为什么不用 `/v1/chat/completions`

OpenAI 的 chat 接口只收发文本，SGLang 侧会自行套 chat template 并 tokenize。这条路径对 RL 是不可用的：训练侧拿到的是文本，重新 tokenize 后极易与采样时的 token 序列错位（空白、特殊 token、think 标记、tool_call 序列化差异都会导致偏移），而且 chat 接口默认不返回逐 token logprob。

因此 `ParserMiddleware` 把请求下沉到 SGLang 原生的 `/generate`，交互单位从"文本"变成"token"：

| 方向 | 内容 |
| --- | --- |
| **token in**（`_build_generate_body`） | `input_ids`（由 `TokenizerManager` 套 chat template 后产出，首轮全量 / 续轮增量）、`rid=trajectory_id#attempt_id`（sticky 路由与精确 abort 的依据）、`return_logprob=True`（强制开启）、`sampling_params`（由 OpenAI 顶层字段按 `_SAMPLING_PARAM_MAP` 映射而来，`max_tokens` / `max_completion_tokens` → `max_new_tokens`）；R3 开启时追加 `return_routed_experts` / `routed_experts_start_len` |
| **token out**（`build_assistant_message`） | `meta_info.output_token_logprobs` 逐项拆成 `response_ids` 与 `logprobs`、`meta_info.weight_version`（本轮生成所用的权重版本）；R3 开启时还有 base64 编码的 `meta_info.routed_experts` |

几个容易踩的细节：

- **`return_logprob` 是硬编码 `True`**，不受调用方请求体影响——没有逐 token logprob 就无法训练。
- **`skip_special_tokens` 不是无条件关闭的**：只有 DeepSeek-V4 模型族会被强制设为 `False`（它把 think 与 DSML tool-call 标记标成了 special token，detokenizer 必须保留它们，reasoning / tool_call parser 才看得到）；其他模型族沿用请求体或 server 默认值。
- **agent 侧完全无感**：Router 最后用 `_to_openai_response()` 把 `/generate` 的返回拼回标准 `chat.completion` 结构（含 `choices[0].message`、`finish_reason`、`usage`），所以 agent 里可以继续用 litellm / openai SDK，不需要知道底层是 token 接口。
- **上游异常一律原样透传**：非 2xx、非 JSON、以及 `finish_reason.type == "abort"` 的响应都直接回传且**不写 TrajectoryStore**，避免把残缺数据落进训练集。
- **支持流式响应**：若 agent 请求体带 `"stream": true`，`ParserMiddleware` 内部仍然用非流式方式完整调用 `/generate` 拿到结果（保证 token/logprob 落盘的完整性），再用 `_to_streaming_response()` 把完整的 `chat.completion` 包装成 SSE chunk 序列返回，对 agent 侧伪装成真流式，不需要额外适配。
- **`request_kind` 请求头**：agent harness 可在请求头里带上 `request_kind`（大小写不敏感，`ParserMiddleware` 通过 `request.headers.get("request_kind")` 读取），用于告知 `TrajectoryParser` 本次请求的语义（目前唯一取值 `"collab_spawn"`，标记这是子 agent/子任务分叉请求而非主线续轮），从而在 `build_turn_input()` 中走对应的分派分支（见 §2.3 情形 3）。

中间件按固定生命周期调用 parser：

```text
get_trajectory()        # 取出当前 attempt 的 Trajectory
  → build_turn_input()  # 组装本轮 input_ids（首轮全量 / 续轮增量 tokenize）
  → [Router 以 input_ids 转发 POST /generate 到选中的 worker]
  → build_assistant_message()  # 从 meta_info.output_token_logprobs 解出 response_ids + logprobs
  → update_trajectory()        # 追加一个 turn，写回 TrajectoryStore
```

`build_turn_input()`（`router/parser.py`）按情形分派，用于支持黑盒 agent（如通过 harness/插件驱动、无法保证消息前缀严格递增）下的多段 trajectory：

1. **首轮**（trajectory 尚无 token）：全量 tokenize 整段 messages，开启 mainline 的首个 segment（`origin="root"`）。
2. **续轮命中当前 active segment 前缀**（最常见路径）：只对新增的消息增量 tokenize，`input_ids` 由"该 segment 已有 token 后缀 + 增量 token"拼成——之所以要带上 segment 内已有 token 而不是只发增量，是因为 SGLang `/generate` 是无状态接口，每次都要收完整上下文；训练侧的隔离靠 `Segment`/`Triplet` 的 `token_start`/`token_end` 索引区间实现，与 `input_ids` 是否带旧 token 无关。
3. **`request_kind == "collab_spawn"`**（子 agent/子任务分叉）：不校验前缀，直接在 `segments` 中新增一个 `origin="subagent"`、`parent_segment_id` 指向当前 active segment 的占位 segment（`is_subagent_placeholder=True`），不更新 `trajectory.active_segment_id`（因为主线并未切换到这个分支）。
4. **前缀不匹配且非 `collab_spawn`**（典型场景：agent 做了 context compaction/Summary Reset，历史消息不再是任何已有 segment 的前缀）：开启一个新的 `origin="compact"` mainline segment，`parent_segment_id` 指向原 active segment；`build_turn_input()` 返回的 `target_segment_id` 指向这个新 segment，随后由落盘阶段（`update_trajectory()`）把 `trajectory.active_segment_id` 切换过去；`input_ids` 对新 segment 做全量 tokenize（这是"新段"，没有历史后缀可复用）。

其中 3 会经由 `request_kind` HTTP 头识别；该头由 agent harness 主动注入（例如通过黑盒 agent 侧的插件机制，在请求发出前读取 session/agent 上下文信息判定是否为子 agent 分叉请求，再设置该请求头）。

### 2.4 TokenizerManager

[tokenizer_manager.py](../../coda/agentflow/tokenizer_manager.py) 把 CPU 密集的 chat-template / encode / decode 封装成**异步接口**，用线程池（`ThreadedTokenizerManager`）或进程池（`ProcessTokenizerManager`）执行，避免阻塞 Router 的事件循环。`create_tokenizer_manager()` 按 `manager.mode` 选择后端。

它还集中缓存与训练对齐必需的元数据：`think_start_ids`/`think_end_ids`（token 级裁剪 think 块用）、`model_family`、`think_tags` 等，对 DeepSeek-V4 会通过 `_wrap_deepseek_v4_tokenizer()` 自动包装 tokenizer 并缓存其 `thinking_mode`。**它是"rollout 采样序列"和"训练回放序列"共享的唯一 tokenize 真源**，避免文本二次 tokenize 造成错位。用于续轮裁掉 system 前缀的 `system_prompt_len` 并不在这里缓存，而是由 `TrajectoryParser` 首次用到时延迟计算并缓存（`parser.py:700-721`）。

> **DeepSeek-V4 解析路径**：dsv4 与其他模型走同一套 SGLang parser。由于它的 tokenizer 被 `_wrap_deepseek_v4_tokenizer()` 换成了官方 encoder，没有 Jinja 模板可供自动检测匹配，`parser.py` 会回退到按模型架构检测：当模板匹配无结果时，`_detect_parser_names()` 调用 `_detect_parser_names_from_arch()`，借助 SGLang 的 `_resolve_architecture_auto_parsers` 按 `config.json` 里的模型架构选出 parser key——因此 parser key 始终由 SGLang 定义，我们这侧不硬编码任何映射。请求侧唯一的特殊处理是 `skip_special_tokens=false`（保留 `<think>` 与 DSML 标记），末尾的 EOS 仍由 SGLang 默认的 `no_stop_trim` 行为裁剪。

### 2.5 TrajectoryStore 与数据模型

[trajectory_store.py](../../coda/agentflow/trajectory_store.py) 定义了核心数据模型和存储。数据模型是三级对象结构：

- **`Trajectory`**：一条完整 rollout。核心是几条**扁平数组**：`tokens`（全序列 token）、`loss_masks`（response 空间，1=计入 loss）、`rollout_log_probs`（response 空间，LLM 位置为真实 logprob、其余为 0）、`rollout_weight_versions`、`token_rewards`；`chat_completions: dict[int, list[dict]]` 按 `segment_id` 分别存各段的 OpenAI 消息历史（不再是单一 list）；`active_segment_id` 指向当前"主线"正在写入的 segment。
- **`Segment`**：共享同一增量上下文（可安全前缀增量 tokenize）的一组轮次，组织成一棵树：`segment_id`/`parent_segment_id` 表达父子关系，`depth` 是树深度，`origin: Literal["root","compact","subagent"]` 记录该 segment 的来源——`root` 是首个 mainline segment，`compact` 是 agent 做 context compaction（如 Summary Reset）后开的新 mainline segment，`subagent` 是子 agent/子任务分叉出的占位分支；`trainable` 标记该 segment 是否计入训练（`TrajectoryGroup.segment_count` 只统计 `trainable=True` 的段）。何时开新 segment、是否更新 `active_segment_id` 由 `build_turn_input()` 的分派逻辑决定（见 §2.3）。
- **`Triplet`**：一个 LLM 交互轮次，以**索引区间**（`token_start/token_end`、`logprob_start/logprob_end`）指向父 `Trajectory` 的扁平数组，不复制 token。

`TrajectoryStore` 用 `dict[trajectory_id -> list[Trajectory]]` 存储，list 里每个元素是一次 attempt（重试）。`update()` 按 `attempt_id` 匹配，避免陈旧后台线程覆盖更新的 attempt；`get()` 不传 `attempt_id` 时返回最新一次。

`TrajectoryStatus` 生命周期：`PENDING → GENERATING → COMPLETED`，错误路径为 `→ FAILED`（重试耗尽）或 `→ ABORTED`（外部中止）。

### 2.6 TrajQueue

[trajectory_queue.py](../../coda/agentflow/trajectory_queue.py) 是 AgentFlow 与下游 collector 之间的**线程安全队列**，按 `prompt_id` 聚合 trajectory，只有凑齐一个 group（`group_sizes[ds_index]` 条）才能出队。用 `threading.Condition` 做生产者-消费者信号，`wait_for_group(will_collect=...)` 支持带条件的阻塞取组，可选 `maxsize` 做背压。

### 2.7 Agent

[agent/base_agent.py](../../coda/agentflow/agent/base_agent.py) 定义抽象基类 `BaseAgent`，agent 只需实现两个方法：

- `async run_trajectory(trajectory: dict) -> Any`：跑完一条完整轨迹并返回 reward。AgentFlow 不约束 `trajectory` 的 schema，各 agent 自取所需字段（`prompt` / `label` / `metadata`）。
- `async clear()`：释放资源。

AgentFlow 统一注入的构造参数有 `router_url`（已带 `/{trajectory_id}/{attempt_id}` 前缀）、`completion_params`、`max_response_len_per_trajectory`、`temperature`，其余 agent 专属参数通过 `**kwargs` 透传（来自数据源的 `agent` 配置块）。

agent 通过 `Registry` + `@register_agent` 装饰器注册，`agent/__init__.py` 用 `pkgutil.walk_packages` 自动发现内置 agent（单个 agent 的依赖缺失不会影响其他 agent 加载）。内置示例：

- `agent/gsm8k/gsm8k_agent.py` —— GSM8K 数学题多轮 agent（注册名 `gsm8k`）。
- `agent/swe/mini_swe_agent.py` —— 基于 mini-swe-agent 的 SWE-bench agent，在沙箱里执行 shell 完成代码修复（注册名 `mini-swe`）。
- `agent/bcp/bcp_agent.py` —— BCP agent（注册名 `bcp`）。
- `agent/opencode/opencode_agent.py` —— 黑盒 opencode agent，在沙箱内驱动 OpenCode CLI 完成 SWE 任务（注册名 `opencode`）。`run_trajectory()` 把 Router 地址包装成 OpenAI-compatible provider 写入 `opencode.json`，再以子进程执行 `opencode run` 跑完整个任务；它是 §2.3 中 `request_kind` 机制的实际生产者——通过写入一个 OpenCode 插件文件（`coda-request-kind.js`，运行时以 base64 编码 `printf` 写入沙箱内 `/root/.config/opencode/coda-request-kind.js`，源码内嵌在 `opencode_agent.py` 的 `_REQUEST_KIND_PLUGIN` 字符串常量中），钩住 OpenCode 的 `chat.headers` 生命周期回调：若本次会话存在 `parentID`（说明是子 session）则设置请求头 `request_kind=collab_spawn`；若当前 agent 为 `compaction` 或消息带 compaction 标记，则设置 `request_kind=compaction`。

### 2.8 Sandbox

[sandbox/base.py](../../coda/agentflow/sandbox/base.py) 定义抽象基类 `SandboxClient`，接口为 `create()` / `execute(command, **kwargs)` / `delete()`。同样用 `Registry` + `@register_sandbox` 注册，`create_sandbox_client(config)` 按 `config.type` 工厂化（`type` 为 `none`/空时返回 `None`，即不启用沙箱）。

内置两种实现：

- **`K8sSandboxClient`（[sandbox/k8s_sandbox.py](../../coda/agentflow/sandbox/k8s_sandbox.py)）**：每个沙箱一个 K8s Pod，通过 `kubectl apply` 创建、`kubectl exec` 执行命令、`delete()` 删除。默认 `working_dir=/rl-sandbox`，`execute(command, workdir=...)` 支持按调用方指定的目录执行（拼成 `cd {workdir} && ...`）。
- **`DockerSandboxClient`（[sandbox/docker_sandbox.py](../../coda/agentflow/sandbox/docker_sandbox.py)）**：本地 Docker 容器实现，默认 `working_dir=/testbed`。

> **Pod 泄漏防护（K8s 沙箱清理机制）**：为避免 apiserver 抖动导致 pod 长期残留，删除路径做了多重加固：
> - `_force_delete()` 用 `kubectl delete --force --grace-period=0`，最多重试 `_FORCE_DELETE_RETRIES`（=3）次并线性退避；`NotFound` / `not found` 视为删除成功；重试耗尽仅打印 "pod may leak" 告警而不阻塞主流程。
> - `_force_delete()` 的 `kubectl delete` 调用带 `--request-timeout=30s`（`_KUBECTL_REQUEST_TIMEOUT`），防止 apiserver i/o 卡死时删除操作无限期挂起。
> - `create()` 中 pod readiness 失败时会立即 `_force_delete` 清理，`delete()` 正常退出时同样走 `_force_delete`。
> - **服务端兜底**：pod manifest（`conf/k8s/pod_manifest.yaml`）设置 `activeDeadlineSeconds: 86400`，即便客户端彻底失联，pod 也会在 24h 后被 K8s 自动回收。

> SWE-bench 场景下，agent 会显式把 `workdir` 指到仓库根（默认 `/testbed`，SWE-bench 镜像的仓库 checkout 位置，可用 `metadata["repo_path"]` 覆盖），因此每条命令实际运行在仓库目录，而不受沙箱通用默认目录影响。

## 3. Rollout 执行流程

`_run_trajectory()`（`agent_flow.py:340`）是单条 trajectory 的主循环，带 `retry_limit` 次重试：

1. `_prepare_attempt_template()` 准备本次提交的可变模板。partial rollout resume 且开启 `mask_offpolicy_in_partial_rollout` 时，会把已有 response token 的 `loss_masks` 全部置 0（旧版本 off-policy 前缀不参与 loss）；同时校验 `loss_masks` / `rollout_weight_versions` 的长度与 `rollout_log_probs` 一致，不一致直接抛 `ValueError`，避免带着损坏的对齐数据续跑。
2. 每次 attempt 置状态 `GENERATING`，写入 store，然后按模式执行：
   - **`_execute_single_turn()`**：不建 agent，直接由 AgentFlow 向 Router 的 `/v1/chat/completions` 发一次请求，再从 store 取回 trajectory 交给 reward 函数打分。`context_length_exceeded` 会被当作"用部分响应"而非硬失败。
   - **`_execute_multi_turn()`**：为该 attempt 创建 agent 实例（`router_url` 带 session 前缀），可选注入 sandbox 与 reward 函数，`await agent.run_trajectory(...)`，`finally` 中 `agent.clear()`。agent 跑完后会立即校验：`completed_attempt.rollout_log_probs` 不能为空，且 `completed_attempt.segments` 中必须至少有一个 `trainable=True`——否则直接抛 `RuntimeError`，走失败重试路径（避免产出一条无法用于训练的空 trajectory）。
3. 成功：`_post_process_reward()` 写回 reward、置 `COMPLETED`、构造 `token_rewards`（仅最后一位为 `final_reward`；若 reward 带 `completion_rewards` 则按 triplet 逐轮赋值并做数量校验）、写入 `is_correct`（reward 未判定时回退为 `final_reward > 0`），emit 终态 trajectory，并 `DELETE /release_session` 释放 sticky 映射。
4. 失败：置 `FAILED`，`POST /abort_session` 中止绑定 worker；重试耗尽则 emit 一条 `is_valid=False` 的 reward。

终态 trajectory 通过 `_emit_terminal_trajectory()` 写入 `TrajQueue` 并从 store 删除（无论入队成功与否都删，避免残留）。

## 4. 训练对齐机制（模块的核心价值）

RL 训练要求"采样时用的 token 序列 + 每个 token 的 logprob"能在训练侧精确回放，否则重要性采样比、KL 等都会失真。AgentFlow 通过 Router 把这条保证做实：

- **token in / token out 是这条保证的地基**：Router 转发时用 **`input_ids`（`TokenizerManager` 产出）而非文本**，并强制 `return_logprob=True`，使推理按精确 token 序列生成并返回逐 token logprob。整条链路上文本只在"给 agent 看"的那一层出现，从不参与训练数据的构造。
- `build_assistant_message()` 直接从 `meta_info.output_token_logprobs` 取 `response_ids` 与 `logprobs`，`update_trajectory()` 据此维护 `tokens` / `rollout_log_probs` / `loss_masks` / `rollout_weight_versions`，并断言 `len(logprobs) == len(response_ids)`。
- **loss_mask 语义**：response 空间里，LLM 生成的 token mask=1（计入 loss），工具输出、重建的 prompt/summary token、以及 partial-rollout 的 off-policy 前缀 mask=0。裁剪或切片时必须保留既有 mask 值，而不能仅按 token 类型重建。
- **think 块裁剪**：当 `enable_thinking and not accumulate_reasoning` 时，续轮前会在 token 级（用 `think_start_ids`/`think_end_ids`）和文本级一致地裁掉 **active segment 最后一个 triplet** 的 think 块，保证 tokens 与 chat 历史同步；只在命中当前 active segment 前缀的续轮（§2.3 情形 2）触发，`collab_spawn` 子 agent 分叉（情形 3）不裁剪（子 agent 轮次不会回到 mainline 历史，无需保持同步），compaction 开新段（情形 4）因新段没有历史 response 也不涉及。
- **weight version 跨越记录**：`start/end_rollout_weight_version` 逐轮打点，衡量该 trajectory 偏离策略的程度，即便 0-token 的 resume 轮也会记录版本跨越。
- **`Segment.trainable`（训练侧行展开）**：`origin="subagent"` 的占位 segment 目前恒为 `trainable=False`。`data_processor.py` 的 `put_dp_shards_to_ray()` 是训练数据从 `Trajectory` 摊平为训练行的唯一入口——按 `trajectory.segments` 遍历，`trainable=False` 的段直接 `continue`（占位段由此被物理排除，不会产生训练行）；每个 `trainable=True` 的 segment 独立展开成一条训练行，各自按 `token_start/token_end`、`logprob_start/logprob_end` 切片 `tokens`/`loss_masks`/`rollout_log_probs`/`rollout_weight_versions`/`token_rewards`/`rollout_routed_experts`。因此一条包含多个 mainline segment（如 `root`+`compact`）的 trajectory 会展开成多条训练行，而不是拼成一条长序列参与训练；`TrajectoryGroup.segment_count` 统计的正是这里最终会产出的训练行数（用于 DP 负载均衡不被占位段带偏）。

`Trajectory` 的 docstring（[trajectory_store.py](../../coda/agentflow/trajectory_store.py)）给出了含 Summary Reset 的完整 token 级示例，是理解 segment/triplet 索引空间的最佳参考。

## 5. 扩展点

AgentFlow 的四个主要扩展维度都是"注册表 + 配置驱动"，无需改动框架代码：

| 扩展点 | 基类 | 注册方式 | 配置字段 |
| --- | --- | --- | --- |
| 自定义 Agent | `BaseAgent` | `@register_agent("name")` | `data_sources[i].agent.name` + 专属参数 |
| 自定义 Reward | `RewardFunction` | `@register_reward("name")` | `data_sources[i].reward.name` + 专属参数 |
| 自定义 Sandbox | `SandboxClient` | `@register_sandbox("type")` | `agentflow.sandbox.type` + 专属参数 |

新增 agent 的最小步骤：继承 `BaseAgent`，实现 `run_trajectory` / `clear`，用 `@register_agent` 装饰，放到 `agent/` 下的包内即可被自动发现；通常还需配套一个 reward 函数（继承 `RewardFunction`、用 `@register_reward` 装饰、放到 `coda/reward/functions/` 下），并在数据源的 `reward.name` 中引用（若能复用已有 reward 则无需新写）。

详细开发指南：

- [自定义 Agent 开发指南](./custom-agent.md) —— 实现工具调用、沙箱执行或多步推理等自定义多轮 rollout 逻辑。
- [自定义 Reward 函数开发指南](./custom-reward.md) —— 为任务专属的打分逻辑编写自定义 reward。
- [自定义 Sandbox 开发指南](./custom-sandbox.md) —— 接入内置 Docker / K8s 之外的代码执行后端。

## 6. 关键配置

AgentFlow 相关配置集中在 `agentflow.*`、`rollout.*`、`data_sources[*]` 下，常用项：

| 配置 | 说明 |
| --- | --- |
| `data_sources[i].agent.name` | 多轮 agent 名；不设则该数据源为单轮模式 |
| `data_sources[i].reward.name` | 该数据源的 reward 函数名（rollout 打分必需） |
| `data_sources[i].max_response_len_per_trajectory` | 每条 trajectory 的 response token 预算 |
| `data_sources[i].num_trajectories_per_prompt` | 每个 prompt 采样的 trajectory 数（group 大小 `N`） |
| `data_sources[i].completion_params` | 透传给 LLM 的采样参数（如 `top_p`、`max_tokens` 等） |
| `agentflow.tokenizer.manager.mode` / `num_workers` | TokenizerManager 后端（`thread`/`process`）与并发数 |
| `agentflow.tokenizer.custom_chat_template_path` | 自定义 chat template 文件路径，相对路径按 `conf/` 解析（留空用 tokenizer 默认） |
| `agentflow.tokenizer.generation_prompt_kwargs` | 生成 prompt 时透传给 chat template 的额外参数 |
| `agentflow.router.ip` / `port` | Router 监听地址；留空自动解析本机 IP、端口 0 自动分配 |
| `agentflow.router.max_connections` | Router 向 worker 转发的最大连接数 |
| `agentflow.router.rollout_worker_load_threshold` | least-loaded 路由的 worker 负载阈值 |
| `agentflow.router.proxy_timeout_seconds` / `abort_timeout_seconds` | 转发请求超时 / abort 操作超时 |
| `agentflow.router.accumulate_reasoning` | 是否在多轮历史中保留 think 块 |
| `agentflow.router.middleware` | 中间件链配置，映射 `{name: params}`（默认 `{parser: null}`） |
| `agentflow.sandbox.type` | 沙箱类型（`k8s` / `docker` / `none`） |
| `agentflow.sandbox.working_dir` | 沙箱默认工作目录（k8s 默认 `/rl-sandbox`，可被 agent/metadata 覆盖） |
| `agentflow.sandbox.command_exec_timeout_seconds` / `sandbox_creation_timeout_seconds` | 命令执行 / pod 创建超时 |
| `agentflow.sandbox.kubeconfig` / `pod_manifest_path` | K8s 沙箱的 kubeconfig 与 pod manifest 路径 |
| `agentflow.dump_trajectory_path` | trajectory 落盘调试路径（留空则不落盘） |
| `rollout.retry_limit` | 单条 trajectory 的重试次数 |
| `rollout.partial` / `mask_offpolicy_in_partial_rollout` | partial rollout 及其 off-policy 掩码 |
| `trainer.temperature` | 采样温度（注入到 agent / 单轮请求） |
| `trainer.use_rollout_routing_replay` | 是否启用 R3（MoE routed experts 回放） |