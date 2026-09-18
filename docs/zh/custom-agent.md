# 自定义 Agent 开发指南

本文说明如何在 LoongSage 中新增自定义 agent。agent 扩展同样是 **“注册表 + 配置驱动”**：继承基类、添加注册装饰器，即可通过 `data_source.agent.name` 引用，无需修改调度代码。

如果一条样本只需请求模型一次即可完成，就不需要 agent，将 `agent.name` 留空即可。只有任务需要模型与工具反复交互时才需要自定义 agent；开始开发前先检查 [agent 目录](../../coda/agentflow/agent/) 中是否已有可复用实现。

## 1. 开发步骤

agent 基类和内置实现位于 [agent 目录](../../coda/agentflow/agent/)。新增 agent 时，按以下步骤操作：

1. 继承 [BaseAgent](../../coda/agentflow/agent/base_agent.py)，实现异步方法 `run_trajectory(trajectory)`。构造函数由 AgentFlow 注入 `router_url`、`reward_fn`、`sandbox_client`、`sandbox_id` 和 `max_response_len_per_trajectory`，并保留 `**kwargs` 接收 `agent` 配置块中的自定义字段。采样参数由 Router 直接读取数据源配置，不传给 agent。
2. 在 `run_trajectory()` 中读取 `prompt`、`label` 和 `metadata`。所有模型请求都发送到 `router_url`，执行结束后调用注入的 `reward_fn` 并返回 `Reward`。`reward_fn` 是同步的，用 `asyncio.to_thread` 调用以免阻塞事件循环。
3. agent 使用 sandbox 时，直接使用注入的 `sandbox_client` 和 `sandbox_id`；同时要把这两个值放进传给 `reward_fn` 的第三个参数（context）里，reward 才能在同一个容器里评测。
4. 使用 `@register_agent("your-name")` 注册，并把实现放到 [coda/custom/](../../coda/custom/) 下，LoongSage 会自动发现，详见[自定义扩展](./custom-extensions.md)。

## 2. 最小示例

以下示例展示一个**用 sandbox** 的工具 agent。工具协议、以及怎么从模型回复里解析工具调用，都由 agent 自己决定，框架不提供解析器。这里用 Chat Completions 的 `tool_calls` 字段：

```python
# coda/custom/my_agent.py
import asyncio
import json
from typing import Any

from coda.agentflow.agent import register_agent
from coda.agentflow.agent.base_agent import BaseAgent
from coda.reward.reward import Reward


@register_agent("my-agent")
class MyAgent(BaseAgent):
    def __init__(self, *args: Any, max_turns: int = 5, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_turns = max_turns

    async def run_trajectory(self, trajectory: dict[str, Any]) -> Reward:
        label = trajectory.get("label")
        metadata = trajectory.get("metadata") or {}
        messages = list(trajectory["prompt"])
        if self.sandbox_id is None:
            self.sandbox_id = await asyncio.to_thread(self.sandbox_client.create, image=metadata["docker_image"])

        for _ in range(self.max_turns):
            response = await self.call_llm(messages)
            assistant = response["choices"][0]["message"]
            messages.append(assistant)
            tool_calls = assistant.get("tool_calls") or []
            if not tool_calls:
                break
            for tc in tool_calls:
                # 本示例只注册了一个 shell 工具；有多个就按 tc["function"]["name"] 分派。
                arguments = json.loads(tc["function"]["arguments"])
                result = await asyncio.to_thread(
                    self.sandbox_client.execute, self.sandbox_id, arguments["command"])
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": f"{result['stdout']}{result['stderr']}",
                })

        return await asyncio.to_thread(
            self.reward_fn,
            messages,
            label,
            {**metadata, "sandbox_client": self.sandbox_client, "sandbox_id": self.sandbox_id},
        )
```

数据源配 `sandbox: {type: none}` 时不注入 sandbox，工具在本地执行（`asyncio.to_thread(your_tool, ...)`），reward 的 context 直接传 `metadata`。这类 agent 可参考 [gsm8k_agent.py](../../coda/agentflow/agent/gsm8k/gsm8k_agent.py)（计算器工具）和 [bcp_agent.py](../../coda/agentflow/agent/bcp/bcp_agent.py)（HTTP 工具）。

## 3. 配置启用

注册后，在数据源中按注册名选择 agent。`agent` 块中的其他字段会直接传给构造函数；reward 和 sandbox 分别独立配置：

```yaml
data_source:
  agent:
    name: my-agent         # ← @register_agent 的注册名
    max_turns: 5           # 最多请求模型的次数
    context_length: 65536  # agent 完成一个任务时可使用的上下文窗口
  reward:
    name: exact-match      # ← 注入为 reward_fn
  max_response_len_per_trajectory: 32768  # 整个任务中回复与工具结果的累计 token 预算
  sandbox: {type: remote}  # ← AgentFlow 注入 client + id；none 表示禁用
```

`agent.context_length` 用于配置 agent 完成一个任务时可使用的上下文窗口。Router 会保证当前上下文和本次生成不超过该值，支持上下文压缩的 agent 也会据此管理上下文；省略或设为 `0` 表示不额外限制。它与 `max_response_len_per_trajectory` 不同，后者控制回复与工具结果的累计 token 预算。

配置生效后，每条使用 agent 的 trajectory 都会创建一个 agent 实例。各 `data_sources[i]` 可以独立选择 agent、reward 与 sandbox 后端。
