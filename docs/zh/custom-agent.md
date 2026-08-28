# 自定义 Agent 开发指南

本文说明如何在 LoongSage 中新增自定义 agent。agent 扩展同样是 **“注册表 + 配置驱动”**：继承基类、添加注册装饰器，即可通过 `data_source.agent.name` 引用，无需修改调度代码。

如果一条样本只需请求模型一次即可完成，例如普通问答或数学题，就不需要 agent，将 `agent.name` 留空即可。只有任务需要模型与工具反复交互，例如执行代码、检索资料或驱动外部 CLI，才需要自定义 agent。开始开发前可先查看内置的 `gsm8k`、`mini-swe`、`bcp` 和 `opencode` 是否能够复用。

## 1. 开发步骤

agent 基类和内置实现位于 [agent 目录](../../coda/agentflow/agent/)。新增 agent 时，按以下步骤操作：

1. 继承 [BaseAgent](../../coda/agentflow/agent/base_agent.py)，实现异步方法 `run_trajectory(trajectory)` 和 `clear()`。构造函数接收 `router_url` 及通用采样参数，并保留 `**kwargs` 接收 `agent` 配置块中的自定义字段。
2. 在 `run_trajectory()` 中读取 `prompt`、`label` 和 `metadata`。所有模型请求都发送到 `router_url`，执行结束后返回 `Reward`。
3. 按需接收 `sandbox_env_client` 和 `reward_fn`。sandbox 的阻塞调用使用 `asyncio.to_thread(...)`；`clear()` 只清理 agent 自身资源。
4. 使用 `@register_agent("your-name")` 注册，并把实现放到 [agent 目录](../../coda/agentflow/agent/)下，LoongSage 会自动发现。

## 2. 最小示例

以下示例展示一个 agent 的基本结构。`_parse_tool_call()` 代表任务自己的工具协议解析逻辑：

```python
# coda/agentflow/agent/my_agent.py
import asyncio
from typing import Any

import httpx

from coda.agentflow.agent import register_agent
from coda.agentflow.agent.base_agent import BaseAgent
from coda.reward.reward import Reward


@register_agent("my-agent")
class MyAgent(BaseAgent):
    def __init__(
        self,
        router_url: str,
        completion_params: dict | None = None,
        max_response_len_per_trajectory: int = 0,
        temperature: float = 1.0,
        sandbox_env_client: Any = None,
        reward_fn: Any = None,
        max_turns: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            router_url=router_url,
            completion_params=completion_params,
            max_response_len_per_trajectory=max_response_len_per_trajectory,
            temperature=temperature,
            **kwargs,
        )
        self.sandbox = sandbox_env_client
        self.reward_fn = reward_fn
        self.max_turns = max_turns
        self.client = httpx.AsyncClient(timeout=120.0)
        self._closed = False

    async def run_trajectory(self, trajectory: dict[str, Any]) -> Reward:
        prompt = trajectory["prompt"]
        label = trajectory.get("label")
        metadata = trajectory.get("metadata") or {}
        messages = prompt if isinstance(prompt, list) else [
            {"role": "user", "content": str(prompt)}
        ]

        if self.sandbox is not None and not self.sandbox.sandbox_id:
            await asyncio.to_thread(
                self.sandbox.create,
                image=metadata.get("docker_image"),
            )

        for _ in range(self.max_turns):
            assistant = await self._complete(messages)
            messages.append(assistant)
            tool_call = _parse_tool_call(assistant)
            if tool_call is None:
                break
            if self.sandbox is None:
                raise RuntimeError("this agent requires a sandbox")
            result = await asyncio.to_thread(
                self.sandbox.execute,
                tool_call["command"],
                workdir=tool_call.get("workdir"),
            )
            messages.append({
                "role": "user",
                "content": f"Tool result:\n{result['stdout']}\n{result['stderr']}",
            })

        if self.reward_fn is None:
            return Reward(final_reward=0.0, is_valid=False)
        return self.reward_fn(messages, label)

    async def _complete(self, messages: list[dict]) -> dict:
        payload = {
            "model": "default",
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_response_len_per_trajectory,
            **self.completion_params,
        }
        response = await self.client.post(
            f"{self.router_url}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]

    async def clear(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.client.aclose()
```

简单工具循环可参考 [gsm8k_agent.py](../../coda/agentflow/agent/gsm8k/gsm8k_agent.py)，代码修复任务可参考 [mini_swe_agent.py](../../coda/agentflow/agent/swe/mini_swe_agent.py)，外部 CLI 驱动可参考 [opencode_agent.py](../../coda/agentflow/agent/opencode/opencode_agent.py)。

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

agentflow:
  sandbox:
    type: remote           # ← 注入为 sandbox_env_client；none 表示禁用
```

`agent.context_length` 用于配置 agent 完成一个任务时可使用的上下文窗口。Router 会保证当前上下文和本次生成不超过该值，支持上下文压缩的 agent 也会据此管理上下文；省略或设为 `0` 表示不额外限制。它与 `max_response_len_per_trajectory` 不同，后者控制回复与工具结果的累计 token 预算。

配置生效后，每条使用 agent 的 trajectory 都会创建一个 agent 实例。多数据源配置下，各 `data_sources[i]` 可以使用不同 agent 和 reward，但 `agentflow.sandbox` 是全局配置。
