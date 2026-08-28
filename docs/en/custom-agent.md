# Custom Agent Development Guide

This document explains how to add a custom agent to LoongSage. Agent extensions are likewise **registry- and config-driven**: inherit the base class, add a registration decorator, and reference the implementation through `data_source.agent.name` without changing scheduling code.

If one sample can be completed with a single model request—for example, ordinary question answering or a math problem—no agent is needed; leave `agent.name` empty. Add a custom agent only when the model must interact with tools repeatedly, such as executing code, searching for information, or driving an external CLI. Before writing one, check whether the built-in `gsm8k`, `mini-swe`, `bcp`, or `opencode` agent can be reused.

## 1. Development Steps

The agent base class and built-in implementations live in the [agent directory](../../coda/agentflow/agent/). Follow these steps to add an agent:

1. Inherit from [BaseAgent](../../coda/agentflow/agent/base_agent.py) and implement the async methods `run_trajectory(trajectory)` and `clear()`. The constructor receives `router_url` and common sampling parameters; keep `**kwargs` for custom fields from the `agent` config block.
2. Read `prompt`, `label`, and `metadata` in `run_trajectory()`. Send every model request to `router_url` and return a `Reward` when execution finishes.
3. Accept `sandbox_env_client` and `reward_fn` when needed. Run blocking sandbox calls through `asyncio.to_thread(...)`; `clear()` should release only agent-owned resources.
4. Register the implementation with `@register_agent("your-name")` and place it in the [agent directory](../../coda/agentflow/agent/); LoongSage discovers it automatically.

## 2. Minimal Example

The following example shows the basic structure of an agent. `_parse_tool_call()` represents task-specific tool-protocol parsing:

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

See [gsm8k_agent.py](../../coda/agentflow/agent/gsm8k/gsm8k_agent.py) for a simple tool loop, [mini_swe_agent.py](../../coda/agentflow/agent/swe/mini_swe_agent.py) for code-repair tasks, and [opencode_agent.py](../../coda/agentflow/agent/opencode/opencode_agent.py) for an external CLI-driven agent.

## 3. Config Enablement

Once registered, select the agent by name in the data source. Other fields in the `agent` block are passed directly to the constructor, while reward and sandbox are configured independently:

```yaml
data_source:
  agent:
    name: my-agent         # ← the @register_agent registered name
    max_turns: 5           # maximum number of model requests
    context_length: 65536  # context window available while the agent completes one task
  reward:
    name: exact-match      # ← injected as reward_fn
  max_response_len_per_trajectory: 32768  # cumulative token budget for replies and tool results

agentflow:
  sandbox:
    type: remote           # ← injected as sandbox_env_client; none disables it
```

`agent.context_length` configures the context window available while the agent completes one task. The Router keeps the current context plus the next generation within this limit, and agents that support compaction can use it to manage their context. Omit it or set it to `0` to disable the additional limit. This differs from `max_response_len_per_trajectory`, which controls the cumulative token budget for replies and tool results.

After configuration, LoongSage creates one agent instance for each trajectory that uses an agent. With multiple data sources, each `data_sources[i]` may use a different agent and reward, while `agentflow.sandbox` is global configuration.
