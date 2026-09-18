# Custom Agent Development Guide

This document explains how to add a custom agent to LoongSage. Agent extensions are likewise **registry- and config-driven**: inherit the base class, add a registration decorator, and reference the implementation through `data_source.agent.name` without changing scheduling code.

If one sample can be completed with a single model request—for example, ordinary question answering or a math problem—no agent is needed; leave `agent.name` empty. Add a custom agent only when the model must interact with tools repeatedly. Check the [agent directory](../../coda/agentflow/agent/) for reusable implementations first.

## 1. Development Steps

The agent base class and built-in implementations live in the [agent directory](../../coda/agentflow/agent/). Follow these steps to add an agent:

1. Inherit from [BaseAgent](../../coda/agentflow/agent/base_agent.py) and implement the async method `run_trajectory(trajectory)`. The constructor receives `router_url`, `reward_fn`, `sandbox_client`, `sandbox_id`, and `max_response_len_per_trajectory`; keep `**kwargs` for custom fields from the `agent` config block. Sampling parameters are applied by Router and are not passed to agents.
2. Read `prompt`, `label`, and `metadata` in `run_trajectory()`. Send every model request to `router_url`, then call the injected `reward_fn` and return its `Reward`. `reward_fn` is synchronous, so call it through `asyncio.to_thread` to keep the event loop free.
3. If the agent uses a sandbox, use the injected `sandbox_client` and `sandbox_id` directly, and put both into the third argument (the context) you pass to `reward_fn` so the reward evaluates inside the same container.
4. Register the implementation with `@register_agent("your-name")` and place it in [coda/custom/](../../coda/custom/); LoongSage discovers it automatically, see [Custom Extensions](./custom-extensions.md).

## 2. Minimal Example

The following example shows a tool agent **that uses a sandbox**. Each agent defines its own tool protocol and decides how to parse tool calls out of a model reply; the framework ships no parser. This example reads the Chat Completions `tool_calls` field:

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
                # This example registers a single shell tool; dispatch on
                # tc["function"]["name"] when there are several.
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

With `sandbox: {type: none}` on the data source no sandbox is injected, tools run locally (`asyncio.to_thread(your_tool, ...)`), and `metadata` is passed as the reward context. See [gsm8k_agent.py](../../coda/agentflow/agent/gsm8k/gsm8k_agent.py) (calculator tool) and [bcp_agent.py](../../coda/agentflow/agent/bcp/bcp_agent.py) (HTTP tools) for that shape.

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
  sandbox: {type: remote}  # ← AgentFlow injects client + id; none disables it
```

`agent.context_length` configures the context window available while the agent completes one task. The Router keeps the current context plus the next generation within this limit, and agents that support compaction can use it to manage their context. Omit it or set it to `0` to disable the additional limit. This differs from `max_response_len_per_trajectory`, which controls the cumulative token budget for replies and tool results.

After configuration, LoongSage creates one agent instance for each trajectory that uses an agent. Each `data_sources[i]` may select its own agent, reward, and sandbox backend.
