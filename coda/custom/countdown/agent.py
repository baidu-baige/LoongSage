"""Countdown numbers-game agent: multi-turn tool loop with a local calculator (GSM8K tool protocol)."""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import Any

from coda.agentflow.agent import register_agent
from coda.agentflow.agent.base_agent import BaseAgent
from coda.agentflow.agent.gsm8k.gsm8k_agent import (
    calculator_tool,
    parse_tool_calls_from_text,
)
from coda.agentflow.utils import ContextLengthExceededError
from coda.reward.reward import Reward

logger = logging.getLogger(__name__)


@register_agent("countdown")
class CountdownAgent(BaseAgent):
    SYSTEM_PROMPT = """You are a puzzle solver playing the Countdown numbers game, with access to a calculator tool.

Given a list of numbers and a target number, find an arithmetic expression that:
1. Uses every given number exactly once (no number may be reused or omitted),
2. Uses only + - * / and parentheses,
3. Evaluates exactly to the target.

Think step by step. Use the calculator tool to try candidate expressions.

You have access to the following tool:
- calculator(expression): Evaluate an arithmetic expression. Returns the numerical result.

To use the tool, respond with JSON in this format:
```json
{
  "tool": "calculator",
  "expression": "your math expression here"
}
```

For example:
```json
{
  "tool": "calculator",
  "expression": "(1 + 2) * 8"
}
```

After getting the tool result, continue reasoning. When you are confident,
give the final expression (and nothing else) inside answer tags:
<answer>(1 + 2) * 8</answer>"""

    def __init__(
        self,
        router_url: str,
        max_response_len_per_trajectory: int = 0,
        max_iterations: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(router_url, max_response_len_per_trajectory=max_response_len_per_trajectory, **kwargs)
        self.max_iterations = max_iterations

        logger.info("CountdownAgent initialized with router_url=%s", router_url)
        logger.info("  max_response_len_per_trajectory=%d", self.max_response_len_per_trajectory)
        logger.info("  max_iterations=%d", max_iterations)

    async def run_trajectory(self, trajectory: Any) -> Reward:
        logger.info("Starting Countdown trajectory")

        if not isinstance(trajectory, dict):
            raise TypeError(f"CountdownAgent expects a dict trajectory, got {type(trajectory).__name__}")

        prompt = trajectory.get("prompt")
        label = trajectory.get("label")

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, dict):
            if isinstance(prompt.get("messages"), list):
                messages = copy.deepcopy(prompt["messages"])
            else:
                messages = [{"role": "user", "content": prompt.get("question", str(prompt))}]
        elif isinstance(prompt, list):
            messages = copy.deepcopy(prompt)
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        if not any(isinstance(msg, dict) and msg.get("role") == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": self.SYSTEM_PROMPT})

        await self._run_with_tools(messages)
        return await asyncio.to_thread(self.reward_fn, messages, label, {})

    async def _run_with_tools(self, messages: list[dict]) -> None:
        iteration = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")

        while iteration < self.max_iterations:
            iteration += 1
            logger.debug("Tool iteration %d/%d", iteration, self.max_iterations)

            try:
                result = await self.call_llm(messages)
            except ContextLengthExceededError as exc:
                logger.warning("CountdownAgent: token budget exhausted: %s", exc)
                break
            response = result["choices"][0]["message"].get("content") or ""

            logger.debug("LLM response: %s", response)

            tool_calls = parse_tool_calls_from_text(response)

            if not tool_calls:
                logger.info("No tool calls; finishing trajectory")
                messages.append({"role": "assistant", "content": response})
                return

            logger.info("Found %d tool call(s)", len(tool_calls))
            messages.append({"role": "assistant", "content": response})

            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                logger.info("Executing tool: %s(%s)", tool_name, tool_args)
                if tool_name == "calculator":
                    result = await asyncio.to_thread(calculator_tool, tool_args.get("expression", ""))
                else:
                    result = f"Error: Unknown tool '{tool_name}'"
                logger.info("Tool result: %s", result)
                messages.append({"role": "user", "content": f"Tool result: {result}"})

        logger.warning("Max iterations (%d) reached", self.max_iterations)
