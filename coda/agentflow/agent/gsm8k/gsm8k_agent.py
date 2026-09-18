"""GSM8K example agent with a calculator tool."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import subprocess
import sys

from typing import Any

from coda.agentflow.agent import register_agent
from coda.agentflow.agent.base_agent import BaseAgent
from coda.agentflow.utils import ContextLengthExceededError
from coda.reward.reward import Reward


logger = logging.getLogger(__name__)

# Reference patterns:
# - verl tool parser keeps parsing logic explicit and protocol-scoped:
#   https://github.com/volcengine/verl/blob/main/verl/verl/experimental/agent_loop/tool_parser.py
# - vLLM tool-calling docs also encourage parsing against a narrow format:
#   https://docs.vllm.ai/en/latest/features/tool_calling.html
_TOOL_JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def calculator_tool(expression: str) -> str:
    """Execute a basic arithmetic expression in a restricted subprocess.

    Validates the expression against an allowlist of characters and rejects patterns that could cause CPU exhaustion
    (e.g. 9**9**9) before running it in a local subprocess.

    Args:
        expression: A plain-text arithmetic expression such as '15 * 7 + 3'.

    Returns:
        The string representation of the result, or an error message.
    """
    if not expression:
        return "Error: No expression provided"

    try:
        # Safe evaluation - only allow basic math operations
        allowed_chars = set("0123456789+-*/().% ") 
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression. Only basic math operations allowed."

        # Reject nested exponentiation like 9**9**9 — right-associative
        # Single ** (e.g. 2**8) is allowed.
        if expression.count('**') > 1:
            return "Error: Nested exponentiation is not allowed."

        # Reject 6+ digit integer literals as an extra safety net.
        if re.search(r'\d{6,}', expression):
            return "Error: Expression too complex (potential CPU exhaustion)"

        program = (
            "expr = " + json.dumps(expression) +
            "; print(eval(expr, {'__builtins__': {}}, {}))"
        )

        # Local execution via subprocess with timeout — avoids in-process
        # eval() so a runaway expression can be killed at the OS level.
        proc = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode != 0:
            return f"Error: {(proc.stderr or 'execution failed').strip()}"
        return proc.stdout.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def _extract_tool_json_blocks(text: str) -> list[str]:
    """Extract JSON blocks that may encode tool calls."""
    blocks = [match.group(1) for match in _TOOL_JSON_BLOCK_PATTERN.finditer(text)]
    if blocks:
        return blocks

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return [stripped]
    return []


def _parse_tool_call_json(block: str) -> dict[str, Any] | None:
    """Parse one tool-call JSON object."""
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict) or "tool" not in data:
        return None

    return {
        "name": str(data["tool"]),
        "arguments": {k: v for k, v in data.items() if k != "tool"},
    }


def parse_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    """Parse tool calls from LLM text output using the example JSON protocol."""
    tool_calls = []
    for block in _extract_tool_json_blocks(text):
        parsed = _parse_tool_call_json(block)
        if parsed is not None:
            tool_calls.append(parsed)
    return tool_calls


# ============================================================================
# GSM8K Agent
# ============================================================================


@register_agent("gsm8k")
class GSM8KAgent(BaseAgent):
    """GSM8K white-box agent demonstrating a text-defined local tool.

    ``max_iterations`` is read from ``data_source.agent`` and limits model
    turns. The calculator runs locally; this agent never uses a sandbox.
    """

    SYSTEM_PROMPT = """You are a helpful math problem solver with access to a calculator tool.

When solving math problems:
1. Think through the problem step by step
2. Use the calculator tool for any arithmetic calculations
3. Show your reasoning clearly

You have access to the following tools:
- calculator(expression): Evaluate mathematical expressions. Returns the numerical result.

To use a tool, respond with JSON in this format:
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
  "expression": "15 * 7 + 3"
}
```

After getting the tool result, continue your reasoning and provide the final answer in the format:
#### <your_answer>"""

    def __init__(
        self,
        router_url: str,
        max_response_len_per_trajectory: int = 0,
        max_iterations: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(router_url, max_response_len_per_trajectory=max_response_len_per_trajectory, **kwargs)
        self.max_iterations = max_iterations

        logger.info("GSM8KAgent initialized with router_url=%s", router_url)
        logger.info("  max_response_len_per_trajectory=%d", self.max_response_len_per_trajectory)
        logger.info("  max_iterations=%d", max_iterations)

    async def run_trajectory(self, trajectory: Any) -> Reward:
        """
        Run a complete trajectory for a GSM8K problem.

        Supports tool calling for multi-turn reasoning.
        """
        logger.info("Starting GSM8K trajectory")

        if not isinstance(trajectory, dict):
            raise TypeError(f"GSM8KAgent expects a dict trajectory, got {type(trajectory).__name__}")

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
        """Append model turns and calculator results to ``messages``."""
        iteration = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")

        while iteration < self.max_iterations:
            iteration += 1
            logger.debug("Tool iteration %d/%d", iteration, self.max_iterations)

            try:
                result = await self.call_llm(messages)
            except ContextLengthExceededError as exc:
                logger.warning("GSM8KAgent: token budget exhausted: %s", exc)
                break
            response = result["choices"][0]["message"].get("content") or ""

            logger.debug("LLM response: %s", response)

            # Check for tool calls
            tool_calls = parse_tool_calls_from_text(response)

            if not tool_calls:
                # No tool calls - this is the final answer
                logger.info("No tool calls; finishing trajectory")
                messages.append({"role": "assistant", "content": response})
                return

            # Execute tool calls
            logger.info("Found %d tool call(s)", len(tool_calls))

            # Add assistant message
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
