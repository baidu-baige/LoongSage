"""GSM8K example agent with a calculator tool."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import shlex
import subprocess
import sys

import httpx
from typing import Any

from coda.agentflow.agent.base_agent import BaseAgent
from coda.agentflow.agent import register_agent
from coda.reward.functions.gsm8k import GSM8KReward, extract_answer, extract_ground_truth
from coda.reward.reward import Reward


logger = logging.getLogger(__name__)

# Reference patterns:
# - verl tool parser keeps parsing logic explicit and protocol-scoped:
#   https://github.com/volcengine/verl/blob/main/verl/verl/experimental/agent_loop/tool_parser.py
# - vLLM tool-calling docs also encourage parsing against a narrow format:
#   https://docs.vllm.ai/en/latest/features/tool_calling.html
_TOOL_JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def calculator_tool(expression: str, sandbox: Any = None) -> str:
    """Execute mathematical expressions safely.

    Validates the expression against an allowlist of characters and rejects patterns that could cause CPU exhaustion
    (e.g. 9**9**9) before delegating to eval or sandbox execution.

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

        if sandbox is not None:
            result = sandbox.execute(
                f"python3 -c {shlex.quote(program)}",
            )
            if not result.get("success"):
                error = (result.get("stderr") or result.get("stdout") or "Sandbox execution failed").strip()
                return f"Error: {error}"
            return str(result.get("stdout", "")).strip()

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
    """
    Agent for solving GSM8K mathematical reasoning problems.

    Configuration (via kwargs):
        completion_params: Sampling parameters for LLM requests.
        temperature: Sampling temperature (default: 0.7)
        max_iterations: Maximum tool calling iterations (default: 5)
        token_budget: Optional agent-side context budget.
    """

    # System prompt for tool-based reasoning
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
        router_url: str = "http://localhost:8000",
        sandbox_env_client: Any = None,
        reward_fn: Any = None,
        completion_params: dict = None,
        max_response_len_per_trajectory: int = 0,
        temperature: float = 0.7,
        max_iterations: int = 5,
        token_budget: dict | None = None,
        **kwargs: Any,
    ):
        """Initialize GSM8KAgent with LLM generation parameters."""
        super().__init__(
            router_url,
            completion_params=completion_params,
            max_response_len_per_trajectory=max_response_len_per_trajectory,
            temperature=temperature,
            **kwargs,
        )
        self.sandbox_env_client = sandbox_env_client
        self.reward_fn = reward_fn
        self.client = httpx.AsyncClient(timeout=120.0)
        self._closed: bool = False
        self._sandbox_ready: bool = False
        self.max_iterations = max_iterations
        budget = token_budget or {}
        self.context_window = int(budget.get("context_window", 0))
        self.reserve_tokens = int(budget.get("reserve_tokens", 512))
        self.budget_ratio = float(budget.get("budget_ratio", 0.85))

        logger.info("GSM8KAgent initialized with router_url=%s", router_url)
        logger.info(
            "  max_response_len_per_trajectory=%d, temperature=%s",
            self.max_response_len_per_trajectory,
            temperature,
        )
        logger.info("  max_iterations=%d", max_iterations)

    async def __aenter__(self) -> "GSM8KAgent":
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb) -> bool:
        await self.clear()
        return False

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
        metadata = trajectory.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        label_answer = label['value']
        ground_truth = extract_ground_truth(label_answer) if label_answer else None

        logger.info("run_trajectory ground_truth: %s", ground_truth)

        if self.sandbox_env_client is not None and not self._sandbox_ready:
            if getattr(self.sandbox_env_client, "sandbox_id", None):
                # Pooled sandbox from a partial-rollout abort: reuse it.
                logger.info(
                    "Reusing existing sandbox %s", self.sandbox_env_client.sandbox_id
                )
                self._sandbox_ready = True
            else:
                image = metadata_dict.get("docker_image")
                if not image and isinstance(prompt, dict):
                    image = prompt.get("docker_image")
                if image:
                    await asyncio.to_thread(self.sandbox_env_client.create, image=image)
                    self._sandbox_ready = True
                else:
                    logger.info("No docker_image provided.")

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

        if not any(
            isinstance(msg, dict) and msg.get("role") == "system" for msg in messages
        ):
            messages.insert(0, {"role": "system", "content": self.SYSTEM_PROMPT})

        question_preview = next(
            (
                msg.get("content", "")
                for msg in reversed(messages)
                if isinstance(msg, dict) and msg.get("role") == "user"
            ),
            "",
        )
        logger.info("Question: %s", question_preview)
        if ground_truth is not None:
            logger.info("Ground truth: %s", ground_truth)

        # Run with tool calling
        final_answer, tool_calls_made = await self._run_with_tools(messages)

        # Calculate reward — prefer injected reward_fn, fall back to inline logic
        if self.reward_fn is not None:
            r = self.reward_fn(messages, {"answer": label_answer})
            return Reward(
                final_reward=r.final_reward,
                completion_rewards=r.completion_rewards,
                is_valid=r.is_valid,
                extra_info=r.extra_info,
            )
        # Fallback: delegate to canonical GSM8KReward so behavior is consistent
        # with the configured reward_fn path (no divergent partial-credit logic).
        return GSM8KReward()(messages, {"answer": label_answer})

    async def _run_with_tools(self, messages: list[dict]) -> tuple[float | None, list[dict]]:
        """Run with tool calling support.

        Returns:
            tuple of (final_answer, tool_calls_made)
        """
        tool_calls_made = []
        iteration = sum(
            isinstance(message, dict) and message.get("role") == "assistant"
            for message in messages
        )
        total_tokens_used = 0
        response = next(
            (
                str(message.get("content") or "")
                for message in reversed(messages)
                if isinstance(message, dict) and message.get("role") == "assistant"
            ),
            "",
        )

        while iteration < self.max_iterations:
            iteration += 1
            logger.debug("Tool iteration %d/%d", iteration, self.max_iterations)

            turn_max_tokens = self.max_response_len_per_trajectory
            if self.context_window > 0:
                remaining = self.context_window - total_tokens_used - self.reserve_tokens
                turn_max_tokens = max(1, min(turn_max_tokens, remaining))

            # Call LLM
            response, usage = await self._call_llm(messages, max_tokens=turn_max_tokens)
            total_tokens_used += usage.get("prompt_tokens", 0) + usage.get(
                "completion_tokens", 0
            )

            logger.debug("LLM response: %s", response)

            # Check token budget before deciding to continue
            if (
                self.context_window > 0
                and total_tokens_used >= self.context_window * self.budget_ratio
            ):
                logger.warning(
                    "Token budget exhausted (%d tokens used), forcing early stop",
                    total_tokens_used,
                )
                messages.append({"role": "assistant", "content": response})
                return extract_answer(response), tool_calls_made

            # Check for tool calls
            tool_calls = parse_tool_calls_from_text(response)

            if not tool_calls:
                # No tool calls - this is the final answer
                answer = extract_answer(response)
                logger.info("No tool calls. Extracted answer: %s", answer)
                messages.append({"role": "assistant", "content": response})
                return answer, tool_calls_made

            # Execute tool calls
            logger.info("Found %d tool call(s)", len(tool_calls))

            # Add assistant message
            messages.append({"role": "assistant", "content": response})

            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                logger.info("Executing tool: %s(%s)", tool_name, tool_args)

                # Execute tool
                if tool_name == "calculator":
                    result = await asyncio.to_thread(
                        calculator_tool,
                        tool_args.get("expression", ""),
                        self.sandbox_env_client if self._sandbox_ready else None,
                    )
                else:
                    result = f"Error: Unknown tool '{tool_name}'"

                logger.info("Tool result: %s", result)
                tool_calls_made.append(
                    {"name": tool_name, "args": tool_args, "result": result}
                )

                # Add tool result message
                messages.append({"role": "user", "content": f"Tool result: {result}"})

        # Max iterations reached — return whatever we have from the last response.
        logger.warning("Max iterations (%d) reached", self.max_iterations)
        return extract_answer(response), tool_calls_made

    async def _call_llm(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, int]]:
        """Call the LLM via the router.

        Args:
            messages: Conversation messages to send.
            max_tokens: Request-level generation cap. If None,
                max_response_len_per_trajectory is used.

        Returns:
            Tuple of(generated_text, usage) where usage contains prompt_tokens and completion_tokens from the response.
            Both fields default to 0 when the worker does not return usage info.
        """
        request_body = {
            "messages": messages,
            "model": "default",
            "temperature": self.temperature,
            **self.completion_params,
        }
        request_body["max_tokens"] = (
            max_tokens if max_tokens is not None else self.max_response_len_per_trajectory
        )

        logger.info("Sending request to %s", self.router_url)

        response = await self.client.post(
            f"{self.router_url}/v1/chat/completions",
            json=request_body,
        )
        response.raise_for_status()
        result = response.json()

        logger.debug("in _call_llm, LLM response: %s", result)

        if "choices" in result and len(result["choices"]) > 0:
            text = result["choices"][0]["message"]["content"] or ""
        else:
            raise ValueError(f"Unexpected response format: {result}")

        usage_raw = result.get("usage") or {}
        usage: dict[str, int] = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens", 0)),
            "completion_tokens": int(usage_raw.get("completion_tokens", 0)),
        }
        return text, usage

    async def clear(self) -> None:
        """Clean up resources. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        # Sandbox deletion is owned by AgentFlow's sandbox pool (release_sandbox),
        # which keeps the sandbox alive across partial-rollout aborts for reuse.
        self._sandbox_ready = False
        await self.client.aclose()
        logger.info("GSM8KAgent resources cleared")
