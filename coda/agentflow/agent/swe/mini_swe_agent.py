"""SWE-bench agent wrapper around mini-swe-agent."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import minisweagent
import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import FormatError, InterruptAgentFlow
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.retry import retry

from coda.agentflow.agent import BaseAgent, register_agent
from coda.agentflow.agent.swe.r2e import with_r2e_environment
from coda.agentflow.sandbox.base import SandboxClient
from coda.agentflow.utils import CONTEXT_LENGTH_EXCEEDED
from coda.reward.reward import Reward

logger = logging.getLogger(__name__)

# Load swebench.yaml once at module level — contains the complete 7-step workflow
# instance_template (with {{task}} Jinja2 placeholder) and observation_template.
_SWEBENCH_CONFIG_PATH = Path(minisweagent.__file__).parent / "config/benchmarks/swebench.yaml"
_SWEBENCH_CONFIG: dict[str, Any] = yaml.safe_load(_SWEBENCH_CONFIG_PATH.read_text()) or {}


class CodaLitellmModel(LitellmModel):
    """LitellmModel variant that preserves the assistant reply on FormatError."""

    def query(self, messages, **kwargs):
        """Query LiteLLM and preserve the raw assistant message on FormatError."""
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": [],
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        try:
            message["extra"]["actions"] = self._parse_actions(response)
        except FormatError as e:
            raise FormatError(message, *e.messages) from e
        return message


def _is_context_length_exceeded_error(exc: BaseException) -> bool:
    """Return True if an agent exception represents Router token-budget exhaustion."""
    response = getattr(exc, "response", None)
    try:
        text = f"{getattr(response, 'text', '')} {exc}"
    except Exception:
        text = str(exc)
    return CONTEXT_LENGTH_EXCEEDED in text or "exhausted" in text



class CodaSandboxEnvironment:
    """mini-swe-agent Environment that delegates shell execution to a CODA SandboxClient.

    Implements the Environment interface expected by mini-swe-agent's DefaultAgent:
    - execute()          — run a bash command and return output dict
    - get_template_vars() — Jinja2 template variables (cwd, platform info)
    - serialize()        — serializable snapshot for logging
    - config.cwd         — working directory attribute accessed by DefaultAgent
    """

    def __init__(
        self,
        sandbox: SandboxClient,
        cwd: str = "/testbed",
        use_r2e_environment: bool = False,
    ) -> None:
        self._sandbox = sandbox
        self._cwd = cwd
        self._use_r2e_environment = use_r2e_environment
        # mini-swe-agent accesses env.config.cwd
        self.config = type("_Cfg", (), {"cwd": cwd, "to_dict": lambda s: {"cwd": cwd}})()

    def execute(self, action: dict | str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a shell command inside the sandbox pod."""
        from minisweagent.exceptions import Submitted

        command = action.get("command", "") if isinstance(action, dict) else action
        if self._use_r2e_environment:
            command = with_r2e_environment(command)
        workdir = cwd or self._cwd
        try:
            result = self._sandbox.execute(command, workdir=workdir)
            output: dict[str, Any] = {
                "output": result.get("stdout", "") + result.get("stderr", ""),
                "returncode": result.get("exit_code", -1),
                "exception_info": "",
            }
        except Exception as e:
            output = {"output": "", "returncode": -1, "exception_info": str(e)}

        # Detect submission sentinel: the agent runs
        #   echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
        # The first non-empty line of stdout is the sentinel; everything after is the patch.
        lines = output["output"].lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted({
                "role": "exit",
                "content": submission,
                "extra": {"exit_status": "Submitted", "submission": submission},
            })

        return output

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        """Return Jinja2 template variables for prompt rendering."""
        return {"cwd": self._cwd, **platform.uname()._asdict(), **kwargs}

    def serialize(self) -> dict[str, Any]:
        """Return a serializable snapshot for trajectory logging."""
        return {"info": {"config": {"environment": {"cwd": self._cwd}}}}


class FormatErrorLimitedAgent(DefaultAgent):
    """DefaultAgent that terminates after N consecutive format errors.

    Prevents training collapse caused by the model repeatedly producing
    empty/garbage responses that never self-correct, filling trajectories
    with thousands of useless tokens.
    """

    def __init__(self, *args, max_consecutive_format_errors: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_consecutive_format_errors = max_consecutive_format_errors
        self._consecutive_format_errors = 0

    def run(self, task: str = "", messages: list[dict] | None = None, **kwargs) -> dict:
        """Run the agent loop, optionally resuming an existing message history."""
        self.extra_template_vars |= {"task": task, **kwargs}
        if messages is None:
            self.messages = []
            self.add_messages(
                self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
                self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
            )
        else:
            self.messages = deepcopy(messages)
            self.n_calls = sum(message.get("role") == "assistant" for message in self.messages)
        while True:
            try:
                self.step()
                self._consecutive_format_errors = 0
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
                if any(
                    isinstance(msg, dict) and msg.get("extra", {}).get("interrupt_type") == "FormatError"
                    for msg in e.messages
                ):
                    self._consecutive_format_errors += 1
                    if self._consecutive_format_errors >= self.max_consecutive_format_errors:
                        logger.warning(
                            "Terminating trajectory: %d consecutive format errors",
                            self._consecutive_format_errors,
                        )
                        self.add_messages({
                            "role": "exit",
                            "content": "FormatErrorLimitExceeded",
                            "extra": {"exit_status": "FormatErrorLimitExceeded", "submission": ""},
                        })
                else:
                    self._consecutive_format_errors = 0
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions and add a submission reminder near the step limit."""
        outputs = [self.env.execute(action) for action in message.get("extra", {}).get("actions", [])]
        remaining = self.config.step_limit - self.n_calls
        if outputs and 1 <= remaining <= 3:
            outputs[-1]["output"] += (
                f"\n\n[SYSTEM WARNING: Only {remaining} step(s) remaining before forced termination. "
                "You MUST submit your patch NOW using separate commands:\n"
                "  1) git diff HEAD -- path/to/changed/file.py > patch.txt\n"
                "  2) cat patch.txt   (verify)\n"
                "  3) echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt]"
            )
        return self.add_messages(
            *self.model.format_observation_messages(message, outputs, self.get_template_vars())
        )


@register_agent("mini-swe")
class SWEAgent(BaseAgent):
    """SWE-bench agent: wraps mini-swe-agent DefaultAgent with a sandbox.

    Uses mini-swe-agent's bundled swebench.yaml config verbatim — system_template,
    instance_template (with full 7-step workflow), and observation_template — so the
    model receives correct submission instructions without any manual overrides.

    Common AgentFlow parameters come from BaseAgent. SWE-specific config includes
    step_limit, request_timeout, and max_consecutive_format_errors.
    """

    def __init__(
        self,
        router_url: str = "http://localhost:8000",
        sandbox_env_client: Any = None,
        reward_fn: Any = None,
        step_limit: int = 100,
        completion_params: dict = None,
        max_response_len_per_trajectory: int = 0,
        temperature: float = 0.0,
        request_timeout: int = 300,
        max_consecutive_format_errors: int = 3,
        **_ignored: Any,
    ) -> None:
        super().__init__(
            router_url,
            completion_params=completion_params,
            max_response_len_per_trajectory=max_response_len_per_trajectory,
            temperature=temperature,
            **_ignored,
        )
        self.sandbox_env_client = sandbox_env_client
        self.reward_fn = reward_fn
        self.step_limit = step_limit
        self.request_timeout = request_timeout
        self.max_consecutive_format_errors = max_consecutive_format_errors
        self._sandbox_created = False
        logger.info(
            "SWEAgent initialized: router_url=%s step_limit=%d "
            "max_response_len_per_trajectory=%d max_consecutive_format_errors=%d",
            router_url,
            step_limit,
            self.max_response_len_per_trajectory,
            max_consecutive_format_errors,
        )

    async def run_trajectory(self, trajectory: Any) -> Reward:
        """Run one SWE-bench trajectory inside a sandbox."""
        from minisweagent.models import get_model

        if not isinstance(trajectory, dict):
            raise TypeError(f"SWEAgent expects a dict trajectory, got {type(trajectory).__name__}")

        metadata = trajectory.get("metadata") or {}
        prompt = trajectory.get("prompt", "")

        # An assistant turn is the authoritative signal that AgentFlow is
        # resuming an existing conversation rather than starting from a
        # dataset-provided [system, user] prompt.
        resume_messages = (
            deepcopy(prompt)
            if isinstance(prompt, list)
            and any(
                isinstance(message, dict) and message.get("role") == "assistant"
                for message in prompt
            )
            else None
        )
        if isinstance(prompt, str):
            problem_statement = prompt
        elif isinstance(prompt, list):
            problem_statement = next(
                (m.get("content", "") for m in prompt if isinstance(m, dict) and m.get("role") == "user"),
                str(prompt),
            )
        else:
            problem_statement = str(prompt)

        image = str(metadata.get("docker_image", ""))
        cwd = str(metadata.get("repo_path", "/testbed"))
        is_r2e = bool(metadata.get("expected_output_json"))

        if self.sandbox_env_client is None:
            raise ValueError("sandbox_env_client is required; set agentflow.sandbox in config.")
        if not image:
            raise ValueError("metadata['docker_image'] is missing.")

        # Build agent config from swebench.yaml — only override step_limit.
        # Crucially, instance_template is left intact so DefaultAgent renders
        # {{task}} → problem_statement via Jinja2, preserving the full workflow.
        agent_cfg = deepcopy(_SWEBENCH_CONFIG.get("agent") or {})
        agent_cfg["step_limit"] = self.step_limit

        # Build model config pointing to the coda SGLang router.
        model_cfg = deepcopy(_SWEBENCH_CONFIG.get("model") or {})
        model_cfg.update({
            "model_name": "openai/default",
            "model_class": "coda.agentflow.agent.swe.mini_swe_agent.CodaLitellmModel",
            "cost_tracking": "ignore_errors",
            "model_kwargs": {
                **(model_cfg.get("model_kwargs") or {}),
                "api_base": f"{self.router_url}/v1",
                "api_key": "not-needed",
                "max_tokens": self.max_response_len_per_trajectory,
                "temperature": self.temperature,
                "timeout": self.request_timeout,
                "drop_params": True,
                "num_retries": 0,  # disable litellm/OpenAI SDK retry to prevent
                # history prefix mismatch on network retry
            },
        })
        os.environ.setdefault(
            "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "1"
        )  # disable tenacity retry for same reason as num_retries=0

        if getattr(self.sandbox_env_client, "sandbox_id", None):
            # Pooled sandbox from a partial-rollout abort: reuse it so tool
            # state from earlier turns is preserved.
            logger.info(
                "Reusing existing sandbox pod %s", self.sandbox_env_client.sandbox_id
            )
        else:
            logger.info("Creating sandbox pod (image=%s)", image)
            await asyncio.to_thread(self.sandbox_env_client.create, image=image)
        self._sandbox_created = True
        env = CodaSandboxEnvironment(
            self.sandbox_env_client,
            cwd=cwd,
            use_r2e_environment=is_r2e,
        )

        model = get_model(config=model_cfg)
        agent = FormatErrorLimitedAgent(
            model, env,
            max_consecutive_format_errors=self.max_consecutive_format_errors,
            **agent_cfg,
        )
        try:
            result = await asyncio.to_thread(
                agent.run,
                task=problem_statement,
                messages=resume_messages,
            )
        except Exception as e:
            if _is_context_length_exceeded_error(e):
                logger.warning(
                    "SWEAgent: token budget exhausted: %s",
                    getattr(getattr(e, "response", None), "text", str(e)),
                )
                result = {"exit_status": CONTEXT_LENGTH_EXCEEDED, "submission": ""}
            else:
                logger.warning("SWEAgent: agent.run raised %s: %s", type(e).__name__, e)
                raise
        messages: list[dict] = agent.messages
        exit_status = result.get("exit_status", "unknown") if isinstance(result, dict) else str(result)
        n_steps = sum(1 for m in messages if m.get("role") == "assistant")
        logger.info(
            "mini-swe-agent done: instance=%s exit=%s steps=%d",
            metadata.get("instance_id", "?"), exit_status, n_steps,
        )
        reward_meta = {**metadata, "sandbox": self.sandbox_env_client}
        return (
            await asyncio.to_thread(
                self.reward_fn, messages, trajectory.get("label", ""), metadata=reward_meta
            )
            if self.reward_fn is not None
            else Reward(final_reward=0.0)
        )

    async def clear(self) -> None:
        """Release agent-local resources.

        Sandbox deletion is owned by AgentFlow's sandbox pool (release_sandbox),
        which keeps the pod alive across partial-rollout aborts for reuse.
        """
        self._sandbox_created = False
