"""Codex CLI black-box agent running inside a Coda sandbox."""

from __future__ import annotations

import asyncio
import base64
import logging
import shlex
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable
from urllib.parse import urlparse

from coda.agentflow.agent import BaseAgent, register_agent
from coda.agentflow.utils import require_docker_image
from coda.reward.reward import Reward

logger = logging.getLogger(__name__)

_CONFIG_PATH = "/root/.codex/config.toml"
_MODEL_INSTRUCTIONS_PATH = "/root/.codex/coda_prompt.md"
_CODEX_PATH = "codex"
_LOG_PATH = "/tmp/coda-codex.log"
_MODEL_INSTRUCTIONS = Path(__file__).with_name("prompt.md").read_text(encoding="utf-8")

_CONFIG_TOML = '''model = "coda-actor"
model_provider = "coda"
model_context_window = {context_length}
model_auto_compact_token_limit = {auto_compact_token_limit}
model_instructions_file = "{model_instructions_path}"
web_search = "disabled"

[skills]
include_instructions = false

[features]
view_image = false
goals = false

[tools.experimental_request_user_input]
enabled = false

[model_providers.coda]
name = "Coda"
base_url = "{router_url}/v1"
env_key = "OPENAI_API_KEY"
request_max_retries = 0
stream_max_retries = 0
stream_idle_timeout_ms = 1800000
'''


def _build_config(router_url: str, context_length: int) -> str:
    return _CONFIG_TOML.format(
        router_url=router_url.rstrip("/"),
        context_length=context_length,
        auto_compact_token_limit=max(1, context_length * 9 // 10),
        model_instructions_path=_MODEL_INSTRUCTIONS_PATH,
    )


def _write_file(execute: Callable[..., dict], path: str, content: str) -> None:
    encoded = base64.b64encode(content.encode()).decode()
    parent = str(PurePosixPath(path).parent)
    command = (
        f"mkdir -p {shlex.quote(parent)} && "
        f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}"
    )
    result = execute(command, workdir="/")
    if result.get("exit_code", -1) != 0:
        output = (result.get("stdout", "") + result.get("stderr", "")).strip()
        raise RuntimeError(f"failed to write {path}: {output[:1000]}")


def _instruction(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        contents = [
            str(message.get("content", ""))
            for message in prompt
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        if contents:
            return "\n\n".join(contents)
    return str(prompt)


async def _probe_router(execute: Callable[..., dict], router_url: str, workdir: str) -> None:
    parsed = urlparse(router_url)
    if not parsed.hostname or not parsed.port:
        return
    command = (
        "for i in 1 2 3; do "
        f"timeout 5 bash -c '</dev/null >/dev/tcp/{parsed.hostname}/{parsed.port}' "
        "2>/dev/null && exit 0; sleep 1; done; exit 1"
    )
    result = await asyncio.to_thread(execute, command, workdir=workdir)
    if result.get("exit_code", -1) != 0:
        raise RuntimeError(f"Sandbox cannot reach Router {parsed.hostname}:{parsed.port}")


@register_agent("codex")
class CodexAgent(BaseAgent):
    """Run the Codex CLI against Coda's Responses-compatible Router."""

    def __init__(
        self,
        router_url: str,
        context_length: int,
        max_response_len_per_trajectory: int = 0,
        **kwargs: Any,
    ) -> None:
        assert context_length > 0, "agent.context_length must be greater than 0"
        super().__init__(
            router_url,
            max_response_len_per_trajectory=max_response_len_per_trajectory,
            **kwargs,
        )
        if self.sandbox_client is None:
            raise ValueError("codex agent requires data_source.sandbox")
        self.context_length = context_length
        self._codex_running = False

    async def run_trajectory(self, trajectory: dict[str, Any]) -> Reward:
        """Run one Codex trajectory inside the configured sandbox."""
        metadata = trajectory.get("metadata") or {}
        repo_path = str(metadata.get("repo_path") or "/testbed")
        if self.sandbox_id is None:
            self.sandbox_id = await asyncio.to_thread(
                self.sandbox_client.create, image=require_docker_image(metadata)
            )
        execute = lambda command, **kwargs: self.sandbox_client.execute(self.sandbox_id, command, **kwargs)

        await _probe_router(execute, self.router_url, repo_path)

        check = await asyncio.to_thread(
            execute,
            f"command -v {shlex.quote(_CODEX_PATH)} && test -d .git",
            workdir=repo_path,
        )
        if check.get("exit_code", -1) != 0:
            output = (check.get("stdout", "") + check.get("stderr", "")).strip()
            raise RuntimeError(f"Codex or repository is missing in sandbox: {output[:1000]}")

        context_length = self.context_length
        await asyncio.to_thread(_write_file, execute, _MODEL_INSTRUCTIONS_PATH, _MODEL_INSTRUCTIONS)
        await asyncio.to_thread(_write_file, execute, _CONFIG_PATH, _build_config(self.router_url, context_length))
        instruction = _instruction(trajectory.get("prompt", ""))
        command = (
            "OPENAI_API_KEY=not-needed "
            f"CODEX_HOME={shlex.quote(str(PurePosixPath(_CONFIG_PATH).parent))} "
            f"{shlex.quote(_CODEX_PATH)} exec --json "
            "--dangerously-bypass-approvals-and-sandbox "
            f"{shlex.quote(instruction)} </dev/null >{shlex.quote(_LOG_PATH)} 2>&1"
        )
        self._codex_running = True
        result = await asyncio.to_thread(execute, command, workdir=repo_path)
        self._codex_running = False
        if result.get("exit_code", -1) != 0:
            tail = await asyncio.to_thread(execute, f"tail -n 100 {shlex.quote(_LOG_PATH)}", workdir=repo_path)
            output = (tail.get("stdout", "") + tail.get("stderr", ""))[-5000:]
            logger.warning(
                "Codex exited with code %s:\n%s",
                result.get("exit_code"),
                output,
            )
            raise RuntimeError(
                f"Codex exited with code {result.get('exit_code')}: {output}"
            )

        return await asyncio.to_thread(
            self.reward_fn,
            [],
            trajectory.get("label"),
            {**metadata, "sandbox_client": self.sandbox_client, "sandbox_id": self.sandbox_id},
        )

    async def close(self) -> None:
        """Interrupt Codex if needed, then close shared resources."""
        await super().close()
        if not self._codex_running:
            return
        command = (
            "pkill -INT -f codex 2>/dev/null || true; sleep 1; "
            "pkill -KILL -f codex 2>/dev/null || true"
        )
        try:
            await asyncio.to_thread(self.sandbox_client.execute, self.sandbox_id, command)
        except Exception as exc:
            logger.warning("[codex] failed to kill lingering process on cleanup: %s", exc)
        finally:
            self._codex_running = False
