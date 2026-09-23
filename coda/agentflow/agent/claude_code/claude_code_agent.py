"""Claude Code black-box agent running inside a Coda sandbox."""

from __future__ import annotations

import asyncio
import base64
import logging
import shlex
from pathlib import PurePosixPath
from typing import Any, Callable
from urllib.parse import urlparse

from coda.agentflow.agent import BaseAgent, register_agent
from coda.agentflow.utils import CONTEXT_LENGTH_EXCEEDED, require_docker_image
from coda.reward.reward import Reward

logger = logging.getLogger(__name__)

_CLAUDE_PATH = "claude"
_LOG_PATH = "/tmp/coda-claude-code.log"


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


# Pure-shell PreToolUse hook: rewrite an unbounded / oversized Read into a
# bounded one so a single tool_result stays around 15k characters, which keeps
# a big-file read from blowing up the context window and triggering the CLI's
# auto-compaction (a frequent cause of exit-code-1 crashes). Read output is
# `cat -n` formatted (~1.2x the raw file bytes), so we cap the raw byte budget
# a bit under the target and let awk translate it into a line `limit`.
_READ_LIMIT_HOOK_SH = r'''#!/bin/sh
BUDGET_RAW=12500
FALLBACK_LINES=250
input=$(cat)
printf '%s' "$input" | grep -q '"tool_name"[[:space:]]*:[[:space:]]*"Read"' || exit 0
file=$(printf '%s' "$input" | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
[ -n "$file" ] || exit 0
limit=""
if [ -f "$file" ]; then
  limit=$(awk -v max="$BUDGET_RAW" '{c+=length($0)+1; if(c>max){print (NR-1>0?NR-1:1); found=1; exit}} END{if(!found) print NR}' "$file" 2>/dev/null)
fi
[ -n "$limit" ] || limit=$FALLBACK_LINES
printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","permissionDecisionReason":"auto-bounded Read","updatedInput":{"file_path":"%s","limit":%s}}}\n' "$file" "$limit"
'''

_READ_HOOK_PATH = "/tmp/coda_read_limit_hook.sh"
_SETTINGS_PATH = "/tmp/coda_claude_settings.json"
_SETTINGS_JSON = (
    '{"hooks":{"PreToolUse":[{"matcher":"Read",'
    '"hooks":[{"type":"command","command":"sh ' + _READ_HOOK_PATH + '"}]}]}}'
)


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
        raise RuntimeError(
            f"Sandbox cannot reach Router {parsed.hostname}:{parsed.port}"
        )


@register_agent("claude_code")
class ClaudeCodeAgent(BaseAgent):
    """Run Claude Code against Coda's Anthropic-compatible Router."""

    def __init__(
        self,
        router_url: str,
        max_response_len_per_trajectory: int = 0,
        context_length: int = 0,
        max_output_tokens: int = 32000,
        auto_compact_ratio: float = 0.8,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            router_url,
            max_response_len_per_trajectory=max_response_len_per_trajectory,
            **kwargs,
        )
        if self.sandbox_client is None:
            raise ValueError("claude_code agent requires data_source.sandbox")
        self.context_length = context_length
        self.max_output_tokens = max_output_tokens
        self.auto_compact_ratio = auto_compact_ratio
        self._claude_running = False

    async def run_trajectory(self, trajectory: dict[str, Any]) -> Reward:
        """Run one Claude Code trajectory inside the configured sandbox."""
        metadata = trajectory.get("metadata") or {}
        repo_path = str(metadata.get("repo_path") or "/testbed")

        if self.sandbox_id is None:
            self.sandbox_id = await asyncio.to_thread(
                self.sandbox_client.create, image=require_docker_image(metadata)
            )
        execute = lambda command, **kwargs: self.sandbox_client.execute(  # noqa: E731
            self.sandbox_id, command, **kwargs
        )

        await _probe_router(execute, self.router_url, repo_path)

        check = await asyncio.to_thread(
            execute,
            f"command -v {shlex.quote(_CLAUDE_PATH)} && test -d .git",
            workdir=repo_path,
        )
        if check.get("exit_code", -1) != 0:
            output = (check.get("stdout", "") + check.get("stderr", "")).strip()
            raise RuntimeError(f"Claude Code or repository is missing in sandbox: {output[:1000]}")

        await asyncio.to_thread(_write_file, execute, _READ_HOOK_PATH, _READ_LIMIT_HOOK_SH)
        await asyncio.to_thread(_write_file, execute, _SETTINGS_PATH, _SETTINGS_JSON)

        instruction = _instruction(trajectory.get("prompt", ""))
        env_vars = {
            "IS_SANDBOX": "1",
            "ANTHROPIC_BASE_URL": self.router_url.rstrip("/"),
            "ANTHROPIC_AUTH_TOKEN": "not-needed",
            "ANTHROPIC_MODEL": "coda-actor",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "coda-actor",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "coda-actor",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "coda-actor",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
            "CLAUDE_CODE_FORK_SUBAGENT": "1",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_MAX_RETRIES": "0",
            "DISABLE_AUTOUPDATER": "1",
            "API_TIMEOUT_MS": "1800000",
        }
        if self.max_output_tokens > 0:
            # Keep the CLI's per-request max_tokens aligned with the rollout
            # budget so the router does not have to truncate-and-400 each turn.
            env_vars["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(self.max_output_tokens)
        if self.context_length > 0:
            compact_window = int(self.context_length * self.auto_compact_ratio)
            env_vars["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(max(1, compact_window))
        env_prefix = " ".join(
            f"{name}={shlex.quote(value)}" for name, value in env_vars.items()
        )
        command = (
            f"printf %s {shlex.quote(instruction)} | {env_prefix} "
            f"{shlex.quote(_CLAUDE_PATH)} --print --verbose "
            "--output-format stream-json "
            "--permission-mode bypassPermissions "
            f"--settings {shlex.quote(_SETTINGS_PATH)} "
            "--disallowedTools WebFetch,WebSearch "
            f">{shlex.quote(_LOG_PATH)} 2>&1"
        )
        self._claude_running = True
        result = await asyncio.to_thread(execute, command, workdir=repo_path)
        if result.get("exit_code", -1) != 0:
            diagnostics = await asyncio.to_thread(
                execute,
                (
                    "ps -eo pid,ppid,pgid,sid,stat,etime,wchan:32,comm,args "
                    "| grep -E '(claude|node|pytest|python|bash)' | grep -v grep; "
                    "printf '\n--- coda claude log ---\n'; "
                    f"tail -n 100 {shlex.quote(_LOG_PATH)}"
                ),
                workdir=repo_path,
            )
            output = (
                diagnostics.get("stdout", "") + diagnostics.get("stderr", "")
            )[-8000:]
            log_output = output.rsplit("--- coda claude log ---", 1)[-1]
            context_rejected = result.get("exit_code") == 1 and any(
                CONTEXT_LENGTH_EXCEEDED in line.lower()
                or (
                    "context" in line.lower()
                    and ("exceed" in line.lower() or "exhaust" in line.lower())
                )
                for line in log_output.splitlines()
            )
            if context_rejected:
                logger.warning(
                    "Claude Code exited with code %s after a context-length "
                    "rejection in sandbox %s; returning valid reward 0",
                    result.get("exit_code"),
                    self.sandbox_id,
                )
                self._claude_running = False
                return Reward(final_reward=0.0, is_valid=True, is_correct=False)
            logger.warning(
                "Claude Code exited with code %s in sandbox %s:\n%s",
                result.get("exit_code"),
                self.sandbox_id,
                output,
            )
            raise RuntimeError(
                f"Claude Code exited with code {result.get('exit_code')}: {output}"
            )
        self._claude_running = False

        return await asyncio.to_thread(
            self.reward_fn,
            [],
            trajectory.get("label"),
            {**metadata, "sandbox_client": self.sandbox_client, "sandbox_id": self.sandbox_id},
        )

    async def close(self) -> None:
        """Interrupt a still-running Claude Code process, then close shared resources."""
        await super().close()
        if not self._claude_running:
            return
        if self.sandbox_id is None:
            return
        command = (
            "pkill -INT -f '(^|/)claude( |$)' 2>/dev/null || true; sleep 1; "
            "pkill -KILL -f '(^|/)claude( |$)' 2>/dev/null || true"
        )
        try:
            await asyncio.to_thread(self.sandbox_client.execute, self.sandbox_id, command)
        except Exception as exc:
            logger.warning("[claude_code] failed to kill lingering process on cleanup: %s", exc)
        finally:
            self._claude_running = False
