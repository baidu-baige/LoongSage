"""OpenCode black-box agent running inside a Coda sandbox."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shlex
from pathlib import PurePosixPath
from typing import Any, Callable
from urllib.parse import urlparse

from coda.agentflow.agent import BaseAgent, register_agent
from coda.agentflow.utils import require_docker_image
from coda.reward.reward import Reward

logger = logging.getLogger(__name__)

_MODEL_ID = "default"
_WIRE_PROVIDERS = {
    "chat_completions": ("coda", "@ai-sdk/openai-compatible"),
    "responses": ("openai", "@ai-sdk/openai"),
    "anthropic_messages": ("anthropic", "@ai-sdk/anthropic"),
}
_CONFIG_PATH = "/root/.config/opencode/opencode.json"
_PLUGIN_PATH = "/root/.config/opencode/coda-request-kind.js"
_OPENCODE_PATH = "/root/.opencode/bin/opencode"
_LOG_PATH = "/tmp/coda-opencode.log"

_REQUEST_KIND_PLUGIN = r'''
export const CodaRequestKind = async ({ client, directory }) => ({
  "chat.headers": async (input, output) => {
    const session = await client.session.get({
      path: { id: input.sessionID },
      query: { directory },
      throwOnError: true,
    }).catch(() => undefined)

    if (session?.data?.parentID) {
      output.headers.request_kind = "collab_spawn"
      return
    }

    if (input.agent === "compaction") {
      output.headers.request_kind = "compaction"
      return
    }

    const message = await client.session.message({
      path: { id: input.message.sessionID, messageID: input.message.id },
      query: { directory },
      throwOnError: true,
    }).catch(() => undefined)
    if (message?.data?.parts?.some((part) =>
      part.type === "compaction" ||
      (part.type === "text" && part.synthetic && part.metadata?.compaction_continue === true)
    )) {
      output.headers.request_kind = "compaction"
    }
  },
})
'''.strip()


def _config(
    router_url: str,
    context_length: int,
    max_output_tokens: int,
    wire_api: str,
) -> dict[str, Any]:
    provider_id, provider_npm = _WIRE_PROVIDERS[wire_api]
    agent = {}
    model: dict[str, Any] = {
        "name": "Coda rollout model",
        "temperature": True,
        "reasoning": True,
        "tool_call": True,
        "interleaved": "reasoning_content",
        "attachment": False,
        "modalities": {
            "input": ["text"],
            "output": ["text"],
        },
    }
    if context_length > 0 and max_output_tokens > 0:
        model["limit"] = {
            "context": context_length,
            "output": max_output_tokens,
        }

    return {
        "$schema": "https://opencode.ai/config.json",
        "plugin": [f"file://{_PLUGIN_PATH}"],
        "provider": {
            provider_id: {
                "npm": provider_npm,
                "name": "Coda Router",
                "options": {
                    "baseURL": f"{router_url.rstrip('/')}/v1",
                    "apiKey": "not-needed",
                },
                "models": {_MODEL_ID: model},
            }
        },
        "model": f"{provider_id}/{_MODEL_ID}",
        "small_model": f"{provider_id}/{_MODEL_ID}",
        "agent": {"build": agent, "compaction": agent},
        "permission": {
            "*": "allow",
            "read": {
                "*": "allow",
                "*.png": "deny",
                "*.jpg": "deny",
                "*.jpeg": "deny",
                "*.gif": "deny",
                "*.webp": "deny",
                "*.pdf": "deny",
            },
            "question": "deny",
            "doom_loop": "deny",
        },
    }


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


@register_agent("opencode")
class OpenCodeAgent(BaseAgent):
    """Run the OpenCode CLI against Coda's OpenAI-compatible Router."""

    def __init__(
        self,
        router_url: str,
        context_length: int,
        max_response_len_per_trajectory: int = 0,
        max_output_tokens: int = 32000,
        wire_api: str = "chat_completions",
        **kwargs: Any,
    ) -> None:
        assert context_length > 0, "agent.context_length must be greater than 0"
        super().__init__(
            router_url,
            max_response_len_per_trajectory=max_response_len_per_trajectory,
            **kwargs,
        )
        if self.sandbox_client is None:
            raise ValueError("opencode agent requires data_source.sandbox")
        self.context_length = context_length
        self.max_output_tokens = max_output_tokens
        self.wire_api = wire_api
        # True only while the `opencode run` subprocess is in flight; reset on a
        # clean return (NOT in a finally), so cancellation leaves it True and
        # close() knows a lingering process must be killed.
        self._opencode_running = False

    async def run_trajectory(self, trajectory: dict[str, Any]) -> Reward:
        """Run one OpenCode SWE trajectory inside the configured sandbox."""
        metadata = trajectory.get("metadata") or {}
        repo_path = str(metadata.get("repo_path") or "")
        if not repo_path:
            raise ValueError("metadata['repo_path'] is missing")
        if self.sandbox_id is None:
            self.sandbox_id = await asyncio.to_thread(
                self.sandbox_client.create, image=require_docker_image(metadata)
            )
        execute = lambda command, **kwargs: self.sandbox_client.execute(self.sandbox_id, command, **kwargs)

        # A pod can pass readiness yet fail to reach the Router. Probe with a
        # raw TCP connect (bypasses http_proxy, like OpenCode's client) so the
        # failure retries with a fresh pod instead of producing an empty trajectory.
        parsed = urlparse(self.router_url)
        router_host, router_port = parsed.hostname, parsed.port
        if router_host and router_port:
            probe_cmd = (
                "for i in 1 2 3; do "
                f"timeout 5 bash -c '</dev/null >/dev/tcp/{router_host}/{router_port}' "
                "2>/dev/null && exit 0; sleep 1; done; exit 1"
            )
            probe = await asyncio.to_thread(execute, probe_cmd, workdir=repo_path)
            if probe.get("exit_code", -1) != 0:
                raise RuntimeError(
                    f"Sandbox cannot reach Router {router_host}:{router_port} "
                    f"(instance={metadata.get('instance_id', '?')}); "
                    "likely a per-pod network/CNI issue — will retry with a new pod"
                )

        check = await asyncio.to_thread(
            execute,
            f"test -x {shlex.quote(_OPENCODE_PATH)} && test -d .git",
            workdir=repo_path,
        )
        if check.get("exit_code", -1) != 0:
            output = (check.get("stdout", "") + check.get("stderr", "")).strip()
            raise RuntimeError(f"OpenCode or repository is missing in sandbox: {output[:1000]}")

        await asyncio.to_thread(_write_file, execute, _PLUGIN_PATH, _REQUEST_KIND_PLUGIN)
        await asyncio.to_thread(
            _write_file,
            execute,
            _CONFIG_PATH,
            json.dumps(
                _config(
                    self.router_url,
                    self.context_length,
                    self.max_output_tokens,
                    self.wire_api,
                ),
                indent=2,
            ),
        )

        instruction = _instruction(trajectory.get("prompt", ""))
        provider_id = _WIRE_PROVIDERS[self.wire_api][0]
        command = (
            "OPENCODE_DISABLE_MODELS_FETCH=1 "
            "OPENCODE_DISABLE_AUTOUPDATE=1 "
            "OPENCODE_DISABLE_LSP_DOWNLOAD=1 "
            "OPENAI_API_KEY=not-needed "
            f"OPENCODE_CONFIG={shlex.quote(_CONFIG_PATH)} "
            f"{shlex.quote(_OPENCODE_PATH)} "
            f"--model={provider_id}/{_MODEL_ID} run "
            "--title=coda --dangerously-skip-permissions -- "
            f"{shlex.quote(instruction)} </dev/null >{shlex.quote(_LOG_PATH)} 2>&1"
        )
        # Mark the process as in flight so clear() can kill it on a partial
        # abort. Reset ONLY on a clean return (below), never in a finally, so a
        # CancelledError leaves the flag True and clear() knows to reap it.
        self._opencode_running = True
        result = await asyncio.to_thread(execute, command, workdir=repo_path)
        self._opencode_running = False
        if result.get("exit_code", -1) != 0:
            tail = await asyncio.to_thread(execute, f"tail -n 100 {shlex.quote(_LOG_PATH)}", workdir=repo_path)
            output = (tail.get("stdout", "") + tail.get("stderr", ""))[-5000:]
            logger.warning(
                "OpenCode exited with code %s for %s:\n%s",
                result.get("exit_code"),
                metadata.get("instance_id", "?"),
                output,
            )
            raise RuntimeError(
                f"OpenCode exited with code {result.get('exit_code')} for "
                f"{metadata.get('instance_id', '?')}: {output}"
            )

        return await asyncio.to_thread(
            self.reward_fn,
            [],
            trajectory.get("label"),
            {**metadata, "sandbox_client": self.sandbox_client, "sandbox_id": self.sandbox_id},
        )

    async def close(self) -> None:
        """Kill a lingering OpenCode process and close shared resources."""
        await super().close()
        if not self._opencode_running:
            return
        kill_cmd = (
            f"pkill -INT -f {shlex.quote(_OPENCODE_PATH)} 2>/dev/null || true; "
            "sleep 1; "
            f"pkill -KILL -f {shlex.quote(_OPENCODE_PATH)} 2>/dev/null || true"
        )
        try:
            await asyncio.to_thread(self.sandbox_client.execute, self.sandbox_id, kill_cmd)
        except Exception as exc:
            logger.warning("[opencode] failed to kill lingering process on cleanup: %s", exc)
        finally:
            self._opencode_running = False
