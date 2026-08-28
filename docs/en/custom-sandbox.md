# Custom Sandbox Development Guide

This document explains how to add a sandbox backend to LoongSage so an agent can execute tool calls in an isolated environment. Sandbox extensions are **registry- and config-driven**: inherit the base class, implement the lifecycle interface, add a registration decorator, and reference the backend through `agentflow.sandbox.type`.

LoongSage provides `k8s` and `docker` sandboxes by default. Use them directly through configuration, and add a custom sandbox only when neither backend fits the target runtime.

LoongSage passes the sandbox client to the agent as `sandbox_env_client`. The agent creates and uses the runtime, while LoongSage performs final deletion. A resumed trajectory can inspect `sandbox_id` to reuse an existing environment.

## 1. Development Steps

The sandbox base class and built-in implementations live in the [sandbox directory](../../coda/agentflow/sandbox/). Follow these steps to add a backend:

1. Inherit from [SandboxClient](../../coda/agentflow/sandbox/base.py) and implement `create()`, `execute()`, `delete()`, and `sandbox_id`.
2. `create()` returns the sandbox ID; `execute()` returns `stdout`, `stderr`, `exit_code`, and `success`; keep `delete()` idempotent. Override `from_config()` only when config fields require extra adaptation.
3. Register the implementation with `@register_sandbox("your-type")` and place it in the [sandbox directory](../../coda/agentflow/sandbox/); LoongSage discovers it automatically.

## 2. Minimal Example

The following example shows the minimum interface for a remote runtime. `_start_runtime`, `_exec_runtime`, and `_destroy_runtime` stand for calls to the target runtime SDK:

```python
# coda/agentflow/sandbox/remote_sandbox.py
from typing import Any

from coda.agentflow.sandbox import register_sandbox
from coda.agentflow.sandbox.base import SandboxClient


@register_sandbox("remote")
class RemoteSandboxClient(SandboxClient):
    def __init__(
        self,
        working_dir: str = "/workspace",
        command_exec_timeout_seconds: int = 600,
        **kwargs: Any,
    ) -> None:
        self.working_dir = working_dir
        self.timeout = command_exec_timeout_seconds
        self._sandbox_id: str | None = None

    @property
    def sandbox_id(self) -> str | None:
        return self._sandbox_id

    def create(self, image: str | None = None, **kwargs: Any) -> str:
        if not image:
            raise ValueError("sandbox image is required")
        self._sandbox_id = _start_runtime(image=image)
        return self._sandbox_id

    def execute(
        self,
        command: str,
        workdir: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self._sandbox_id:
            raise RuntimeError("sandbox has not been created")
        result = _exec_runtime(
            sandbox_id=self._sandbox_id,
            command=command,
            workdir=workdir or self.working_dir,
            timeout=self.timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "success": result.exit_code == 0,
        }

    def delete(self, **kwargs: Any) -> None:
        if not self._sandbox_id:
            return
        _destroy_runtime(self._sandbox_id)  # “not found” should count as success
        self._sandbox_id = None
```

See [docker_sandbox.py](../../coda/agentflow/sandbox/docker_sandbox.py) for a local container backend and [k8s_sandbox.py](../../coda/agentflow/sandbox/k8s_sandbox.py) for remote K8s execution.

## 3. Config Enablement

Once registered, select the backend by name under `agentflow.sandbox`. Every field except `type` is passed to `from_config()`:

```yaml
agentflow:
  sandbox:
    type: remote                         # ← the @register_sandbox registered name
    working_dir: /workspace              # default working directory for tool calls
    command_exec_timeout_seconds: 600    # timeout for one tool call, in seconds
```

Set `type: none` (or leave it empty) to disable sandboxes entirely; the agent then does not receive `sandbox_env_client`. If an agent requires a sandbox, check this explicitly at the beginning of `run_trajectory()` and raise a clear error.
