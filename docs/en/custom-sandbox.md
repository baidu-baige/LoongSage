# Custom Sandbox Development Guide

This document explains how to add a sandbox backend to LoongSage so an agent can execute tool calls in an isolated environment. Sandbox extensions are **registry- and config-driven**: inherit the base class, implement its stateless backend interface, add a registration decorator, and select the backend through `data_source.sandbox.type` or `data_sources[i].sandbox.type`.

LoongSage provides `k8s` and `docker` sandboxes by default. Use them directly through configuration, and add a custom sandbox only when neither backend fits the target runtime.

`SandboxClient` has no managed-instance state: `create()` returns an ID, and every later operation receives that ID explicitly.

## 1. Development Steps

The sandbox base class and built-in implementations live in the [sandbox directory](../../coda/agentflow/sandbox/). Follow these steps to add a backend:

1. Inherit from [SandboxClient](../../coda/agentflow/sandbox/base.py) and implement `create()`, `execute(sandbox_id, command, **kwargs)`, and `delete(sandbox_id, **kwargs)`.
2. `create()` returns the backend ID. `execute()` returns `stdout`, `stderr`, `exit_code`, and `success`. `delete()` destroys the explicitly identified instance and should treat an already-missing backend resource as success. Do not store the current sandbox ID on the client; one client may address multiple instances. Override `from_config()` only when configuration fields require extra adaptation.
3. Register the implementation with `@register_sandbox("your-type")` and place it in [coda/custom/](../../coda/custom/); LoongSage discovers it automatically, see [Custom Extensions](./custom-extensions.md).

## 2. Minimal Example

The following example shows the minimum stateless interface for a remote runtime. `_start_runtime`, `_exec_runtime`, and `_destroy_runtime` stand for calls to the target runtime SDK:

```python
# coda/custom/remote_sandbox.py
from typing import Any

from coda.agentflow.sandbox import register_sandbox
from coda.agentflow.sandbox.base import SandboxClient


@register_sandbox("remote")
class RemoteSandboxClient(SandboxClient):
    def __init__(self, working_dir: str = "/workspace", command_exec_timeout_seconds: int = 600, **kwargs: Any) -> None:
        self.working_dir = working_dir
        self.timeout = command_exec_timeout_seconds

    def create(self, image: str | None = None, **kwargs: Any) -> str:
        if not image:
            raise ValueError("sandbox image is required")
        return _start_runtime(image=image)

    def execute(self, sandbox_id: str, command: str, workdir: str | None = None, **kwargs: Any) -> dict[str, Any]:
        result = _exec_runtime(
            sandbox_id=sandbox_id,
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

    def delete(self, sandbox_id: str, **kwargs: Any) -> None:
        # Treat a "not found" backend as success so cleanup stays idempotent.
        _destroy_runtime(sandbox_id)
```

See [docker_sandbox.py](../../coda/agentflow/sandbox/docker_sandbox.py) for a local container backend and [k8s_sandbox.py](../../coda/agentflow/sandbox/k8s_sandbox.py) for remote K8s execution.

## 3. Config Enablement

Once registered, select the backend by name under the data source's `sandbox` block. Every field except `type` is passed to `from_config()`:

```yaml
data_source:
  sandbox:
    type: remote                         # ← the @register_sandbox registered name
    working_dir: /workspace              # default working directory for tool calls
    command_exec_timeout_seconds: 600    # timeout for one tool call, in seconds
```

With multiple data sources, configure each `data_sources[i].sandbox` independently. Set `type: none` (or leave it empty) to disable the sandbox for that data source. Sandbox image selection remains per trajectory, typically via `metadata["docker_image"]`.
