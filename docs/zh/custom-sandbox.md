# 自定义 Sandbox 开发指南

本文说明如何在 LoongSage 中新增一个 sandbox 后端，供 agent 在隔离环境中执行工具调用。sandbox 扩展是 **“注册表 + 配置驱动”**：继承基类、实现无状态后端接口、添加注册装饰器，即可通过 `data_source.sandbox.type` 或 `data_sources[i].sandbox.type` 选择后端。

LoongSage 默认提供 `k8s` 和 `docker` 两种 sandbox，可直接通过配置使用；只有这两种后端无法满足运行环境要求时，才需要自定义 sandbox。

`SandboxClient` 不保存受管实例状态：`create()` 返回 ID，之后的每次操作都显式接收该 ID。

## 1. 开发步骤

sandbox 基类和内置实现位于 [sandbox 目录](../../coda/agentflow/sandbox/)。新增后端时，按以下步骤操作：

1. 继承 [SandboxClient](../../coda/agentflow/sandbox/base.py)，实现 `create()`、`execute(sandbox_id, command, **kwargs)` 和 `delete(sandbox_id, **kwargs)`。
2. `create()` 返回后端 ID；`execute()` 返回 `stdout`、`stderr`、`exit_code` 和 `success`；`delete()` 销毁显式指定的实例，并应把后端资源已经不存在视为成功。不要在 client 中保存当前 sandbox ID；一个 client 可以寻址多个实例。只有配置字段需要额外转换时才覆写 `from_config()`。
3. 使用 `@register_sandbox("your-type")` 注册，并把实现放到 [coda/custom/](../../coda/custom/) 下，LoongSage 会自动发现，详见[自定义扩展](./custom-extensions.md)。

## 2. 最小示例

以下示例展示一个远程运行时 sandbox 的最小无状态接口。`_start_runtime`、`_exec_runtime` 和 `_destroy_runtime` 代表目标运行时的 SDK 调用：

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
        # “资源不存在”应按成功处理，保证清理幂等。
        _destroy_runtime(sandbox_id)
```

本地容器可参考 [docker_sandbox.py](../../coda/agentflow/sandbox/docker_sandbox.py)，K8s 远端执行可参考 [k8s_sandbox.py](../../coda/agentflow/sandbox/k8s_sandbox.py)。

## 3. 配置启用

注册后，在数据源的 `sandbox` 配置块中按注册名选择后端。除 `type` 外的字段会交给 `from_config()`：

```yaml
data_source:
  sandbox:
    type: remote                         # ← @register_sandbox 的注册名
    working_dir: /workspace              # 工具调用的默认工作目录
    command_exec_timeout_seconds: 600    # 单次工具调用的超时时间，单位为秒
```

多数据源时，每个 `data_sources[i].sandbox` 可独立配置。设置 `type: none`（或留空）会为该数据源禁用 sandbox。sandbox 镜像仍按 trajectory 选择，通常来自 `metadata["docker_image"]`。
