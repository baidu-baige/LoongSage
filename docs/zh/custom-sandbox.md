# 自定义 Sandbox 开发指南

本文说明如何在 LoongSage 中新增一个 sandbox 后端，供 agent 在隔离环境中执行工具调用。sandbox 扩展是 **“注册表 + 配置驱动”**：继承基类、实现生命周期接口、添加注册装饰器，即可通过 `agentflow.sandbox.type` 引用。

LoongSage 默认提供 `k8s` 和 `docker` 两种 sandbox，可直接通过配置使用；只有这两种后端无法满足运行环境要求时，才需要自定义 sandbox。

LoongSage 将 sandbox client 作为 `sandbox_env_client` 传给 agent。agent 负责创建和使用运行环境，LoongSage 负责最终删除；续跑时可通过 `sandbox_id` 判断是否复用已有环境。

## 1. 开发步骤

sandbox 基类和内置实现位于 [sandbox 目录](../../coda/agentflow/sandbox/)。新增后端时，按以下步骤操作：

1. 继承 [SandboxClient](../../coda/agentflow/sandbox/base.py)，实现 `create()`、`execute()`、`delete()` 和 `sandbox_id`。
2. `create()` 返回 sandbox ID；`execute()` 返回 `stdout`、`stderr`、`exit_code` 和 `success`；`delete()` 保持幂等。只有配置字段需要额外转换时才覆写 `from_config()`。
3. 使用 `@register_sandbox("your-type")` 注册，并把实现放到 [coda/custom/](../../coda/custom/) 下，LoongSage 会自动发现，详见[自定义扩展](./custom-extensions.md)。

## 2. 最小示例

以下示例展示一个远程运行时 sandbox 的最小接口。`_start_runtime`、`_exec_runtime` 和 `_destroy_runtime` 代表目标运行时的 SDK 调用：

```python
# coda/custom/remote_sandbox.py
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
        _destroy_runtime(self._sandbox_id)  # “not found” 应按成功处理
        self._sandbox_id = None
```

本地容器可参考 [docker_sandbox.py](../../coda/agentflow/sandbox/docker_sandbox.py)，K8s 远端执行可参考 [k8s_sandbox.py](../../coda/agentflow/sandbox/k8s_sandbox.py)。

## 3. 配置启用

注册后，在 `agentflow.sandbox` 中按注册名选择后端。除 `type` 外的字段会交给 `from_config()`：

```yaml
agentflow:
  sandbox:
    type: remote                         # ← @register_sandbox 的注册名
    working_dir: /workspace              # 工具调用的默认工作目录
    command_exec_timeout_seconds: 600    # 单次工具调用的超时时间，单位为秒
```

设置 `type: none`（或留空）会完全禁用 sandbox，此时 agent 不会收到 `sandbox_env_client`。如果某个 agent 强依赖 sandbox，应在 `run_trajectory()` 开始时明确检查并报错。
