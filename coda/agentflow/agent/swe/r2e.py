"""R2E testbed command environment for mini-SWE agents."""

from __future__ import annotations

import shlex

R2E_SHELL_PREFIX = (
    "source /root/.bashrc >/dev/null 2>&1 || true; "
    "export PATH=/testbed/.venv/bin:/opt/miniconda3/envs/testbed/bin:"
    "/root/.local/bin:$PATH; "
)


def with_r2e_environment(command: str) -> str:
    """Run *command* with the R2E testbed Python environment on ``PATH``."""
    return f"bash -c {shlex.quote(R2E_SHELL_PREFIX + command)}"
