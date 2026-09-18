"""Docker-backed sandbox client implementation."""

from __future__ import annotations

import logging
import shlex
import subprocess
from typing import Any

from coda.agentflow.sandbox.base import SandboxClient
from coda.agentflow.sandbox import register_sandbox

logger = logging.getLogger(__name__)


@register_sandbox("docker")
class DockerSandboxClient(SandboxClient):
    """
    SandboxClient backed by a local Docker container.

    Usage::

        client = DockerSandboxClient(
            image="example.com/path/sweb.eval.x86_64.xxx",
            working_dir="/testbed",
            timeout=120,
        )
        sandbox_id = client.create()
        try:
            result = client.execute(sandbox_id, "ls /testbed")
            print(result["stdout"])
        finally:
            client.delete(sandbox_id)
    """

    def __init__(
        self,
        image: str | None = None,
        working_dir: str = "/testbed",
        timeout: int = 120,
    ) -> None:
        """
        Initialize DockerSandboxClient config (does not start container yet).

        Args:
            image: Docker image name or tag.
            working_dir: Default working directory inside the container.
            timeout: Default command timeout in seconds.
        """
        self.image = image or None
        self.working_dir = working_dir
        self.timeout = timeout

    # ------------------------------------------------------------------
    # SandboxClient interface
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        sandbox_config: dict[str, Any],
        **kwargs: Any,
    ) -> "DockerSandboxClient | None":
        """Create a DockerSandboxClient from environment-level config."""
        return cls(
            image=sandbox_config.get("image"),
            working_dir=sandbox_config.get("working_dir", "/testbed"),
            timeout=sandbox_config.get("command_exec_timeout_seconds", sandbox_config.get("timeout", 120)),
        )

    def create(self, **kwargs) -> str:
        """
        Pull image if needed, start a detached container.

        Returns:
            Container ID string.
        """
        image = kwargs.get("image") or self.image
        if not image:
            raise ValueError("DockerSandboxClient.create() requires an image")

        logger.info(f"Starting Docker container: {image}")
        result = subprocess.run(
            ["docker", "run", "--rm", "-d", image, "sleep", "infinity"],
            capture_output=True,
            text=True,
            check=True,
        )
        sandbox_id = result.stdout.strip()
        logger.info(f"Container started: {sandbox_id[:12]}")

        # Create /root/.venv symlink so tool scripts with #!/root/.venv/bin/python
        # shebang can resolve to the testbed conda environment (which has chardet
        # and other tool dependencies pre-installed in the SWE-bench image).
        # This mirrors the setup step that baidubce performs after pod creation.
        venv_result = subprocess.run(
            [
                "docker", "exec", sandbox_id,
                "ln", "-sf", "/opt/miniconda3/envs/testbed", "/root/.venv",
            ],
            capture_output=True,
        )
        if venv_result.returncode != 0:
            logger.warning(
                "Could not create /root/.venv symlink in container %s: %s",
                sandbox_id[:12],
                venv_result.stderr.decode(errors="replace").strip(),
            )
        else:
            logger.debug(f"Created /root/.venv symlink in container {sandbox_id[:12]}")

        return sandbox_id

    def execute(self, sandbox_id: str, command: str, workdir: str | None = None, **kwargs) -> dict[str, Any]:
        """
        Run a bash command inside the container.

        Args:
            command: Shell command string.
            workdir: Working directory override (defaults to self.working_dir).

        Returns:
            Dict with stdout, stderr, exit_code, success keys.
        """
        cwd = workdir or self.working_dir
        full_cmd = f"cd {shlex.quote(cwd)} && timeout {self.timeout} {command}"
        logger.debug(f"[{sandbox_id[:12]}] {full_cmd}")

        try:
            result = subprocess.run(
                ["docker", "exec", sandbox_id, "bash", "-c", full_cmd],
                capture_output=True,
                text=True,
                timeout=self.timeout + 10,
            )
        except subprocess.TimeoutExpired as e:
            logger.error("Command timed out after %ds: %s", self.timeout + 10, full_cmd)
            return {
                "stdout": e.stdout or "",
                "stderr": e.stderr or "",
                "exit_code": -1,
                "success": False,
            }
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "success": result.returncode == 0,
        }

    def delete(self, sandbox_id: str, **kwargs) -> None:
        """Stop and remove the identified container."""
        logger.info(f"Removing container {sandbox_id[:12]}")
        subprocess.run(["docker", "rm", "-f", sandbox_id], capture_output=True)
