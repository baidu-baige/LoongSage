"""Kubernetes-backed sandbox client implementation."""

from __future__ import annotations

import copy
import logging
import re
import shlex
import subprocess
import time
import uuid
from typing import Any

from coda.agentflow.sandbox import register_sandbox
from coda.agentflow.sandbox.base import SandboxClient
from coda.utils.path_utils import resolve_conf_path

logger = logging.getLogger(__name__)

# kubectl diagnostic noise that leaks into stderr but is unrelated to the
# command executed inside the pod (e.g. API discovery warnings).
_KUBECTL_NOISE_RE = re.compile(
    r"^E\d{4} \d{2}:\d{2}:\d{2}\.\d+ .+ memcache\.go:\d+\]",
)
_KUBECTL_REQUEST_TIMEOUT = "30s"
_FORCE_DELETE_RETRIES = 3
_FORCE_DELETE_BACKOFF_SECONDS = 2


@register_sandbox("k8s")
class K8sSandboxClient(SandboxClient):
    """
    SandboxClient backed by a Kubernetes pod.

    Creates one pod per sandbox instance using `kubectl apply`, executes
    commands via `kubectl exec`, and deletes the pod on `delete()`.
    """

    def __init__(
        self,
        namespace: str = "default",
        working_dir: str = "/rl-sandbox",
        command_exec_timeout_seconds: int = 120,
        sandbox_creation_timeout_seconds: int = 120,
        kubeconfig: str | None = None,
        pod_manifest: dict | None = None,
    ) -> None:
        """Initialize K8sSandboxClient with namespace, working dir, timeouts, and credentials.

        Args:
            pod_manifest: Pod manifest applied via `kubectl apply`. Follows standard
                Kubernetes pod manifest structure. namespace is read from
                `metadata.namespace`; the container image is injected at create() time.
                `metadata.generateName` is used as the pod name prefix
                (default "rl-sandbox"). See conf/k8s/pod_manifest.yaml for the full example.
        """
        self.namespace = namespace
        self.working_dir = working_dir
        self.command_exec_timeout_seconds = command_exec_timeout_seconds
        self.sandbox_creation_timeout_seconds = sandbox_creation_timeout_seconds
        self.kubeconfig = kubeconfig
        self.pod_manifest = pod_manifest or {}
        self._pod_name: str | None = None

    # ------------------------------------------------------------------
    # SandboxClient interface
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        sandbox_config: dict[str, Any],
        **kwargs: Any,
    ) -> "K8sSandboxClient":
        """Construct a K8sSandboxClient from sandbox config dict."""
        kubeconfig = resolve_conf_path(sandbox_config.get("kubeconfig"))
        pod_manifest_path = resolve_conf_path(sandbox_config.get("pod_manifest_path"))
        pod_manifest = None
        if pod_manifest_path:
            import yaml
            with open(pod_manifest_path) as f:
                pod_manifest = yaml.safe_load(f)

        namespace = (pod_manifest or {}).get("metadata", {}).get("namespace", "default")
        return cls(
            namespace=namespace,
            working_dir=sandbox_config.get("working_dir", "/rl-sandbox"),
            command_exec_timeout_seconds=sandbox_config.get("command_exec_timeout_seconds", 120),
            sandbox_creation_timeout_seconds=sandbox_config.get("sandbox_creation_timeout_seconds", 120),
            kubeconfig=kubeconfig,
            pod_manifest=pod_manifest,
        )

    @property
    def sandbox_id(self) -> str | None:
        """Name of the live pod, or None when not created."""
        return self._pod_name

    def create(self, image: str | None = None, **kwargs) -> str:
        """Start a pod with the given image.

        Args:
            image: Docker image to run.

        Returns:
            Pod name.
        """
        if not image:
            raise ValueError("K8sSandboxClient.create() requires an image")

        # Derive pod name from pod_manifest.metadata.generateName if set.
        generate_name = (
            self.pod_manifest.get("metadata", {}).get("generateName", "rl-sandbox-")
        )
        prefix = generate_name.rstrip("-")
        pod_name = f"{prefix}-{uuid.uuid4().hex[:10]}"
        logger.info("Create pod %s from image %s", pod_name, image)

        # Build a complete pod manifest from pod_manifest so that all fields
        # (command, env, imagePullPolicy, tolerations, nodeSelector, etc.) are applied correctly.
        manifest = copy.deepcopy(self.pod_manifest)
        manifest.setdefault("apiVersion", "v1")
        manifest.setdefault("kind", "Pod")
        metadata = manifest.setdefault("metadata", {})
        # generateName was only used to derive the pod name above; keep it out
        # of the applied manifest so it doesn't conflict with name.
        metadata.pop("generateName", None)
        metadata["name"] = pod_name
        metadata.setdefault("namespace", self.namespace)
        # inject image into the first container
        containers = manifest.setdefault("spec", {}).setdefault("containers", [{}])
        containers[0]["image"] = image

        import yaml as _yaml
        manifest_yaml = _yaml.dump(manifest, default_flow_style=False)
        apply_cmd = self._kubectl(["apply", "-f", "-"])
        result = subprocess.run(apply_cmd, input=manifest_yaml, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"kubectl apply failed for pod {pod_name}: {result.stderr.strip()}"
            )

        # Wait for pod to be Running
        logger.info("Waiting for pod %s to be ready ...", pod_name)
        wait_cmd = self._kubectl([
            "wait",
            f"pod/{pod_name}",
            "--for", "condition=Ready",
            "--namespace", self.namespace,
            f"--timeout={self.sandbox_creation_timeout_seconds}s",
        ])
        wait_result = subprocess.run(wait_cmd, capture_output=True, text=True)
        if wait_result.returncode != 0:
            self._force_delete(pod_name)
            raise RuntimeError(
                f"Pod {pod_name} did not become Ready within {self.sandbox_creation_timeout_seconds}s: "
                f"{wait_result.stderr.strip()}"
            )

        self._pod_name = pod_name
        logger.info("Pod %s is ready", pod_name)

        # Create /root/.venv → /opt/miniconda3/envs/testbed
        venv_result = subprocess.run(
            self._kubectl([
                "exec", pod_name,
                "--namespace", self.namespace,
                "--",
                "ln", "-sf", "/testbed/.venv", "/root/.venv",
            ]),
            capture_output=True,
        )
        if venv_result.returncode != 0:
            logger.warning(
                "Could not create /root/.venv symlink in pod %s: %s",
                pod_name,
                venv_result.stderr.decode(errors="replace").strip(),
            )

        return pod_name

    def execute(self, command: str, workdir: str | None = None, **kwargs) -> dict[str, Any]:
        """Run a bash command inside the pod.

        Args:
            command: Shell command string.
            workdir: Working directory override (defaults to self.working_dir).

        Returns:
            Dict with stdout, stderr, exit_code, success keys.
        """
        if not self._pod_name:
            raise RuntimeError("Sandbox not running. Call create() first.")

        cwd = workdir or self.working_dir
        cmd = f"cd {shlex.quote(cwd)} && timeout {self.command_exec_timeout_seconds} bash -c {shlex.quote(command)}"
        logger.debug("[%s] %s", self._pod_name, cmd)

        exec_cmd = self._kubectl([
            "exec", self._pod_name,
            "--namespace", self.namespace,
            "--",
            "bash", "-c", cmd,
        ])
        try:
            result = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=self.command_exec_timeout_seconds + 10,
            )
        except subprocess.TimeoutExpired as e:
            logger.error("Command timed out after %ds: %s", self.command_exec_timeout_seconds + 10, cmd)
            return {
                "stdout": e.stdout or "",
                "stderr": e.stderr or "",
                "exit_code": -1,
                "success": False,
            }
        return {
            "stdout": result.stdout,
            "stderr": self._filter_kubectl_noise(result.stderr),
            "exit_code": result.returncode,
            "success": result.returncode == 0,
        }

    def delete(self, **kwargs) -> None:
        """Delete the pod."""
        if not self._pod_name:
            return
        self._force_delete(self._pod_name)
        self._pod_name = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _kubectl(self, args: list[str], *, request_timeout: str | None = None) -> list[str]:
        """Build a kubectl command list, prepending kubeconfig and optional request timeout."""
        cmd = ["kubectl"]
        if self.kubeconfig:
            cmd += ["--kubeconfig", self.kubeconfig]
        if request_timeout:
            cmd += [f"--request-timeout={request_timeout}"]
        return cmd + args

    def _force_delete(self, pod_name: str) -> None:
        """Force-delete a pod by name, retrying on transient apiserver failures."""
        delete_cmd = self._kubectl(
            [
                "delete", "pod", pod_name,
                "--namespace", self.namespace,
                "--force",
                "--grace-period=0",
            ],
            request_timeout=_KUBECTL_REQUEST_TIMEOUT,
        )
        last_stderr = ""
        for attempt in range(1, _FORCE_DELETE_RETRIES + 1):
            logger.info("Deleting pod %s (attempt %d/%d)", pod_name, attempt, _FORCE_DELETE_RETRIES)
            result = subprocess.run(delete_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return
            last_stderr = (result.stderr or "").strip()
            if "NotFound" in last_stderr or "not found" in last_stderr:
                logger.info("Pod %s already gone: %s", pod_name, last_stderr)
                return
            logger.warning(
                "Delete pod %s failed (attempt %d/%d, rc=%d): %s",
                pod_name, attempt, _FORCE_DELETE_RETRIES, result.returncode, last_stderr,
            )
            if attempt < _FORCE_DELETE_RETRIES:
                time.sleep(_FORCE_DELETE_BACKOFF_SECONDS * attempt)
        logger.error(
            "Giving up deleting pod %s after %d attempts; pod may leak. Last error: %s",
            pod_name, _FORCE_DELETE_RETRIES, last_stderr,
        )

    @staticmethod
    def _filter_kubectl_noise(stderr: str) -> str:
        """Remove kubectl diagnostic lines (e.g. memcache.go) from stderr."""
        if not stderr:
            return stderr
        return "\n".join(
            line for line in stderr.splitlines()
            if not _KUBECTL_NOISE_RE.match(line)
        )
