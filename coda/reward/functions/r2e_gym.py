"""R2E-Gym reward function.

Evaluation protocol:

1. Keep the sandbox worktree exactly as the agent left it.
2. Run the evaluation script ``run_tests.sh``.
   (The script and test files ship with the Docker image — under ``/r2e_tests``,
    ``/testbed/r2e_tests`` or ``/root/r2e_tests`` — and are arranged under
    ``/root/`` during container initialization.)
3. Parse pytest's stdout summary into {Class.method: PASSED/FAILED/ERROR}.
4. Require every entry of ``metadata["expected_output_json"]`` to be present in
   the parsed results with the same status. Extra entries in the parsed results
   are ignored.

reward = 1.0  iff  every expected entry matches
reward = 0.0  otherwise

Reference implementations:
- SkyRL: https://github.com/novasky-ai/skyrl
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from coda.reward import register_reward
from coda.reward.base import RewardFunction
from coda.reward.reward import Reward

logger = logging.getLogger(__name__)

R2E_ASSET_ARCHIVE = "/root/.coda-r2e-eval-assets.tar"
_R2E_ASSET_DIGEST_MARKER = "CODA_R2E_ASSET_SHA256="
_R2E_SHELL_PREFIX = (
    "source /root/.bashrc >/dev/null 2>&1 || true; "
    "export PATH=/testbed/.venv/bin:/opt/miniconda3/envs/testbed/bin:"
    "/root/.local/bin:$PATH; "
)

_R2E_SETUP_COMMAND = r"""
set -eu

marker=/tmp/coda-r2e-initialized
if [ -f "$marker" ]; then
    exit 0
fi

if [ -x /testbed/.venv/bin/python ]; then
    testbed_env=/testbed/.venv
elif [ -x /opt/miniconda3/envs/testbed/bin/python ]; then
    testbed_env=/opt/miniconda3/envs/testbed
else
    echo "R2E setup: testbed Python environment not found" >&2
    exit 1
fi

mkdir -p /root/.local/bin
if [ -L /root/.venv ]; then
    ln -sfn "$testbed_env" /root/.venv
elif [ ! -e /root/.venv ]; then
    ln -s "$testbed_env" /root/.venv
fi
ln -sfn "$testbed_env/bin/python" /root/.local/bin/python
ln -sfn "$testbed_env/bin/python" /root/.local/bin/python3

if [ ! -e /root/run_tests.sh ] && [ -f /testbed/run_tests.sh ]; then
    mv /testbed/run_tests.sh /root/run_tests.sh
fi

if [ ! -e /root/r2e_tests ]; then
    if [ -d /r2e_tests ]; then
        mv /r2e_tests /root/r2e_tests
    elif [ -d /testbed/r2e_tests ] && [ ! -L /testbed/r2e_tests ]; then
        mv /testbed/r2e_tests /root/r2e_tests
    fi
fi

if [ -d /root/r2e_tests ]; then
    if [ -L /testbed/r2e_tests ]; then
        ln -sfn /root/r2e_tests /testbed/r2e_tests
    elif [ ! -e /testbed/r2e_tests ]; then
        ln -s /root/r2e_tests /testbed/r2e_tests
    fi
fi

if [ ! -f /root/run_tests.sh ]; then
    echo "R2E setup: run_tests.sh not found in /testbed or /root" >&2
    exit 1
fi
if [ ! -d /root/r2e_tests ]; then
    echo "R2E setup: r2e_tests not found in /r2e_tests, /testbed, or /root" >&2
    exit 1
fi

find /testbed -path /testbed/.venv -prune -o -name '*.pyc' -exec rm -f {} +
find /root/r2e_tests -name '*.pyc' -delete

tar -C /root -cf /root/.coda-r2e-eval-assets.tar run_tests.sh r2e_tests
touch "$marker"
printf 'CODA_R2E_ASSET_SHA256=%s\n' "$(sha256sum /root/.coda-r2e-eval-assets.tar | awk '{print $1}')"
"""


def initialize_r2e_sandbox(sandbox: Any, *, workdir: str = "/testbed") -> str:
    """Create the pristine R2E evaluation archive and return its SHA-256."""
    result = sandbox.execute(
        f"bash -c {shlex.quote(_R2E_SHELL_PREFIX + _R2E_SETUP_COMMAND)}",
        workdir=workdir,
    )
    if result.get("exit_code", -1) != 0:
        output = (result.get("stdout", "") + result.get("stderr", "")).strip()
        raise RuntimeError(f"R2E sandbox initialization failed: {output[:1000]}")
    match = re.search(
        rf"^{re.escape(_R2E_ASSET_DIGEST_MARKER)}([0-9a-f]{{64}})$",
        result.get("stdout", ""),
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("R2E sandbox initialization did not return an asset digest")
    return match.group(1)

# ── executor abstraction ──────────────────────────────────────────────────────


class _LocalExecutor:
    """Run commands on the local host via subprocess."""

    python_cmd: str = "python"

    def __init__(self, repo_path: str) -> None:
        """Initialize with the local repository path."""
        self.repo_path = repo_path

    def run(self, cmd: list[str] | str, timeout: int = 120) -> tuple[int, str, str]:
        """Run a command in the repo directory; return (returncode, stdout, stderr)."""
        if isinstance(cmd, list):
            result = subprocess.run(
                cmd, cwd=self.repo_path, capture_output=True, text=True, timeout=timeout,
            )
        else:
            result = subprocess.run(
                cmd, cwd=self.repo_path, capture_output=True, text=True,
                timeout=timeout, shell=True,
            )
        return result.returncode, result.stdout, result.stderr


class _SandboxExecutor:
    """Run commands inside a CODA SandboxClient (K8s / Docker pod)."""

    _CONDA_PYTHON = "/opt/miniconda3/envs/testbed/bin/python"
    python_cmd: str = _CONDA_PYTHON

    def __init__(self, sandbox: Any, workdir: str = "/testbed") -> None:
        """Initialize with a SandboxClient instance and working directory."""
        self._sandbox = sandbox
        self._workdir = workdir

    def run(self, cmd: list[str] | str) -> tuple[int, str, str]:
        """Run a command using the sandbox client's configured execution timeout."""
        if isinstance(cmd, list):
            cmd = [self._CONDA_PYTHON if c == "python" else c for c in cmd]
            command = subprocess.list2cmdline(cmd)
        else:
            command = cmd
        result = self._sandbox.execute(command, workdir=self._workdir)
        return result.get("exit_code", -1), result.get("stdout", ""), result.get("stderr", "")

    def restore_r2e_assets(self, expected_sha256: str) -> tuple[bool, str]:
        """Verify the pristine archive and restore evaluation assets from it."""
        archive = shlex.quote(R2E_ASSET_ARCHIVE)
        expected = shlex.quote(expected_sha256)
        command = f"""
set -eu
archive={archive}
expected={expected}
if [ ! -f "$archive" ]; then
    echo "R2E evaluation asset archive is missing" >&2
    exit 1
fi
actual="$(sha256sum "$archive" | awk '{{print $1}}')"
if [ "$actual" != "$expected" ]; then
    echo "R2E evaluation asset archive digest mismatch" >&2
    exit 1
fi
rm -rf /root/r2e_tests /root/run_tests.sh /testbed/r2e_tests
tar -C /root -xf "$archive"
ln -s /root/r2e_tests /testbed/r2e_tests
echo CODA_R2E_ASSETS_RESTORED
"""
        rc, stdout, stderr = self.run(command)
        output = (stdout + stderr).strip()
        return rc == 0 and "CODA_R2E_ASSETS_RESTORED" in stdout, output


# ── pytest log parser ─────────────────────────────────────────────────────────


def _parse_pytest_log(log: str) -> dict[str, str]:
    """Parse pytest summary into {ClassName.test_method: PASSED/FAILED/ERROR}.

    Strips ANSI colour codes before parsing. Follows SkyRL parse_log_pytest logic.
    """
    log = re.sub(r"\x1b\[[0-9;]*m|\r", "", log)
    if "short test summary info" not in log:
        return {}
    results: dict[str, str] = {}
    for line in log.split("short test summary info")[1].splitlines():
        line = line.strip()
        if "PASSED" in line:
            results[".".join(line.split("::")[1:]).split(" - ")[0]] = "PASSED"
        elif "FAILED" in line:
            results[".".join(line.split("::")[1:]).split(" - ")[0]] = "FAILED"
        elif "ERROR" in line:
            results[".".join(line.split("::")[1:]).split(" - ")[0]] = "ERROR"
    return results


# ── reward function ───────────────────────────────────────────────────────────


@register_reward("r2e_gym")
class R2EGymReward(RewardFunction):
    """Test the current worktree with the official R2E result comparison.

    reward = 1.0  iff  the official R2E comparison accepts the test results
    reward = 0.0  otherwise
    """

    def prepare_sandbox(self, sandbox: Any, metadata: dict | None = None) -> None:
        """Create the trusted evaluation-asset snapshot before the agent runs.

        This hook is called by AgentFlow after the sandbox is created and before
        the selected agent starts.  Keeping it here means black-box agents do
        not need to identify or initialize R2E-Gym themselves.
        """
        if getattr(sandbox, "_coda_r2e_asset_sha256", None) is not None:
            return
        metadata = metadata or {}
        digest = initialize_r2e_sandbox(
            sandbox,
            workdir=str(metadata.get("repo_path") or "/testbed"),
        )
        sandbox._coda_r2e_asset_sha256 = digest

    def __call__(
        self,
        messages: list[dict],
        label: Any,
        metadata: dict | None = None,
        **kwargs: Any,
    ) -> Reward:
        """Run tests in the current worktree; 1.0 when every expected entry matches."""
        meta = metadata or {}
        extra: dict[str, Any] = {"instance_id": meta.get("instance_id", "")}

        # 1. parse expected outputs (done first so all early returns can include it)
        expected_json = meta.get("expected_output_json") or ""
        if not expected_json:
            raise RuntimeError("R2EGymReward: expected_output_json missing from metadata")
        _ansi_re = re.compile(r"\x1b\[[0-9;]*m|\r")
        expected = {
            _ansi_re.sub("", k).split(" - ")[0]: v
            for k, v in json.loads(expected_json).items()
        }
        extra["expected"] = expected

        # 2. Build an executor for the same worktree the agent modified.
        sandbox = meta.get("sandbox")
        if sandbox is not None:
            executor: _LocalExecutor | _SandboxExecutor = _SandboxExecutor(
                sandbox, workdir="/testbed"
            )
            logger.info("R2EGymReward: using sandbox executor")
        else:
            repo_path = meta.get("repo_path", "")
            if not repo_path or not Path(repo_path).is_dir():
                raise RuntimeError(
                    f"R2EGymReward: repo_path missing or invalid: {repo_path!r}"
                )
            executor = _LocalExecutor(repo_path)
            logger.info("R2EGymReward: using local executor at %s", repo_path)

        # 3. R2E images may intentionally ship with dirty tracked files. Test
        # the current worktree directly; checkout/reset corrupts that baseline.
        if isinstance(executor, _SandboxExecutor):
            expected_asset_sha256 = getattr(
                sandbox, "_coda_r2e_asset_sha256", None
            )
            if expected_asset_sha256 is None:
                raise RuntimeError(
                    "R2EGymReward: sandbox was not prepared before agent execution"
                )
            restored, restore_output = executor.restore_r2e_assets(
                expected_asset_sha256
            )
            if not restored:
                logger.warning(
                    "R2EGymReward: evaluation asset integrity check failed: %s",
                    restore_output[:1000],
                )
                extra.update(
                    {
                        "actual": {},
                        "matched": 0,
                        "reason": "evaluation_assets_modified",
                        "asset_check": restore_output[:1000],
                    }
                )
                return Reward(final_reward=0.0, extra_info=extra)
            run_cmd = "bash /root/run_tests.sh"
        else:
            run_cmd = "bash ./run_tests.sh"
        rc, stdout, stderr = executor.run(run_cmd)
        logger.info(
            "R2EGymReward: run_tests.sh rc=%d stdout (first 2000):\n%s\n"
            "stderr (first 2000):\n%s",
            rc,
            stdout[:2000],
            stderr[:2000],
        )

        actual = _parse_pytest_log(stdout)
        matched = sum(1 for k, v in expected.items() if actual.get(k) == v)
        score = 1.0 if matched == len(expected) else 0.0

        extra.update({"actual": actual, "rc": rc, "matched": matched})
        logger.info(
            "R2EGymReward %s: score=%.1f  matched=%d/%d",
            meta.get("instance_id", "?"), score, matched, len(expected),
        )
        return Reward(final_reward=score, extra_info=extra)
