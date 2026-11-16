"""Unit tests for AgentFlow utility helpers."""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from coda.agentflow.utils import (
    ContextLengthExceededError,
    build_request_id,
    cli_process_pattern,
    kill_process_tree_command,
    require_docker_image,
    spawn_process_pool,
)

# Workers re-import coda.agentflow; without this litellm blocks fetching its cost map.
_LOCAL_COST_MAP = {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}


def test_build_request_id() -> None:
    """build_request_id should generate a stable attempt-scoped identifier."""
    assert build_request_id("traj-001", 3) == "traj-001#3"


def test_context_length_exceeded_error() -> None:
    """ContextLengthExceededError is a RuntimeError and preserves the error payload."""
    err = ContextLengthExceededError("budget exhausted", error={"type": "context_length_exceeded"})
    assert isinstance(err, RuntimeError)
    assert str(err) == "budget exhausted"
    assert err.error == {"type": "context_length_exceeded"}

    default = ContextLengthExceededError()
    assert default.error == {}
    assert str(default)


def test_require_docker_image() -> None:
    """require_docker_image passes through valid values, raises ValueError otherwise."""
    assert require_docker_image({"docker_image": "example/image:latest"}) == "example/image:latest"
    with pytest.raises(ValueError, match="docker_image"):
        require_docker_image({})
    with pytest.raises(ValueError, match="docker_image"):
        require_docker_image(None)
    with pytest.raises(ValueError, match="docker_image"):
        require_docker_image({"docker_image": ""})
    with pytest.raises(ValueError, match="docker_image"):
        require_docker_image({"docker_image": None})


def test_cli_process_pattern() -> None:
    """The pattern escapes the path, brackets the first letter and anchors the name."""
    assert cli_process_pattern("/root/.opencode/bin/opencode") == r"/root/\.opencode/bin/[o]pencode( |$)"
    assert cli_process_pattern("claude") == "(^|/)[c]laude( |$)"


def test_kill_process_tree_command_is_a_plain_command() -> None:
    command = kill_process_tree_command("(^|/)[c]odex( |$)", grace_seconds=3)
    # The docker backend runs `timeout N {command}`, so no leading assignment.
    assert command.startswith("bash -c ")
    assert command.endswith(" coda-kill '(^|/)[c]odex( |$)' 30")


def _gone(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return True
    return state in ("Z", "X")


_FAKE_CLI = """#!/bin/sh
trap '' INT
setsid sleep 3301 &
echo $! > "$1"
sleep 3302 &
echo $! >> "$1"
wait
"""


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or not (shutil.which("pgrep") and shutil.which("setsid")),
    reason="needs Linux /proc, pgrep and setsid",
)
def test_kill_process_tree_reaps_detached_children(tmp_path: Path) -> None:
    """A CLI that ignores SIGINT is killed with its same-group and setsid children."""
    cli = tmp_path / "bin" / "fakecli"
    cli.parent.mkdir()
    cli.write_text(_FAKE_CLI)
    cli.chmod(0o755)
    pid_file = tmp_path / "pids"
    # argv launch keeps the CLI path out of this test process's command line.
    proc = subprocess.Popen([str(cli), str(pid_file)])
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if pid_file.exists() and len(pid_file.read_text().split()) == 2:
                break
            time.sleep(0.05)
        children = [int(pid) for pid in pid_file.read_text().split()]
        # $! is written right after fork, before the child has run setsid(); wait for it.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and os.getpgid(children[0]) == os.getpgid(proc.pid):
            time.sleep(0.05)
        assert os.getpgid(children[0]) != os.getpgid(proc.pid)

        command = kill_process_tree_command(cli_process_pattern(str(cli)), grace_seconds=1)
        result = subprocess.run(command, shell=True, timeout=30)

        assert result.returncode == 0
        assert proc.wait(timeout=5) != 0
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not all(_gone(pid) for pid in children):
            time.sleep(0.05)
        assert all(_gone(pid) for pid in children)
    finally:
        if proc.poll() is None:
            proc.kill()


# Spawns a detached child only after SIGINT, i.e. after the first process-tree
# snapshot, so only the post-grace re-collection can find it.
_FAKE_CLI_SPAWN_ON_INT = """#!/bin/bash
trap 'setsid sleep 3303 & echo $! > "$1"' INT
echo ready > "$1.ready"
while :; do sleep 0.05; done
"""


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or not (shutil.which("pgrep") and shutil.which("setsid")),
    reason="needs Linux /proc, pgrep and setsid",
)
def test_kill_process_tree_reaps_children_spawned_during_grace(tmp_path: Path) -> None:
    """A detached child spawned after SIGINT is found by the post-grace re-collection."""
    cli = tmp_path / "bin" / "fakecli"
    cli.parent.mkdir()
    cli.write_text(_FAKE_CLI_SPAWN_ON_INT)
    cli.chmod(0o755)
    pid_file = tmp_path / "pids"
    proc = subprocess.Popen([str(cli), str(pid_file)])
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not Path(f"{pid_file}.ready").exists():
            time.sleep(0.05)

        command = kill_process_tree_command(cli_process_pattern(str(cli)), grace_seconds=1)
        assert subprocess.run(command, shell=True, timeout=30).returncode == 0

        assert proc.wait(timeout=5) != 0
        child = int(pid_file.read_text())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not _gone(child):
            time.sleep(0.05)
        assert _gone(child)
    finally:
        if proc.poll() is None:
            proc.kill()
        if pid_file.exists():
            subprocess.run(["kill", "-KILL", pid_file.read_text().strip()], check=False)


# Set only in the test process; a forked worker would inherit it, a spawned one re-imports this module.
_SET_IN_PARENT: str | None = None
_INITIALIZED_WITH: str | None = None


def _record_init(value: str) -> None:
    global _INITIALIZED_WITH
    _INITIALIZED_WITH = value


def _worker_state() -> tuple[str | None, str | None]:
    return _SET_IN_PARENT, _INITIALIZED_WITH


def test_spawn_process_pool_spawns_and_runs_initializer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Workers start from a fresh interpreter and still run the caller's initializer."""
    for key, value in _LOCAL_COST_MAP.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys.modules[__name__], "_SET_IN_PARENT", "parent")
    pool = spawn_process_pool(1, initializer=_record_init, initargs=("tokenizer",))
    try:
        assert pool.submit(_worker_state).result(timeout=120) == (None, "tokenizer")
    finally:
        pool.shutdown(wait=True)


_POOL_OWNER = """
import os, sys, time
from coda.agentflow.utils import spawn_process_pool
pool = spawn_process_pool(1)
print("worker=%d" % pool.submit(os.getpid).result(), flush=True)
time.sleep(600)
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_spawn_process_pool_workers_die_with_parent() -> None:
    """SIGKILL on the pool owner takes its idle worker down too, leaving no orphan."""
    repo_root = Path(__file__).resolve().parents[3]
    owner = subprocess.Popen(
        [sys.executable, "-c", _POOL_OWNER],
        cwd=repo_root,
        env={**os.environ, **_LOCAL_COST_MAP},
        stdout=subprocess.PIPE,
        text=True,
    )
    worker = None
    try:
        # Imports may print banners (mini-swe-agent) to stdout before the pid line.
        for line in owner.stdout:
            if line.startswith("worker="):
                worker = int(line.removeprefix("worker="))
                break
        assert worker is not None, "pool owner exited before reporting its worker"
        owner.send_signal(signal.SIGKILL)
        owner.wait(timeout=5)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not _gone(worker):
            time.sleep(0.05)
        assert _gone(worker)
    finally:
        if owner.poll() is None:
            owner.kill()
        if worker is not None and not _gone(worker):
            os.kill(worker, signal.SIGKILL)
