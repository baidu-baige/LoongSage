"""Utility helpers for AgentFlow internals."""

from __future__ import annotations

import asyncio
import copy
import ctypes
import multiprocessing
import os
import shlex
import signal
import sys
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from coda.agentflow.sandbox import SandboxClient

_REQUEST_ID_SEPARATOR = "#"
CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
# User turn a black-box CLI sends after reloading its own persisted session.
RESUME_PROMPT = "Continue working on the task."


_ERE_SPECIAL = set(".[]()*+?{}|^$\\")


def cli_process_pattern(executable: str) -> str:
    """``pgrep -f`` regex matching processes launched from *executable* (a path or a bare name).

    The first letter of the name is bracketed so the pattern never matches the
    shell command that carries it.
    """
    head, _, name = executable.rpartition("/")
    escape = lambda text: "".join("\\" + c if c in _ERE_SPECIAL else c for c in text)  # noqa: E731
    prefix = escape(head + "/") if head else "(^|/)"
    return f"{prefix}[{escape(name[0])}]{escape(name[1:])}( |$)"


# Black-box CLIs spawn tool commands in their own process groups (OpenCode's
# shell tool runs them detached), so killing only the CLI leaves e.g. a running
# pytest editing the sandbox that partial resume later reuses. The process tree
# and its groups are snapshotted while the CLI is still alive (orphans are
# re-parented to init afterwards); only groups led by a descendant are killed,
# never a group the CLI merely inherited from the sandbox exec.
_KILL_PROCESS_TREE_SH = r'''
pattern=$1
ticks=$2
collect() {
  # Each pass walks the whole tree again; $pids only accumulates, so a second
  # pass after the grace period still finds children spawned in between.
  seen=""
  queue=$(pgrep -f "$pattern")
  while [ -n "$queue" ]; do
    next=""
    for p in $queue; do
      case " $seen " in *" $p "*) continue ;; esac
      seen="$seen $p"
      case " $pids " in *" $p "*) ;; *) pids="$pids $p" ;; esac
      next="$next $(pgrep -P "$p")"
    done
    queue=$(echo $next)
  done
  for p in $pids; do
    g=$(sed 's/.*) //' "/proc/$p/stat" 2>/dev/null | cut -d' ' -f3)
    case " $pids " in *" $g "*) case " $groups " in *" $g "*) ;; *) groups="$groups $g" ;; esac ;; esac
  done
}
alive() {
  for p in $roots; do kill -0 "$p" 2>/dev/null && return 0; done
  return 1
}
pids=""
groups=""
roots=$(echo $(pgrep -f "$pattern"))
[ -n "$roots" ] || exit 0
collect
kill -INT $roots 2>/dev/null
n=0
while alive && [ "$n" -lt "$ticks" ]; do sleep 0.1; n=$((n + 1)); done
alive && collect
kill -KILL $pids 2>/dev/null
for g in $groups; do kill -KILL -- "-$g" 2>/dev/null; done
exit 0
'''


def kill_process_tree_command(pattern: str, grace_seconds: int = 10) -> str:
    """Shell command that interrupts a sandboxed CLI, then kills it and every process it spawned.

    The CLI gets SIGINT first and up to *grace_seconds* to exit on its own, so
    it can record interrupted tool calls in its session before the SIGKILL.
    """
    # Starts with a command word (not an assignment) because the docker backend
    # prefixes commands with `timeout N`.
    return shlex.join(
        ["bash", "-c", _KILL_PROCESS_TREE_SH.strip(), "coda-kill", pattern, str(int(grace_seconds * 10))]
    )


_PR_SET_PDEATHSIG = 1


def _init_pool_worker(parent_pid: int, initializer: Callable[..., None] | None, initargs: tuple) -> None:
    """Make a pool worker die with the process that spawned it, then run *initializer*."""
    if sys.platform.startswith("linux"):
        # The parent fires on SIGTERM/SIGKILL without shutting the pool down, and
        # idle workers block on the call queue forever, so let the kernel kill them.
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
        # The parent may have died before prctl took effect.
        if os.getppid() != parent_pid:
            os._exit(1)
    if initializer is not None:
        initializer(*initargs)


def spawn_process_pool(
    max_workers: int, initializer: Callable[..., None] | None = None, initargs: tuple = ()
) -> ProcessPoolExecutor:
    """ProcessPoolExecutor whose workers are spawned and exit when this process dies.

    spawn, not fork: workers are created lazily while sandbox threads have kubectl
    subprocesses in flight; forked workers inherit those pipe fds (and the router's
    listening socket), so the subprocess calls never see EOF and hang.
    """
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_pool_worker,
        initargs=(os.getpid(), initializer, initargs),
    )


class ContextLengthExceededError(RuntimeError):
    """Raised when the router rejects a request because the trajectory token
    budget is exhausted (HTTP 400 with ``error.type == context_length_exceeded``).

    Callers treat this as a graceful stop of the trajectory, not a failure.
    """

    def __init__(self, message: str = "", *, error: dict | None = None) -> None:
        super().__init__(message or "context length exceeded (router token budget exhausted)")
        self.error: dict = error or {}


def build_request_id(trajectory_id: str, attempt_id: int) -> str:
    """Build a stable per-attempt request id.

    Used by the router's abort_session endpoint and AgentFlow's failure-handling
    path to identify the worker request to cancel.
    """
    return f"{trajectory_id}{_REQUEST_ID_SEPARATOR}{attempt_id}"


def require_docker_image(metadata: Mapping[str, Any] | None) -> str:
    """Return ``metadata['docker_image']`` or raise ValueError when missing."""
    image = str((metadata or {}).get("docker_image") or "")
    if not image:
        raise ValueError("metadata['docker_image'] is missing")
    return image


async def setup_sandbox(
    client: SandboxClient, metadata: Mapping[str, Any], prepare: Callable,
) -> tuple[str, dict[str, Any]]:
    """Return (sandbox_id, reward state) for a newly created, prepared sandbox.

    Creation and preparation share one worker thread so they happen as a unit: a
    sandbox that was created but not prepared is useless, and once retained it is
    indistinguishable from a prepared one whose preparation returned no state.
    """
    def create_and_prepare() -> tuple[str, dict[str, Any]]:
        sandbox_id = client.create(image=require_docker_image(metadata))
        try:
            return sandbox_id, dict(prepare(client, sandbox_id, metadata=copy.deepcopy(dict(metadata))))
        except BaseException:
            client.delete(sandbox_id)
            raise

    return await asyncio.to_thread(create_and_prepare)
