"""Sandbox backend interface.

A client keeps no per-sandbox state beyond the names of the sandboxes it created
and has not deleted yet, so the owning process can delete them all before exiting.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)


def decode_output(data: bytes | str | None) -> str:
    """Return ``data`` as text; ``TimeoutExpired`` carries bytes (or None) even with ``text=True``."""
    if isinstance(data, bytes):
        return data.decode(errors="replace")
    return data or ""


class SandboxClient(ABC):
    """Backend connector that can manage multiple sandbox instances."""

    @classmethod
    def from_config(cls, sandbox_config: Mapping[str, Any], **kwargs: Any) -> "SandboxClient":
        """Construct a backend client from framework configuration."""
        config = dict(sandbox_config)
        config.pop("type", None)
        return cls(**config)

    @abstractmethod
    def create(self, **kwargs: Any) -> str:
        """Create a sandbox instance and return its backend identifier."""

    @abstractmethod
    def execute(self, sandbox_id: str, command: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a shell command in the identified sandbox."""

    @abstractmethod
    def delete(self, sandbox_id: str, **kwargs: Any) -> None:
        """Destroy the identified sandbox; a missing resource is a success."""

    # ------------------------------------------------------------------
    # Live-sandbox tracking. Backends call _track before asking the backend to
    # create a sandbox, so one still being created is covered, and _untrack once
    # it is deleted. State is created lazily: subclasses need not call
    # super().__init__().
    # ------------------------------------------------------------------

    def _live_state(self) -> tuple[threading.RLock, set[str]]:
        state = self.__dict__.get("_live")
        if state is None:
            # RLock: delete_all may run in a signal handler that interrupted this
            # thread while it held the lock.
            state = self.__dict__.setdefault("_live", (threading.RLock(), set()))
        return state

    def _track(self, sandbox_id: str) -> None:
        """Record a sandbox this client is about to create; refuse once closed."""
        lock, live = self._live_state()
        with lock:
            if self.closed:
                raise RuntimeError(f"{type(self).__name__} is closed; not creating {sandbox_id}")
            live.add(sandbox_id)

    def _untrack(self, sandbox_id: str) -> None:
        """Forget a sandbox that has been deleted."""
        lock, live = self._live_state()
        with lock:
            live.discard(sandbox_id)

    @property
    def closed(self) -> bool:
        """True once delete_all has started; no sandbox is created afterwards."""
        return bool(self.__dict__.get("_closed", False))

    def live_sandboxes(self) -> list[str]:
        """Names of sandboxes created (or being created) and not yet deleted."""
        lock, live = self._live_state()
        with lock:
            return sorted(live)

    def delete_all(self, max_workers: int = 32) -> tuple[int, list[str]]:
        """Close the client and delete every tracked sandbox.

        Returns:
            (number deleted, ids whose delete raised).
        """
        lock, live = self._live_state()
        with lock:
            self.__dict__["_closed"] = True
            pending = sorted(live)
        if not pending:
            return 0, []

        def _delete(sandbox_id: str) -> str | None:
            try:
                self.delete(sandbox_id)
            except Exception as exc:
                logger.warning("delete_all: delete %s failed: %s", sandbox_id, exc)
                return sandbox_id
            self._untrack(sandbox_id)
            return None

        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(pending)))) as pool:
            failed = [sid for sid in pool.map(_delete, pending) if sid is not None]
        return len(pending) - len(failed), failed
