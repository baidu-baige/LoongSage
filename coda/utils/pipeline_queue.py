"""Thread-safe pipeline buffer for producer-consumer pattern between rollout and training.

The producer (rollout_loop) runs in a dedicated thread with its own event loop.
The consumer (train_loop) runs in the main thread's event loop.
Communication is via thread-safe queue.Queue.
"""
import asyncio
import logging
import queue
from typing import Callable

logger = logging.getLogger(__name__)


class PipelineBuffer:
    """Thread-safe unbounded priority queue.

    Designed for cross-thread usage:
      - Producer thread: calls put() (non-blocking; the queue is unbounded)
      - Consumer thread (main): calls async_get() (non-blocking via asyncio.to_thread)

    All lifecycle control (pause/resume/stop) is managed externally (by RolloutSampler).
    ``is_stopped`` is injected at construction time so that put()/get() can react to
    external state without owning it.
    """

    def __init__(
        self,
        is_stopped: Callable[[], bool],
    ):
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._is_stopped = is_stopped

    # ── Properties ──────────────────────────────────────────────────────────────

    @property
    def qsize(self) -> int:
        """Current number of items in the queue."""
        return self._queue.qsize()

    # ── Producer API (called from producer thread) ─────────────────────────────

    def put(self, item) -> bool:
        """Put an item into the buffer as a priority entry (non-blocking, thread-safe).

        Priority is derived from prompt_id (e.g. ``"epoch0_step1_ds0_prompt116"``):
        ordered by epoch first, then by the per-epoch prompt index.

        Since the buffer is unlimited, this always succeeds immediately
        unless the system is stopped.
        """
        if self._is_stopped():
            return False
        prompt_id = item.prompt_id
        epoch_num = int(prompt_id.split("epoch")[1].split("_")[0])
        prompt_num = int(prompt_id.split("prompt")[1])
        self._queue.put((epoch_num, prompt_num, item))
        return True

    # ── Consumer API (called from main thread's event loop) ────────────────────

    def _blocking_get(self):
        """Blocking get that waits until item available or stopped."""
        while not self._is_stopped():
            try:
                _a, _b, item = self._queue.get(timeout=1.0)
                return item
            except queue.Empty:
                continue
        return None

    async def async_get(self):
        """Async get for consumer in main event loop. Non-blocking to event loop."""
        return await asyncio.to_thread(self._blocking_get)

    def snapshot(self) -> list:
        """Return a non-destructive snapshot of the queued items.

        The order is the priority queue's internal heap order, NOT priority
        order — only ``async_get`` pops in priority order. Callers that need a
        deterministic sequence must sort the result themselves.

        The returned list contains only payloads; internal ``(epoch_num,
        prompt_num)`` priorities are stripped. Payload objects are not copied,
        but adding or removing entries from the returned list does not affect
        the queue.
        """
        with self._queue.mutex:
            entries = list(self._queue.queue)

        return [item for _, _, item in entries]
