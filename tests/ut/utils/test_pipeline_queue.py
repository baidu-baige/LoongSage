"""Unit tests for coda/utils/pipeline_queue.py.

Run: python -m pytest tests/ut/utils/test_pipeline_queue.py -v
"""
import asyncio

import pytest  # noqa: F401
from coda.utils.pipeline_queue import PipelineBuffer


class FakeItem:
    """Minimal item stub with a prompt_id attribute."""

    def __init__(self, prompt_id):
        """Initialize FakeItem with the given prompt_id."""
        self.prompt_id = prompt_id


class TestPipelineBufferPut:
    """Tests for PipelineBuffer.put method."""

    def test_put_success(self):
        """Verify put returns True and increments qsize."""
        buf = PipelineBuffer(is_stopped=lambda: False)
        item = FakeItem("epoch0_step0_prompt5")
        assert buf.put(item) is True
        assert buf.qsize == 1

    def test_put_stopped_returns_false(self):
        """Verify put returns False when buffer is stopped."""
        buf = PipelineBuffer(is_stopped=lambda: True)
        item = FakeItem("epoch0_step0_prompt0")
        assert buf.put(item) is False
        assert buf.qsize == 0

    def test_put_priority_ordering(self):
        """Verify items are dequeued in priority order by prompt_id."""
        buf = PipelineBuffer(is_stopped=lambda: False)
        buf.put(FakeItem("epoch1_step0_prompt10"))
        buf.put(FakeItem("epoch0_step0_prompt5"))
        buf.put(FakeItem("epoch0_step0_prompt1"))

        # Get items in priority order
        _, _, item1 = buf._queue.get()
        _, _, item2 = buf._queue.get()
        _, _, item3 = buf._queue.get()

        assert item1.prompt_id == "epoch0_step0_prompt1"
        assert item2.prompt_id == "epoch0_step0_prompt5"
        assert item3.prompt_id == "epoch1_step0_prompt10"


class TestPipelineBufferAsyncGet:
    """Tests for PipelineBuffer.async_get method."""

    def test_async_get_returns_item(self):
        """Verify async_get returns the buffered item."""
        buf = PipelineBuffer(is_stopped=lambda: False)
        buf.put(FakeItem("epoch0_step0_prompt3"))

        async def _run():
            return await buf.async_get()

        item = asyncio.run(_run())
        assert item is not None
        assert item.prompt_id == "epoch0_step0_prompt3"

    def test_async_get_returns_none_when_stopped(self):
        """Verify async_get returns None when buffer becomes stopped."""
        stopped = [False]
        buf = PipelineBuffer(is_stopped=lambda: stopped[0])

        async def _run():
            async def stop_after_delay():
                await asyncio.sleep(0.1)
                stopped[0] = True

            asyncio.create_task(stop_after_delay())
            return await buf.async_get()

        item = asyncio.run(_run())
        assert item is None


class TestPipelineBufferQsize:
    """Tests for PipelineBuffer.qsize property."""

    def test_qsize_empty(self):
        """Verify qsize is 0 for a fresh buffer."""
        buf = PipelineBuffer(is_stopped=lambda: False)
        assert buf.qsize == 0

    def test_qsize_after_puts(self):
        """Verify qsize reflects the number of items put."""
        buf = PipelineBuffer(is_stopped=lambda: False)
        buf.put(FakeItem("epoch0_step0_prompt0"))
        buf.put(FakeItem("epoch0_step0_prompt1"))
        assert buf.qsize == 2


class TestPipelineBufferSnapshot:
    """Tests for PipelineBuffer.snapshot method."""

    def test_snapshot_returns_all_items_without_consuming_them(self):
        """snapshot() is non-destructive; it does not promise priority order.

        Only async_get / _blocking_get pop in priority order, so the assertion
        here is on the item set, while the pop order is checked separately.
        """
        buf = PipelineBuffer(is_stopped=lambda: False)
        buf.put(FakeItem("epoch1_step0_prompt10"))
        buf.put(FakeItem("epoch0_step0_prompt5"))
        buf.put(FakeItem("epoch0_step0_prompt1"))

        items = buf.snapshot()

        assert sorted(item.prompt_id for item in items) == [
            "epoch0_step0_prompt1",
            "epoch0_step0_prompt5",
            "epoch1_step0_prompt10",
        ]
        assert buf.qsize == 3

        # Priority order is (epoch_num, prompt_num); step is not part of the key.
        queued_items = [buf._queue.get()[2] for _ in range(3)]
        assert [item.prompt_id for item in queued_items] == [
            "epoch0_step0_prompt1",
            "epoch0_step0_prompt5",
            "epoch1_step0_prompt10",
        ]

    def test_snapshot_list_is_independent(self):
        """Verify mutating the snapshot list does not change the queue."""
        buf = PipelineBuffer(is_stopped=lambda: False)
        buf.put(FakeItem("epoch0_step0_prompt0"))

        items = buf.snapshot()
        items.clear()

        assert buf.qsize == 1
