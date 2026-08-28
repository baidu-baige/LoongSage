"""Multi-process integration tests for transfer_mesh.channel.

These tests require CUDA GPUs and use torch.multiprocessing to validate
real IPC-based tensor transfer through TransferMeshChannel.
"""

import queue as _queue

import pytest
import torch
import torch.multiprocessing as mp

from tests.conftest import requires_cuda, requires_multi_gpu


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _drain_results(result_queue, procs, expected_ranks, timeout=30):
    """Collect one result per expected rank, join the children, assert clean exits.

    ``mp.Queue`` is not flushed synchronously by ``join()``, so draining must
    happen before joining, and polling ``empty()`` can miss results still in
    transit — hence a fixed number of blocking ``get()`` calls. Checking
    ``exitcode`` afterwards catches a child that died instead of reporting.
    """
    results = {}
    for _ in expected_ranks:
        try:
            r = result_queue.get(timeout=timeout)
        except _queue.Empty:
            break
        results[r["rank"]] = r

    for p in procs:
        p.join(timeout=timeout)
        assert p.exitcode is not None, f"child pid={p.pid} did not exit in {timeout}s"
        assert p.exitcode == 0, f"child pid={p.pid} exited with {p.exitcode}"

    for rank in expected_ranks:
        assert rank in results, f"Rank {rank} did not produce results"
        if not results[rank]["success"]:
            pytest.fail(
                f"Rank {rank} failed: {results[rank].get('error')}\n"
                f"{results[rank].get('tb', '')}"
            )
    return results


def _sender_proc(rank, world_size, gpu_id, gloo_port, tensors_to_send,
                 buffer_size, result_queue):
    """Sender worker: create channel, send tensors, close."""
    torch.cuda.set_device(gpu_id)
    try:
        from coda.transfer_mesh.channel import create_channel, Role
        ch = create_channel(
            master_addr="127.0.0.1", addr="127.0.0.1",
            gpu_id=gpu_id, world_size=world_size, rank=rank,
            role=Role.SENDER, src_rank=rank,
            buffer_size_bytes=buffer_size,
            gloo_port=gloo_port,
        )
        for name, data, dtype, shape in tensors_to_send:
            t = torch.tensor(data, dtype=dtype, device=f"cuda:{gpu_id}").reshape(shape)
            ch.send((name, t))
        ch.send(None)
        ch.close()
        result_queue.put({"rank": rank, "success": True})
    except Exception as e:
        import traceback
        result_queue.put({"rank": rank, "success": False, "error": str(e),
                          "tb": traceback.format_exc()})


def _receiver_proc(rank, world_size, gpu_id, gloo_port, src_rank,
                   buffer_size, yield_buckets, result_queue):
    """Receiver worker: create channel, receive tensors, return results."""
    torch.cuda.set_device(gpu_id)
    try:
        from coda.transfer_mesh.channel import create_channel, Role
        ch = create_channel(
            master_addr="127.0.0.1", addr="127.0.0.1",
            gpu_id=gpu_id, world_size=world_size, rank=rank,
            role=Role.RECEIVER, src_rank=src_rank,
            buffer_size_bytes=buffer_size,
            gloo_port=gloo_port,
        )
        received = []
        if yield_buckets:
            for payload, meta in ch.get_iterator(yield_buckets=True):
                for spec in meta.tensor_specs:
                    t = payload[spec.offset:spec.offset + spec.numel()].view(spec.shape).clone()
                    received.append({
                        "name": spec.name,
                        "shape": tuple(t.shape),
                        "dtype": str(t.dtype),
                        "values": t.cpu().tolist(),
                    })
        else:
            for name, tensor in ch.get_iterator():
                received.append({
                    "name": name,
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "values": tensor.cpu().tolist(),
                })
        ch.close()
        result_queue.put({"rank": rank, "success": True, "received": received})
    except Exception as e:
        import traceback
        result_queue.put({"rank": rank, "success": False, "error": str(e),
                          "tb": traceback.format_exc()})


def _run_transfer(tensors_to_send, world_size=2, sender_rank=0, receiver_rank=1,
                  sender_gpu=0, receiver_gpu=0,
                  buffer_size=64 * 1024 * 1024, yield_buckets=False, timeout=60):
    """Run a single sender/receiver transfer and return received tensors."""
    gloo_port = _get_free_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    sender = ctx.Process(
        target=_sender_proc,
        args=(sender_rank, world_size, sender_gpu, gloo_port,
              tensors_to_send, buffer_size, result_queue),
    )
    receiver = ctx.Process(
        target=_receiver_proc,
        args=(receiver_rank, world_size, receiver_gpu, gloo_port, sender_rank,
              buffer_size, yield_buckets, result_queue),
    )

    sender.start()
    receiver.start()

    results = _drain_results(
        result_queue, [sender, receiver], [sender_rank, receiver_rank], timeout=timeout
    )

    return results[receiver_rank]["received"]


# ── Tests ────────────────────────────────────────────────────────────────────


@requires_cuda
class TestBasicIPCTransfer:
    """Single sender + single receiver on the same GPU."""

    def test_single_tensor(self):
        tensors = [("weight", [1.0, 2.0, 3.0, 4.0], torch.float32, (2, 2))]
        received = _run_transfer(tensors)

        assert len(received) == 1
        r = received[0]
        assert r["name"] == "weight"
        assert r["shape"] == (2, 2)
        assert r["dtype"] == "torch.float32"
        assert r["values"] == [[1.0, 2.0], [3.0, 4.0]]

    def test_multiple_tensors_order_preserved(self):
        tensors = [
            ("a", [1.0, 2.0], torch.float32, (2,)),
            ("b", [3.0, 4.0, 5.0], torch.float32, (3,)),
            ("c", [6.0], torch.float32, (1,)),
        ]
        received = _run_transfer(tensors)

        assert len(received) == 3
        assert received[0]["name"] == "a"
        assert received[1]["name"] == "b"
        assert received[2]["name"] == "c"
        assert received[0]["values"] == [1.0, 2.0]
        assert received[1]["values"] == [3.0, 4.0, 5.0]
        assert received[2]["values"] == [6.0]

    def test_bfloat16_dtype(self):
        tensors = [("bf16", [1.0, 2.0, 3.0], torch.bfloat16, (3,))]
        received = _run_transfer(tensors)

        assert received[0]["dtype"] == "torch.bfloat16"
        assert len(received[0]["values"]) == 3

    def test_mixed_dtypes(self):
        """Send tensors with different dtypes (triggers bucket flush on dtype change)."""
        tensors = [
            ("f32", [1.0, 2.0], torch.float32, (2,)),
            ("bf16", [3.0, 4.0], torch.bfloat16, (2,)),
            ("f16", [5.0, 6.0], torch.float16, (2,)),
        ]
        received = _run_transfer(tensors)

        assert len(received) == 3
        assert received[0]["dtype"] == "torch.float32"
        assert received[1]["dtype"] == "torch.bfloat16"
        assert received[2]["dtype"] == "torch.float16"

    def test_2d_and_3d_shapes(self):
        tensors = [
            ("mat", list(range(12)), torch.float32, (3, 4)),
            ("cube", list(range(24)), torch.float32, (2, 3, 4)),
        ]
        received = _run_transfer(tensors)

        assert received[0]["shape"] == (3, 4)
        assert received[1]["shape"] == (2, 3, 4)


@requires_cuda
class TestBucketIteratorMode:

    def test_yield_buckets_true(self):
        tensors = [
            ("a", [1.0, 2.0], torch.float32, (2,)),
            ("b", [3.0, 4.0], torch.float32, (2,)),
        ]
        received = _run_transfer(tensors, yield_buckets=True)

        assert len(received) == 2
        names = [r["name"] for r in received]
        assert "a" in names
        assert "b" in names


@requires_cuda
class TestLargeTensor:

    def test_tensor_larger_than_buffer(self):
        """A single tensor larger than buffer_size_bytes should still transfer."""
        # Buffer = 1024 bytes, tensor = 1000 float32 = 4000 bytes
        data = list(range(1000))
        tensors = [("big", data, torch.float32, (1000,))]
        received = _run_transfer(tensors, buffer_size=1024)

        assert len(received) == 1
        assert received[0]["name"] == "big"
        assert len(received[0]["values"]) == 1000
        # Spot check first and last values
        assert received[0]["values"][0] == 0.0
        assert received[0]["values"][999] == 999.0


@requires_cuda
class TestEmptyTransfer:

    def test_send_none_immediately(self):
        """Sender sends None immediately -> receiver gets empty iterator."""
        received = _run_transfer([])
        assert received == []


@requires_multi_gpu
class TestMultiSenderReceiver:
    """2 senders + 2 receivers, each pair on a different GPU."""

    def test_parallel_transfer(self):
        gloo_port = _get_free_port()
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        world_size = 4

        # sender 0 on GPU 0, sender 1 on GPU 1
        # receiver 2 on GPU 0, receiver 3 on GPU 1
        # src_rank=0 (the primary sender for NCCL broadcast)
        tensors_gpu0 = [("gpu0_w", [1.0, 2.0], torch.float32, (2,))]
        tensors_gpu1 = [("gpu1_w", [3.0, 4.0], torch.float32, (2,))]

        procs = []

        # Sender rank 0, GPU 0
        p = ctx.Process(target=_sender_proc, args=(
            0, world_size, 0, gloo_port,
            tensors_gpu0, 64 * 1024 * 1024, result_queue,
        ))
        procs.append(p)

        # Sender rank 1, GPU 1
        p = ctx.Process(target=_sender_proc, args=(
            1, world_size, 1, gloo_port,
            tensors_gpu1, 64 * 1024 * 1024, result_queue,
        ))
        procs.append(p)

        # Receiver rank 2, GPU 0 (receives from sender 0 via IPC)
        p = ctx.Process(target=_receiver_proc, args=(
            2, world_size, 0, gloo_port, 0,
            64 * 1024 * 1024, False, result_queue,
        ))
        procs.append(p)

        # Receiver rank 3, GPU 1 (receives from sender 1 via IPC)
        p = ctx.Process(target=_receiver_proc, args=(
            3, world_size, 1, gloo_port, 0,
            64 * 1024 * 1024, False, result_queue,
        ))
        procs.append(p)

        for p in procs:
            p.start()

        results = _drain_results(
            result_queue, procs, list(range(world_size)), timeout=60
        )

        # Receiver on GPU 0 should have received from sender 0
        recv0 = results[2]["received"]
        assert len(recv0) >= 1
        assert recv0[0]["name"] == "gpu0_w"
        assert recv0[0]["values"] == [1.0, 2.0]

        # Receiver on GPU 1 should have received from sender 1
        recv1 = results[3]["received"]
        assert len(recv1) >= 1
        assert recv1[0]["name"] == "gpu1_w"
        assert recv1[0]["values"] == [3.0, 4.0]


@requires_cuda
class TestSmallBuffer:
    """Test with very small buffer to force many flush rounds."""

    def test_many_flushes(self):
        # Buffer = 32 bytes = 8 float32 elements
        # Send 5 tensors of 4 elements each -> 5 * 16 = 80 bytes -> multiple flushes
        tensors = [
            (f"t{i}", [float(i)] * 4, torch.float32, (4,))
            for i in range(5)
        ]
        received = _run_transfer(tensors, buffer_size=32)

        assert len(received) == 5
        for i, r in enumerate(received):
            assert r["name"] == f"t{i}"
            assert r["values"] == [float(i)] * 4


# ── Merged data + is_end test ────────────────────────────────────────────────


def _receiver_proc_bucket_meta(rank, world_size, gpu_id, gloo_port, src_rank,
                               buffer_size, result_queue):
    """Receiver that records per-bucket metadata (tensor count + is_end flag)."""
    torch.cuda.set_device(gpu_id)
    try:
        from coda.transfer_mesh.channel import create_channel, Role
        ch = create_channel(
            master_addr="127.0.0.1", addr="127.0.0.1",
            gpu_id=gpu_id, world_size=world_size, rank=rank,
            role=Role.RECEIVER, src_rank=src_rank,
            buffer_size_bytes=buffer_size,
            gloo_port=gloo_port,
        )
        buckets = []
        for payload, meta in ch.get_iterator(yield_buckets=True):
            buckets.append({
                "num_specs": len(meta.tensor_specs),
                "spec_names": [s.name for s in meta.tensor_specs],
                "is_end": meta.is_end,
            })
        ch.close()
        result_queue.put({"rank": rank, "success": True, "buckets": buckets})
    except Exception as e:
        import traceback
        result_queue.put({"rank": rank, "success": False, "error": str(e),
                          "tb": traceback.format_exc()})


@requires_cuda
class TestMergedEndWithData:
    """Verify that the last data bucket and is_end are merged into one MetaFrame."""

    def test_data_and_end_in_single_frame(self):
        """All tensors fit in buffer -> send(None) flushes data+is_end together.

        The receiver uses yield_buckets=True and checks that the final
        (and only) bucket carries both tensor_specs AND is_end=True.
        """
        tensors = [
            ("w1", [1.0, 2.0], torch.float32, (2,)),
            ("w2", [3.0, 4.0], torch.float32, (2,)),
            ("w3", [5.0, 6.0], torch.float32, (2,)),
        ]

        gloo_port = _get_free_port()
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        sender = ctx.Process(
            target=_sender_proc,
            args=(0, 2, 0, gloo_port, tensors, 64 * 1024 * 1024, result_queue),
        )
        receiver = ctx.Process(
            target=_receiver_proc_bucket_meta,
            args=(1, 2, 0, gloo_port, 0, 64 * 1024 * 1024, result_queue),
        )

        sender.start()
        receiver.start()

        results = _drain_results(
            result_queue, [sender, receiver], [0, 1], timeout=60
        )

        buckets = results[1]["buckets"]

        # All 3 tensors fit in one 64MB buffer -> exactly 1 bucket
        assert len(buckets) == 1, f"Expected 1 bucket, got {len(buckets)}"

        # That single bucket should carry all data AND is_end=True
        assert buckets[0]["num_specs"] == 3
        assert buckets[0]["spec_names"] == ["w1", "w2", "w3"]
        assert buckets[0]["is_end"] is True, (
            "Expected data and is_end to be merged in a single MetaFrame"
        )
