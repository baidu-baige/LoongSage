""" Memory utils """
import gc
import logging

import torch
import torch.distributed as dist
import psutil

logger = logging.getLogger(__name__)


def clear_memory():
    """ Clear cuda memory cache """
    # First, collect Python garbage so unreferenced tensors are freed
    gc.collect()
    # Synchronize all CUDA streams to ensure pending ops complete before releasing memory
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def available_memory():
    """Collect CUDA memory stats plus a host-RSS breakdown."""
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
        "host_rss_GB": _byte_to_gb(_host_rss()),
        **_host_rss_breakdown(),
    }


def _host_rss_breakdown():
    """Split ``host_rss_GB`` into the pools that need different fixes (~1.5 ms).

    * ``rss_anon_GB`` — malloc heap plus anonymous ``mmap``. A per-step climb here
      is the one that needs real investigation.
    * ``rss_shmem_GB`` — shared memory: Ray plasma, ``/dev/shm`` checkpoint
      staging, and torch's page-locked host blocks.
    * ``rss_pinned_alloc_GB`` — page-locked host memory owned by torch's caching host
      allocator, **in use plus cached**. 
      ``allocated_bytes.current`` does NOT decrement on free, so this single field
      already is the pool size, and only ``torch._C._host_emptyCache()`` brings it
      down. 
    """
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in ("RssAnon", "RssShmem"):
                    out[f"rss_{key[3:].lower()}_GB"] = _byte_to_gb(int(rest.split()[0]) * 1024)
    except (OSError, ValueError, IndexError):
        pass
    stats = torch.cuda.memory.host_memory_stats()
    if "allocated_bytes.current" in stats:
        out["rss_pinned_alloc_GB"] = _byte_to_gb(stats["allocated_bytes.current"])
    return out

def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)

def _host_rss():
    return psutil.Process().memory_info().rss

def print_memory(msg):
    """ Print cuda memory stats"""
    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    logger.info(
        f"[Rank {rank}] Memory-Usage {msg}: {memory_info}"
    )
    return memory_info
