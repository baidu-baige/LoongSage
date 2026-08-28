""" Distributed utils """
import torch.distributed as dist

GLOO_GROUP = None


def init_gloo_group(timeout=None):
    """Initialize Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        GLOO_GROUP = dist.new_group(backend="gloo", timeout=timeout)
    return GLOO_GROUP


def get_gloo_group():
    """Get the Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        raise RuntimeError("Gloo group has not been initialized. Call init_gloo_group() first.")
    return GLOO_GROUP