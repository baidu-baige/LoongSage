"""Checkpoint path and tracker utilities (backend-agnostic)."""

import os
import logging

import torch.distributed as dist

logger = logging.getLogger(__name__)

LATEST_ITERATION_FILE = "latest_checkpointed_iteration.txt"
DIST_CKPT_CONFIG_FILE = "metadata.json"


def get_ckpt_dir(checkpoint_base_dir, step):
    """Build the dist_ckpt directory path for a given step."""
    return os.path.join(checkpoint_base_dir, f"train_step_{step}", "dist_ckpt")


def get_hf_dir(checkpoint_base_dir, step):
    """Build the hf_model directory path for a given step."""
    return os.path.join(checkpoint_base_dir, f"train_step_{step}", "hf_model")


def get_data_source_dir(checkpoint_base_dir, step):
    """Build the data_source directory path for a given step.

    Holds the per-datasource cursor state_dict used for resuming, colocated
    with the model checkpoint under the same ``train_step_{step}`` directory.
    """
    return os.path.join(checkpoint_base_dir, f"train_step_{step}", "data_source")


def find_latest_ckpt_path(checkpoint_base_dir):
    """Locate the latest checkpoint by tracker_file, return path or None."""
    if not os.path.isdir(checkpoint_base_dir):
        return None
    latest_file = get_tracker_file(checkpoint_base_dir)
    if not os.path.isfile(latest_file):
        return None
    try:
        with open(latest_file, "r") as f:
            step = f.read().strip()
        ckpt_dir = get_ckpt_dir(checkpoint_base_dir, step)
        if not os.path.isdir(ckpt_dir):
            logger.warning(f"Tracker file points to non-existent checkpoint dir: {ckpt_dir}")
            return None
        return ckpt_dir
    except (IOError, OSError) as e:
        logger.warning(f"Failed to read tracker file: {e}")
    return None


def resolve_dist_ckpt_dir(path, config_key):
    """Validate a configured dist checkpoint dir for the ref model / OPD teachers."""
    if not path:
        return None
    example = "<run_dir>/train_step_100/dist_ckpt"
    if not os.path.isdir(path):
        raise ValueError(
            f"{config_key}='{path}' is not a directory. It must name a Megatron "
            f"dist checkpoint dir, e.g. {example}"
        )
    if not os.path.isfile(os.path.join(path, DIST_CKPT_CONFIG_FILE)):
        raise ValueError(
            f"{config_key}='{path}' is not a Megatron dist checkpoint "
            f"(no {DIST_CKPT_CONFIG_FILE}). It must name the dist_ckpt dir "
            f"itself, e.g. {example}"
        )
    return os.path.normpath(path)


def update_latest(checkpoint_base_dir, step):
    """Write the step to the tracker file (rank 0 only)."""
    if dist.get_rank() == 0:
        latest_file = get_tracker_file(checkpoint_base_dir)
        os.makedirs(checkpoint_base_dir, exist_ok=True)
        with open(latest_file, "w") as f:
            f.write(str(step))
        logger.info(f"Updated 'latest' to step={step}")


def get_tracker_file(checkpoint_base_dir):
    """Build the path to the checkpoint tracker file in the base directory."""
    return os.path.join(checkpoint_base_dir, LATEST_ITERATION_FILE)
