"""Data source abstractions for the RL rollout pipeline.

Defines the abstract interface `DataSource` and two concrete implementations:

- RolloutDataSource: Sequentially streams prompts from a `Dataset` file, assigning group/trajectory indices and
                     wrapping epochs.
- RolloutDataSourceWithBuffer: Extends `RolloutDataSource` with an in-memory replay buffer so that leftover or
                               externally generated trajectories can be re-consumed before fetching new prompts.

Built-in buffer replay strategies:
    fifo: Default buffer-drain function, pops up to *num* oldest groups; registered as fifo in the package-level
          BUFFER_REPLAY_STRATEGY_REGISTRY.

Extending buffer replay strategies:
    Use the @register_buffer_replay_strategy("name") decorator to register a custom strategy, then reference its name
    via config.data_sources[].dataset.buffer_replay_strategy.
"""
import abc
import copy
import glob
import logging
import os

import torch

from coda.agentflow.trajectory_store import Trajectory, TrajectoryGroup
from coda.data_factory import (
    get_buffer_replay_strategy,
    get_data_pre_processor,
    register_buffer_replay_strategy,
)
from coda.data_factory.dataset import Dataset
from coda.utils.checkpoint_utils import get_data_source_dir

logger = logging.getLogger(__name__)

# Separator used when concatenating segments of prompt_id and trajectory_id.
_ID_SEP = "_"


class DataSource(abc.ABC):
    """Abstract base class for all rollout data sources.

    A `DataSource` manages the lifecycle of `Trajectory` groups that feed the rollout engine. Subclasses must implement
    the five abstract methods below.

    A *trajectory group* is a TrajectoryGroup where every trajectory shares the same prompt but may diverge during
    generation (e.g. multiple rollouts per prompt).The list[TrajectoryGroup] returned by get() therefore has
    length *num*, each TrajectoryGroup having num_trajectories_per_prompt trajectories.
    """

    @abc.abstractmethod
    def get(self, num: int) -> list[TrajectoryGroup]:
        """Return *num* trajectory groups ready for rollout.

        Args:
            num: Number of prompt groups to return.

        Returns:
            A list of *num* TrajectoryGroup objects, each containing num_trajectories_per_prompt trajectories.
        """

    @abc.abstractmethod
    def add(self, groups: list[TrajectoryGroup]):
        """Inject externally produced trajectory groups into this data source.

        Not all implementations support this operation.  Read-only sources raise `RuntimeError` when called.

        Args:
            groups: List of TrajectoryGroup objects to add.
        """

    @abc.abstractmethod
    def save(self, step):
        """Persist the current iteration state to disk.

        Args:
            step: Training step identifier used to name the checkpoint file.
        """

    @abc.abstractmethod
    def load(self, step=None):
        """Restore iteration state from a previously saved checkpoint.

        Args:
            step: Training step of the checkpoint to restore.  When `None` the implementation may choose a default
                  (e.g. the latest available checkpoint).
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of prompts currently available.

        The value may change across calls when groups are added or consumed
        (e.g. after add or epoch wrapping).
        """


class RolloutDataSource(DataSource):
    """Read-only data source that streams prompts from a `Dataset`.

    Iterates through the underlying `Dataset` sequentially, assigning each `Trajectory` an ID that encodes its epoch,
    training step, and a per-epoch prompt counter.  When the end of the dataset is reached the source wraps around
    to a new epoch (optionally re-shuffling).

    Args:
        ds_config: A single data source unit config dict/DictConfig with keys:
                   dataset, agent, reward, max_response_len_per_trajectory,
                   num_trajectories_per_prompt, num_prompts_per_step.
        global_config: The top-level config (for checkpoint_path, seed, etc.).
        ds_index: Index of this data source, used in the trajectory ID prefix.
        is_eval: Read the eval split (`eval_prompt_data_path`) instead of the train split.

    Expected `ds_config.dataset` fields
    ------------------------------------
    - `prompt_data_path` (str):         Path to the dataset file, or to a directory of dataset files.
    - `eval_prompt_data_path` (str):    Path to the eval split, used when `is_eval` is set.
    - `max_prompt_len` (int | None):    Max prompt character length.
    - `input_key` (str):                Dataset column for the prompt text.
    - `label_key` (str | None):         Dataset column for ground-truth labels.
    - `metadata_key` (str):             Dataset column for per-row metadata dicts.
    - `data_pre_processor` (str|None):  Registered name of a raw-record pre-processor
                                        (see `coda.data_factory.data_pre_processor`).  null = disabled.
    - `shuffle` (bool):                 Whether to shuffle between epochs.
    """

    def __init__(self, ds_config, global_config, ds_index: int = 0, is_eval: bool = False):
        self.config = global_config
        self.ds_config = ds_config
        self.ds_index = ds_index
        self.is_eval = is_eval
        self.dataset_config = ds_config.dataset

        self.epoch_id = 0
        self.step = 0
        self.prompt_offset = 0
        self.prompt_index = 0
        self.trajectory_index = 0
        self.trajectory_count = 0

        pre_processor_name = self.dataset_config.data_pre_processor
        self.dataset = Dataset(
            self.dataset_config.eval_prompt_data_path if is_eval else self.dataset_config.prompt_data_path,
            max_length=self.dataset_config.max_prompt_len,
            prompt_key=self.dataset_config.input_key,
            label_key=self.dataset_config.label_key,
            metadata_key=self.dataset_config.metadata_key,
            seed=global_config.seed,
            data_pre_processor=get_data_pre_processor(pre_processor_name) if pre_processor_name else None,
        )

        if self.dataset_config.shuffle:
            self.dataset.shuffle(self.epoch_id)

    def get(self, num):
        """Return *num* trajectory groups from the dataset.

        Delegates to `_fetch_prompts` to advance the read position, then to `_build_trajectory_groups` to replicate
        each prompt into `num_trajectories_per_prompt` copies.

        Args:
            num: Number of prompt groups to return.

        Returns:
            List of *num* trajectory groups.
        """
        prompts = self._fetch_prompts(num)
        return self._build_trajectory_groups(prompts)

    def _fetch_prompts(self, num: int) -> list:
        """Fetch *num* prompts from the dataset, wrapping epochs as needed."""
        dataset_size = len(self.dataset)
        available = dataset_size - self.prompt_offset

        if num <= available:
            prompts = self.dataset.prompts[self.prompt_offset : self.prompt_offset + num]
            self.prompt_offset += num
            self._assign_prompt_and_trajectory_id_prefix(prompts)
        else:
            # Consume the tail of the current epoch (still old epoch_id). Deep-copy immediately so that head items
            # from the next epoch cannot overwrite the epoch prefix on shared dataset objects.
            tail = [copy.deepcopy(s) for s in self.dataset.prompts[self.prompt_offset :]]
            self._assign_prompt_and_trajectory_id_prefix(tail)

            # Start a new epoch.
            remaining = num - len(tail)
            self.epoch_id += 1
            self.prompt_index = 0
            if self.dataset_config.shuffle:
                self.dataset.shuffle(self.epoch_id)

            head = list(self.dataset.prompts[:remaining])
            self.prompt_offset = remaining
            self._assign_prompt_and_trajectory_id_prefix(head)

            prompts = tail + head

        return prompts

    def _assign_prompt_and_trajectory_id_prefix(self, prompts: list) -> None:
        """Stamp each prompt with the current epoch and step as prompt_id / trajectory_id prefix.

        The prefix format is epoch{epoch_id}_step{step}_ds{ds_index}[_eval]_. Both prompt_id and
        trajectory_id are set to this value so that the suffix methods can append the group- and
        trajectory-level segments independently. The ``eval`` marker keeps eval ids from
        colliding with the training prompts of the same source and step.
        """
        for prompt in prompts:
            prompt.ds_index = self.ds_index
            prompt.is_eval = self.is_eval
            eval_tag = f"{_ID_SEP}eval" if self.is_eval else ""
            prompt.prompt_id = prompt.trajectory_id = (
                f"epoch{self.epoch_id}{_ID_SEP}step{self.step}{_ID_SEP}ds{self.ds_index}{eval_tag}{_ID_SEP}"
            )

    def _build_trajectory_groups(self, prompts: list) -> list[TrajectoryGroup]:
        """Expand each prompt into a group of *num_trajectories_per_prompt* trajectory copies."""
        traj_groups = []
        for prompt in prompts:
            trajectories = []
            self.trajectory_index = 0
            for _ in range(self.ds_config.num_trajectories_per_prompt):
                traj = copy.deepcopy(prompt)
                self._assign_prompt_and_trajectory_id_suffix(
                        traj, self.prompt_index, self.trajectory_index, self.trajectory_count)
                self.trajectory_index += 1
                self.trajectory_count += 1
                trajectories.append(traj)

            self.prompt_index += 1
            traj_groups.append(TrajectoryGroup(prompt_id=trajectories[0].prompt_id, trajectories=trajectories))

        return traj_groups

    def _assign_prompt_and_trajectory_id_suffix(
            self,
            traj,
            prompt_index,
            trajectory_index,
            trajectory_count
        ) -> None:
        """Stamp each trajectory with a unique (within the current epoch) id."""
        traj.prompt_id += f"prompt{prompt_index}"
        traj.trajectory_id += (
            f"prompt{prompt_index}{_ID_SEP}trajectory{trajectory_index}{_ID_SEP}count{trajectory_count}"
        )

    def add(self, groups: list[TrajectoryGroup]):
        """Not supported — `RolloutDataSource` is read-only.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(
            f"Cannot add groups to {self.__class__.__name__}. This is a read-only data source."
        )

    def _state_dict_path(self, step) -> str:
        """Return the on-disk path of this datasource's persisted state_dict for *step*."""
        return os.path.join(
            get_data_source_dir(self.config.checkpoint_path, step),
            f"global_dataset_state_dict_ds{self.ds_index}.pt",
        )

    def save(self, step):
        """Persist iteration counters to a `.pt` checkpoint file."""
        state_dict = {
            "prompt_data_path": self.dataset_config.prompt_data_path,
            "epoch_id":      self.epoch_id,
            "step":          self.step,
            "prompt_offset": self.prompt_offset,
            "prompt_index":     self.prompt_index,
            "trajectory_count": self.trajectory_count,
        }
        path = self._state_dict_path(step)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, step=None):
        """Restore iteration counters from a previously saved checkpoint.

        Datasource indices may change when the configuration is reordered, so
        every datasource checkpoint under ``train_step_{step}/data_source`` is
        inspected until its stored ``prompt_data_path`` matches this
        datasource. Returns the matching raw state_dict (or ``None`` when no
        match exists) so subclasses can extend the payload with additional
        keys (e.g. buffer contents).
        """
        pattern = os.path.join(
            get_data_source_dir(self.config.checkpoint_path, step),
            "global_dataset_state_dict_*.pt",
        )
        prompt_data_path = self.dataset_config.prompt_data_path
        state_dict = None
        matched_path = None
        for path in sorted(glob.glob(pattern)):
            candidate = torch.load(path)
            if candidate.get("prompt_data_path") == prompt_data_path:
                state_dict = candidate
                matched_path = path
                break

        if state_dict is None:
            logger.warning(
                "Dataset state checkpoint not found for prompt_data_path=%s at step=%s under %s; "
                "starting this datasource from a fresh cursor.",
                prompt_data_path, step, os.path.dirname(pattern),
            )
            return None

        logger.info("Loading dataset state from %s", matched_path)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.step = state_dict.get("step", 0)
        self.prompt_offset = state_dict.get("prompt_offset", 0)
        self.prompt_index = state_dict.get("prompt_index", 0)
        self.trajectory_count = state_dict.get("trajectory_count", 0)

        if self.dataset_config.shuffle:
            self.dataset.shuffle(self.epoch_id)

        return state_dict

    def __len__(self) -> int:
        """Return the total number of prompts in the underlying dataset."""
        return len(self.dataset)


class RolloutDataSourceWithBuffer(RolloutDataSource):
    """A data source that combines an experience replay buffer with the prompt dataset.

    This class acts as a *dataloader + replay buffer*: each call to `get` first drains groups from the
    in-memory buffer, and only falls back to the parent `RolloutDataSource` (i.e. the prompt dataset) when the buffer
    cannot satisfy the full request.

    Typical buffer contents
    -----------------------
    - Leftover groups from a partial rollout (e.g. the generator was interrupted before the full batch was consumed).
    - Unused groups from the previous training step (oversampling surplus).
    - Candidate trajectories selected by an external policy (e.g. priority replay or curriculum selection).

    Sampling flow
    -------------
    1. Ask the pluggable `buffer_replay_strategy` function to pop up to *num* groups from self.buffer.
    2. If the buffer supplied fewer groups than requested, fetch the remainder from the underlying dataset via
       `super().get()`.

    This design enables advanced sampling strategies such as:

    - *Dynamic sampling*: inject externally generated trajectories at any point during training.
    - *Partial rollout*: resume an incomplete rollout batch by re-queuing leftover groups into the buffer.
    - *Oversampling / priority replay*: accumulate more rollouts than consumed per step and replay high-value groups in
      later steps.

    Attributes:
        buffer:                 List of trajectory groups (`list[TrajectoryGroup]`) waiting to be consumed.
        buffer_replay_strategy: Callable with signature (config, step, buffer, num) -> list[TrajectoryGroup]
                                that decides which groups to pop from the buffer.  Defaults to the `"fifo"` strategy.
                                A custom strategy can be supplied by setting
                                `data_sources[].dataset.buffer_replay_strategy` to its registered name
                                (see `BUFFER_REPLAY_STRATEGY_REGISTRY` in `coda.data_factory`).
    """

    def __init__(self, ds_config, global_config, ds_index: int = 0):
        super().__init__(ds_config, global_config, ds_index=ds_index)
        self.buffer = []
        strategy_name = self.dataset_config.buffer_replay_strategy or "fifo"
        self.buffer_replay_strategy = get_buffer_replay_strategy(strategy_name)

    def get(self, num: int, step: int) -> list[TrajectoryGroup]:
        """Return *num* trajectory groups, preferring buffer over dataset.

        Drains up to *num* groups from the buffer first.  Any shortfall is filled by delegating to the
        parent dataset source.  The current training step is recorded as `self.step` so that subsequent calls
        to `_assign_prompt_and_trajectory_id_prefix` embed it in every new prompt's ID.

        Args:
            num:  Number of trajectory groups to return.
            step: Current training step index.  Stored as `self.step` and stamped into
                  the `prompt_id` / `trajectory_id` prefix of any newly fetched prompts.

        Returns:
            List of `num` trajectory groups.
        """
        self.step = step
        groups = self._get_from_buffer(num)
        num -= len(groups)

        if num == 0:
            return groups

        groups += super().get(num=num)
        return groups

    def _get_from_buffer(self, num: int) -> list[TrajectoryGroup]:
        """Pop up to *num* groups from the buffer via `buffer_replay_strategy`."""
        if len(self.buffer) == 0 or num == 0:
            return []

        groups = self.buffer_replay_strategy(self.config, self.step, self.buffer, num)
        return groups

    def add(self, groups: list[TrajectoryGroup]):
        """Append trajectory groups to the buffer for future consumption.

        Args:
            groups: List of TrajectoryGroup objects to enqueue.

        Raises:
            AssertionError: If *groups* is not a list of TrajectoryGroup, or if any group's trajectory count does not
                            equal num_trajectories_per_prompt.
        """
        if not groups:
            return

        assert isinstance(groups, list), f"groups must be a list, got {type(groups)}"
        assert isinstance(groups[0], TrajectoryGroup), (
            f"the elements of groups must be TrajectoryGroup, got {type(groups[0])}"
        )

        for i in range(0, len(groups)):
            expected = self.ds_config.num_trajectories_per_prompt
            actual = len(groups[i].trajectories)

            assert actual == expected, (
                f"the length of the elements of groups must be equal to num_trajectories_per_prompt, "
                f"got {actual} != {expected}"
            )

            self.buffer.append(groups[i])

    def get_buffer_length(self):
        """Return the number of trajectory groups currently held in the buffer."""
        return len(self.buffer)

    def save(self, step, additional_groups: list[TrajectoryGroup] | None = None):
        """Persist counters and a snapshot of prompts waiting to be consumed.

        The buffer typically holds groups that were already rolled out but were not
        consumed by the trainer before the checkpoint boundary (e.g. `pipeline_buf`
        overflow in fully_async mode, or oversampling surplus in dynamic sampling).
        Only prompt-level information is preserved — trajectory tokens / log_probs /
        rewards are dropped and will be regenerated after restore.

        ``additional_groups`` are included only in the persisted snapshot. They
        are not appended to the live buffer, so saving a checkpoint cannot cause
        them to be consumed twice if training continues.
        """
        groups_to_save = [*self.buffer, *(additional_groups or [])]
        state_dict = {
            "prompt_data_path":  self.dataset_config.prompt_data_path,
            "epoch_id":         self.epoch_id,
            "step":             self.step,
            "prompt_offset":    self.prompt_offset,
            "prompt_index":     self.prompt_index,
            "trajectory_count": self.trajectory_count,
            "unused_prompt_list": [
                {
                    "prompt_id": g.prompt_id,
                    "prompt":    g.trajectories[0].prompt,
                    "label":     g.trajectories[0].label,
                    "metadata":  g.trajectories[0].metadata,
                }
                for g in groups_to_save
            ],
        }
        path = self._state_dict_path(step)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, step=None):
        """Restore counters and reconstruct buffer groups from persisted prompt info.

        For every entry in ``unused_prompt_list`` we rebuild a full group of
        ``num_trajectories_per_prompt`` Trajectory copies. Each new trajectory is
        assigned a fresh ``trajectory_id`` whose ``count`` suffix continues from the
        current ``self.trajectory_count`` and is bumped once per trajectory so that
        ids remain unique across restart.
        """
        state_dict = super().load(step)
        if state_dict is None:
            return None

        unused = state_dict.get("unused_prompt_list", [])
        rollout_n = int(self.ds_config.num_trajectories_per_prompt)
        for prompt_info in unused:
            prompt_id = prompt_info["prompt_id"]
            trajectories = []
            for ti in range(rollout_n):
                traj = Trajectory(
                    prompt_id=prompt_id,
                    trajectory_id=(
                        f"{prompt_id}{_ID_SEP}trajectory{ti}"
                        f"{_ID_SEP}count{self.trajectory_count}"
                    ),
                    ds_index=self.ds_index,
                    prompt=copy.deepcopy(prompt_info["prompt"]),
                    label=copy.deepcopy(prompt_info["label"]),
                    metadata=copy.deepcopy(prompt_info["metadata"]),
                )
                self.trajectory_count += 1
                trajectories.append(traj)
            self.buffer.append(TrajectoryGroup(prompt_id=prompt_id, trajectories=trajectories))

        logger.info(
            "Restored %d group(s) for datasource %s",
            len(unused), self.dataset_config.prompt_data_path,
        )
        return state_dict


@register_buffer_replay_strategy("fifo")
def fifo(
    config, step, buffer: list[TrajectoryGroup], num: int
) -> list[TrajectoryGroup]:
    """Remove and return up to *num* oldest groups from *buffer* (FIFO).

    This is the default `buffer_replay_strategy` used by `RolloutDataSourceWithBuffer`. It mutates *buffer* in-place.

    Args:
        config: Training config namespace (unused; present for API compatibility).
        step:   Current training step (unused; present for API compatibility).
        buffer: Mutable list of trajectory groups to drain from.
        num:    Maximum number of groups to pop.

    Returns:
        List of up to *num* groups removed from the front of *buffer*.
    """
    num_to_pop = min(len(buffer), num)
    groups = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return groups
