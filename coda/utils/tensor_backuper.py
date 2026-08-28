# Adapted from
# https://github.com/THUDM/slime/blob/6961f5970e9dbb4716a10ba4a54a28fa3876d274/slime/utils/tensor_backper.py
# Copyright 2025 Zhipu AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#
# Modifications: keeps only the pinned-memory backup path (slime's
# ``_TensorBackuperNormal``) as a single concrete class, drops the abstract base
# class and the no-op variant, and resizes the parameter storage in ``restore``
# so the class also works with offloaded parameters.
"""Tensor backup/restore utility for multi-model weight switching.

Manages multiple sets of model weights using a single GPU model with
CPU pinned-memory backups. Only one model variant occupies GPU at a
time; switching is done by overwriting GPU params from CPU backup.

Usage:
    backuper = TensorBackuper(
        source_getter=lambda: model.named_parameters(),
    )
    backuper.backup("model_a")   # GPU -> CPU snapshot
    backuper.restore("model_b")  # CPU -> GPU load
"""

from collections import defaultdict
from collections.abc import Callable, Iterable

import torch

# Returns an iterable of (name, tensor) pairs from the live GPU model.
_SourceGetter = Callable[[], Iterable[tuple[str, torch.Tensor]]]


class TensorBackuper:
    """Tag-based model weight backup/restore using CPU pinned memory."""

    def __init__(self, source_getter: _SourceGetter):
        self._source_getter = source_getter
        # tag -> {param_name -> cpu_pinned_tensor}
        self._backups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

    @property
    def backup_tags(self):
        """List of tags that have been backed up."""
        return list(self._backups)

    def get(self, tag: str):
        """Get the CPU tensor dict for a given tag."""
        return self._backups[tag]

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        """Copy GPU params -> CPU pinned memory (async + sync)."""
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            if name not in backup_dict:
                # Allocate pinned CPU buffer on first backup
                backup_dict[name] = torch.empty_like(
                    param, device=torch.device("cpu"), pin_memory=True
                )
            backup_dict[name].copy_(param.detach(), non_blocking=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def copy(self, *, src_tag: str, dst_tag: str):
        """CPU-to-CPU copy between backup slots."""
        for name in self._backups[dst_tag]:
            self._backups[dst_tag][name].copy_(self._backups[src_tag][name])

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        """Copy CPU pinned memory -> GPU params (async + sync)."""
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            assert name in backup_dict
            backup = backup_dict[name]
            if param.data.storage().size() < backup.numel():
                param.data.storage().resize_(backup.numel())
            param.copy_(backup, non_blocking=True)
        torch.cuda.synchronize()
