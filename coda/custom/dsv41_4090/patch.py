"""Hooks the streaming recipe into the stock controller without editing it.

Every patched method checks ``dsv41_4090.stream`` in its own config and otherwise runs the
original, so other recipes are unaffected:

  TrainManager.__init__                builds MegatronStreamActorWorker (and validates the config
                                       with it) instead of MegatronActorWorker
  TrainManager.async_update_weights    no channel push: the stream worker already wrote the
                                       merged weights to hf_model_path during train()
  RolloutManager.async_update_weights  the engines reload hf_model_path from disk; the first call
                                       (the initial sync) only sets the weight version, since
                                       the engines started from that checkpoint

The controller modules are only patched in the process that already imported them (the training
entry point imports them before ``coda.custom``); Ray worker processes are left alone.
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

_MARK = "_dsv41_4090_patched"


def enabled(config) -> bool:
    section = config.get("dsv41_4090") if config is not None else None
    return bool(section is not None and section.get("stream", False))


def _patch_train_manager(tm):
    cls = tm.TrainManager
    if getattr(cls, _MARK, False):
        return
    orig_init, orig_update = cls.__init__, cls.async_update_weights

    def __init__(self, config, resource_scheduler):
        if not enabled(config):
            return orig_init(self, config, resource_scheduler)
        from coda.custom.dsv41_4090.stream_worker import MegatronStreamActorWorker

        stock = tm.MegatronActorWorker
        tm.MegatronActorWorker = MegatronStreamActorWorker
        try:
            orig_init(self, config, resource_scheduler)
        finally:
            tm.MegatronActorWorker = stock

    def async_update_weights(self, channel_meta):
        if not enabled(self.config):
            return orig_update(self, channel_meta)
        return []

    cls.__init__, cls.async_update_weights = __init__, async_update_weights
    setattr(cls, _MARK, True)


def _patch_rollout_manager(rm):
    cls = rm.RolloutManager
    if getattr(cls, _MARK, False):
        return
    orig_update = cls.async_update_weights

    def async_update_weights(self, meta, weight_version):
        if not enabled(self.config):
            return orig_update(self, meta, weight_version)
        engines = [engine for replica in self.replica_groups for engine in replica.engines]
        version = str(weight_version)
        if not getattr(self, "_dsv41_4090_synced", False):
            self._dsv41_4090_synced = True
            payload = {"new_version": version, "abort_all_requests": False}
            return [e._make_request.remote("update_weight_version", payload, 60.0) for e in engines]
        model_path = str(self.config.hf_model_path)
        payload = {"model_path": model_path, "load_format": "auto", "weight_version": version}
        logger.info("dsv41_4090: rollout engines reload %s (version %s)", model_path, version)
        # an engine actor runs its calls in submission order: flush, then reload
        return [ref for e in engines
                for ref in (e._flush_cache.remote(),
                            e._make_request.remote("update_weights_from_disk", payload, 1800.0))]

    cls.async_update_weights = async_update_weights
    setattr(cls, _MARK, True)


def apply():
    tm = sys.modules.get("coda.controller.train_manager")
    rm = sys.modules.get("coda.controller.rollout_manager")
    if tm is not None:
        _patch_train_manager(tm)
    if rm is not None:
        _patch_rollout_manager(rm)


apply()
