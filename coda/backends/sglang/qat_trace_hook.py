"""SGLang side of rollout-guided activation QAT (``trainer.qat_trace``, see coda/utils/qat_trace.py).

Installed through SGLang's ``forward_hooks`` server arg with ``"capture": true``, i.e. attached
before CUDA graph capture (docker/patch/sglang-0.5.23.patch). ``make_hook`` runs once per scheduler
process: it wraps ``ModelRunner._forward_raw`` (forward boundaries) and the output streamer's
``build_additional_customized_info`` (request finish), connects to the Mooncake Store and returns a
forward hook that, for each FP8 linear of a traced layer, packs the low code bits of the e4m3 input
its GEMM uses.

* Eager forward (prefill, decode without graph): the hook packs the codes of the batch rows.
* CUDA graph capture: the hook packs every row of the captured batch into a static per-module
  buffer, so a replay rewrites it; after a replay (``can_run_graph``) its first n rows are the batch.
* Either way ``end()`` joins the modules into one [n, bytes] tensor. With attention TP > 1 every
  rank then all-gathers its K-sharded columns over the attention TP device group (NCCL), so rank 0
  holds whole rows; rank 0 copies them to pinned host memory without blocking, and a worker thread
  keys the rows (prefix hash per request) and buffers them per ``rid``. Rows get a key only if the
  request prefilled its whole prefix itself.
* When a request finishes, rank 0 puts the rows of the request that no earlier request of the rid
  put as one object and returns its key in ``meta_info["qat_trace"]`` before the response leaves
  the server. The other ranks keep no records.

Replicated inputs are recorded by attention TP rank 0 only, K-sharded inputs (row-parallel
linears) by every rank. Failures never take the rollout down: recording stops and the trainer
falls back to round-to-nearest for the missing rows.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
import uuid
import weakref
from collections import Counter

import numpy as np
import torch

from coda.utils import qat_trace as qt

logger = logging.getLogger(__name__)

INFO_KEY = "qat_trace"  # request-level customized_info / meta_info key
_STAGING_BYTES = 256 << 20  # put staging buffer; grows to the largest batch of objects
_CATCH_UP_S = 60.0  # longest wait for the worker thread at a request finish
_RID_IDLE_S = 3600.0  # drop the state of a rid untouched for this long
_LOG_EVERY = 100  # finished requests between two stats lines (tp 0)
_UNSET = object()

_recorder = None


def make_hook(config: dict):
    """SGLang ``forward_hooks`` factory (one recorder per process)."""
    global _recorder
    if _recorder is None:
        _recorder = _Recorder(config)
        _recorder.install()
    return _recorder.hook


def _attn_tp_group():
    """SGLang's attention TP group (the dense linears are sharded over it)."""
    from sglang.srt.distributed.parallel_state import get_attn_tp_group

    return get_attn_tp_group()


def _check_graph_config() -> None:
    """Prefill must run eager; a decode graph must be a full graph (no eager pieces inside replay)."""
    from sglang.srt.runtime_context import get_exec

    cfg = get_exec().graph.cuda_graph_config
    if cfg is None:
        return
    if cfg.prefill.backend != "disabled":
        raise RuntimeError(f"[QAT trace] needs the prefill CUDA graph disabled (got {cfg.prefill.backend}); "
                           "set rollout.sglang_args.disable_prefill_cuda_graph: true")
    if cfg.decode.backend not in ("full", "disabled"):
        raise RuntimeError(f"[QAT trace] the decode CUDA graph backend must be full (got {cfg.decode.backend})")


class _Recorder:
    def __init__(self, config: dict):
        self.bits = int(config["bits"])
        self.layers = sorted(int(i) for i in config["layers"])
        self.mc = config["mooncake"]
        self.max_host_bytes = int(self.mc["max_host_gb"] * (1 << 30))
        self.group = _attn_tp_group()
        self.tp_rank, self.tp_size = int(self.group.rank_in_group), int(self.group.world_size)
        self.inst = uuid.uuid4().hex[:12]  # recorder id in the object keys (rank 0's is the one used)
        self.store = None
        # modules (fixed after the first complete forward)
        self.kind: dict[int, object] = {}  # id(module) -> sharded flag, or None (not recorded)
        self.slot_of: dict[int, int] = {}  # id(module) -> slot (module whose input is recorded)
        self.alias_of: dict[int, int] = {}  # id(module) -> id(module) fed the same input tensor
        self.slot_mod: list[int] = []
        self.slot_rb: list[int] = []
        self.slot_sharded: list[bool] = []
        self.static: list[torch.Tensor | None] = []  # per slot, written inside CUDA graphs
        self.frozen = False
        self.names: dict[int, str] = {}
        self.broken: str | None = None
        # K-sharded columns gathered to rank 0: None until the ranks agreed on their layout, then
        # [(name, row bytes)] of the sharded slots (the same order on every rank)
        self.gather_layout: list | None = None
        self.gather_tries = 0
        # running forward (scheduler thread)
        self.cur = None
        self.fwd_seen: dict = {}
        self.fwd_mods: set[int] = set()
        self.enq_seq = 0  # forwards handed to the worker
        self.inflight: collections.deque = collections.deque()  # (seq, cuda event) not known done
        self.ready_seq = 0  # every forward below it has completed on the GPU
        self.finished = 0  # finished requests (object sequence number)
        # host side, shared with the worker thread under ``lock``
        self.queue: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.done_cv = threading.Condition(self.lock)
        self.done_seq = 0  # forwards the worker has ingested
        self.reqs: dict[str, list] = {}  # rid -> [next position, prefix hash]
        self.rows: dict[str, list] = {}  # rid -> [(positions int64[m], keys int64[m], rows uint8[m, B])]
        self.put_keys: dict[str, np.ndarray] = {}  # rid -> sorted row keys already put
        self.touched: collections.OrderedDict = collections.OrderedDict()  # rid -> last use (time)
        self.host_bytes = 0
        self.stats: Counter = Counter()
        self.warned_incomplete = False
        self.staging = None

    # -- setup
    def install(self):
        from sglang.srt.managers.scheduler_components.output_streamer import SchedulerOutputStreamer
        from sglang.srt.model_executor.model_runner import ModelRunner

        _check_graph_config()
        if self.tp_rank == 0:  # only rank 0 puts
            self.store = qt.connect_store(self.mc, int(self.mc["segment_gb"] * (1 << 30)))
        rec = self
        orig_forward = ModelRunner._forward_raw

        def _forward_raw(runner, forward_batch, *args, **kwargs):
            rec.begin(runner, forward_batch)
            try:
                out = orig_forward(runner, forward_batch, *args, **kwargs)
            except BaseException:
                rec.cur = None
                raise
            rec.end(out)
            return out

        orig_build = SchedulerOutputStreamer.build_additional_customized_info
        orig_level = SchedulerOutputStreamer.is_request_level_customized_info_key
        had_info = SchedulerOutputStreamer.has_additional_customized_info

        def build_additional_customized_info(streamer, reqs):
            if streamer.rust_server is not None:
                raise RuntimeError("[QAT trace] SGLang's Rust egress does not carry customized_info")
            info = orig_build(streamer, reqs) if had_info else {}
            info[INFO_KEY] = rec.finish(reqs)
            return info

        def is_request_level_customized_info_key(streamer, key):
            return key == INFO_KEY or orig_level(streamer, key)

        ModelRunner._forward_raw = _forward_raw
        SchedulerOutputStreamer.has_additional_customized_info = True
        SchedulerOutputStreamer.build_additional_customized_info = build_additional_customized_info
        SchedulerOutputStreamer.is_request_level_customized_info_key = is_request_level_customized_info_key
        threading.Thread(target=self._worker, daemon=True, name="coda-qat-trace").start()
        logger.info(f"[QAT trace] recorder {self.inst} on attn tp rank {self.tp_rank}/{self.tp_size}: "
                    f"bits={self.bits} layers={self.layers[0]}..{self.layers[-1]} "
                    f"store={self.store.get_hostname() if self.store is not None else None}")

    def _fail(self, msg: str):
        if self.broken is None:
            self.broken = msg
            logger.error(f"[QAT trace] recording stopped: {msg}; the trainer falls back to round-to-nearest")

    # -- forward boundaries (scheduler thread). Whether a forward is recorded depends only on the
    # batch, the same on every attention TP rank, because recorded forwards run a collective.
    def begin(self, runner, fb):
        self.cur = None
        self.fwd_seen.clear()
        self.fwd_mods.clear()
        if not fb.rids:
            return
        mode = fb.forward_mode
        if not (mode.is_decode() or mode.is_extend()):
            return
        if not self.names:
            self.names = {id(m): name for name, m in runner.model.named_modules()}
        n = int(fb.input_ids.shape[0])
        if mode.is_decode():
            owners = list(fb.rids)
        else:
            owners = [r for r, k in zip(fb.rids, fb.extend_seq_lens_cpu) for _ in range(int(k))]
        if len(owners) != n or fb.positions is None or int(fb.positions.shape[0]) < n:
            self.stats["skipped_forwards"] += 1
            return
        self.cur = {"n": n, "owners": owners, "ids": fb.input_ids, "pos": fb.positions, "out": {}}

    def end(self, output):
        cur, self.cur = self.cur, None
        if cur is None:
            return
        n = cur["n"]
        parts, flags = self._parts(cur, output), None
        if self.tp_size > 1:
            parts, flags = self._gather(n, parts)
            if self.tp_rank != 0:
                return
        if parts is None:
            return
        codes = torch.cat(parts, dim=1)
        host = torch.empty(codes.shape, dtype=torch.uint8, pin_memory=True)
        host.copy_(codes, non_blocking=True)
        if flags is not None:
            hflags = torch.empty(flags.shape, dtype=torch.uint8, pin_memory=True)
            hflags.copy_(flags, non_blocking=True)
            flags = hflags
        ids = torch.empty(n, dtype=cur["ids"].dtype, pin_memory=True)
        ids.copy_(cur["ids"][:n], non_blocking=True)
        pos = torch.empty(n, dtype=cur["pos"].dtype, pin_memory=True)
        pos.copy_(cur["pos"][:n], non_blocking=True)
        done = torch.cuda.Event()
        done.record()
        self.enq_seq += 1
        self._pop_completed()
        self.inflight.append((self.enq_seq, done))
        self.queue.put((self.enq_seq, done, host, flags, ids, pos, cur["owners"]))

    def _pop_completed(self):
        while self.inflight and self.inflight[0][1].query():
            self.ready_seq = self.inflight.popleft()[0]

    def _parts(self, cur, output):
        """This rank's [n, row bytes] record of every slot for the forward, or None."""
        n = cur["n"]
        if cur.get("bad") or self.broken or not self.slot_rb:
            return None
        if getattr(output, "can_run_graph", False):
            if any(s is None or s.shape[0] < n for s in self.static):
                self._fail(f"a CUDA graph replay of {n} rows has no captured record buffer for every module")
                return None
            parts = [s[:n] for s in self.static]
        else:
            if len(cur["out"]) != len(self.slot_rb):
                self.stats["incomplete_forwards"] += 1
                return None
            parts = [cur["out"][s] for s in range(len(self.slot_rb))]
        self.frozen = True
        return parts

    def _sharded_layout(self) -> list:
        return [(self.names.get(self.slot_mod[s], ""), self.slot_rb[s])
                for s in range(len(self.slot_rb)) if self.slot_sharded[s]]

    def _gather(self, n: int, parts):
        """Every rank, every recorded forward: all-gather the K-sharded columns over the attention TP
        device group. Rank 0 gets (its slots followed by ranks 1.. sharded slots, [tp-1, n] flags that
        are 0 where a rank had no record); the other ranks get (None, None)."""
        if self.gather_layout is None:  # agree on the column layout (CPU collective, first forwards only)
            sigs = self.group.all_gather_object((self.frozen, self._sharded_layout()))
            self.gather_tries += 1
            if all(f for f, _ in sigs):
                if all(s == sigs[0][1] for _, s in sigs):
                    self.gather_layout = sigs[0][1]
                else:
                    self.gather_layout = False
                    self._fail("the attention TP ranks record different K-sharded modules")
            elif self.gather_tries >= 100:
                self.gather_layout = False
                self._fail("some attention TP rank never completed a recorded forward")
            if not self.gather_layout:
                return None, None
        if self.gather_layout is False:
            return None, None
        width = sum(rb for _, rb in self.gather_layout)
        if not width:
            return (parts, None) if self.tp_rank == 0 else (None, None)
        ok = parts is not None and self._sharded_layout() == self.gather_layout
        dev = torch.device("cuda", torch.cuda.current_device())
        if ok:
            ones = torch.ones((n, 1), dtype=torch.uint8, device=dev)
            x = torch.cat([p for p, sh in zip(parts, self.slot_sharded) if sh] + [ones], dim=1)
        else:
            x = torch.zeros((n, width + 1), dtype=torch.uint8, device=dev)
        out = torch.empty((self.tp_size * n, width + 1), dtype=torch.uint8, device=dev)
        torch.distributed.all_gather_into_tensor(out, x, group=self.group.device_group)
        if self.tp_rank != 0 or parts is None:
            return None, None
        out = out.view(self.tp_size, n, width + 1)
        return parts + [out[r, :, :width] for r in range(1, self.tp_size)], out[1:, :, width]

    # -- per linear
    def _classify(self, module):
        qm = getattr(module, "quant_method", None)
        if qm is None or "Fp8LinearMethod" not in type(qm).__name__:
            return None
        full, local = getattr(module, "input_size", None), getattr(module, "input_size_per_partition", None)
        sharded = full is not None and local is not None and full != local
        return sharded if (self.tp_rank == 0 or sharded) else None

    def _mismatch(self, mid: int, capturing: bool):
        """The input sharing of ``mid`` differs from the first forward's: the records of this forward
        would not fit the fixed slot layout. Skip the forward (fatal while capturing a graph)."""
        what = f"module {self.names.get(mid, mid)} shares its input differently from the first forward"
        if capturing:
            self._fail(what)
        else:
            if not self.stats["inconsistent_forwards"]:
                logger.warning(f"[QAT trace] {what}; such forwards are not recorded")
            self.stats["inconsistent_forwards"] += 1
            self.cur["bad"] = True

    @torch.no_grad()
    def hook(self, module, inputs, output):
        capturing = torch.cuda.is_current_stream_capturing()
        cur = self.cur
        if self.broken or (cur is None and not capturing) or not inputs:
            return
        mid = id(module)
        sharded = self.kind.get(mid, _UNSET)
        if sharded is _UNSET:
            sharded = self.kind[mid] = self._classify(module)
        if sharded is None:
            return
        x = inputs[0]
        if torch.is_tensor(x):
            t = x
        elif hasattr(x, "data") and hasattr(x, "scales"):  # Mxfp8SwizzledInput: fused quant upstream
            t = x.data
        else:
            return
        if mid in self.fwd_mods:  # the module again: a new forward (capture has no begin())
            self.fwd_seen.clear()
            self.fwd_mods.clear()
        self.fwd_mods.add(mid)
        # Same input tensor OBJECT as an earlier linear of this forward (e.g. wq_a / wkv): record
        # once. A weakref (not the address) so a freed and reused buffer is never taken for it.
        prev = self.fwd_seen.get(id(t))
        if prev is not None and prev[1]() is t:
            if self.alias_of.get(mid) != prev[0]:
                if self.frozen or mid in self.slot_of or mid in self.alias_of:
                    self._mismatch(mid, capturing)
                else:
                    self.alias_of[mid] = prev[0]
            return
        self.fwd_seen[id(t)] = (mid, weakref.ref(t))
        if mid in self.alias_of:
            self._mismatch(mid, capturing)
            return
        n = None if capturing else cur["n"]
        codes = self._codes(x, n)
        if codes is None:
            self.stats["short_inputs"] += 1
            return
        packed = qt.pack_low_bits(codes, self.bits)
        slot = self.slot_of.get(mid)
        if slot is None:
            if self.frozen:
                self._mismatch(mid, capturing)
                return
            slot = self.slot_of[mid] = len(self.slot_rb)
            self.slot_mod.append(mid)
            self.slot_rb.append(int(packed.shape[1]))
            self.slot_sharded.append(bool(sharded))
            self.static.append(None)
        if capturing:
            st = self.static[slot]
            if st is None:  # SGLang captures the largest batch first
                st = self.static[slot] = torch.empty(packed.shape, dtype=torch.uint8, device=packed.device)
            elif packed.shape[0] > st.shape[0] or packed.shape[1] != st.shape[1]:
                self._fail(f"captured batch {tuple(packed.shape)} exceeds the first one {tuple(st.shape)}")
                return
            st[: packed.shape[0]].copy_(packed)
        else:
            cur["out"][slot] = packed

    def _codes(self, x, n: int | None):
        """[rows, K] uint8 e4m3 codes of the GEMM input; the first ``n`` rows (None: all, when capturing)."""
        if torch.is_tensor(x):
            from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

            x2 = x.reshape(-1, x.shape[-1])
            if n is not None:
                if x2.shape[0] < n:
                    return None
                x2 = x2[:n]
            codes, _ = sglang_per_token_group_quant_fp8(x2.contiguous(), 32, scale_ue8m0=True)
            return codes.view(torch.uint8)
        k = x.data.shape[-1]
        codes = x.data.view(torch.uint8).reshape(-1, k)
        if n is not None:
            if codes.shape[0] < n:
                return None
            codes = codes[:n]
        return codes

    # -- host side (worker thread)
    def _worker(self):
        while True:
            seq, done, host, flags, ids, pos, owners = self.queue.get()
            try:
                done.synchronize()
                if flags is not None and not bool(flags.numpy().all()):
                    with self.lock:  # some rank had no record of this forward: its rows break prefixes
                        self.stats["forwards_missing_rank"] += 1
                        for rid in set(owners):
                            self.reqs.pop(rid, None)
                    continue
                codes, ids, pos = host.numpy(), ids.numpy().astype(np.int64), pos.numpy().astype(np.int64)
                with self.lock:
                    self._ingest(codes, ids, pos, owners)
            except Exception:  # noqa: BLE001 - never take the rollout down
                logger.exception("[QAT trace] failed to buffer rollout records")
            finally:
                with self.done_cv:
                    self.done_seq = seq
                    self.done_cv.notify_all()

    def _ingest(self, codes: np.ndarray, ids: np.ndarray, pos: np.ndarray, owners: list):
        i, n = 0, len(owners)
        now = time.monotonic()
        while i < n:
            j = i + 1
            while j < n and owners[j] == owners[i]:
                j += 1
            self._ingest_run(owners[i], codes[i:j], ids[i:j], pos[i:j], now)
            i = j

    def _ingest_run(self, rid, codes, ids, pos, now):
        p0, m = int(pos[0]), len(ids)
        st = self.reqs.get(rid)
        if p0 == 0:  # (re-)prefill from the start: a new turn of the rid, or after a retract
            st = self.reqs[rid] = [0, 0]
        if st is None or st[0] != p0 or (m > 1 and not np.array_equal(pos, np.arange(p0, p0 + m))):
            self.reqs.pop(rid, None)  # prefix not seen by this request (radix hit, gap, after finish)
            self.stats["rows_without_prefix"] += m
            return
        if m == 1:
            h = qt.prefix_hash_step(st[1], int(ids[0]))
            keys = np.array([qt.to_int64(h)], dtype=np.int64)
        else:
            keys = qt.prefix_hashes(ids, st[1])
            h = int(keys.view(np.uint64)[-1])
        st[0], st[1] = p0 + m, h
        self._touch(rid, now)
        pos = np.arange(p0, p0 + m, dtype=np.int64)
        copied = False
        prev = self.put_keys.get(rid)
        if prev is not None and prev.size:  # rows an earlier request of the rid already put (re-prefilled history)
            at = np.minimum(np.searchsorted(prev, keys), prev.size - 1)
            new = prev[at] != keys
            self.stats["rows_already_put"] += m - int(new.sum())
            if not new.all():
                pos, keys, codes, copied = pos[new], keys[new], codes[new], True
            m = keys.size
            if not m:
                return
        if self.host_bytes + codes.nbytes > self.max_host_bytes:
            self.stats["rows_over_host_cap"] += m
            return
        self.rows.setdefault(rid, []).append((pos, keys, codes if copied else codes.copy()))
        self.host_bytes += codes.nbytes + keys.nbytes
        self.stats["rows"] += m

    def _touch(self, rid, now):
        """Mark ``rid`` used and drop the state of rids idle for longer than ``_RID_IDLE_S`` (lock held)."""
        self.touched[rid] = now
        self.touched.move_to_end(rid)
        while self.touched:
            old, t = next(iter(self.touched.items()))
            if now - t < _RID_IDLE_S:
                break
            del self.touched[old]
            self.reqs.pop(old, None)
            self.put_keys.pop(old, None)
            for _, k, c in self.rows.pop(old, ()):
                self.host_bytes -= k.nbytes + c.nbytes
                self.stats["rows_dropped_idle"] += k.size

    # -- object encoding and put (rank 0, scheduler thread)
    def _table(self) -> tuple[np.ndarray, int]:
        """(record table, slot count) of the rows rank 0 holds: its own slots, then for every other
        rank the K-sharded slots in ``gather_layout`` order."""
        name = lambda mid: self.names.get(mid, f"<module {mid}>")  # noqa: E731
        own = len(self.slot_rb)
        entries = [(name(m), s, self.slot_sharded[s], self.slot_rb[s], 0) for s, m in enumerate(self.slot_mod)]
        aliases = [(name(mid), self.slot_of[c]) for mid, c in self.alias_of.items()]
        entries += [(a, s, self.slot_sharded[s], self.slot_rb[s], 0) for a, s in aliases]
        sharded = [s for s in range(own) if self.slot_sharded[s]] if self.tp_size > 1 and self.gather_layout else []
        for r in range(1, self.tp_size if sharded else 1):
            base = own + (r - 1) * len(sharded)
            entries += [(name(self.slot_mod[s]), base + j, True, self.slot_rb[s], r) for j, s in enumerate(sharded)]
            entries += [(a, base + sharded.index(s), True, self.slot_rb[s], r) for a, s in aliases if s in sharded]
        return qt.make_table(entries), own + (self.tp_size - 1) * len(sharded)

    def _staging_for(self, size: int) -> np.ndarray:
        if self.staging is None or self.staging.nbytes < size:
            if self.staging is not None:
                self.store.unregister_buffer(self.staging.ctypes.data)
            self.staging = np.empty(max(size, _STAGING_BYTES), dtype=np.uint8)
            rc = self.store.register_buffer(self.staging.ctypes.data, self.staging.nbytes)
            if rc != 0:
                raise RuntimeError(f"register_buffer failed ({rc})")
        return self.staging

    # -- request finish (scheduler thread)
    def finish(self, reqs) -> list:
        """``meta_info["qat_trace"]`` value of every emitted request: ``[{"key", "n_rows"}]`` for a
        finished request whose new rows rank 0 put, else ``[]``. Only rank 0 records and puts."""
        out = [[] for _ in reqs]
        done = [(i, r) for i, r in enumerate(reqs) if r.finished()]
        if not done or self.tp_rank != 0:
            return out
        t0 = time.perf_counter()
        calls = [f"{self.inst}.{self.finished + k}" for k in range(len(done))]
        self.finished += len(done)
        try:
            self._catch_up()
            objs = self._put([r for _, r in done], calls)
        except Exception:  # noqa: BLE001 - never take the rollout down
            logger.exception("[QAT trace] failed to put rollout records")
            objs = [None] * len(done)
        with self.lock:
            for k, (i, req) in enumerate(done):
                self.reqs.pop(req.rid, None)  # the next turn prefills from position 0
                if objs[k] is None:
                    self.stats["requests_failed"] += 1
                    continue
                if not objs[k].size:
                    continue
                prev = self.put_keys.get(req.rid)
                self.put_keys[req.rid] = objs[k] if prev is None else np.union1d(prev, objs[k])
                self.stats["requests_put"] += 1
                out[i] = [{"key": qt.object_key(req.rid, calls[k]), "n_rows": int(objs[k].size)}]
        self.stats["put_s"] += time.perf_counter() - t0
        if self.stats["requests_failed"] and not self.warned_incomplete:
            self.warned_incomplete = True
            logger.warning(f"[QAT trace] some requests have no record object; {dict(self.stats)}")
        if self.finished // _LOG_EVERY != (self.finished - len(done)) // _LOG_EVERY:
            logger.info(f"[QAT trace] {self.finished} requests finished; {dict(self.stats)}")
        return out

    def _catch_up(self):
        """Wait until the worker has ingested every forward that completed on the GPU (including the
        one that sampled the last token of the finishing requests, but not the next one in flight)."""
        self._pop_completed()
        with self.done_cv:
            if not self.done_cv.wait_for(lambda: self.done_seq >= self.ready_seq, timeout=_CATCH_UP_S):
                raise RuntimeError(f"worker thread is {self.ready_seq - self.done_seq} forwards behind")

    def _put(self, reqs, calls) -> list:
        """Put the new rows of each finished request (one object on this rank); per request the sorted
        row keys put, an empty array if it has none, or None if its put failed."""
        with self.lock:
            taken = []
            for req in reqs:
                chunks = self.rows.pop(req.rid, ())
                for _, k, c in chunks:
                    self.host_bytes -= k.nbytes + c.nbytes
                taken.append((chunks, self.put_keys.get(req.rid)))
        empty = np.zeros(0, dtype=np.int64)
        result, todo = [empty] * len(reqs), []
        for j, (req, (chunks, prev)) in enumerate(zip(reqs, taken)):
            if self.broken or not chunks:
                continue
            keys = np.concatenate([k for _, k, _ in chunks])
            data = np.concatenate([c for _, _, c in chunks])
            pos = np.concatenate([p for p, _, _ in chunks])
            # the last sampled token was never fed back (except by one extra overlap forward on some ranks)
            keep = pos < len(req.origin_input_ids) + len(req.output_ids) - 1
            _, first = np.unique(keys, return_index=True)  # a retract re-prefills: keep the first record
            mask = np.zeros(keys.size, dtype=bool)
            mask[first] = True
            keep &= mask
            if prev is not None and prev.size:
                at = np.minimum(np.searchsorted(prev, keys), prev.size - 1)
                keep &= prev[at] != keys  # rows an earlier request of the rid already put
            if keep.any():
                todo.append((j, calls[j], req.rid, keys[keep], data[keep]))
        if todo:
            for (j, _, _, keys, _), ok in zip(todo, self._put_objects(todo)):
                result[j] = np.sort(keys) if ok else None
        return result

    def _put_objects(self, todo) -> list[bool]:
        """Encode ``(_, call, rid, keys, rows)`` objects into the staging buffer and put them at once."""
        table, n_slots = self._table()
        sizes = [qt.ObjectLayout(table, n_slots, t[3].size).nbytes for t in todo]
        staging = self._staging_for(sum(sizes))
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int).tolist()
        for (_, _, _, keys, data), off, size in zip(todo, offsets, sizes):
            qt.encode_object(staging[off: off + size], bits=self.bits, tp_rank=0,
                             tp_size=self.tp_size, table=table, n_slots=n_slots, keys=keys, rows=data)
        names = [qt.object_key(rid, call) for _, call, rid, _, _ in todo]
        base = staging.ctypes.data
        rcs = self.store.batch_put_from(names, [base + o for o in offsets], sizes, qt.hard_pin_config(self.store))
        oks = []
        for name, size, rc in zip(names, sizes, rcs):
            oks.append(rc == 0)
            if rc == 0:
                self.stats["put_bytes"] += size
            else:
                if not self.stats["put_failed"]:
                    logger.warning(f"[QAT trace] put {name} failed ({rc})")
                self.stats["put_failed"] += 1
        return oks

