"""Trainer side of rollout-guided activation QAT (``trainer.qat_trace``, see coda/utils/qat_trace.py).

:class:`TraceStore` brings the rollout records of one step onto the node. The ranks of a node that
share a PP stage and DP rank need the same records (the SGLang linears of their layers, for the record
objects of their rollout shard), so the lowest of them (the leader) reads them from the Mooncake Store
into one shared-memory buffer and the others map it. The buffer holds, per (SGLang module, rollout
rank), one [rows, bytes] segment over the rows of all objects of the step, plus the row keys.

Every training row lists the record objects of its calls (``rollout_data["qat_trace_objs"]``: per
call the object key and row count SGLang returned with the response). One object holds every
rollout attention TP rank's records of a call; objects of one SGLang engine share their record
table, read once per step.

:class:`TraceHooks` replaces the activation fake-quant of the traced linears. ``model.py`` attaches
the row keys and record objects of each micro-batch to its ``packed_seq_params``; a pre-hook on every
decoder layer picks them up, so pipeline interleaving and full recompute see their own micro-batch.
On the first traced linear of a micro-batch the keys are mapped to buffer rows (matched only among
the rows of each training row's own objects, the first object listed winning a duplicate key); every
linear then gathers its local rows (sequence-parallel shard or not) and K slice (row-parallel or not)
and calls :func:`coda.utils.qat_trace.guided_fake_quant`. Rows without a record fall back to
round-to-nearest.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import re
import socket
import time
import zlib

import numpy as np
import torch
import torch.distributed as dist

from coda.utils import qat_trace as qt

logger = logging.getLogger(__name__)

SRC_ATTR = "qat_trace_src"  # (per-sample row keys, per-sample record object keys, cp partition mode)
ROWS_ATTR = "qat_trace_rows"  # CP-local packed buffer row per token (-1: no record)
_GET_BATCH = 256  # objects per get_into_ranges call


def attach_keys(packed_seq_params, tokens_list, objs, cp_partition_mode) -> None:
    """Attach the row keys and record objects (``qat_trace_objs`` rows) of a micro-batch to its
    ``packed_seq_params``; the rows are resolved by :class:`TraceHooks` on first use."""
    keys = [qt.prefix_hashes(t.detach().cpu().numpy()) for t in tokens_list]
    groups = [tuple(k for k, _ in o) for o in objs] if objs is not None else [()] * len(keys)
    setattr(packed_seq_params, SRC_ATTR, (keys, groups, cp_partition_mode))
    if hasattr(packed_seq_params, ROWS_ATTR):
        delattr(packed_seq_params, ROWS_ATTR)


class TraceStore:
    """Node-local view of the rollout records of the current step for the SGLang linears ``wanted``."""

    def __init__(self, mc: dict, bits: int, wanted: set[str]):
        from megatron.core import parallel_state as ps

        self.mc, self.bits = mc, bits
        rank = dist.get_rank()
        me = (socket.gethostname(), ps.get_pipeline_model_parallel_rank(), ps.get_data_parallel_rank())
        everyone = [None] * dist.get_world_size()
        dist.all_gather_object(everyone, (me, sorted(wanted)))
        mine = [(r, w) for r, (k, w) in enumerate(everyone) if k == me]
        self.leader = rank == min(r for r, _ in mine)
        self.wanted = set().union(*(set(w) for _, w in mine))
        tag = f"{zlib.crc32(str(mc['master_server']).encode()):08x}_pp{me[1]}_dp{me[2]}"
        self.base = os.path.join(mc["shm_dir"], f"coda_trace_{tag}")
        need_client = (self.leader and self.wanted) or rank == 0  # rank 0 also deletes the step's objects
        self.client = qt.connect_store(mc, 0) if need_client else None
        self.buf, self.buf_gen = None, 0  # leader: registered shared-memory buffer
        self._reset()

    def _reset(self):
        self.obj_rows: dict[str, tuple[int, int]] = {}  # group id -> (first buffer row, rows)
        self.keys = np.zeros(0, dtype=np.int64)
        self.segs: dict[str, tuple[bool, list[np.ndarray]]] = {}  # name -> (sharded, [view per rollout rank])
        self.sorted: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self.view = None

    # -- step boundaries (every rank)
    def load(self, objs) -> dict:
        """Fetch (leader) and map (every rank of the group) the record objects listed in ``objs``
        (``rollout_data["qat_trace_objs"]``: per row, (key, rows) of each call)."""
        self._reset()
        stats = {}
        dist.barrier()  # nobody still reads the previous step's buffer
        if self.leader and self.wanted:
            stats = self._fetch({str(k): int(n) for row in (objs or []) for k, n in row})
        dist.barrier()
        if self.wanted:
            self._open()
        return stats

    def clear(self):
        self._reset()

    def delete_step_objects(self) -> int:
        """Remove every record object of the store (rank 0, once the step is trained)."""
        if self.client is None or dist.get_rank() != 0:
            return 0
        return self.client.remove_by_regex("^" + re.escape(qt.KEY_PREFIX), True)

    # -- leader
    def _scratch(self, nbytes: int) -> np.ndarray:
        """Registered scratch memory for object headers (reuses the shared buffer)."""
        self._ensure_buffer(nbytes)
        return self.buf

    def _ensure_buffer(self, nbytes: int):
        if self.buf is not None and self.buf.size >= nbytes:
            return
        if self.buf is not None:
            self.client.unregister_buffer(self.buf.ctypes.data)
            old, self.buf = self.buf_path, None
            os.remove(old)
        self.buf_gen += 1
        self.buf_path = f"{self.base}_{os.getpid()}_{self.buf_gen}.buf"
        cap = max(nbytes + nbytes // 4, 64 << 20)
        self.buf = np.memmap(self.buf_path, dtype=np.uint8, mode="w+", shape=(cap,))
        atexit.register(_remove_quietly, self.buf_path)
        rc = self.client.register_buffer(self.buf.ctypes.data, cap)
        if rc != 0:
            raise RuntimeError(f"[QAT trace] register_buffer of {cap} bytes failed ({rc})")

    def _ranges(self, keys, dsts, srcs, sizes) -> list[bool]:
        """get_into_ranges over the shared buffer; per object: every range complete."""
        ok = []
        for a in range(0, len(keys), _GET_BATCH):
            b = a + _GET_BATCH
            res = self.client.get_into_ranges([self.buf.ctypes.data], [keys[a:b]], [dsts[a:b]], [srcs[a:b]], [sizes[a:b]])
            ok += [all(int(g) == int(s) for g, s in zip(got, want)) for got, want in zip(res[0], sizes[a:b])]
        return ok

    def _preambles(self, keys: list[str]) -> list:
        n = qt.PREAMBLE.itemsize
        buf = self._scratch(n * len(keys))
        ok = self._ranges(keys, [[i * n] for i in range(len(keys))], [[0]] * len(keys), [[n]] * len(keys))
        out = []
        for i, good in enumerate(ok):
            out.append(qt.decode_preamble(bytes(buf[i * n:(i + 1) * n])) if good else None)
        return out

    def _table(self, key: str, n_entries: int) -> np.ndarray:
        n = n_entries * qt.TABLE_ENTRY.itemsize
        buf = self._scratch(n)
        if not self._ranges([key], [[0]], [[qt.PREAMBLE.itemsize]], [[n]])[0]:
            raise RuntimeError(f"[QAT trace] could not read the record table of {key}")
        return qt.decode_table(bytes(buf[:n]), n_entries)

    def _tables(self, keys: list[str]) -> dict:
        """Record table of every recorder (SGLang engine) among ``keys``: the preamble and table of one
        readable object each (up to 3 tried). Returns {recorder id: (table, slot count, crc)}."""
        cands: dict[str, list[str]] = {}
        for k in keys:
            cands.setdefault(qt.key_instance(k), []).append(k)
        tables = {}
        for attempt in range(3):
            todo = [(inst, ks[attempt]) for inst, ks in cands.items() if inst not in tables and len(ks) > attempt]
            if not todo:
                break
            for (inst, k), p in zip(todo, self._preambles([k for _, k in todo])):
                if p is None:
                    continue
                if int(p["bits"]) != self.bits:
                    raise ValueError(f"[QAT trace] rollout records have {int(p['bits'])} bits, trainer expects {self.bits}")
                tables[inst] = (self._table(k, int(p["n_entries"])), int(p["n_slots"]), int(p["table_crc"]))
        return tables

    def _fetch(self, objs: dict[str, int]) -> dict:
        """``objs``: object key -> rows, as returned with each response (no per-object header read)."""
        t0 = time.time()
        stats = {"objects": len(objs)}
        tables = self._tables(list(objs))
        if not tables:
            self._write_index([], [], [], {}, 0)
            return {**stats, "complete": 0}
        # every engine runs the same model and code, so the tables should match; others are dropped
        table, n_slots, crc = next(iter(tables.values()))
        good = {inst for inst, (_, _, c) in tables.items() if c == crc}
        complete = [k for k in objs if qt.key_instance(k) in good]
        stats.update(complete=len(complete), engines=len(tables), table_mismatch=len(tables) - len(good))
        if not complete:
            self._write_index([], [], [], {}, 0)
            return stats
        return {**stats, **self._fetch_rows(complete, objs, table, n_slots), "fetch_s": round(time.time() - t0, 1)}

    def _fetch_rows(self, keys: list[str], n_rows: dict, table: np.ndarray, n_slots: int) -> dict:
        # buffer segments: one per slot (= (rollout rank, module input)) that holds a wanted module
        seg_of, seg_rb, names = {}, [], {}
        for e in table:
            name = e["name"].decode()
            if name not in self.wanted:
                continue
            slot = int(e["slot"])
            if slot not in seg_of:
                seg_of[slot] = len(seg_rb)
                seg_rb.append(int(e["row_bytes"]))
            names.setdefault(name, (bool(e["sharded"]), []))[1].append((int(e["rank"]), seg_of[slot]))
        starts, total = {}, 0
        for k in keys:
            starts[k], total = total, total + int(n_rows[k])
        seg_base = np.cumsum([total * 8] + [total * rb for rb in seg_rb])
        self._ensure_buffer(int(seg_base[-1]))
        # object of n rows: slot s at data_start + n * prefix[s], keys at data_start + n * width
        lay = qt.ObjectLayout(table, n_slots, 0)
        rb = lay.slot_row_bytes
        prefix, width = np.concatenate([[0], np.cumsum(rb)[:-1]]), int(rb.sum())
        gets, dsts, srcs, sizes = [], [], [], []
        for k in keys:
            n, s0 = int(n_rows[k]), starts[k]
            if not n:
                continue
            d, s, z = [s0 * 8], [lay.data_start + n * width], [n * 8]
            for slot, g in seg_of.items():
                d.append(int(seg_base[g]) + s0 * seg_rb[g])
                s.append(lay.data_start + n * int(prefix[slot]))
                z.append(n * seg_rb[g])
            gets.append(k)
            dsts.append(d)
            srcs.append(s)
            sizes.append(z)
        ok = self._ranges(gets, dsts, srcs, sizes)
        failed = {k for k, good in zip(gets, ok) if not good}
        good = [k for k in keys if k not in failed]
        self._write_index(good, [starts[k] for k in good], [int(n_rows[k]) for k in good],
                          {name: (sh, [(r, int(seg_base[g]), seg_rb[g]) for r, g in lst]) for name, (sh, lst) in names.items()},
                          total)
        return {"read_failed": len(failed), "rows": sum(int(n_rows[k]) for k in good), "bytes": int(seg_base[-1])}

    def _write_index(self, gids, starts, counts, segs: dict, total: int):
        names = sorted(segs)
        flat = [(k, int(segs[name][0]), r, off, rb) for k, name in enumerate(names) for r, off, rb in segs[name][1]]
        arrays = {
            "gids": np.asarray(gids, dtype=str), "starts": np.asarray(starts, dtype=np.int64),
            "counts": np.asarray(counts, dtype=np.int64), "names": np.asarray(names, dtype=str),
            "segs": np.asarray(flat, dtype=np.int64).reshape(-1, 5), "total": np.int64(total),
            "path": np.asarray(self.buf_path if self.buf is not None else ""),
        }
        if self.buf is not None:
            self.buf.flush()
        tmp = f"{self.base}.idx.tmp.npz"
        np.savez(tmp, **arrays)
        if not getattr(self, "_idx_cleanup", False):
            atexit.register(_remove_quietly, f"{self.base}.idx.npz")
            self._idx_cleanup = True
        os.replace(tmp, f"{self.base}.idx.npz")

    # -- every rank of the group
    def _open(self):
        with np.load(f"{self.base}.idx.npz", allow_pickle=False) as idx:
            gids, starts, counts = idx["gids"].tolist(), idx["starts"], idx["counts"]
            names, segs, total, path = idx["names"].tolist(), idx["segs"], int(idx["total"]), str(idx["path"])
        if not total or not path:
            return
        self.view = np.memmap(path, dtype=np.uint8, mode="r")
        self.keys = self.view[: total * 8].view(np.int64)
        self.obj_rows = {g: (int(s), int(c)) for g, s, c in zip(gids, starts, counts)}
        for k, sharded, r, off, rb in segs.tolist():
            entry = self.segs.setdefault(names[k], (bool(sharded), []))
            entry[1].append((r, self.view[off: off + total * rb].reshape(total, rb)))
        for name, (sharded, lst) in self.segs.items():
            lst.sort(key=lambda t: t[0])
            self.segs[name] = (sharded, [v for _, v in lst])

    def rows(self, groups, keys: np.ndarray) -> np.ndarray:
        """Buffer row of every key among the rows of the record objects ``groups`` (keys) of one
        training row (-1: none); a key recorded by several objects maps to the first listed."""
        out = np.full(keys.shape[0], -1, dtype=np.int64)
        groups = tuple(groups or ())
        cached = self.sorted.get(groups)
        if cached is None:
            spans = [self.obj_rows[g] for g in groups if g in self.obj_rows and self.obj_rows[g][1]]
            rows = np.concatenate([np.arange(s, s + c) for s, c in spans]) if spans else np.zeros(0, np.int64)
            ks = np.asarray(self.keys[rows]) if rows.size else np.zeros(0, np.int64)
            order = np.argsort(ks, kind="stable")
            cached = self.sorted[groups] = (ks[order], rows[order])
        sk, at = cached
        if not sk.size:
            return out
        pos = np.clip(np.searchsorted(sk, keys), 0, sk.shape[0] - 1)  # leftmost match: first group
        hit = sk[pos] == keys
        out[hit] = at[pos[hit]]
        return out

    def segments(self, name: str):
        """(sharded, [rows x bytes view per rollout rank]) of SGLang module ``name``, or None."""
        return self.segs.get(name)


def _remove_quietly(path: str):
    with contextlib.suppress(FileNotFoundError):
        os.remove(path)


class _GuidedFakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, low_bits, found, bits, stat):
        shape, K = x.shape, x.shape[-1]
        y, changed = qt.guided_fake_quant(x.reshape(-1, K), group, low_bits, found, bits)
        hit = int(found.sum())
        stat[0] += hit
        stat[1] += found.numel()
        stat[2] += changed
        stat[3] += hit * K
        return y.view(shape)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None, None, None


class TraceHooks:
    """Guided activation fake-quant for the traced linears of one rank (see module docstring)."""

    def __init__(self, cfg: dict, model_chunks, linears: list[tuple[str, torch.nn.Module, int, str, int]]):
        """``linears``: (megatron name, module, group, SGLang module name, global layer) to guide."""
        from megatron.core import parallel_state as ps

        self.bits = int(cfg["bits"])
        self.tp = ps.get_tensor_model_parallel_world_size()
        self.tp_rank = ps.get_tensor_model_parallel_rank()
        self.log_rank = (self.tp_rank == 0 and ps.get_context_parallel_rank() == 0
                         and ps.get_data_parallel_rank() == 0)
        self.pp_rank = ps.get_pipeline_model_parallel_rank()
        self.store = TraceStore(cfg["mooncake"], self.bits, {s for _, _, _, s, _ in linears})
        self.linears = linears
        self.psp = None  # packed_seq_params of the running decoder layer's micro-batch
        self.enabled = True
        self.stats: dict[str, list[int]] = {}
        self.warned: set[str] = set()
        self._register(model_chunks)

    def _register(self, model_chunks):
        def layer_hook(module, args, kwargs):
            self.psp = kwargs.get("packed_seq_params")

        for chunk in model_chunks:
            for name, m in chunk.named_modules():
                if re.search(r"(^|\.)decoder\.layers\.\d+$", name):
                    m.register_forward_pre_hook(layer_hook, with_kwargs=True)
        for pname, module, group, sname, _ in self.linears:
            module.register_forward_pre_hook(self._make_hook(pname, sname, group))

    def _warn(self, pname, msg):
        if pname not in self.warned:
            self.warned.add(pname)
            logger.warning(f"[QAT trace] {pname}: {msg} -> round-to-nearest")

    def _rows(self) -> np.ndarray | None:
        """Buffer row of every CP-local packed token of the running micro-batch (computed once)."""
        psp = self.psp
        if psp is None:
            return None
        rows = getattr(psp, ROWS_ATTR, None)
        if rows is None:
            src = getattr(psp, SRC_ATTR, None)
            if src is None:
                return None
            from coda.backends.megatron.cp_utils import slice_cp_packed

            keys, groups, cp_mode = src
            per = [torch.from_numpy(self.store.rows(g, k)) for g, k in zip(groups, keys)]
            packed, _ = slice_cp_packed(per, cp_mode, -1, 128)  # same padding as prepare_packed_seq_params
            rows = packed.cpu().numpy()
            setattr(psp, ROWS_ATTR, rows)
        return rows

    def _make_hook(self, pname: str, sname: str, group: int):
        from coda.backends.megatron.qat import _FakeQuantActFP8

        def hook(module, args):
            if not args or not torch.is_tensor(args[0]):
                return None
            x = args[0]
            rows = self._rows() if self.enabled else None
            res = self._rollout_bits(pname, sname, x, rows) if rows is not None else None
            if res is None:
                return (_FakeQuantActFP8.apply(x, group), *args[1:])
            found, low_bits = res
            stat = self.stats.setdefault(pname, [0, 0, 0, 0])
            if low_bits is None:  # no row of this micro-batch has a record
                stat[1] += found.numel()
                return (_FakeQuantActFP8.apply(x, group), *args[1:])
            return (_GuidedFakeQuant.apply(x, group, low_bits, found, self.bits, stat), *args[1:])

        return hook

    def _rollout_bits(self, pname, sname, x, rows):
        """(found [R] bool, low code bits [R, K] uint8 or None) of the local input rows, or None."""
        K = x.shape[-1]
        R, S = x.numel() // K, rows.shape[0]
        if R == S:
            rk = rows
        elif R * self.tp == S:  # sequence-parallel shard
            rk = rows[self.tp_rank * R:(self.tp_rank + 1) * R]
        else:
            self._warn(pname, f"{R} input rows vs {S} keys (tp {self.tp})")
            return None
        seg = self.store.segments(sname)
        if seg is None:
            self._warn(pname, f"no rollout records for {sname}")
            return None
        found = rk >= 0
        if not found.any():
            return torch.from_numpy(found), None
        _, views = seg
        Kr = views[0].shape[1] * 8 // self.bits  # K of one rollout rank's record
        Kf = Kr * len(views)
        if Kf == K:
            lo, chosen = 0, views
        elif Kf == K * self.tp:  # row-parallel: this rank holds a contiguous K slice of the full record
            first, last = self.tp_rank * K // Kr, ((self.tp_rank + 1) * K - 1) // Kr
            lo, chosen = self.tp_rank * K - first * Kr, views[first:last + 1]
        else:
            self._warn(pname, f"rollout K {Kf} ({len(views)} ranks) vs local K {K} (tp {self.tp})")
            return None
        idx = np.where(found, rk, 0)
        packed = np.concatenate([v[idx] for v in chosen], axis=1) if len(chosen) > 1 else chosen[0][idx]
        low = qt.unpack_low_bits(torch.from_numpy(packed).to(x.device, non_blocking=True), self.bits)
        if low.shape[1] != K:
            low = low[:, lo: lo + K]
        return torch.from_numpy(found).to(x.device, non_blocking=True), low.contiguous()

    # -- step boundaries (every rank)
    def new_step(self, rollout_data) -> None:
        """Fetch and map the record objects of this step's rollout (``rollout_data["qat_trace_objs"]``)."""
        self.stats = {}
        objs = rollout_data.get("qat_trace_objs") if rollout_data is not None else None
        t0 = time.time()
        stats = self.store.load(objs)
        if stats:
            logger.info(f"[QAT trace] pp {self.pp_rank}: fetched rollout records in {time.time() - t0:.1f}s: {stats}")

    def end_step(self) -> None:
        """Report this step's guidance and delete its records from the store (rank 0)."""
        self._report()
        self.store.clear()
        removed = self.store.delete_step_objects()
        if removed:
            logger.info(f"[QAT trace] removed {removed} record objects from the store")

    def _report(self):
        if self.log_rank and self.stats:
            tot = [sum(v[i] for v in self.stats.values()) for i in range(4)]
            logger.info(f"[QAT trace] pp {self.pp_rank}: rows with a rollout record {tot[0]}/{tot[1]} "
                        f"({tot[0] / max(tot[1], 1):.4f}); elements re-rounded {tot[2] / max(tot[3], 1):.4f} "
                        f"(bits={self.bits}, {len(self.linears)} linears)")
        self.stats = {}

    @contextlib.contextmanager
    def paused(self):
        """Plain round-to-nearest inside (e.g. reference-model log-probs: other weights)."""
        prev, self.enabled = self.enabled, False
        try:
            yield
        finally:
            self.enabled = prev

