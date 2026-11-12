"""Rollout-guided activation QAT (TRACE, arXiv 2610.07767): pieces shared by rollout, trainer and driver.

With plain activation QAT both sides round their own bf16 GEMM inputs to the nearest e4m3 code;
elements whose two bf16 values straddle a rounding boundary land on different codes, so the
fake-quant adds a second, independent quantization error. TRACE lets the rollout tell the trainer
which way it rounded: the rollout records the lowest ``bits`` code bits of every e4m3 element it
feeds a W8A8 GEMM, and the trainer picks, of the two e4m3 codes bracketing its own value, the one
consistent with those bits (see :func:`guided_fake_quant`).

* Rows are matched by a 64-bit polynomial hash of the token prefix ``tokens[0..pos]`` (inclusive),
  computed by the rollout per request and by the trainer per training row (:func:`prefix_hashes`).
  Only rows a request prefilled itself get a key, so the rollout must run with ``disable_radix_cache``.
* ``bits`` <= 3 low code bits are independent of the ue8m0 group scale (a scale of 2x shifts the
  e4m3 code ordinal by 8), so no scale is transferred and the trainer uses its own scale.
* Records travel through a Mooncake Store whose master the driver starts (:func:`start_master`).
  Every rollout attention-TP rank records its K slices, gathers them to rank 0 after each forward,
  and rank 0 buffers the rows of each request (SGLang ``rid``); when a request finishes it puts the
  rows no earlier request of the rid put as one object (:func:`encode_object`), and the response
  carries the object key in ``meta_info["qat_trace"]``.
  agentflow keeps them per turn, every training row lists the objects of its turns
  (``rollout_data["qat_trace_objs"]``), and the trainer fetches those and deletes the step's objects
  after training (docs/zh/design/dsv41-trace-mooncake.md).
"""

from __future__ import annotations

import json
import os
import zlib

import numpy as np

HASH_MUL = 0x100000001B3  # odd -> invertible mod 2**64
_MASK = (1 << 64) - 1
_HASH_MUL_INV = pow(HASH_MUL, -1, 1 << 64)
SUPPORTED_BITS = (1, 2)
LAYER_MODES = ("back_half", "all")

KEY_PREFIX = "coda_trace/"
FORMAT_MAGIC = 0x43525443  # "CTRC"
FORMAT_VERSION = 3
# Object = preamble | table (one entry per recorded module and rollout rank) | one [n, row_bytes]
# segment per slot | keys int64[n]. One object per request holds every attention TP rank's records.
PREAMBLE = np.dtype([
    ("magic", "<u4"), ("version", "<u4"), ("bits", "<u4"), ("tp_rank", "<u4"), ("tp_size", "<u4"),
    ("n_entries", "<u4"), ("n_slots", "<u4"), ("table_crc", "<u4"), ("n_rows", "<u8"), ("_pad", "V24"),
])
# ``slot``: the segment holding the module's input; modules fed the same input tensor (DSV4.1
# wq_a / wkv) share one slot. ``sharded``: K-sharded (row-parallel) input, recorded on every rank;
# ``rank``: the rollout attention TP rank whose K slice (or, unsharded, whose input) the slot holds.
TABLE_ENTRY = np.dtype([("name", "S112"), ("slot", "<i4"), ("sharded", "<u4"), ("row_bytes", "<u4"), ("rank", "<u4")])
assert PREAMBLE.itemsize == 64 and TABLE_ENTRY.itemsize == 128


# ---------------------------------------------------------------- config

def trace_config(config) -> dict | None:
    """``trainer.qat_trace`` as a plain dict when enabled, else None."""
    cfg = config.trainer.get("qat_trace")
    if not cfg or not cfg.get("enable"):
        return None
    mc = cfg.get("mooncake") or {}
    return {
        "bits": cfg.get("bits"),
        "layers": cfg.get("layers"),
        "mooncake": {
            "master_server": mc.get("master_server"),
            "metadata_server": mc.get("metadata_server"),
            "protocol": mc.get("protocol") or "rdma",
            "rdma_devices": mc.get("rdma_devices") or "",
            "segment_gb": float(mc.get("segment_gb") or 64),
            "max_host_gb": float(mc.get("max_host_gb") or 32),
            "shm_dir": mc.get("shm_dir") or "/dev/shm",
        },
    }


def num_hidden_layers(hf_model_path: str) -> int:
    with open(os.path.join(hf_model_path, "config.json")) as f:
        cfg = json.load(f)
    cfg = cfg.get("text_config", cfg)
    return int(cfg["num_hidden_layers"])


def trace_layers(mode: str, num_layers: int) -> list[int]:
    """Decoder layers whose FP8 linear inputs are guided."""
    if mode == "all":
        return list(range(num_layers))
    if mode == "back_half":
        return list(range(num_layers // 2, num_layers))
    raise ValueError(f"trainer.qat_trace.layers must be one of {LAYER_MODES}, got {mode!r}")


def sglang_hook_spec(config) -> dict:
    """SGLang ``forward_hooks`` entry that installs the rollout-side recorder. ``capture``: attached
    before CUDA graph capture (docker/patch/sglang-0.5.23.patch), the hook is graph-safe."""
    cfg = trace_config(config)
    layers = trace_layers(cfg["layers"], num_hidden_layers(config.hf_model_path))
    return {
        "name": "coda_qat_trace",
        "target_modules": [f"model.layers.{i}.*" for i in layers],  # the hook keeps the FP8 linears
        "hook_factory": "coda.backends.sglang.qat_trace_hook:make_hook",
        "capture": True,
        "config": {"bits": int(cfg["bits"]), "layers": layers, "mooncake": cfg["mooncake"]},
    }


def _prefill_graph_disabled(args) -> bool:
    if args.get("disable_cuda_graph") or args.get("disable_prefill_cuda_graph"):
        return True
    if args.get("cuda_graph_backend_prefill") == "disabled":
        return True
    graph_cfg = args.get("cuda_graph_config")
    if isinstance(graph_cfg, str):
        graph_cfg = json.loads(graph_cfg)
    return bool(graph_cfg) and (graph_cfg.get("prefill") or {}).get("backend") == "disabled"


def validate(config) -> None:
    cfg = trace_config(config)
    if cfg is None:
        return
    problems = []
    if not config.trainer.qat:
        problems.append("needs trainer.qat: true")
    if int(cfg["bits"]) not in SUPPORTED_BITS:
        problems.append(f"trainer.qat_trace.bits must be one of {SUPPORTED_BITS}")
    if cfg["layers"] not in LAYER_MODES:
        problems.append(f"trainer.qat_trace.layers must be one of {LAYER_MODES}")
    if cfg["mooncake"]["protocol"] not in ("rdma", "tcp"):
        problems.append("trainer.qat_trace.mooncake.protocol must be rdma or tcp")
    if config.rollout.backend != "sglang":
        problems.append("only the SGLang rollout records its quantized activations")
    args = config.rollout.sglang_args
    if not args.get("disable_radix_cache"):
        problems.append("rollout.sglang_args.disable_radix_cache must be true (rows are keyed by the prefix a request prefilled itself)")
    if not _prefill_graph_disabled(args):
        problems.append("rollout.sglang_args.disable_prefill_cuda_graph must be true (prefill runs eager; "
                        "the decode CUDA graph is supported)")
    if args.get("cuda_graph_backend_decode") == "breakable" or args.get("debug_cuda_graph"):
        problems.append("the decode CUDA graph backend must be full (breakable graphs run hooks inside replay)")
    if args.get("enable_torch_compile"):
        problems.append("rollout.sglang_args.enable_torch_compile is not supported")
    if not config.colocate:
        problems.append("needs colocate: true (the step's record objects are deleted by prefix after training)")
    if problems:
        raise ValueError("trainer.qat_trace: " + "; ".join(problems))


# ---------------------------------------------------------------- row keys

def prefix_hash_step(h: int, tok: int) -> int:
    """Hash of ``prefix + [tok]`` from the hash ``h`` of ``prefix`` (0 for the empty prefix)."""
    return (h * HASH_MUL + tok + 1) & _MASK


def to_int64(h: int) -> int:
    return h - (1 << 64) if h >= (1 << 63) else h


def prefix_hashes(tokens: np.ndarray, h0: int = 0) -> np.ndarray:
    """int64 key of every position of ``prefix + tokens`` where ``h0`` is the hash of ``prefix``:
    ``h_t = h0 * MUL**(t + 1) + sum_{i<=t} (tok_i + 1) * MUL**(t - i) mod 2**64``."""
    t = np.asarray(tokens, dtype=np.uint64) + np.uint64(1)
    n = t.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    with np.errstate(over="ignore"):
        pw = np.full(n, HASH_MUL, dtype=np.uint64)
        pw[0] = 1
        pw = np.cumprod(pw, dtype=np.uint64)
        ipw = np.full(n, _HASH_MUL_INV, dtype=np.uint64)
        ipw[0] = 1
        ipw = np.cumprod(ipw, dtype=np.uint64)
        h = pw * np.cumsum(t * ipw, dtype=np.uint64)
        if h0:
            h = h + np.uint64(h0) * pw * np.uint64(HASH_MUL)
    return h.view(np.int64)


# ---------------------------------------------------------------- low code bits

def pack_low_bits(codes, bits: int):
    """[N, K] uint8 e4m3 code bytes -> [N, K * bits // 8] uint8 holding their lowest ``bits`` bits
    (the sign is bit 7, so for bits <= 3 these are the low magnitude/mantissa bits)."""
    import torch

    per = 8 // bits
    v = (codes & ((1 << bits) - 1)).reshape(codes.shape[0], -1, per).to(torch.int32)
    shifts = torch.arange(per, device=codes.device, dtype=torch.int32) * bits
    return (v << shifts).sum(-1).to(torch.uint8)


def unpack_low_bits(packed, bits: int):
    """Inverse of :func:`pack_low_bits`: [N, B] uint8 -> [N, B * 8 // bits] uint8."""
    import torch

    per = 8 // bits
    shifts = torch.arange(per, device=packed.device, dtype=torch.int32) * bits
    v = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    return v.reshape(packed.shape[0], -1).to(torch.uint8)


_E4M3_TABLE: dict = {}


def _e4m3_magnitudes(device):
    import torch

    t = _E4M3_TABLE.get(device)
    if t is None:  # magnitude of every e4m3fn code 0x00..0x7e (0x7f is NaN)
        t = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float().to(device)
        _E4M3_TABLE[device] = t
    return t


def guided_fake_quant(x2, group: int, low_bits, found, bits: int):
    """Fake-quant ``x2`` [R, K] like SGLang (per-token group ``group``, ue8m0 scale, e4m3) but, on
    the ``found`` rows, round each element to whichever of its two bracketing codes agrees with the
    rollout's lowest ``bits`` code bits ``low_bits`` [R, K] (ties: round-to-nearest).
    Returns (y [R, K] in x2.dtype, #elements that differ from round-to-nearest)."""
    import torch
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

    R, K = x2.shape
    q, s = sglang_per_token_group_quant_fp8(x2.contiguous(), group, scale_ue8m0=True)
    s = s.float().view(R, -1, 1)
    mag = _e4m3_magnitudes(x2.device)
    b = q.view(torch.uint8).to(torch.int32).view(R, -1, group)
    o_rtn = torch.where((b & 0x80) != 0, -(b & 0x7F), b & 0x7F)
    v_rtn = q.float().view(R, -1, group)
    z = x2.float().view(R, -1, group) / s
    on_grid = v_rtn == z
    below = v_rtn < z
    lo = torch.where(on_grid | below, o_rtn, o_rtn - 1)
    hi = torch.where(on_grid | ~below, o_rtn, o_rtn + 1)
    # rollout reference: magnitude code nearest |z| whose low bits match, with the training sign
    mod = 1 << bits
    c = low_bits.to(torch.int32).view(R, -1, group)
    base = o_rtn.abs()
    up = base + (c - base) % mod
    dn = up - mod
    za = z.abs()
    use_dn = (dn >= 0) & ((za - mag[dn.clamp(0, 126)]).abs() <= (mag[up.clamp(max=126)] - za).abs())
    ref = torch.where(use_dn, dn, up).clamp(max=126)
    ref = torch.where(z < 0, -ref, ref)
    d_lo, d_hi = (lo - ref).abs(), (hi - ref).abs()
    pick = torch.where(d_lo < d_hi, lo, torch.where(d_hi < d_lo, hi, o_rtn))
    pick = torch.where(found.view(R, 1, 1), pick, o_rtn)
    v = torch.sign(pick).float() * mag[pick.abs().clamp(max=126)]
    changed = int((pick != o_rtn).sum())
    return (v * s).view(R, K).to(x2.dtype), changed


# ---------------------------------------------------------------- record objects

def object_key(rid: str, call: str) -> str:
    """Key of the object of one request (``call`` = ``{recorder id}.{seq}``) of ``rid``. Only the
    rollout builds keys; the trainer reads the keys returned with each response."""
    return f"{KEY_PREFIX}{rid}/{call}"


def key_instance(key: str) -> str:
    """Recorder id (one per SGLang attention TP group) of an object key: objects of one recorder
    share their record table."""
    return key.rsplit("/", 1)[-1].split(".", 1)[0]


def make_table(entries) -> np.ndarray:
    """``entries``: (module name, slot, sharded, row_bytes, rollout rank) -> TABLE_ENTRY array."""
    t = np.zeros(len(entries), dtype=TABLE_ENTRY)
    for i, (name, slot, sharded, row_bytes, rank) in enumerate(entries):
        raw = name.encode()
        if len(raw) > TABLE_ENTRY["name"].itemsize:
            raise ValueError(f"[QAT trace] module name too long for the record table: {name}")
        t["name"][i], t["slot"][i], t["sharded"][i], t["row_bytes"][i], t["rank"][i] = raw, slot, int(sharded), row_bytes, rank
    return t


def table_crc(table: np.ndarray) -> int:
    return zlib.crc32(np.ascontiguousarray(table).tobytes())


class ObjectLayout:
    """Byte offsets inside one record object of ``n_rows`` rows."""

    def __init__(self, table: np.ndarray, n_slots: int, n_rows: int):
        rb = np.zeros(n_slots, dtype=np.int64)
        for e in table:
            s = int(e["slot"])
            if rb[s] and rb[s] != int(e["row_bytes"]):
                raise ValueError(f"[QAT trace] slot {s}: row bytes {rb[s]} vs {int(e['row_bytes'])}")
            rb[s] = int(e["row_bytes"])
        if n_slots and not rb.all():
            raise ValueError("[QAT trace] record table leaves a slot without a module")
        self.n_rows = int(n_rows)
        self.slot_row_bytes = rb
        self.data_start = PREAMBLE.itemsize + table.nbytes
        starts = np.concatenate([[0], np.cumsum(rb)[:-1]]) if n_slots else np.zeros(0, np.int64)
        self.slot_offsets = self.data_start + self.n_rows * starts
        self.keys_offset = self.data_start + self.n_rows * int(rb.sum())
        self.nbytes = self.keys_offset + 8 * self.n_rows


def encode_object(dst: np.ndarray, *, bits: int, tp_rank: int, tp_size: int, table: np.ndarray, n_slots: int,
                  keys: np.ndarray, rows: np.ndarray) -> int:
    """Write one record object into the uint8 buffer ``dst``; returns its size. ``rows`` [n, sum of
    the slot row bytes] holds the slots side by side (slot order), ``keys`` int64[n] the row keys."""
    n = keys.shape[0]
    lay = ObjectLayout(table, n_slots, n)
    if rows.shape != (n, int(lay.slot_row_bytes.sum())):
        raise ValueError(f"[QAT trace] rows {rows.shape} vs {n} keys and slot row bytes {lay.slot_row_bytes.tolist()}")
    pre = np.zeros(1, dtype=PREAMBLE)
    pre["magic"], pre["version"], pre["bits"] = FORMAT_MAGIC, FORMAT_VERSION, bits
    pre["tp_rank"], pre["tp_size"], pre["n_entries"], pre["n_slots"] = tp_rank, tp_size, len(table), n_slots
    pre["table_crc"], pre["n_rows"] = table_crc(table), n
    dst[: PREAMBLE.itemsize] = pre.view(np.uint8)
    dst[PREAMBLE.itemsize: lay.data_start] = np.ascontiguousarray(table).view(np.uint8)
    col = 0
    for s in range(n_slots):
        rb, off = int(lay.slot_row_bytes[s]), int(lay.slot_offsets[s])
        dst[off: off + n * rb].reshape(n, rb)[:] = rows[:, col: col + rb]
        col += rb
    dst[lay.keys_offset: lay.nbytes] = np.ascontiguousarray(keys, dtype=np.int64).view(np.uint8)
    return lay.nbytes


def decode_preamble(raw) -> np.void:
    pre = np.frombuffer(raw, dtype=PREAMBLE, count=1)[0]
    if int(pre["magic"]) != FORMAT_MAGIC or int(pre["version"]) != FORMAT_VERSION:
        raise ValueError(f"[QAT trace] not a version {FORMAT_VERSION} record object")
    return pre


def decode_table(raw, n_entries: int) -> np.ndarray:
    return np.frombuffer(raw, dtype=TABLE_ENTRY, count=n_entries).copy()


# ---------------------------------------------------------------- Mooncake Store

def local_ip(peer_host: str) -> str:
    """Address of the interface this host uses to reach ``peer_host``."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((peer_host, 9))
        return s.getsockname()[0]


def nearest_rdma_device() -> str:
    """Active RDMA physical function sharing the deepest PCIe path with the current GPU ("" if none)."""
    root = "/sys/class/infiniband"
    if not os.path.isdir(root):
        return ""
    devs = {}
    for name in sorted(os.listdir(root)):
        dev = os.path.join(root, name, "device")
        path = os.path.realpath(dev)
        if os.path.basename(path).startswith("mlx5_core.sf") or os.path.exists(os.path.join(dev, "physfn")):
            continue  # sub-function / virtual function
        try:
            with open(os.path.join(root, name, "ports", "1", "state")) as f:
                if "ACTIVE" not in f.read():
                    continue
        except OSError:
            continue
        devs[name] = path
    if not devs:
        return ""
    import torch

    if not torch.cuda.is_available():
        return ",".join(devs)
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    gpu = os.path.realpath(f"/sys/bus/pci/devices/{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0")
    if not os.path.exists(gpu):
        return ",".join(devs)
    return max(devs, key=lambda d: len(os.path.commonpath([devs[d], gpu])))


def connect_store(mc: dict, segment_bytes: int, local_buffer_bytes: int = 64 << 20):
    """Mooncake client of this process; ``segment_bytes`` of host memory it contributes to the store."""
    from mooncake.store import MooncakeDistributedStore

    master, metadata = mc.get("master_server"), mc.get("metadata_server")
    if not master or not metadata:
        raise RuntimeError("[QAT trace] trainer.qat_trace.mooncake.master_server / metadata_server are unset "
                           "(the driver fills them in when it starts mooncake_master)")
    devices = ""
    if mc["protocol"] == "rdma":
        devices = mc.get("rdma_devices") or nearest_rdma_device()
    store = MooncakeDistributedStore()
    host = local_ip(master.rsplit(":", 1)[0])
    rc = store.setup(host, metadata, int(segment_bytes), int(local_buffer_bytes), mc["protocol"], devices, master)
    if rc != 0:
        raise RuntimeError(f"[QAT trace] Mooncake setup failed ({rc}): host={host} master={master} "
                           f"protocol={mc['protocol']} devices={devices!r}")
    return store


def hard_pin_config(store):
    """One replica in this client's own segment, never evicted (removed by the trainer)."""
    from mooncake.store import ReplicateConfig

    cfg = ReplicateConfig()
    cfg.replica_num = 1
    cfg.with_hard_pin = True
    cfg.preferred_segment = store.get_hostname()
    return cfg


def _free_ports(n: int) -> list[int]:
    import socket

    socks = [socket.socket() for _ in range(n)]
    try:
        for s in socks:
            s.bind(("", 0))
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


def _port_open(host: str, port: int) -> bool:
    import socket

    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _die_with_parent():
    import ctypes
    import signal

    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def start_master(config):
    """Start ``mooncake_master`` (with its HTTP metadata server) on this node for the run, and write
    its addresses into ``trainer.qat_trace.mooncake`` before any worker reads the config. Each run
    gets a fresh, empty store; the process dies with the driver."""
    import atexit
    import shutil
    import subprocess
    import sys
    import tempfile
    import time

    import ray

    exe = shutil.which("mooncake_master") or os.path.join(os.path.dirname(sys.executable), "mooncake_master")
    if not os.access(exe, os.X_OK):
        raise FileNotFoundError("[QAT trace] mooncake_master not found (install mooncake-transfer-engine)")
    host = ray.util.get_node_ip_address()
    rpc, http, metrics = _free_ports(3)
    log = os.path.join(tempfile.gettempdir(), f"coda_mooncake_master_{os.getpid()}.log")
    cmd = [exe, f"-rpc_port={rpc}", "-enable_http_metadata_server=true",
           f"-http_metadata_server_port={http}", f"-metrics_port={metrics}"]
    with open(log, "ab") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, preexec_fn=_die_with_parent)
    atexit.register(proc.terminate)
    deadline = time.time() + 60
    while not (_port_open(host, rpc) and _port_open(host, http)):
        if proc.poll() is not None:
            raise RuntimeError(f"[QAT trace] mooncake_master exited with {proc.returncode}, see {log}")
        if time.time() > deadline:
            proc.terminate()
            raise TimeoutError(f"[QAT trace] mooncake_master did not come up in 60 s, see {log}")
        time.sleep(0.2)
    mc = config.trainer.qat_trace.mooncake
    mc.master_server = f"{host}:{rpc}"
    mc.metadata_server = f"http://{host}:{http}/metadata"
    return proc, log



