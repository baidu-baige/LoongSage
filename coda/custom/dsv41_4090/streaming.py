"""Training side of ``MegatronStreamActorWorker``: one GRPO step in a short-lived process.

The weights stay on disk: every decoder layer's parameters are views into one shared GPU pool
and are paged in from the pruned checkpoint before the layer runs (and again before its
backward). Only the LoRA adapters of the tail layers, the embedding/head and small buffers are
resident.

The colocated SGLang engine needs ~19.5 GB to resume its weights after training, so the trainer
must hand back the whole GPU, CUDA context included. ``MegatronStreamActorWorker`` therefore runs
every step as a subprocess calling ``main``: build the streaming model, restore adapters + Adam
state, run GRPO on the trajectories, merge the adapters into the serving checkpoint, save state,
exit.

    python -c "from coda.custom.dsv41_4090.streaming import main; main()" \
        --work-dir DIR --step 1
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import struct
import sys
import time

import torch

TAIL = (36, 37, 38, 39)


def build_streaming_model(ckpt: str, engram_ckpt: str,
                          tail=TAIL, lora_dim: int = 16, lora_alpha: int = 32,
                          log=print, routing_replay=False):
    """Build the meta model, page weights per layer, and wrap the tail with LoRA."""
    from megatron.bridge import AutoBridge
    import megatron.bridge.models.deepseek.deepseek_v41_bridge  # noqa: F401
    from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
    from megatron.bridge.peft.lora import LoRA

    # The engram hasher keeps small derived buffers (token map, primes, multipliers) that must be
    # real even though every weight is built on the meta device.
    from megatron.core.transformer.experimental_attention_variant.dsv41.engram import (
        DeepSeekV41Engram,
    )
    if not getattr(DeepSeekV41Engram.__init__, "_coda_cuda", False):
        _engram_init = DeepSeekV41Engram.__init__

        def _engram_init_cuda(self, *a, **k):
            with torch.device("cuda"):
                _engram_init(self, *a, **k)

        _engram_init_cuda._coda_cuda = True
        DeepSeekV41Engram.__init__ = _engram_init_cuda

    provider = AutoBridge.from_hf_pretrained(ckpt).to_megatron_provider(load_weights=True)
    provider._pre_wrap_hooks = []
    for name, val in (("tensor_model_parallel_size", 1), ("pipeline_model_parallel_size", 1),
                      ("context_parallel_size", 1), ("expert_model_parallel_size", 1),
                      ("expert_tensor_parallel_size", 1), ("perform_initialization", False),
                      ("init_model_with_meta_device", True), ("gradient_accumulation_fusion", False),
                      ("recompute_granularity", None), ("moe_token_dispatcher_type", "alltoall")):
        setattr(provider, name, val)
    provider.moe_flex_dispatcher_backend = None
    provider.moe_enable_routing_replay = routing_replay
    provider.calculate_per_token_loss = True
    provider.variable_seq_lengths = True
    provider.moe_router_load_balancing_type = "none"
    from megatron.core.utils import init_method_normal, scaled_init_method_normal
    provider.init_method = init_method_normal(0.02)
    provider.output_layer_init_method = scaled_init_method_normal(0.02, provider.num_layers)
    provider.embedding_init_method = init_method_normal(0.02)
    provider.v41_hf_checkpoint = engram_ckpt

    model = provider.provide_distributed_model(wrap_with_ddp=False, mixed_precision_wrapper=None)[0]
    model.eval()
    layers = model.decoder.layers

    # one pool: every layer's params are views into it
    def nbytes(layer):
        return sum(((p.numel() * p.element_size() + 255) // 256) * 256 for m in layer.modules()
                   for p in m.parameters(recurse=False))

    pool = torch.empty(max(nbytes(l) for l in layers), dtype=torch.uint8, device="cuda")
    for l in layers:
        off = 0
        for _, mod in l.named_modules():
            for pname, p in list(mod.named_parameters(recurse=False)):
                view = pool[off: off + p.numel() * p.element_size()].view(p.dtype).view(p.shape)
                setattr(mod, pname, torch.nn.Parameter(view, requires_grad=False))
                off += ((p.numel() * p.element_size() + 255) // 256) * 256

    # resident params (embedding, head, norms) are real
    for name, p in list(model.named_parameters()):
        if ".layers." in name or p.device.type != "meta":
            continue
        owner, parts = model, name.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part)
        setattr(owner, parts[-1], torch.nn.Parameter(
            torch.empty(p.shape, dtype=p.dtype, device="cuda"), requires_grad=False))

    hf = PreTrainedCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16)
    state = hf.state
    qcfg = json.load(open(os.path.join(ckpt, "config.json")))["quantization_config"]
    block = qcfg["weight_block_size"][0]
    keys = set(state.source.get_all_keys())

    def one(name):
        t = state[name].cuda()
        if name.endswith(".weight"):
            sn = name[:-len(".weight")] + ".scale"
            if sn in keys:
                w, s = t, state[sn].cuda()
                O, I = w.shape
                b = s.shape[0]
                return (w.float().view(b, O // b, s.shape[1], I // s.shape[1])
                        * s.float().view(b, 1, s.shape[1], 1)).view(O, I).bfloat16()
        return t

    def fetch(hp):
        return {k: one(v) for k, v in hp.items()} if isinstance(hp, dict) else one(hp)

    # single rank: the bridge's cross-PP all_gathers are no-ops
    import torch.distributed as dist
    if not getattr(dist, "_coda_stream_patched", False):
        _ago = dist.all_gather_object

        def all_gather_object(out, obj, group=None):
            if (group is None and dist.get_world_size() == 1) or (group is not None and group.size() == 1):
                out[0] = obj
                return
            return _ago(out, obj, group=group)

        dist.all_gather_object = all_gather_object
        dist._coda_stream_patched = True

    tasks = AutoBridge.from_hf_pretrained(ckpt).get_conversion_tasks(model, hf_path=ckpt)
    by_layer, resident = collections.defaultdict(list), []
    for t in tasks:
        m = re.search(r"\.layers\.(\d+)\.", t.param_name)
        (by_layer[int(m.group(1))] if m else resident).append(t)

    with torch.no_grad():
        for t in resident:
            conv = t.mapping.hf_to_megatron(fetch(t.mapping.hf_param), t.megatron_module)
            if conv is not None:
                t.param_weight.data.copy_(conv)

    def load_layer(i):
        with torch.no_grad():
            for t in by_layer[i]:
                conv = t.mapping.hf_to_megatron(fetch(t.mapping.hf_param), t.megatron_module)
                if conv is not None:
                    t.param_weight.data.copy_(conv)

    patterns = [f"*.layers.{i}.*{m}" for i in tail for m in
                ("linear_q_up_proj", "linear_kv_proj", "linear_proj",
                 "mlp.shared_experts.linear_fc1", "mlp.shared_experts.linear_fc2")]
    lora = LoRA(target_modules=patterns, dim=lora_dim, alpha=lora_alpha)
    model = lora([model], training=True)[0]
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith("adapter.linear_in.weight"):
                p.normal_(0.0, 0.02)
            elif name.endswith("adapter.linear_out.weight"):
                p.zero_()
    _forward_with_served_weights(model)

    for i, l in enumerate(layers):
        l.register_forward_pre_hook(lambda m, a, i=i: load_layer(i))
        l.register_full_backward_pre_hook(lambda m, gin, i=i: load_layer(i))

    meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
            if t.device.type == "meta"]
    if meta:
        raise RuntimeError(f"streaming model still has meta tensors: {meta[:6]}")
    log(f"streaming model ready: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B, "
        f"LoRA {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M, "
        f"pool {pool.numel() / 2**30:.2f}GiB")
    return model, layers, load_layer, pool


# Merge the LoRA adapters into the dense FP8 weights the rollout engine serves. Base weights are
# always read from the pristine checkpoint and the merged result is written to the serving
# checkpoint at the same byte offsets, so repeated merges never accumulate. Dense weights are E4M3
# with 32x32 E8M0 block scales; the merged weight is requantized to that exact layout.

# Megatron adapter module -> HF tensor stems (fused Megatron weights are row-concatenated)
ATTN = {"linear_q_up_proj": ["attn.wq_b"], "linear_kv_proj": ["attn.wkv"], "linear_proj": ["attn.wo_b"]}
MLP = {"mlp.shared_experts.linear_fc1": ["ffn.shared_experts.w1", "ffn.shared_experts.w3"],
       "mlp.shared_experts.linear_fc2": ["ffn.shared_experts.w2"]}


def _headers(ckpt):
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    meta = {}
    for fn in sorted(set(idx.values())):
        path = os.path.join(ckpt, fn)
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                s, e = v["data_offsets"]
                meta[k] = (path, 8 + n + s, 8 + n + e, v["dtype"], tuple(v["shape"]))
    return meta


def _read(entry, dtype):
    path, s, e, _, shape = entry
    with open(path, "rb") as f:
        f.seek(s)
        return torch.frombuffer(bytearray(f.read(e - s)), dtype=torch.uint8).view(dtype).view(shape)


def _write(entry, tensor):
    path, s, e, _, _ = entry
    data = tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    assert len(data) == e - s, (path, len(data), e - s)
    with open(path, "r+b") as f:
        f.seek(s)
        f.write(data)


def _dequant(w, s, block=32):
    O, I = w.shape
    exp = s.view(torch.uint8).to(torch.int32) - 127
    scale = torch.pow(2.0, exp.float())
    return (w.float().view(O // block, block, I // block, block)
            * scale.view(O // block, 1, I // block, 1)).view(O, I)


def _quant(w, block=32):
    N, K = w.shape
    x = w.float().view(N // block, block, K // block, block)
    amax = x.abs().amax(dim=(1, 3)).clamp_min(1e-30)
    exp = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q = (x / torch.pow(2.0, exp).view(N // block, 1, K // block, 1)).clamp(-448, 448)
    return q.view(N, K).to(torch.float8_e4m3fn), (exp + 127).to(torch.uint8).view(torch.float8_e8m0fnu)


def _forward_with_served_weights(model):
    """Make every LoRA layer compute with the weight the rollout engine serves.

    The merge requantizes W + BA to FP8, which keeps only about half of a typical LoRA delta (most
    of it is below the E4M3 step). Adding x @ (Q(W + BA) - (W + BA))^T to the forward makes the
    trainer see exactly the served weight; the correction carries no gradient, so the adapters
    train straight through the quantizer.
    """
    from megatron.bridge.peft.lora_layers import LoRALinear

    def forward(self, x, *args, **kwargs):
        out, bias, ln_out = self.base_linear_forward(x, *args, **kwargs)
        lora = self.adapter_forward(self.adapter, ln_out.contiguous(), *args, **kwargs)
        with torch.no_grad():
            delta = self.adapter.linear_out.weight.float() @ self.adapter.linear_in.weight.float()
            merged = self.to_wrap.weight.float() + delta * (self.adapter.alpha / self.adapter.dim)
            residual = (_dequant(*_quant(merged)) - merged).to(ln_out.dtype)
        correction = torch.nn.functional.linear(ln_out, residual)
        return out + lora.reshape(out.shape) + correction.reshape(out.shape), bias

    wrapped = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            if type(module) is not LoRALinear:
                raise TypeError(f"unsupported LoRA wrapper {type(module).__name__}")
            module.forward = forward.__get__(module)
            wrapped += 1
    return wrapped


def merge_adapters(adapters, ckpt, base_ckpt, layers=TAIL, alpha_over_r=2.0):
    """adapters: Megatron param name -> tensor. Returns the number of tensors written."""
    base, serve = _headers(base_ckpt), _headers(ckpt)
    written = 0
    for layer in layers:
        for prefix, table in (("self_attention", ATTN), (None, MLP)):
            for mod, stems in table.items():
                path = f"decoder.layers.{layer}.{prefix + '.' if prefix else ''}{mod}.adapter"
                A, B = adapters.get(f"{path}.linear_in.weight"), adapters.get(f"{path}.linear_out.weight")
                if A is None or B is None:
                    continue
                delta = (B.float() @ A.float()) * alpha_over_r
                row = 0
                for stem in stems:
                    wn, sn = f"layers.{layer}.{stem}.weight", f"layers.{layer}.{stem}.scale"
                    w = _read(base[wn], torch.float8_e4m3fn)
                    s = _read(base[sn], torch.float8_e8m0fnu)
                    rows = w.shape[0]
                    q, e = _quant(_dequant(w, s) + delta[row:row + rows])
                    row += rows
                    _write(serve[wn], q)
                    _write(serve[sn], e)
                    written += 2
                assert row == delta.shape[0], (path, row, delta.shape)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    a = ap.parse_args()

    os.environ.setdefault("MEGATRON_SKIP_PARAM_INIT", "1")
    t0 = time.time()
    import ray
    from omegaconf import OmegaConf
    from types import SimpleNamespace
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from coda.backends.megatron.megatron_actor_worker import MegatronActorWorker
    from coda.custom.dsv41_4090.stream_worker import MegatronStreamActorWorker
    from coda.data_factory.data_processor import Box

    config = OmegaConf.load(os.path.join(a.work_dir, "config.yaml"))
    config = MegatronStreamActorWorker.validate_config(config)
    batch = torch.load(os.path.join(a.work_dir, "rollout_batch.pt"), weights_only=False)
    model, _, _, _ = build_streaming_model(ckpt=config.dsv41_4090.base_ckpt,
                                           engram_ckpt=config.dsv41_4090.orig_ckpt,
                                           routing_replay=config.trainer.use_rollout_routing_replay)
    t_build = time.time() - t0
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    adapter_path = os.path.join(a.work_dir, "adapter.pt")
    if os.path.exists(adapter_path):
        sd = torch.load(adapter_path, map_location="cuda")
        for k, v in sd.items():
            if k in trainable and trainable[k].shape == v.shape:
                trainable[k].data.copy_(v)
    names = sorted(trainable)
    params = [trainable[n] for n in names]
    # Only adapters require gradients, so standard DDP allocates buffers for them alone.
    class StreamingActor(MegatronActorWorker):
        """Specialize initialization/metric transport; inherit the complete training cycle."""

        def __init__(self, config, model):
            self.config, self.rank, self.world_size = config, 0, 1
            self.model = [DistributedDataParallel(
                model.config, DistributedDataParallelConfig(**OmegaConf.to_container(config.megatron.ddp_config)),
                model, disable_bucketing=True,
            )]
            self._forward_only = False
            self.kl_policies = {}
            self._quant = SimpleNamespace(qat=False)
            self._build_optimizer()

        def _track_metrics(self, step, rollout_data, metrics):
            for key in ("m2po_metrics", "is_metrics", "kl_metrics", "adv_metrics"):
                metrics.update(rollout_data.get(key, {}))
            self.step_metrics = metrics

    actor = StreamingActor(config, model)
    opt_path = os.path.join(a.work_dir, "optimizer.pt")
    if os.path.exists(opt_path):
        state = torch.load(opt_path, map_location="cuda")
        if state.get("format") != "standard_actor_v1" or state.get("names") != names:
            raise ValueError("Optimizer checkpoint is not from the standard streaming actor; use a new work directory")
        actor.optimizer.load_state_dict(state["state"])
        actor.scheduler.load_state_dict(state["scheduler"])

    ray.init(address="auto", ignore_reinit_error=True)
    actor.train(a.step, [Box(ray.put(batch))])
    if not all(bool(torch.isfinite(p).all()) for p in params):
        raise RuntimeError("Non-finite adapter weights after standard actor training")
    adapters = {n: trainable[n].detach().cpu() for n in names}
    torch.save(adapters, adapter_path + ".tmp")
    os.replace(adapter_path + ".tmp", adapter_path)
    torch.save({"format": "standard_actor_v1", "names": names, "state": actor.optimizer.state_dict(),
                "scheduler": actor.scheduler.state_dict()}, opt_path + ".tmp")
    os.replace(opt_path + ".tmp", opt_path)
    merged = merge_adapters(adapters, ckpt=config.hf_model_path, base_ckpt=config.dsv41_4090.base_ckpt)
    rewards = batch["rewards"]
    metrics = {
        **actor.step_metrics,
        "train/merged_tensors": merged,
        "train/reward_mean": sum(rewards) / max(1, len(rewards)),
        "train/adapter_norm": float(sum(p.detach().float().pow(2).sum() for p in params) ** 0.5),
        "perf/train_build_seconds": t_build, "perf/train_seconds": time.time() - t0,
        "perf/train_memory_allocated_max": torch.cuda.max_memory_allocated() / 2**30,
    }
    json.dump(metrics, open(os.path.join(a.work_dir, "metrics.json"), "w"))
    print("METRICS", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    sys.exit(main())
