<div align="center">

<div align="right">
  <a href="https://baidu-baige.github.io/LoongSage/zh/">中文</a> | <strong>English</strong>
</div>

# LoongSage: the coda of LLM training

**Agentic · Scalable · Lightweight**

A production-grade, minimalist reinforcement learning framework for ultra-large models and agents.

<p>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/code-14k%20lines-brightgreen.svg" alt="LOC">
  <img src="https://img.shields.io/badge/training-Megatron--Core-76B900.svg" alt="Megatron-Core">
  <img src="https://img.shields.io/badge/rollout-SGLang-orange.svg" alt="SGLang">
</p>

</div>

______________________________________________________________________

**LoongSage** is a post-training reinforcement learning framework for Large Language Models (LLMs), with Ray as the scheduling foundation, Megatron-Core as the training backend, and SGLang as the inference backend. It focuses on: **Training ultra-large models with a minimalist architecture**, **supporting zero-intrusion access for any agent**, **and providing system-level training-inference consistency guarantees.**

______________________________________________________________________

## 🌟 Features

### 🤖 Versatility — Any agent, any paradigm, any data mix

- **Zero-Intrusion Agent Access** — Agents only need to call the framework's inference service via a standard chat completion interface. Multi-turn context concatenation, tool output recognition, loss masking, trajectory collection, and sandbox full-lifecycle management are all handled automatically by the built-in AgentFlow module, achieving complete decoupling between the framework foundation and agent business logic.
- **Multi-Paradigm Online Distillation** — Natively supports PG-Style, GKD-Style, and mixed distillation modes, with built-in TopK, full-vocabulary, and JSD KL distillation strategies as well as multi-teacher distillation.
- **Multi-Data-Source Mixed Training** — Agents, reward functions, teacher models, and sampling hyperparameters can be configured independently per data source, while the proportion of each source is kept strictly globally consistent across every Rank and every Mini-Batch.

### ✅ Correctness — Reproducible convergence, observable bias

- **Out-of-the-Box Training Recipes** — Tuned end-to-end configurations are provided for typical scenarios such as math reasoning, agentic code repair, and full-vocabulary online distillation. Each recipe is hardened by real production training on MoE models inside Baidu (DeepSeek-V4-Flash, Qwen3-30B-A3B), so reaching stable convergence takes no hyperparameter guesswork.
- **System-Level Training-Inference Consistency Guarantee** — Built-in Router Replay (R3) for experts, FP32 output layer, TITO, importance sampling correction, M2PO, OPSM, and other consistency alignment techniques, together with real-time monitoring of distribution bias. In MoE architectures and low-precision scenarios, training-inference probability bias is stably held at the 1e-4 magnitude, significantly reducing the risk of training collapse.

### ⚡ Efficiency — Throughput tuned for ultra-large scale

- **Fully Asynchronous Training** — Supports partial rollout and oversampling. Sampling and training are fully decoupled, advancing in parallel and overlapping in time, with flexible data staleness and sliding window strategies. This effectively eliminates long-tail requests slowing down global training, improving throughput by 39% in our experiments.
- **Adaptive Weight Synchronization** — Weight synchronization between the training and inference engines automatically selects the optimal path: CUDA IPC zero-copy direct transfer for colocated deployment, and NCCL broadcast for cross-node distribution. Tensors are aggregated into a shared buffer by buckets, significantly improving transmission bandwidth utilization.
- **Flexible Resource Placement** — Fine-grained GPU allocation built on Ray Placement Groups lets you switch between intra-node multiplexing and disaggregated deployment with a single flag, and precisely coordinates training/inference VRAM in colocated mode.

### 🧩 Usability & Extensibility — Lightweight core, changeable anywhere

- **Highly Extensible Plugin System** — Provides up to 12 core extension points (covering agents, reward models, sandboxes, advantage estimation, policy loss, KL divergence, routing middleware, data filtering, asynchronous scheduling strategies, and more). Simply place a new implementation in the corresponding directory and reference it in the configuration to take effect via hot-plugging, without modifying the framework's main trunk.
- **Loosely Coupled Backend Integration** — The core is kept exceptionally lightweight, relying exclusively on Megatron-Core, Megatron-Bridge, and SGLang. Each backend module can be upgraded independently and smoothly while transparently inheriting all upstream training and inference features, with no tedious framework-layer adaptation. The library of supported models likewise expands automatically alongside the Megatron-Bridge ecosystem.

______________________________________________________________________

## 📰 News

- **[2026-08]** LoongSage is officially open-sourced! 

______________________________________________________________________

## 🏗️ Architecture

![LoongSage Architecture Diagram](./imgs/arch.png)

Targeting ultra-large model scenarios, LoongSage adopts a structure of **Single-Controller orchestration and multi-role Ray Actor execution**, which greatly reduces system maintenance complexity:

| Layer | Module | Responsibility |
| :--- | :--- | :--- |
| **Orchestration** | [`controller/`](coda/controller/) | The sole entry point of the system. Drives training, sampling, teacher inference, and weight synchronization, and precisely coordinates VRAM allocation in intra-node mode. |
| **Agent** | [`agentflow/`](coda/agentflow/) | Handles request routing, trajectory storage, tokenization, and agent/sandbox lifecycles. Produces standardized trajectories that can be accurately replayed on the training side. |
| **Backend** | [`backends/`](coda/backends/) | Universal abstraction interfaces for Training, Inference, and Teacher Workers, along with their underlying implementations based on Megatron and SGLang. |
| **Algorithm** | [`algorithms/`](coda/algorithms/) | Covers advantage function estimation, policy loss computation, divergence constraint strategies, and off-policy protection mechanisms. |
| **Data** | [`data_factory/`](coda/data_factory/) | Manages dataset loading, resumable data sources, sampling filtering mechanisms, and data partitioning/distribution based on load balancing. |
| **Transfer** | [`transfer_mesh/`](coda/transfer_mesh/) | A high-performance unified data transmission channel between training and inference engines. |
| **Scheduler** | [`resource_scheduler/`](coda/resource_scheduler/) | Based on Ray Placement Groups to achieve fine-grained GPU resource allocation, perfectly supporting both intra-node multiplexing and disaggregated deployment. |

______________________________________________________________________

## 🛠️ Installation

It is recommended to run using a container image, as the dependencies for CUDA, PyTorch, Megatron-Core, Megatron-Bridge, SGLang, and Ray have been strictly aligned, and necessary SGLang patches have been applied.

```bash
# Pull the image
docker pull loongsage/loongsage:latest

# Or build locally from source
docker build -t loongsage/loongsage:latest docker/

docker run -it --gpus all --ipc=host --network=host \
  -v /path/to/workspace:/root loongsage/loongsage:latest bash
```

## ⚡ Quick Start

All training shares the same entry point, [`examples/start.sh`](examples/start.sh). Tasks are distinguished by Hydra config name, and any Hydra overrides can follow it:

```bash
bash examples/start.sh <config-name> [key=value ...]
```

Config names map to files under [`conf/`](conf/), subdirectories included (e.g. `qwen3_30b_a3b/dapo_h20_1node`). The script launches training in the background and writes to `log/trainer_<timestamp>.log`.

### Your First Run: Qwen3-4B + DAPO

Download the model and dataset:

```bash
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
hf download Haitao999/DAPO-Math-17k-unique --repo-type=dataset \
  --local-dir /root/DAPO-Math-17k-unique
```

Launch the single-node 8-GPU preset [`qwen3_4b/dapo_h800_1node`](conf/qwen3_4b/dapo_h800_1node.yaml), passing the model and data paths on the command line:

```bash
bash examples/start.sh qwen3_4b/dapo_h800_1node \
  hf_model_path=/root/Qwen3-4B \
  data_source.dataset.prompt_data_path=/root/DAPO-Math-17k-unique
```

For everything else, see the documentation:

- Full first run, from environment setup and data download to verifying the results → [Quick Start](docs/en/quick-start.md)
- Multi-node setup, the remaining example tasks (DAPO / BCP / MOPD with Qwen3-30B-A3B, SWE with DeepSeek-V4-Flash, the OpenCode black-box agent), monitoring and resume → [Run Guide](docs/en/run-guide.md)

______________________________________________________________________

## 📚 Documentation

| Document | Content |
| :--- | :--- |
| [AgentFlow Framework](docs/en/agentflow-framework.md) | Router, trajectory storage, tokenization service, multi-turn & tool calls |
| [Custom Agent](docs/en/custom-agent.md) · [Reward](docs/en/custom-reward.md) · [Sandbox](docs/en/custom-sandbox.md) | Core extension point tutorials |
| [Custom RL Algorithm](docs/en/custom-algorithm.md) · [KL Algorithm](docs/en/custom-kl.md) · [Sliding Window Strategy](docs/en/custom-sliding-window.md) | Algorithm-side extension point tutorials |
| [Fully Asynchronous Training](docs/en/fully-async-mode.md) | Architecture, sliding window strategies, metrics, and tuning |
| [Training Algorithms](docs/en/training-algorithms.md) | Advantages, losses, off-policy protection, and execution order |
| [On-Policy Distillation](docs/en/on-policy-distillation.md) | PG / GKD, full vocabulary, multi-teacher |
| [Consistency](docs/en/train-inference-consistency.md) | Router replay, FP32 output layer, and monitoring |
| [TransferMesh](docs/en/transfer-mesh.md) | Weight transfer design and performance data |
| [Resource Scheduling](docs/en/resource-scheduler.md) | Placement groups, bundle ordering, colocated/disaggregated placement |
| [Model Loading & Saving](docs/en/model-checkpointing.md) | Directory layout, resuming, HF export, and optimizer sharding formats |
| [Config Reference](docs/en/config-reference.md) | Field-by-field walkthrough of `conf/default.yaml` with cross-field constraints |

Chinese documentation is available in [`docs/zh/`](docs/zh/).

## 🗺️ Roadmap

Below are our main plans for upcoming work. Discussions and contributions via Issues are welcome:

- [ ] Claude Code & Codex support
- [ ] Low-precision training support
- [ ] PPO support
- [ ] MTP / DSpark
- [ ] Prefill / Decode disaggregated deployment
- [ ] Multi-data-source and multi-agent support in fully asynchronous mode
- [ ] SFT support
- [ ] LORA support
- [ ] Distributed storage for multi-teacher weights
- [ ] Omni-modal support

______________________________________________________________________

## 🚀 Performance

All curves below come from a single real training run; the metric names in the charts match the framework's built-in metrics. `train/is_approx_k3_kl` is the K3 estimate of the train-inference probability gap, and `timing/step` is the end-to-end time per step.

#### DeepSeek-V4-Flash · SWE ([`dsv4_flash_bf16/swe_h20_8node`](conf/dsv4_flash_bf16/swe_h20_8node.yaml))

![DeepSeek-V4-Flash SWE](./imgs/performance-dsv4-swe.png)

#### DeepSeek-V4-Flash · DAPO ([`dsv4_flash_bf16/dapo_h20_6node`](conf/dsv4_flash_bf16/dapo_h20_6node.yaml))

![DeepSeek-V4-Flash DAPO](./imgs/performance-dsv4-dapo.png)

#### Qwen3-30B-A3B · MOPD Multi-Teacher Full-Vocabulary Distillation ([`qwen3_30b_a3b/mopd_h20_1node`](conf/qwen3_30b_a3b/mopd_h20_1node.yaml))

![Qwen3-30B-A3B MOPD](./imgs/performance-qwen3_30b_a3b-mopd.png)

#### Qwen3-Coder-30B-A3B · OpenCode ([`qwen3_coder_30b_a3b/opencode_h20_4node`](conf/qwen3_coder_30b_a3b/opencode_h20_4node.yaml))

![Qwen3-Coder-30B-A3B OpenCode](./imgs/performance-qwen3coder-opencode.png)

______________________________________________________________________

## 👨‍💻 Contributing

```bash
bash build.sh test                        # Full unit tests and branch coverage
bash build.sh test tests/ut/algorithms    # Specific directory
```

We warmly welcome Issues and Pull Requests from the community! It is recommended to access new capabilities via the aforementioned extension points to avoid modifying the framework's main trunk. For the complete contribution process, commit message conventions, and pre-commit checklists, please carefully read [CONTRIBUTING.md](CONTRIBUTING.md) ([中文](CONTRIBUTING_zh.md)).

______________________________________________________________________

## 🙏 Acknowledgments

The birth of LoongSage is inseparable from a thriving open-source community. We would like to extend our special thanks to the following excellent projects:

LoongSage is built on top of the following foundational projects, which provide powerful capabilities for training, inference, and distributed scheduling:
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) — Core backend for large-scale distributed training and HF ↔ Megatron weight bridging.
- [SGLang](https://github.com/sgl-project/sglang) — High-performance, high-throughput inference engine.
- [Ray](https://github.com/ray-project/ray) — Flexible and highly efficient distributed task scheduler.

In terms of architecture design and code implementation, LoongSage was deeply inspired by and heavily references the following outstanding works:
- [slime](https://github.com/THUDM/slime) — The core code implementation of this project references the architecture and logic of slime, which provided us with significant inspiration and reference, particularly in designing our minimalist architecture. We express our sincere gratitude to the THUDM team!
- [verl](https://github.com/verl-project/verl) — Provided an excellent design paradigm for an extensible RL training and inference framework.
- [Agent-Lightning](https://github.com/microsoft/agent-lightning) — Inspired our design to decouple Agent frameworks from RL post-training platforms.

## 📜 Citation

If LoongSage is helpful to your research, please consider citing our project:

```bibtex
@software{loongsage,
  title  = {LoongSage: An Agent-Native Asynchronous Reinforcement Learning Framework for LLM Post-Training},
  author = {LoongSage Contributors},
  year   = {2026},
  url    = {https://github.com/baidu-baige/LoongSage/}
}
```

## 📄 License

This project is open-sourced under the [Apache License 2.0](./LICENSE).
