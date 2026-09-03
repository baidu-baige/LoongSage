LoongSage Documentation
=======================

**LoongSage** is a post-training reinforcement learning framework for large language
models: Ray as the scheduling foundation, Megatron-Core as the training
backend, SGLang as the inference backend. It aims at training ultra-large
models with a minimalist architecture, zero-intrusion access for any agent, and
system-level train/inference consistency guarantees.

Highlights
----------

- **Zero-intrusion agents** — an agent only talks to the framework's inference
  service through a standard chat-completion API; multi-turn context
  concatenation, loss masking, trajectory collection and sandbox lifecycles are
  all handled by AgentFlow.
- **Consistency by construction** — Router Replay, FP32 output layer, TITO,
  importance-sampling correction and other alignment techniques keep the
  train/inference probability bias at the 1e-4 magnitude, see
  :doc:`train-inference-consistency`.
- **Fully asynchronous training** — sampling and training are decoupled,
  advance in parallel and overlap in time, with configurable staleness and
  sliding windows, see :doc:`fully-async-mode`.
- **Highly extensible plugin system** — 12 extension points covering agents,
  reward models, sandboxes, advantage estimation, policy loss, asynchronous
  scheduling and more; add an implementation and reference it from yaml to
  hot-plug it in, starting with :doc:`custom-agent`.
- **Multi-paradigm on-policy distillation** — native support for PG-style,
  GKD-style and hybrid distillation, with built-in TopK, full-vocabulary and
  JSD KL strategies, as well as multi-teacher distillation, see
  :doc:`on-policy-distillation`.

Where to start
--------------

- *Getting a first training run going?* → :doc:`quick-start`
- *How do I launch, and which run modes exist?* → :doc:`run-guide`
- *How do I plug in my own agent, reward function or sandbox?* →
  :doc:`custom-agent`, :doc:`custom-reward`, :doc:`custom-sandbox`
- *What does a given yaml field do?* → :doc:`config-reference`

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   quick-start.md
   run-guide.md

.. toctree::
   :maxdepth: 1
   :caption: Training

   training-algorithms.md
   on-policy-distillation.md
   fully-async-mode.md
   train-inference-consistency.md
   model-checkpointing.md

.. toctree::
   :maxdepth: 1
   :caption: Extension Guides

   custom-extensions.md
   custom-agent.md
   custom-reward.md
   custom-sandbox.md
   custom-algorithm.md
   custom-kl.md
   custom-sliding-window.md

.. toctree::
   :maxdepth: 1
   :caption: Reference

   config-reference.md
