LoongSage 文档
==============

**LoongSage** 是面向大语言模型的后训练强化学习框架：以 Ray 为调度底座，Megatron-Core
为训练后端，SGLang 为推理后端。目标是用极简架构训练超大模型、支持任意 Agent
零侵入接入，并提供系统级的训推一致性保障。

核心特性
--------

- **Agent 零侵入接入** —— Agent 只需通过标准 chat completion 接口调用框架的推理
  服务，多轮上下文拼接、loss mask、轨迹收集与 Sandbox 全生命周期管理均由
  AgentFlow 自动完成。
- **系统级训推一致性** —— Router Replay、FP32 输出层、TITO、重要性采样修正等一致性
  对齐手段，把训推概率偏差稳定控制在 1e-4 量级，详见
  :doc:`train-inference-consistency`。
- **全异步训练** —— 采样与训练完全解耦、并行推进、时间重叠，数据陈旧度与滑动窗口
  策略可配，详见 :doc:`fully-async-mode`。
- **高可扩展的插件系统** —— 覆盖 Agent、奖励模型、Sandbox、优势估计、策略损失、异步调度等
  12 个扩展点，新增实现后在 yaml 中引用即可热插拔生效，可从
  :doc:`custom-agent` 开始。
- **多范式在线蒸馏** — 原生支持 PG-Style、GKD-Style 及混合蒸馏模式，内置 TopK、全词表、JSD 
  等 KL 蒸馏策略，并支持多教师蒸馏。详见 :doc:`on-policy-distillation`。

从哪里开始
----------

- *第一次跑通一个训练任务？* → :doc:`quick-start`
- *怎么启动、有哪些运行模式？* → :doc:`run-guide`
- *怎么接入自己的 Agent / 奖励函数 / Sandbox？* → :doc:`custom-agent`、
  :doc:`custom-reward`、:doc:`custom-sandbox`
- *某个 yaml 字段是什么含义？* → :doc:`config-reference`

.. toctree::
   :maxdepth: 1
   :caption: 开始使用

   quick-start.md
   run-guide.md

.. toctree::
   :maxdepth: 1
   :caption: 训练

   training-algorithms.md
   on-policy-distillation.md
   fully-async-mode.md
   train-inference-consistency.md
   model-checkpointing.md

.. toctree::
   :maxdepth: 1
   :caption: 扩展开发

   custom-agent.md
   custom-reward.md
   custom-sandbox.md
   custom-algorithm.md
   custom-kl.md
   custom-sliding-window.md

.. toctree::
   :maxdepth: 1
   :caption: 配置参考

   config-reference.md
