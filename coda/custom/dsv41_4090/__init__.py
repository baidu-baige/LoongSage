"""Full DeepSeek-V4.1-Flash GRPO on one RTX 4090 (22 GB, SM89).

  reward.py         ``gsm8k_answer_only`` reward
  patch.py          wires the worker into the stock controller when ``dsv41_4090.stream`` is set
  stream_worker.py  actor worker; runs each training step in a subprocess
  streaming.py      that subprocess: layer-paged Megatron model, LoRA GRPO, merge into the
                    serving checkpoint
  conf/             Hydra recipe; launch with ``--config-dir coda/custom/dsv41_4090/conf``
  tools/            offline scripts (pruned checkpoint, serving checkpoint, dataset); not a
                    package, so ``coda.custom`` does not import them
"""
