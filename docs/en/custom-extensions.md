# Custom Extensions

Every LoongSage extension point is **registry- and config-driven**: add a `@register_*` decorator to your implementation and reference it by its registered name from configuration. A decorator only runs when its module is imported, so the framework has to discover your module first.

[coda/custom/](../../coda/custom/) is the single directory for that: every module placed there (sub-packages included) is imported automatically, whether it registers an agent, a reward, a sandbox, an advantage estimator, a policy loss, a KL method, a data filter, or a sliding window strategy.

```
coda/custom/
├── my_reward.py        # @register_reward("my-reward")
├── my_agent.py         # @register_agent("my-agent")
├── my_advantage.py     # @register_advantage("my-adv")
└── my_project/         # sub-packages are scanned too
    ├── __init__.py
    └── sandbox.py      # @register_sandbox("my-sandbox")
```

You never have to import your own module by hand, and you never have to touch framework code.

## 1. When It Is Loaded

Registries are per-process, so `coda/custom/` is loaded once in every process that resolves names from a registry:

| Process | Load site | Extension points covered |
| --- | --- | --- |
| Driver (training entry point) | `main()` in [trainer.py](../../coda/controller/trainer.py) | agent, reward, sandbox, data filter, sliding window strategy |
| Train / teacher Ray actors | [TrainWorker](../../coda/backends/train_worker.py) / [TeacherWorker](../../coda/backends/teacher_worker.py) constructors | advantage, policy loss, KL method |

## 2. Error Handling

If a module fails to import — a missing third-party dependency, for instance — it is skipped with a single WARNING and the remaining modules still register. When a custom extension does not take effect, grep the log for `Skipped custom module` first; if the registered name never appears at all, the registry's error message lists every name currently available.

## 3. Relationship to the Built-in Directories

The reward, agent, sandbox, and algorithms packages each also scan themselves, so an implementation placed in [coda/reward/functions/](../../coda/reward/functions/), [coda/agentflow/agent/](../../coda/agentflow/agent/), [coda/agentflow/sandbox/](../../coda/agentflow/sandbox/), or [coda/algorithms/](../../coda/algorithms/) is discovered as well — that is where the built-in implementations live, and they are worth reading as references.

For your own extensions, prefer `coda/custom/`: one directory covers every extension point and keeps your code out of the framework tree. 
