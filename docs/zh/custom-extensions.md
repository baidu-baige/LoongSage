# 自定义扩展

LoongSage 的扩展点都是"注册表 + 配置驱动"：实现里加一个 `@register_*` 装饰器，配置里按注册名引用。装饰器只在模块被 import 时才执行，所以框架必须先发现你的模块。

[coda/custom/](../../coda/custom/) 就是统一的扩展目录：放进去的每个模块（含子包）都会被自动 import，无论注册的是 agent、reward、sandbox、advantage、policy loss、KL 方法、data filter 还是滑动窗口策略。

```
coda/custom/
├── my_reward.py        # @register_reward("my-reward")
├── my_agent.py         # @register_agent("my-agent")
├── my_advantage.py     # @register_advantage("my-adv")
└── my_project/         # 子包同样会被扫描
    ├── __init__.py
    └── sandbox.py      # @register_sandbox("my-sandbox")
```

不需要在任何地方手动 import 自己的模块，也不需要改框架代码。

## 1. 加载时机

注册表是进程内的，因此 `coda/custom/` 会在每个需要查表的进程各加载一次：

| 进程 | 加载位置 | 覆盖的扩展点 |
| --- | --- | --- |
| driver（训练入口） | [trainer.py](../../coda/controller/trainer.py) 的 `main()` | agent、reward、sandbox、data filter、滑动窗口策略 |
| 训练 / teacher Ray actor | [TrainWorker](../../coda/backends/train_worker.py) / [TeacherWorker](../../coda/backends/teacher_worker.py) 的构造函数 | advantage、policy loss、KL 方法 |

## 2. 错误处理

某个模块 import 失败（例如缺少三方依赖）时只记一条 WARNING 并跳过，其余模块照常注册。自定义扩展没生效时，先在日志里搜 `Skipped custom module`；如果注册名压根没出现，注册表报错信息里会列出当前所有可用名字。

## 3. 与内建目录的关系

reward、agent、sandbox、algorithms 这四个内建包各自也会扫描本包，所以把实现放到 [coda/reward/functions/](../../coda/reward/functions/)、[coda/agentflow/agent/](../../coda/agentflow/agent/)、[coda/agentflow/sandbox/](../../coda/agentflow/sandbox/)、[coda/algorithms/](../../coda/algorithms/) 下同样会被发现 —— 内置实现就放在那里，阅读时可作参考。

新增自己的扩展推荐统一用 `coda/custom/`：一个目录覆盖全部扩展点，且不与框架代码混在一起。
