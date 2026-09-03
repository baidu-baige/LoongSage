# 自定义滑动窗口策略开发指南

本文说明如何在全异步模式下新增一个滑动窗口策略，用于控制 prompt group 的派发与收集节奏。策略同样是 **"注册表 + 配置驱动"**：继承基类、加一个装饰器，即可通过 `fully_async.sliding_window` 引用。三种内置策略（`no-window` / `window-gated` / `windowed-fifo`）的语义、取舍与适用场景见[全异步模式](./fully-async-mode.md)第 3 节。

## 1. 开发步骤

策略实现在 [rollout_sampler.py](../../coda/controller/rollout_sampler.py) 的 `SlidingWindowStrategy` 一节。新增策略时，按以下步骤操作：

1. 继承 `SlidingWindowStrategy`，并至少实现 `compute_dispatch_count(buf_qsize)`。基类已经通过 `on_dispatched()` 和 `on_collected()` 维护 `_running_count`。
2. 如果策略需要维护序号、优先级或其他状态，请覆写 `on_dispatched()`、`on_collected()` 和 `on_reset()`；覆写前两个方法时必须调用 `super()`，以确保计数正确。
3. 如果需要限制 collector 的出队顺序，请覆写 `will_collect(group)`。该方法在 `TrajQueue` 条件锁和策略锁内执行，因此不能执行网络或磁盘 I/O、sleep 以及其他耗时计算，也不能再次获取 queue 锁或修改传入的 group。
4. 使用 `@register_sliding_window_strategy("your-name")` 装饰策略类完成注册，例如 `@register_sliding_window_strategy("latency-aware")`。`create_strategy()` 会自动为其添加线程安全代理。模块放到 [coda/custom/](../../coda/custom/) 下即会被自动 import，详见[自定义扩展](./custom-extensions.md)。
5. 在 `conf/default.yaml` 的注释中补充新的可选值，并为派发、收集、reset、未知 prompt 以及线程安全行为补充单元测试。

## 2. 最小示例

以下是一个最小化的容量控制策略示例：

```python
@register_sliding_window_strategy("fixed-capacity")
class FixedCapacityStrategy(SlidingWindowStrategy):
    def __init__(self, config):
        super().__init__()
        self._capacity = int(config.data_sources[0].num_prompts_per_step)

    def compute_dispatch_count(self, buf_qsize: int) -> int:
        return max(0, self._capacity - self._running_count - buf_qsize)
```

## 3. 配置启用

注册后在全异步配置中按注册名引用即可：

```yaml
fully_async:
  enable: true
  sliding_window: fixed-capacity   # ← @register_sliding_window_strategy 的注册名
  stale_steps: 1.0
```
