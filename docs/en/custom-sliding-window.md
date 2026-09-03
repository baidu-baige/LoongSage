# Custom Sliding Window Strategy Development Guide

This document explains how to add a sliding window strategy for fully async mode, controlling the pace at which prompt groups are dispatched and collected. Strategies are likewise **"registry + config-driven"**: inherit the base class, add a decorator, and it can be referenced via `fully_async.sliding_window`. For the semantics, trade-offs and applicable scenarios of the three built-in strategies (`no-window` / `window-gated` / `windowed-fifo`), see Section 3 of [Fully Async Mode](./fully-async-mode.md).

## 1. Development Steps

The strategy implementations live in the `SlidingWindowStrategy` section of [rollout_sampler.py](../../coda/controller/rollout_sampler.py). To add a new strategy, follow these steps:

1. Inherit from `SlidingWindowStrategy` and implement at least `compute_dispatch_count(buf_qsize)`. The base class already maintains `_running_count` through `on_dispatched()` and `on_collected()`.
2. If the strategy needs to maintain sequence numbers, priorities, or other state, override `on_dispatched()`, `on_collected()`, and `on_reset()`; when overriding the first two, `super()` must be called to keep counts correct.
3. If the collector's dequeue order needs to be constrained, override `will_collect(group)`. This method is executed inside the `TrajQueue` condition lock and the strategy lock, so it must not perform network or disk I/O, sleep, or other time-consuming computation, nor re-acquire the queue lock or modify the incoming group.
4. Register the strategy class by decorating it with `@register_sliding_window_strategy("your-name")`, e.g., `@register_sliding_window_strategy("latency-aware")`. `create_strategy()` will automatically add a thread-safe proxy for it. Place the module in [coda/custom/](../../coda/custom/) so it is imported automatically, see [Custom Extensions](./custom-extensions.md).
5. Add the new option to the comments in `conf/default.yaml`, and add unit tests for dispatch, collection, reset, unknown prompt, and thread safety behavior.

## 2. Minimal Example

Below is a minimal capacity-control strategy example:

```python
@register_sliding_window_strategy("fixed-capacity")
class FixedCapacityStrategy(SlidingWindowStrategy):
    def __init__(self, config):
        super().__init__()
        self._capacity = int(config.data_sources[0].num_prompts_per_step)

    def compute_dispatch_count(self, buf_qsize: int) -> int:
        return max(0, self._capacity - self._running_count - buf_qsize)
```

## 3. Config Enablement

Once registered, reference it by name in the fully-async configuration:

```yaml
fully_async:
  enable: true
  sliding_window: fixed-capacity   # ← the @register_sliding_window_strategy registered name
  stale_steps: 1.0
```
