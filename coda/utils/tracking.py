"""
A unified tracking interface that supports logging data to different backend
Ref: https://github.com/verl-project/verl/blob/main/verl/utils/tracking.py
"""
import dataclasses
import functools
import inspect
import json
import logging
import os
import re
import tempfile
import time
import torch
import pprint
import numbers
from contextlib import contextmanager
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Optional

import mlflow
import pandas as pd
import wandb
from omegaconf import OmegaConf

from coda.utils.logging_utils import redact_secrets

logger = logging.getLogger(__name__)

# Retry configuration for MLflow initialization
MLFLOW_MAX_ATTEMPTS = 3
MLFLOW_SLEEP_SECONDS = 5

def configure_tracking(config):
    """Initialize the Tracking singletons from the given config."""
    Tracking.init(
        config.tracking.project_name,
        config.tracking.experiment_name,
        config.tracking.tracking_backend,
        config=config,
    )
    return Tracking.get_instance().run_id

def track(metrics, step):
    """Log a dictionary of metrics to all configured tracking backends.

    Args:
        metrics (dict): Key-value pairs of metric names and their values,
            e.g. ``{"loss": 0.5, "reward": 1.2}``.
        step (int): The training step associated with the metrics.

    Raises:
        RuntimeError: If ``configure_tracking()`` has not been called yet.
    """
    tracker = Tracking.get_instance()
    if tracker is None:
        raise RuntimeError("Tracker is not initialized. Call configure_tracking() first.")
    tracker.log(data=metrics, step=step)

class TimeMarkerAcc:
    """Accumulating variant of time_marker that collects timing across
    multiple invocations and reports once on exit.

    Usage::

        with TimeMarkerAcc(step=step) as timers:
            for mb_idx in range(num_mini_batches):
                with timers("rollout"):
                    ...
                with timers("train"):
                    ...
        # auto-reports {"timing/rollout": ..., "timing/train": ...} on exit
    """

    def __init__(self, step: int):
        self.step = step
        self._timers: dict[str, float] = {}
        self._start_time: dict[str, float] = {}

    def __enter__(self):
        # Start the "wait" clock on region entry; inverse_timer("wait") pauses it
        # during training, so it accumulates all non-train time.
        self.start("wait")
        return self

    def elapsed(self, name: str) -> float:
        """Return accumulated seconds for ``name`` (0.0 if never timed)."""
        return self._timers.get(name, 0.0)

    def start(self, name: str):
        """Start (or resume) the ``name`` clock; must not already be running."""
        assert name not in self._start_time, f"Timer {name} already started."
        self._start_time[name] = time.time()

    def end(self, name: str):
        """Stop the ``name`` clock and bank its interval into ``_timers``."""
        assert name in self._start_time, f"Timer {name} not started."
        self._timers[name] = self._timers.get(name, 0.0) + (time.time() - self._start_time.pop(name))

    @contextmanager
    def inverse_timer(self, name: str):
        """Accumulate time spent *outside* the wrapped block.

        The named clock must already be running; it is ended on entry (banking the
        elapsed wait) and restarted on exit. Wrapping the train call therefore
        measures everything except training as the wait.
        """
        self.end(name)
        try:
            yield
        finally:
            self.start(name)

    def __exit__(self, *exc):
        # End any still-running clocks (e.g. an inverse_timer's wait clock) so
        # their final interval is banked before reporting.
        for name in list(self._start_time):
            self.end(name)
        for name, elapsed in self._timers.items():
            # "wait" is internal (drives perf/wait_ratio), not a timing/* metric.
            if name == "wait":
                continue
            track({f"timing/{name}": f"{elapsed:.4f}"}, step=self.step)
            logger.info("[step %d] %s done, elapsed=%.4fs", self.step, name, elapsed)
        return False

    @contextmanager
    def __call__(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.end(name)


@contextmanager
def time_marker(name: str, step: int):
    """Context manager that measures a code block's wall-clock time, logs it,
    and tracks the metric via the configured tracking backends.

    Usage::

        with time_marker("rollout", step=step):
            result = await self.rollout_sampler(step, timeout=timeout)

    This replaces the repetitive pattern of::

        t0 = time.time()
        ...
        track({"timing/<name>": elapsed}, step=step)
        logger.info("[step %d] <name> done, elapsed=%.4fs", step, elapsed)

    Args:
        name: A short label for the timed block (e.g. "rollout", "train").
            Used in both the metric key (``timing/<name>``) and the log message.
        step: The training step, forwarded to ``track()`` and included in logs.
    """
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    track({f"timing/{name}": f"{elapsed:.4f}"}, step=step)
    logger.info("[step %d] %s done, elapsed=%.4fs", step, name, elapsed)


class TrainMetricsAggregator:
    """Process-local accumulator for training metrics in fully_async mode.

    Aggregation is decided purely by the metric key, following the unified
    ``<domain>/<object_statistic>`` two-level naming convention:

    * ``timing/*``               -> sum   (wall-clock durations accumulate)
    * key ending in ``_max``     -> max   (e.g. ``perf/train_memory_allocated_max``)
    * key ending in ``_min``     -> min
    * everything else            -> mean
    """

    def __init__(self) -> None:
        self._step: Optional[int] = None
        self._values: dict[str, list[float]] = {}

    def add(self, data: dict[str, Any], step: int) -> None:
        """Buffer a batch of metrics for ``step``.

        If ``step`` differs from the currently buffered step, emit a warning and
        drop the previously buffered values -- callers are expected to call
        flush() at every step boundary.
        """
        if self._step is None:
            self._step = step
        elif self._step != step:
            logger.warning(
                "TrainMetricsAggregator: step changed from %s to %s without "
                "explicit flush; discarding %d buffered keys.",
                self._step, step, len(self._values),
            )
            self._values = {}
            self._step = step

        for k, v in data.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            self._values.setdefault(k, []).append(fv)

    def flush(self) -> Optional[tuple[int, dict[str, float]]]:
        """Return ``(step, aggregated_dict)`` and reset state, or ``None`` if empty"""
        if self._step is None or not self._values:
            self._step = None
            self._values = {}
            return None
        step = self._step
        result: dict[str, float] = {}
        for k, vals in self._values.items():
            if k.startswith("timing/"):
                result[k] = sum(vals)
            elif k.endswith("_max"):
                result[k] = max(vals)
            elif k.endswith("_min"):
                result[k] = min(vals)
            else:
                result[k] = sum(vals) / len(vals)
        self._step = None
        self._values = {}
        return step, result


class _TrackBuffer:
    """Module-level holder for the optional process-local TrainMetricsAggregator"""
    _aggregator: Optional["TrainMetricsAggregator"] = None

    @classmethod
    def install(cls, aggregator: "TrainMetricsAggregator") -> None:
        """Register the process-local aggregator that intercepts ``Tracking.log()`` calls."""
        cls._aggregator = aggregator

    @classmethod
    def get(cls) -> Optional["TrainMetricsAggregator"]:
        """Return the currently installed aggregator, or ``None`` if none has been installed."""
        return cls._aggregator


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including Wandb, MLflow, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    _instance: Optional["Tracking"] = None
    run_id: Optional[str] = None
    supported_backend = [
        "wandb",
        "mlflow",
        "console",
    ]

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance is created (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, project_name, experiment_name, default_backend: str | list[str] = "console", config=None):
        """Initialize the Tracking instance and set up all requested backend loggers.

        Skips re-initialization if the singleton has already been configured.

        Args:
            project_name: Name of the project, used as the experiment group in backends.
            experiment_name: Name of the specific run or experiment.
            default_backend: One or more backend names to enable ('wandb', 'mlflow', 'console').
            config: OmegaConf config object containing trainer and hyperparameter settings.
        """
        if self._initialized:
            return

        # Normalize backend to a list for uniform handling
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}

        # Convert OmegaConf config to a plain dict for use with backend SDKs
        config_dict = OmegaConf.to_container(config, resolve=True)

        if "wandb" in default_backend:
            self._init_wandb(config_dict, project_name, experiment_name)

        if "mlflow" in default_backend:
            # Retry MLflow initialization in case of transient connection errors
            for _mlflow_attempt in range(1, MLFLOW_MAX_ATTEMPTS + 1):
                try:
                    mlflow_tracking_uri = config_dict["tracking"].get("mlflow_tracking_uri", "sqlite:////tmp/mlruns.db")
                    logger.info("Using MLFlow tracking URI: %s", mlflow_tracking_uri)
                    mlflow.set_tracking_uri(mlflow_tracking_uri)

                    # Project_name is actually experiment_name in MLFlow
                    # If experiment does not exist, will create a new experiment
                    experiment = mlflow.set_experiment(project_name)
                    if config_dict["tracking"].get("run_id") is None:
                        active_run = mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)
                        if active_run:
                            self.run_id = active_run.info.run_id
                            logger.info(f"view mlflow run at: "
                                  f"{mlflow_tracking_uri}#/experiments/{experiment.experiment_id}/runs/{self.run_id}")
                            # Log all hyperparameters as Mlflow params at run start
                            # (credentials stripped -- params are visible to all viewers).
                            params = _compute_mlflow_params_from_objects(redact_secrets(config_dict))
                            # mlflow.log_params only accepts str/int/float/bool values;
                            # stringify anything else to avoid "must be of type dict" errors.
                            params = {k: v if isinstance(v, (str, int, float, bool)) else str(v)
                                    for k, v in params.items()}
                            mlflow.log_params(params)
                    else:
                        active_run = mlflow.start_run(
                            run_id=config_dict["tracking"].get("run_id"),
                            experiment_id=experiment.experiment_id,
                            run_name=experiment_name
                        )
                        if active_run:
                            self.run_id = active_run.info.run_id
                    self.logger["mlflow"] = _MlflowLoggingAdapter(run_id=self.run_id)
                    break  # Success
                except Exception as e:
                    logger.warning(
                        "MLflow initialization attempt %d/%d failed: %s", _mlflow_attempt, MLFLOW_MAX_ATTEMPTS, e
                    )
                    if _mlflow_attempt < MLFLOW_MAX_ATTEMPTS:
                        time.sleep(MLFLOW_SLEEP_SECONDS)
                    else:
                        logger.warning("All MLflow initialization attempts failed. Proceeding without MLflow tracking.")

        if "console" in default_backend:
            # Use a local console logger that prints metrics to stdout
            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        self._initialized = True

    # https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
    def _init_wandb(self, config_dict: dict, project_name: str, experiment_name: str) -> None:
        """Initialize wandb backend.

        ``config.tracking.wandb_args`` is forwarded as a single dict: its keys are unpacked verbatim
        into ``wandb.Settings(**wandb_args)`` and thus take effect in ``wandb.init``. Do not put
        ``x_primary`` in it — this method passes ``x_primary`` itself, and a duplicate would raise.

        ``x_primary`` is derived from ``config.tracking.run_id``:
        - no run_id: primary (creates a new run)
        - run_id present: not primary (attach to existing run)

        NOTE: ``run_id`` is not a ``conf/`` key; nothing populates it before this runs, so today the
        primary branch is always taken. Resuming or assigning a run_id is not supported yet.
        """
        tracking = config_dict["tracking"]
        # run_id is None if this is a new run, otherwise it's the run_id to attach to.
        run_id = tracking.get("run_id")
        wandb_args = dict(tracking.get("wandb_args", {}))

        active_run = wandb.init(
            project=project_name,
            name=experiment_name,
            id=run_id,
            # The run config is visible to everyone with project access, so upload a
            # redacted copy; wandb_args below still carries the real credentials.
            config=redact_secrets(config_dict),
            settings=wandb.Settings(
                **wandb_args,
                x_primary=not run_id,
            ),
        )

        # All metrics use step as x-axis (matches Tracking.log() which
        # injects step into every wandb.log call).
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

        self.run_id = active_run.id
        self.logger["wandb"] = wandb

    @classmethod
    def init(
        cls,
        project_name: str,
        experiment_name: str,
        default_backend: str | list[str] = "console",
        config=None,
    ) -> "Tracking":
        """Initialize and return the singleton instance.

        Subsequent calls are no-ops and return the already-initialized instance.
        """
        return cls(project_name, experiment_name, default_backend, config)

    @classmethod
    def get_instance(cls) -> Optional["Tracking"]:
        """Return the singleton instance, or None if not yet initialized."""
        return cls._instance if (cls._instance is not None and cls._instance._initialized) else None

    @classmethod
    def reset(cls):
        """Clear the singleton so it can be re-initialized (mainly for testing)."""
        cls._instance = None

    def log(self, data, step, backend=None, _bypass_buffer=False):
        """Log data to all configured backends, or only to the specified one(s).

        When a ``TrainMetricsAggregator`` is installed (see
        ``install_train_metrics_aggregator``), incoming metrics are buffered and
        not forwarded to backends until ``flush_train_metrics()`` is called.
        ``_bypass_buffer=True`` is the escape hatch the flush path uses to write
        the aggregated record straight through.
        """
        if not _bypass_buffer:
            aggregator = _TrackBuffer.get()
            if aggregator is not None:
                aggregator.add(data, step)
                return
        for default_backend, logger_instance in self.logger.items():
            # Log to all backends if no filter is specified, otherwise match by name
            if backend is None or default_backend in backend:
                if default_backend == "wandb":
                    # wandb's step argument is not used here; step is included as a
                    # data field so it works under shared-mode runs too.
                    logger_instance.log(data={**data, "step": step})
                else:
                    logger_instance.log(data=data, step=step)

    def __del__(self):
        """Gracefully finish the Wandb run on object destruction."""
        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)

class _MlflowLoggingAdapter:
    """Adapter that wraps MLflow metric logging with key sanitization."""

    def __init__(self, run_id: Optional[str] = None):
        """Set up regex patterns and caches for MLflow metric key sanitization."""
        self.run_id = run_id
        self.logger = logging.getLogger(__name__)
        # Suppress noisy "Found credentials from IAM Role" on every MLflow request
        logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
        # MLflow metric key validation logic:
        # https://github.com/mlflow/mlflow/blob/master/mlflow/utils/validation.py#L157C12-L157C44
        # Only characters allowed: slashes, alphanumerics, underscores, periods, dashes, colons,
        # and spaces.
        self._invalid_chars_pattern = re.compile(
            r"[^/\w.\- :]"
        )  # Allowed: slashes, alphanumerics, underscores, periods, dashes, colons, and spaces.
        self._consecutive_slashes_pattern = re.compile(r"/+")
        # Cache sanitized keys to avoid redundant regex operations
        self._sanitized_key_cache = {}

    def _sanitize_key(self, key):
        """Return a sanitized copy of the metric key that is valid for MLflow."""
        # Return cached result if available
        if key in self._sanitized_key_cache:
            return self._sanitized_key_cache[key] or key
        # First replace @ with _at_ for backward compatibility
        sanitized = key.replace("@", "_at_")
        # Replace consecutive slashes with a single slash (MLflow treats them as file paths)
        sanitized = self._consecutive_slashes_pattern.sub("/", sanitized)
        # Then replace any other invalid characters with _
        sanitized = self._invalid_chars_pattern.sub("_", sanitized)
        # Cache None to indicate no sanitization was needed (avoids string duplication)
        if sanitized == key:
            self._sanitized_key_cache[key] = None
        else:
            self.logger.warning("[MLflow] Metric key '%s' sanitized to '%s' due to invalid characters.", key, sanitized)
            self._sanitized_key_cache[key] = sanitized
        return sanitized

    def log(self, data, step):
        """Sanitize metric keys and log them to MLflow."""
        # Sanitize all keys before logging to prevent MLflow validation errors
        results = {self._sanitize_key(k): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step, run_id=self.run_id)


def install_train_metrics_aggregator() -> None:
    """Install a process-local ``TrainMetricsAggregator``.

    While installed, every ``Tracking.log()`` call (e.g. ``track()``, the
    ``time_tracker`` decorator, etc.) is intercepted and buffered instead of
    being written to the configured backends. ``flush_train_metrics()`` drains
    the buffer and emits one aggregated record.
    """
    if _TrackBuffer.get() is None:
        _TrackBuffer.install(TrainMetricsAggregator())


def flush_train_metrics() -> None:
    """Aggregate buffered metrics and emit one record to the configured backends.

    No-op when no aggregator is installed or when nothing has been buffered.
    """
    aggregator = _TrackBuffer.get()
    if aggregator is None:
        return
    flushed = aggregator.flush()
    if flushed is None:
        return
    step, agg_dict = flushed
    tracker = Tracking.get_instance()
    if tracker is None:
        return
    tracker.log(data=agg_dict, step=step, _bypass_buffer=True)


def time_tracker(func=None, func_name: Optional[str] = None):
    """Decorator that measures a function's wall-clock execution time and logs it
    to the configured tracking backends (wandb / mlflow / console) via the
    ``Tracking`` singleton.

    Can be used with or without arguments::

        @time_tracker
        def train_step(self, batch, step): ...

        @time_tracker("custom_train")
        def train_step(self, batch, step): ...

        @time_tracker(func_name="custom_train")
        def train_step(self, batch, step): ...

    Step resolution (in priority order):

    1. If the wrapped function declares a ``step`` parameter, its runtime value
       is used directly as the logging step — the caller already manages the
       training step counter.
    2. Otherwise an internal ``call_count`` is incremented on every invocation
       and used as the step, so no external step management is required.

    Args:
        func: The function to wrap. Supplied automatically when used as
            ``@time_tracker`` without parentheses.
        func_name: Optional metric name override. When provided, the logged
            metric key becomes ``timing/<func_name>`` instead of
            ``timing/<func.__qualname__>``.
    """
    # Support @time_tracker("name") syntax
    if isinstance(func, str):
        func_name = func
        func = None

    def decorator(fn):
        call_count = {"value": 0}
        sig = inspect.signature(fn)
        has_step = "step" in sig.parameters
        metric_name = func_name if func_name is not None else fn.__qualname__

        def _resolve_step(args, kwargs) -> Optional[int]:
            """Extract the ``step`` value from call arguments when available."""
            if not has_step:
                return None
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return bound.arguments.get("step", None)

        def _log_elapsed(elapsed: float, step: Optional[int]) -> None:
            # Only main rank reports timing metrics
            if torch.distributed.get_rank() != 0:
                return

            
            tracker = Tracking.get_instance()
            if tracker is None:
                return
            if step is None:
                call_count["value"] += 1
                step = call_count["value"]
            tracker.log(
                data={f"timing/{metric_name}": elapsed},
                step=step,
            )

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            step = _resolve_step(args, kwargs)
            start = time.time()
            try:
                return await fn(*args, **kwargs)
            finally:
                _log_elapsed(time.time() - start, step)

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            step = _resolve_step(args, kwargs)
            start = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                _log_elapsed(time.time() - start, step)

        return async_wrapper if inspect.iscoroutinefunction(fn) else sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def _compute_mlflow_params_from_objects(params) -> dict[str, Any]:
    """Convert arbitrary config objects to a flat dict suitable for MLflow params."""
    if params is None:
        return {}

    # Flatten the JSON-serializable config into a single-level dict using '/' as separator
    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    """Recursively convert objects to JSON-serializable types.

    Lists are optionally converted to dicts keyed by index so that MLflow
    can store them as individual params rather than as a raw list string.
    """
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        # Convert dataclass to dict first, then recurse
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            # Store list length alongside indexed entries for full fidelity
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        # Serialize Path objects as plain strings
        return str(x)
    if isinstance(x, Enum):
        # Use the enum's underlying value for serialization
        return x.value

    return x


def _flatten_dict(raw: dict[str, Any], *, sep: str) -> dict[str, Any]:
    """Flatten a nested dict into a single-level dict using the given separator."""
    if not raw:
        return {}
    # pandas json_normalize handles arbitrary nesting depth
    records = pd.json_normalize(raw, sep=sep).to_dict(orient="records")
    ans = records[0] if records else {}
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:
    """Logger for recording model-generated text during validation steps."""

    project_name: str = None
    experiment_name: str = None

    def log(self, loggers, trajectories, step):
        """Dispatch validation generation logging to all active backends."""
        if "wandb" in loggers:
            self.log_generations_to_wandb(trajectories, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(trajectories, step)

    def log_generations_to_wandb(self, trajectories, step):
        """Public entry point for logging generations to Wandb."""
        self._log_generations_to_wandb(trajectories, step, wandb)

    def _log_generations_to_wandb(self, trajectories, step, wandb):
        """Log trajectories to wandb as a table"""

        # Create column names for all trajectories
        columns = ["step"] + sum(
            [[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(trajectories))], []
        )

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for traj in trajectories:
            row_data.extend(traj)

        new_table.add_data(*row_data)

        # Update reference and log
        if wandb.run is not None:
            wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_mlflow(self, trajectories, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Write generation results to a temporary JSON file before uploading
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for traj in trajectories:
                    data = {"input": traj[0], "output": traj[1], "score": traj[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

def concat_dict_to_str(dict: dict, step):
    """Format a metrics dictionary into a human-readable single-line string.

    Only numeric values are included. The step is prepended as the first field.

    Args:
        dict (dict): Key-value pairs of metric names and values. Non-numeric
            values are silently skipped.
        step (int): Current training step, displayed at the start of the string.

    Returns:
        str: A formatted string such as ``"step:10 - loss:0.5 - reward:1.2"``.
    """
    output = [f"step:{step}"]
    for k, v in dict.items():
        if isinstance(v, numbers.Number):
            output.append(f"{k}:{pprint.pformat(v)}")
    output_str = " - ".join(output)
    return output_str

class LocalLogger:
    """
    A local logger that logs messages to the console.

    Args:
        print_to_console (bool): Whether to print to the console.
    """

    def __init__(self, print_to_console=True):
        self.print_to_console = print_to_console

    def flush(self):
        """No-op flush for interface compatibility with other logger backends."""

    def log(self, data, step):
        """Print metrics to the console if ``print_to_console`` is enabled.

        Args:
            data (dict): Key-value pairs of metric names and values.
            step (int): Current training step associated with the metrics.
        """
        if self.print_to_console:
            print(concat_dict_to_str(data, step=step), flush=True)
