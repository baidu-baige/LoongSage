# Adapted from
# https://github.com/THUDM/slime/blob/6961f5970e9dbb4716a10ba4a54a28fa3876d274/slime/utils/health_monitor.py
# Copyright 2025 Zhipu AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#
# Modifications: monitors a ReplicaGroup instead of a slime ServerGroup, and the
# check interval / timeout / first-wait values are instance attributes rather than CLI args.
"""Health monitoring module for rollout engines.

Provides a RolloutHealthMonitor class that monitors the health of
inference engines and can restart failed engines.
"""

import logging
import threading

import ray


logger = logging.getLogger(__name__)


class RolloutHealthMonitor:
    """Health monitor for rollout engines.

    The monitor runs continuously once started, but can be paused/resumed
    based on whether the engines are offloaded (cannot health check when offloaded).

    Lifecycle:
    - start(): Start the monitor thread (called once during initialization)
    - pause(): Pause health checking (called when offloading engines)
    - resume(): Resume health checking (called when onloading engines)
    - stop(): Stop the monitor thread completely (called during dispose)
    """

    def __init__(self, replica_group):
        """Initialize the health monitor.

        Args:
            replica_group: The replica groups to monitor.
        """
        self._replica_group = replica_group

        self._thread = None
        self._stop_event = None
        self._pause_event = None  # When set, health checking is paused
        self._check_interval = 30
        self._check_timeout = 30
        self._check_first_wait = 60
        self._need_first_wait = True  # Need to wait after each resume
        self._is_checking_enabled = False  # Track if health checking should be active

    def start(self) -> bool:
        """Start the health monitor thread. Called once during initialization.

        Returns:
            True if the monitor was started, False if there are no engines to monitor.
        """
        if not self._replica_group.all_engines:
            return False

        if self._thread is not None:
            logger.warning("Health monitor thread is already running.")
            return True

        logger.info("Starting RolloutHealthMonitor...")
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread = threading.Thread(
            target=self._health_monitor_loop,
            name="RolloutHealthMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("RolloutHealthMonitor started (in paused state).")
        return True

    def stop(self) -> None:
        """Stop the health monitor thread completely. Called during dispose."""
        if not self._thread:
            return

        logger.info("Stopping RolloutHealthMonitor...")
        assert self._stop_event is not None
        self._stop_event.set()
        # Also clear pause to let the thread exit
        if self._pause_event:
            self._pause_event.clear()
        timeout = self._check_timeout + self._check_interval + 5
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logging.warning("Rollout health monitor thread did not terminate within %.1fs", timeout)
        else:
            logger.info("RolloutHealthMonitor stopped.")

        self._thread = None
        self._stop_event = None
        self._pause_event = None
        self._is_checking_enabled = False

    def pause(self) -> None:
        """Pause health checking. Called when engines are offloaded."""
        if self._pause_event is None:
            return
        logger.info("Pausing health monitor...")
        self._pause_event.set()
        self._is_checking_enabled = False

    def resume(self) -> None:
        """Resume health checking. Called when engines are onloaded."""
        if self._pause_event is None:
            return
        logger.info("Resuming health monitor...")
        self._need_first_wait = True  # Need to wait after each resume
        self._pause_event.clear()
        self._is_checking_enabled = True

    def is_checking_enabled(self) -> bool:
        """Return whether health checking is currently enabled (not paused)."""
        return self._is_checking_enabled

    def _health_monitor_loop(self) -> None:
        """Main health monitor loop that runs in a separate thread."""
        assert self._stop_event is not None
        assert self._pause_event is not None
        while not self._stop_event.is_set():
            # Wait while paused
            while self._pause_event.is_set() and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)

            if self._stop_event.is_set():
                break
            # Do first wait after each resume (for large MoE models to be ready)
            if self._need_first_wait:
                logger.info(f"Health monitor doing first wait after resume: {self._check_first_wait}s")
                if self._stop_event.wait(self._check_first_wait):
                    logger.info("Health monitor stopped during first wait.")
                    break
                if self._pause_event.is_set():
                    # Got paused during first wait, skip this round and wait again next resume
                    logger.info("Health monitor paused during first wait, will wait again next resume.")
                    continue
                self._need_first_wait = False

            # Run health checks
            if not self._pause_event.is_set() and not self._stop_event.is_set():
                self._run_health_checks()

            # Wait for next check interval
            if self._stop_event.wait(self._check_interval):
                break

    def _run_health_checks(self) -> None:
        """Run health checks on the leader engine of each replica in the group."""
        for rollout_engine_id, engine in enumerate(self._replica_group.engines):
            if self._stop_event is not None and self._stop_event.is_set():
                break
            if self._pause_event is not None and self._pause_event.is_set():
                break
            self._check_engine_health(rollout_engine_id, engine)

    def _check_engine_health(self, rollout_engine_id, engine) -> None:
        """Check the health of a single rollout engine.

        Args:
            rollout_engine_id: The ID of the engine to check.
            engine: The Ray actor for the engine.
        """
        if engine is None:
            logger.info(f"Skipping health check for engine {rollout_engine_id} (None)")
            return

        # Record pause state before blocking call; pause/resume may happen during ray.get()
        was_paused = self._pause_event is not None and self._pause_event.is_set()
        try:
            ray.get(engine.health_generate.remote(timeout=self._check_timeout))
        except Exception as e:

            is_paused = self._pause_event is not None and self._pause_event.is_set()
            if was_paused or is_paused or not self._is_checking_enabled:
                logger.info(
                    f"Health check failed, but paused/disabled, will not kill engine:{rollout_engine_id}."
                )
                return
            logger.error(
                f"Health check failed for rollout engine {rollout_engine_id} "
                f"(ray timeout or error). Killing actor. Exception: {e}"
            )
            self._replica_group.kill_engine(rollout_engine_id=rollout_engine_id)
        else:
            logger.info(f"Health check passed for rollout engine {rollout_engine_id}")

