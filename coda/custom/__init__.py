"""Single drop-in directory for user extensions.

Every module placed under ``coda/custom/`` is imported when this package is
imported, so the ``@register_*`` decorators inside it execute.  This gives one
location for all registry-based extension points instead of one directory per
subsystem::

    coda/custom/my_reward.py     ->  @register_reward("my_reward")
    coda/custom/my_agent.py      ->  @register_agent("my_agent")
    coda/custom/my_advantage.py  ->  @register_advantage("my_adv")
    coda/custom/my_filter.py     ->  @register_data_filter("my_filter")

Sub-packages work too, so a larger extension can keep its own directory.

Registries are per-process, so this package is imported once per process that
resolves names from a registry: the training entry point (``coda.controller.trainer``)
for the driver-side registries (reward, agent, sandbox, sliding window, data
filter), and the ``TrainWorker`` / ``TeacherWorker`` base constructors for the
algorithm-side registries (advantage, policy loss, KL policy) that are looked up
inside Ray actors.

Import ordering: this package must only ever be imported from an entry point,
never from a framework module that a custom module might import back, otherwise
a custom module importing ``coda.<subsystem>`` can hit a partially initialised
package.
"""

from __future__ import annotations

import logging
import pkgutil

logger = logging.getLogger(__name__)

for _, _module_name, _ in pkgutil.walk_packages(__path__, __name__ + "."):
    try:
        __import__(_module_name)
    except ImportError as _exc:
        # One custom module with an unavailable dependency must not stop the
        # others from registering.  Warn rather than debug: unlike the built-in
        # packages, everything here was put there deliberately by the user.
        logger.warning("Skipped custom module %s: %s", _module_name, _exc)
    else:
        logger.info("Loaded custom module %s", _module_name)
