"""Training algorithms: advantage estimators, policy losses, KL policies (OPD).

The registries themselves live in :mod:`coda.algorithms.registry`; every module
in this package registers its implementations there through the ``@register_*``
decorators.
"""

# Auto-discover built-in algorithms so their @register_* decorators execute:
# importing any ``coda.algorithms.*`` module is enough to populate the
# registries, so a new algorithm only has to be dropped into this package.
import pkgutil as _pkgutil

for _, _name, _ in _pkgutil.walk_packages(__path__, __name__ + "."):
    __import__(_name)
