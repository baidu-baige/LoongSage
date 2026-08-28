"""Filesystem-path helpers shared across Coda.

Config files reference auxiliary files (chat templates, k8s manifests, kubeconfig, ...) by
path. Those paths are resolved relative to the project's `conf/` directory so that a config
stays portable regardless of the process's working directory; absolute paths are used as-is.
"""

from pathlib import Path

# coda/utils/path_utils.py -> parents[2] is the repo root that holds conf/.
CONF_DIR = Path(__file__).resolve().parents[2] / "conf"


def resolve_conf_path(path: str | Path | None) -> str | None:
    """Resolve a config-declared path against the project `conf/` directory.

    Args:
        path: Absolute path (returned unchanged), path relative to `conf/`, or `None`.

    Returns:
        The absolute path as a string, or `None` when *path* is empty.
    """
    if not path:
        return None

    path = Path(path)
    if path.is_absolute():
        return str(path)

    return str((CONF_DIR / path).resolve())
