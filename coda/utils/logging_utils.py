""" Logging utils """
import logging
from typing import Any

from omegaconf import OmegaConf

_LOGGER_CONFIGURED = False

# Credentials live in the same config object as hyperparameters. Their values must
# never reach a log file or a tracking backend's run config/params, both of which
# are readable by anyone with project (or log) access.
_SECRET_KEY_MARKERS = ("api_key", "token", "password", "secret", "credential")
_REDACTED = "***REDACTED***"


def _is_secret_key(key: Any) -> bool:
    """Return True if ``key`` names a credential field."""
    return isinstance(key, str) and any(marker in key.lower() for marker in _SECRET_KEY_MARKERS)


def redact_secrets(obj: Any) -> Any:
    """Deep-copy ``obj`` with credential values replaced by a placeholder.

    Key matching is substring-based, so ``wandb_api_key`` and ``access_token`` are
    covered too. Empty values (``None`` / ``""``) are left untouched so an unset
    credential still reads as unset.
    """
    if isinstance(obj, dict):
        return {k: (_REDACTED if _is_secret_key(k) and v else redact_secrets(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_secrets(v) for v in obj]
    return obj


def redacted_config_yaml(config) -> str:
    """Render an OmegaConf config as YAML with credential values masked."""
    plain = redact_secrets(OmegaConf.to_container(config, resolve=True))
    return OmegaConf.to_yaml(OmegaConf.create(plain))


def configure_logger(prefix: str = "", level="INFO"):
    """ Set logging basic config"""
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging_level = getattr(logging, level.upper(), None)

    logging.basicConfig(
        level=logging_level,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )