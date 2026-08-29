""" Logging utils """
import logging
import sys
from typing import Any

from omegaconf import OmegaConf

_LOGGER_CONFIGURED = False

# Credentials live in the same config object as hyperparameters. Their values must
# never reach a log file or a tracking backend's run config/params, both of which
# are readable by anyone with project (or log) access.
# A new credential field shall be added here explicitly.
_SECRET_KEYS = frozenset({"api_key", "token", "password", "secret", "credential"})
_REDACTED = "***REDACTED***"


def _is_secret_key(key: Any) -> bool:
    """Return True if ``key`` names a credential field."""
    return isinstance(key, str) and key.lower() in _SECRET_KEYS


def redact_secrets(obj: Any) -> Any:
    """Deep-copy ``obj`` with credential values replaced by a placeholder.

    Empty values (``None`` / ``""``) are left untouched so an unset credential
    still reads as unset.
    """
    if isinstance(obj, dict):
        return {k: (_REDACTED if _is_secret_key(k) and v else redact_secrets(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_secrets(v) for v in obj]
    return obj


def _redact_cli_arg(arg: str) -> str:
    """Mask the value of a ``key=value`` CLI override that carries a credential."""
    key, sep, _ = arg.partition("=")
    if not sep or not _is_secret_key(key.lstrip("+~").rpartition(".")[2]):
        return arg
    return f"{key}{sep}{_REDACTED}"


def redact_argv() -> None:
    """Mask credentials in ``sys.argv`` in place.

    wandb copies ``sys.argv`` into the run's config and metadata (see
    ``Settings._args`` in ``wandb.sdk.wandb_settings``), so a credential passed as
    a Hydra override leaks into the run record even though the config we upload is
    redacted. Call this once the CLI has been parsed.
    """
    sys.argv[1:] = [_redact_cli_arg(a) for a in sys.argv[1:]]


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