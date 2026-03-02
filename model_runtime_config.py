"""Helpers for loading runtime model routing settings from config.yaml."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ModelRuntimeConfig:
    """Resolved model/runtime routing config."""

    model: str
    base_url: str
    provider: str = ""
    extra_body: Optional[Dict[str, Any]] = None


def load_model_runtime_config(
    config_path: Path,
    *,
    default_model: str,
    default_base_url: str,
    logger=None,
) -> ModelRuntimeConfig:
    """Resolve model routing settings from ``config.yaml``.

    Supports both styles:
    - ``model: "provider/model"``
    - ``model: {default, base_url, provider, extra_body}``
    """
    resolved = ModelRuntimeConfig(
        model=default_model,
        base_url=default_base_url,
    )

    if not config_path.exists():
        return resolved

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except UnicodeDecodeError:
        with open(config_path, "r", encoding="latin-1") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        if logger:
            logger.debug("Could not read %s: %s", config_path, e)
        return resolved

    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, str):
        resolved.model = model_cfg
        return resolved

    if not isinstance(model_cfg, dict):
        return resolved

    resolved.model = model_cfg.get("default", resolved.model)
    resolved.base_url = model_cfg.get("base_url", resolved.base_url)
    resolved.provider = str(model_cfg.get("provider", "") or "")

    raw_extra_body = model_cfg.get("extra_body")
    if isinstance(raw_extra_body, dict):
        resolved.extra_body = copy.deepcopy(raw_extra_body)
    elif raw_extra_body is not None and logger:
        logger.warning(
            "Ignoring model.extra_body because it is not a mapping (got %s)",
            type(raw_extra_body).__name__,
        )

    return resolved
