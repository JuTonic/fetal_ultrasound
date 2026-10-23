from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from config.app import AppConfig

DEFAULT_YAML_PATH = Path("./config.yaml")


def read_yaml_config(path: Path = DEFAULT_YAML_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        raw_config: Any = yaml.safe_load(file)

    if raw_config is None:
        raw_config = {}

    if not isinstance(raw_config, dict):
        raise TypeError(f"Config YAML root must be a mapping: {path!s}")

    return raw_config


def load_yaml_config(path: Path = DEFAULT_YAML_PATH) -> AppConfig:
    """Compatibility entry point; now includes environment and .env overrides."""
    from config.load import load_config

    return load_config(path)
