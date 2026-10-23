import json
from pathlib import Path

from fetal.config.app import AppConfig

DEFAULT_CONFIG_PATH = Path("configs/pipeline.json")


def load_config(
    path: Path = DEFAULT_CONFIG_PATH,
    *,
    env_file: Path | None = Path(".env"),
) -> AppConfig:
    """Read JSON, overlay dotenv/process environment, then validate once."""
    with path.open(encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, dict):
        raise TypeError(f"Config JSON root must be an object: {path}")
    return AppConfig(_env_file=env_file, **values)
