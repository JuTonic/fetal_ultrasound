from pathlib import Path

from config.app import AppConfig
from config.providers.yaml import DEFAULT_YAML_PATH, read_yaml_config


def load_config(
    path: Path = DEFAULT_YAML_PATH,
    *,
    env_file: Path | None = Path(".env"),
) -> AppConfig:
    """Load YAML, overlay dotenv and process environment, then validate.

    Environment wins over dotenv; dotenv wins over YAML, including individual
    nested fields. Set env_file=None to disable dotenv in production or tests.
    Missing YAML files and invalid settings fail startup; missing .env is fine.
    """
    return AppConfig(_env_file=env_file, **read_yaml_config(path))
