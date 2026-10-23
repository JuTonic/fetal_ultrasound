from __future__ import annotations

from pydantic_settings import (BaseSettings, PydanticBaseSettingsSource,
                               SettingsConfigDict)

from config.infrastructure import InfrastructureConfig


class AppConfig(BaseSettings):
    """Validated settings: environment > dotenv > YAML/init > field defaults."""

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
    )

    default: str = "default"
    infrastructure: InfrastructureConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # YAML is passed as init data by load_config(). Pydantic deep-merges
        # source dictionaries before validating the nested infrastructure models.
        return env_settings, dotenv_settings, init_settings
