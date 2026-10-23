from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from fetal.config.models import Architecture, ClusterConfig, Seed, TeamModelConfig, TrainingConfig


class AppConfig(BaseSettings):
    """Settings precedence: environment > dotenv > JSON > field defaults."""

    model_config = SettingsConfigDict(
        env_prefix="FETAL_",
        env_nested_delimiter="__",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
    )

    architectures: tuple[Architecture, ...] = Field(min_length=1)
    training: TrainingConfig = TrainingConfig()
    finalist_seeds: tuple[Seed, ...] = Field(default=(0, 1), min_length=2)
    team_model: TeamModelConfig = TeamModelConfig()
    cluster: ClusterConfig = ClusterConfig()
    note: str = ""

    @field_validator("architectures", "finalist_seeds")
    @classmethod
    def validate_unique(cls, values: tuple) -> tuple:
        if len(values) != len(set(values)):
            raise ValueError("Architectures and finalist seeds must not contain duplicates")
        return values

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return env_settings, dotenv_settings, init_settings
