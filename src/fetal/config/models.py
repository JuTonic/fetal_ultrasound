"""Frozen configuration records validated at the file/environment boundary."""
from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Architecture = Literal["resnet18", "resnet50", "densenet121", "convnext_tiny", "swin_t"]
Seed = Annotated[int, Field(ge=0)]


class FrozenConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)


class TrainingConfig(FrozenConfig):
    epochs: int = Field(default=6, ge=1)
    batch_size: int = Field(default=32, ge=2)
    num_workers: int = Field(default=4, ge=0)
    lr: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    precision: Literal["fp32"] = "fp32"


class TeamModelConfig(FrozenConfig):
    enabled: bool = False
    architecture: Architecture | None = None
    checkpoint: str | None = None
    download_url: str | None = None

    @model_validator(mode="after")
    def validate_checkpoint(self) -> TeamModelConfig:
        if self.enabled and (self.architecture is None or not self.checkpoint):
            raise ValueError("An enabled team model requires architecture and local checkpoint")
        return self


class WorkerConfig(FrozenConfig):
    host: str
    directory: str = "/home/semyon/neuroml"
    speed: Literal["fast", "small"]
    storage_mount: str | None = None


class ClusterConfig(FrozenConfig):
    workers: tuple[WorkerConfig, ...] = ()
    profiling_host: str = "kr54-pc"
    image: str = "neuroml-pipeline:distributed"
