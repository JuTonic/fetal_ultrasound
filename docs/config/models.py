from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class FrozenConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore",
        frozen=True,
        use_enum_values=False,
    )
