"""JSON configuration follows the reference's env > dotenv > file convention."""
import json
import os

import pytest
from pydantic import ValidationError

from fetal.config import load_config


@pytest.fixture(autouse=True)
def clear_settings_environment(monkeypatch):
    for name in os.environ:
        if name.startswith("FETAL_"):
            monkeypatch.delenv(name)


def test_environment_overrides_dotenv_overrides_json(tmp_path, monkeypatch):
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps({"architectures": ["resnet50"], "training": {"epochs": 6, "batch_size": 32}}))
    dotenv = tmp_path / ".env"
    dotenv.write_text("FETAL_TRAINING__BATCH_SIZE=16\nFETAL_TRAINING__EPOCHS=3\n")
    monkeypatch.setenv("FETAL_TRAINING__EPOCHS", "2")
    config = load_config(path, env_file=dotenv)
    assert config.training.epochs == 2
    assert config.training.batch_size == 16
    assert config.training.lr == 0.0003
    assert config.architectures == ("resnet50",)
    assert json.loads(config.model_dump_json())["training"]["epochs"] == 2
    with pytest.raises(ValidationError, match="frozen"):
        config.training.epochs = 10


def test_dotenv_can_be_disabled_without_disabling_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "pipeline.json"
    path.write_text('{"architectures": ["resnet50"]}')
    (tmp_path / ".env").write_text("FETAL_TRAINING__BATCH_SIZE=16\n")
    monkeypatch.setenv("FETAL_TRAINING__EPOCHS", "2")
    config = load_config(path, env_file=None)
    assert config.training.epochs == 2
    assert config.training.batch_size == 32


@pytest.mark.parametrize("override", [
    {"training": {"epochs": 0}},
    {"training": {"precision": "fp16"}},
    {"architectures": ["unknown"]},
    {"architectures": ["resnet50", "resnet50"]},
    {"finalist_seeds": [0, 0]},
    {"team_model": {"enabled": True}},
])
def test_invalid_settings_fail_at_load(tmp_path, override):
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps({"architectures": ["resnet50"], **override}))
    with pytest.raises(ValidationError):
        load_config(path, env_file=None)


def test_json_root_must_be_an_object(tmp_path):
    path = tmp_path / "pipeline.json"
    path.write_text("[]")
    with pytest.raises(TypeError, match="object"):
        load_config(path, env_file=None)
