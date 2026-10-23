"""Test optimizer/scheduler recovery on synthetic tensors, without dataset runs."""
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from fetal import experiment


class ToyDataset(Dataset):
    def __init__(self, meta, raw_dir, split_col, part, train=False):
        self.df = meta[meta[split_col] == part].reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = int(self.df.iloc[index].label)
        x = torch.tensor([label / 9, index / 10, 0.5, 1.0])
        if self.train:
            x = x + torch.rand(4) * 0.1
        return x, label


def tiny_model(arch, pretrained=False):
    return nn.Sequential(nn.Linear(4, 12), nn.ReLU(), nn.Dropout(0.2), nn.Linear(12, 9))


@pytest.fixture
def training_case(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    meta = pd.DataFrame({"split_grouped": ["train"] * 18 + ["val"] * 9,
                         "label": list(range(9)) * 3, "patient_group": [1] * 18 + [2] * 9})
    monkeypatch.setattr(experiment.pd, "read_csv", lambda *args: meta)
    monkeypatch.setattr(experiment, "FetalPlanesDataset", ToyDataset)
    monkeypatch.setattr(experiment, "build_model", tiny_model)

    def reload_checkpoint(path, expected):
        checkpoint = torch.load(path, weights_only=True)
        assert all(checkpoint["contract"][key] == value for key, value in expected.items())
        model = tiny_model("test")
        model.load_state_dict(checkpoint["state_dict"])
        return model, checkpoint["contract"]

    monkeypatch.setattr(experiment, "load_checkpoint", reload_checkpoint)
    job = {"id": "test", "arch": "resnet18", "protocol": "split_grouped", "seed": 7,
           "pretrained": False, "smoke": False,
           "config": {"epochs": 3, "batch_size": 6, "num_workers": 0,
                      "lr": 0.003, "weight_decay": 0.0001, "precision": "fp32"}}
    dataset = {"splits_sha256": "split", "subsets_sha256": "subset", "metadata_sha256": "metadata"}
    return job, dataset


def assert_equal_state(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal_state(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal_state(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("workers", [0, 2])
def test_resume_replays_interrupted_epoch_and_matches_uninterrupted_training(training_case, tmp_path, monkeypatch, workers):
    job, dataset = training_case
    job["config"]["num_workers"] = workers
    uninterrupted = tmp_path / "uninterrupted"
    interrupted = tmp_path / "interrupted"
    uninterrupted.mkdir()
    interrupted.mkdir()
    experiment.train_checkpoint(job, dataset, uninterrupted, "cpu")
    original_predict = experiment.predict
    calls = 0

    def stop_in_second_epoch(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("simulated stop before epoch 2 is committed")
        return original_predict(*args)

    monkeypatch.setattr(experiment, "predict", stop_in_second_epoch)
    with pytest.raises(KeyboardInterrupt):
        experiment.train_checkpoint(job, dataset, interrupted, "cpu")
    saved = torch.load(interrupted / "resume.pt", weights_only=True)
    assert saved["completed_epoch"] == 1
    (interrupted / "resume.tmp").write_bytes(b"interrupted write")
    monkeypatch.setattr(experiment, "predict", original_predict)
    experiment.train_checkpoint(job, dataset, interrupted, "cpu")
    expected = torch.load(uninterrupted / "resume.pt", weights_only=True)
    actual = torch.load(interrupted / "resume.pt", weights_only=True)
    for key in ("model", "optimizer", "scheduler", "best_score", "best_checkpoint", "completed_epoch"):
        assert_equal_state(expected[key], actual[key])
    assert [row["epoch"] for row in actual["learning_curve"]] == [1, 2, 3]
    assert [row["loss"] for row in actual["learning_curve"]] == [row["loss"] for row in expected["learning_curve"]]


def test_stop_after_last_epoch_restores_exports_without_training(training_case, tmp_path, monkeypatch):
    job, dataset = training_case
    job["config"]["epochs"] = 1
    original_save = experiment.save_training_state

    def stop_after_commit(path, state):
        original_save(path, state)
        if path.name == "resume.pt":
            raise KeyboardInterrupt("simulated stop after last commit")

    monkeypatch.setattr(experiment, "save_training_state", stop_after_commit)
    with pytest.raises(KeyboardInterrupt):
        experiment.train_checkpoint(job, dataset, tmp_path, "cpu")
    assert not (tmp_path / "model.pt").exists()
    monkeypatch.setattr(experiment, "save_training_state", original_save)
    monkeypatch.setattr(experiment, "predict", lambda *args: pytest.fail("Finished epochs must not run again"))
    path = experiment.train_checkpoint(job, dataset, tmp_path, "cpu")
    assert path.is_file()
    assert (tmp_path / "training_seconds.txt").exists()
    changed = {**job, "seed": 8}
    with pytest.raises(ValueError, match="differs"):
        experiment.train_checkpoint(changed, dataset, tmp_path, "cpu")


def test_failed_atomic_write_keeps_previous_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "resume.pt"
    experiment.save_training_state(path, {"completed_epoch": 1})

    def stop_write(state, stream):
        stream.write(b"incomplete")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(experiment.torch, "save", stop_write)
    with pytest.raises(OSError, match="simulated"):
        experiment.save_training_state(path, {"completed_epoch": 2})
    assert torch.load(path, weights_only=True)["completed_epoch"] == 1
