"""Torch-часть: FetalPlanesDataset + DataLoader на реальных картинках. Запуск: python -m pytest -q"""
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from fetal.datasets import FetalPlanesDataset, make_loader  # noqa: E402

META = Path("data/processed/metadata.csv")
RAW = Path("data/raw")
pytestmark = pytest.mark.skipif(not (META.exists() and (RAW / "Images").exists()),
                                reason="нужны data/raw и python -m fetal.prepare")


def _ds(part="train", train=True, n=24, **kw):
    meta = pd.read_csv(META)
    return FetalPlanesDataset(meta[meta.split_grouped == part].head(n), RAW, "split_grouped", part,
                              train=train, **kw)


def _epoch(ds, seed, workers):
    torch.manual_seed(seed)
    return list(make_loader(ds, batch_size=8, shuffle=True, num_workers=workers))


@pytest.mark.parametrize("workers", [0, 2])
def test_loader_batch_shape_dtype_labels(workers):
    ds = _ds()
    x, y = next(iter(make_loader(ds, batch_size=8, shuffle=False, num_workers=workers)))
    assert x.shape == (8, 3, 224, 224) and x.dtype == torch.float32
    assert y.dtype == torch.int64 and y.tolist() == ds.df.label.head(8).tolist()


@pytest.mark.parametrize("workers", [0, 2])
def test_augmentation_reproducible_with_seed(workers):
    ds = _ds()
    a, b, c = _epoch(ds, 0, workers), _epoch(ds, 0, workers), _epoch(ds, 1, workers)
    same = lambda p, q: all(torch.equal(x1, x2) and torch.equal(y1, y2) for (x1, y1), (x2, y2) in zip(p, q))
    assert same(a, b), "тот же seed должен давать те же батчи"
    assert not same(a, c), "другой seed должен давать другие батчи"


def test_eval_is_deterministic_and_size_param():
    ds = _ds("val", train=False, n=2, size=320)
    x1, _ = ds[0]
    x2, _ = ds[0]
    assert x1.shape == (3, 320, 320) and torch.equal(x1, x2)
