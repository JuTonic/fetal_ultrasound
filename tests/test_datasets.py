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


def _epoch(ds, seed, workers, shuffle=False):
    torch.manual_seed(seed)
    return list(make_loader(ds, batch_size=8, shuffle=shuffle, num_workers=workers))


@pytest.mark.parametrize("workers", [0, 2])
def test_loader_batch_shape_dtype_labels(workers):
    ds = _ds()
    x, y = next(iter(make_loader(ds, batch_size=8, shuffle=False, num_workers=workers)))
    assert x.shape == (8, 3, 224, 224) and x.dtype == torch.float32
    assert y.dtype == torch.int64 and y.tolist() == ds.df.label.head(8).tolist()


def _same(p, q):
    return all(torch.equal(x1, x2) and torch.equal(y1, y2) for (x1, y1), (x2, y2) in zip(p, q))


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("shuffle", [False, True])
def test_augmentation_reproducible_with_seed(workers, shuffle):
    ds = _ds()
    a, b, c = (_epoch(ds, sd, workers, shuffle) for sd in (0, 0, 1))
    assert _same(a, b), "тот же seed должен давать те же батчи"
    assert not _same(a, c), "другой seed должен давать другие батчи"


@pytest.mark.parametrize("workers", [0, 2])
def test_seed_changes_pixels_not_just_order(workers):
    """shuffle=False: порядок картинок фиксирован, значит различия между seed — только от аугментаций."""
    ds = _ds()
    a, c = _epoch(ds, 0, workers), _epoch(ds, 1, workers)
    assert all(torch.equal(y1, y2) for (_, y1), (_, y2) in zip(a, c))      # порядок меток тот же
    assert not any(torch.equal(x1, x2) for (x1, _), (x2, _) in zip(a, c))  # а пиксели разные


def test_eval_is_deterministic_and_size_param():
    ds = _ds("val", train=False, n=2, size=320)
    x1, _ = ds[0]
    x2, _ = ds[0]
    assert x1.shape == (3, 320, 320) and torch.equal(x1, x2)
