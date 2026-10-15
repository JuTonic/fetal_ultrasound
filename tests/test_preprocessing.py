"""Запуск из корня репозитория:  python -m pytest -q"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fetal import data, splits
from fetal.prepare import changed_columns
from fetal.transforms import letterbox_square, preprocess

META = Path("data/processed/metadata.csv")


@pytest.mark.skipif(not META.exists(), reason="сначала: python -m fetal.prepare")
@pytest.mark.parametrize("col", ["split_grouped", "split_official"])
def test_no_patient_leakage(col):
    meta = pd.read_csv(META)
    assert all(v == 0 for v in splits.check_no_leakage(meta, col).values())


@pytest.mark.skipif(not META.exists(), reason="сначала: python -m fetal.prepare")
def test_naive_split_leaks():
    """Негативный контроль: image-level разбиение обязано давать утечку."""
    assert splits.check_no_leakage(pd.read_csv(META), "split_naive")["patients"] > 0


@pytest.mark.parametrize("wh", [(959, 661), (640, 392), (300, 800)])
def test_letterbox_keeps_aspect_ratio(wh):
    out = letterbox_square(Image.fromarray(np.full(wh[::-1], 200, np.uint8)), 224)
    assert out.size == (224, 224)
    ys, xs = np.nonzero(np.asarray(out) > 100)
    ratio = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
    assert abs(ratio - wh[0] / wh[1]) < 0.05


def test_preprocess_shape_and_rgba():
    rgba = Image.fromarray(np.random.randint(0, 255, (400, 600, 4), np.uint8), "RGBA")
    x = preprocess(rgba, 224)
    assert x.shape == (3, 224, 224) and x.dtype == np.float32
    assert np.allclose(x[0] * 0.229 + 0.485, x[1] * 0.224 + 0.456, atol=1e-5)  # 3 одинаковых серых канала


def test_load_metadata_strips_strings(tmp_path):
    """В pandas 3 строки имеют dtype str (не object): пробелы по краям всё равно должны обрезаться."""
    (tmp_path / "FETAL_PLANES_DB_data.csv").write_text(
        "Image_name;Patient_num;Plane;Brain_plane;Operator;US_Machine;Train \n"
        "a;1;Fetal brain ; Trans-thalamic;Op. 1 ;Aloka;1 \n"
        "b;2;Fetal femur;Not A Brain;Other;Voluson E6;0 \n")
    meta = data.load_metadata(tmp_path)
    assert meta.cls9.tolist() == ["Brain: TT", "Fetal femur"]
    assert meta.operator[0] == "Op. 1" and meta.split_official_raw.tolist() == ["train", "test"]


def test_changed_columns_detects_split_change():
    old = pd.DataFrame({"image": ["a", "b", "c"], "label": [0, 1, 2], "split_grouped": ["train", "val", "test"]})
    new = old.iloc[::-1].reset_index(drop=True)                  # порядок строк не важен
    assert changed_columns(old, new) == {}
    assert changed_columns(old, new.assign(extra=1)) == {}        # новая колонка — не поломка
    new.loc[new.image == "a", "split_grouped"] = "test"
    assert changed_columns(old, new) == {"split_grouped": 1}
    assert changed_columns(old, new.drop(columns="label")) == {"label": 3, "split_grouped": 1}
