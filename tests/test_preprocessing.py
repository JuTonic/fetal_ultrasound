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
    assert set(meta[col]) == set(splits.SPLITS) and meta[col].notna().all()      # части полные и непустые


@pytest.mark.skipif(not META.exists(), reason="сначала: python -m fetal.prepare")
def test_metadata_table_is_consistent():
    meta = pd.read_csv(META)
    assert len(meta) == 12400 and meta["image"].is_unique and meta["patient"].nunique() == 1792
    splits.validate_splits(meta)


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


def test_changed_columns_detects_removed_and_duplicated_images():
    old = pd.DataFrame({"image": ["a", "b", "c"], "label": [0, 1, 2]})
    assert changed_columns(old, old.iloc[:-1]) == {"image": 1}                     # удалили строку
    assert changed_columns(old, old.iloc[:0]) == {"image": 3}                      # пустая таблица
    assert changed_columns(old, pd.concat([old, old.iloc[:1]])) == {"image": 1}    # дубль


def test_preprocess_train_requires_explicit_rng():
    with pytest.raises(ValueError):
        preprocess(Image.fromarray(np.zeros((50, 80), np.uint8)), 64, train=True)


def _toy_meta(n_patients=60, per=10, seed=0):
    rng = np.random.default_rng(seed)
    pat = np.repeat(np.arange(n_patients), per)
    return pd.DataFrame({
        "image": [f"i{k}" for k in range(len(pat))], "patient": pat, "patient_group": pat,
        "label": rng.integers(0, 3, len(pat)), "cls9": rng.integers(0, 3, len(pat)).astype(str),
        "machine": rng.choice(["A", "B"], len(pat)), "md5_gray": [f"h{k}" for k in range(len(pat))],
    })


def test_grouped_split_complete_and_leak_free():
    m = _toy_meta()
    m["split_grouped"] = splits.grouped_split(m, 0)
    assert set(m.split_grouped) == {"train", "val", "test"}
    assert all(v == 0 for v in splits.check_no_leakage(m, "split_grouped").values())


def test_validate_splits_catches_leak_and_missing_part():
    m = _toy_meta()
    m["split_grouped"] = splits.grouped_split(m, 0)
    m["split_official"] = m["split_grouped"]
    m["split_naive"] = m["split_grouped"]
    splits.validate_splits(m)                                         # корректная таблица проходит
    leaky = m.copy(); leaky.loc[0, "split_official"] = "test" if m.split_official[0] != "test" else "train"
    with pytest.raises(AssertionError, match="утечка"):
        splits.validate_splits(leaky)
    with pytest.raises(AssertionError):
        splits.validate_splits(m.assign(split_naive="train"))        # части не полные


def test_link_duplicate_patients_merges_only_with_enough_pairs():
    meta = pd.DataFrame({"patient": [1, 1, 1, 2, 2, 2, 3, 4]})
    pairs = pd.DataFrame({"i": [0, 1, 2, 0], "j": [3, 4, 5, 7]})     # 1↔2: 3 пары, 1↔4: одна
    out, linked = data.link_duplicate_patients(meta, pairs, min_pairs=3)
    assert out.patient_group.tolist() == [1, 1, 1, 1, 1, 1, 3, 4] and len(linked) == 1


def test_split_columns_match_frozen_checksum():
    """Разбиения — зафиксированный протокол: меняются только осознанно (fetal.prepare --force)
    с обновлением data/processed/splits.sha256."""
    sha = Path("data/processed/splits.sha256")
    if not (META.exists() and sha.exists()):
        pytest.skip("нет metadata.csv / splits.sha256")
    assert splits.splits_checksum(pd.read_csv(META)) == sha.read_text().strip()
