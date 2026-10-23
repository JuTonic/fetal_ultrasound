"""Вложенные стратифицированные подвыборки обучающих пациентов (25 / 50 / 100 %) для пункта (e)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fetal import subsets

META = Path("data/processed/metadata.csv")
SHA = Path("data/processed/subsets.sha256")


def _toy(n_groups=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        part = "train" if g < int(0.7 * n_groups) else ("val" if g < int(0.85 * n_groups) else "test")
        mach = ["Aloka", "Voluson E6", "Voluson S10"][g % 3]
        for k in range(int(rng.integers(1, 12))):
            rows.append({"image": f"g{g}_{k}", "patient": g, "patient_group": g, "machine": mach,
                         "cls9": f"c{int(rng.choice(5, p=[.4, .25, .2, .1, .05]))}", "split_grouped": part})
    return pd.DataFrame(rows)


def test_folds_are_balanced_by_size_class_and_machine():
    m = subsets.add_subset_columns(_toy())
    subsets.validate_subsets(m, max_class_dev_pp=3.0)
    sizes = m[m.train_fold >= 0].train_fold.value_counts()
    assert sizes.max() / sizes.min() < 1.1
    rep = subsets.subset_report(m).set_index("доля")
    assert rep.loc["25 %", "макс. откл. классы, п.п."] < 1.5 and rep.loc["25 %", "макс. откл. аппараты, п.п."] < 1.5
    assert abs(rep.loc["25 %", "кадров"] / rep.loc["100 %", "кадров"] - 0.25) < 0.02


def test_nested_group_integrity_and_val_test_untouched():
    m = subsets.add_subset_columns(_toy())
    base = m.split_grouped
    u25, u50 = m.split_grouped_f25 == "train", m.split_grouped_f50 == "train"
    assert (u25 <= u50).all() and (u50 <= (base == "train")).all() and u25.sum() < u50.sum() < (base == "train").sum()
    for col in ("split_grouped_f25", "split_grouped_f50"):
        assert (m.loc[base != "train", col] == base[base != "train"]).all()
        assert set(m.loc[base == "train", col]) <= {"train", "unused"}
    g_used = m[u25].patient_group.unique()
    assert not m[(base == "train") & ~u25].patient_group.isin(g_used).any()          # группа целиком внутри или вне
    assert (m.loc[base != "train", "train_fold"] == -1).all()


def test_deterministic_seed_dependent_and_does_not_change_existing_columns():
    m = _toy()
    a, b = subsets.add_subset_columns(m), subsets.add_subset_columns(m)
    assert subsets.subsets_checksum(a) == subsets.subsets_checksum(b)
    assert subsets.subsets_checksum(subsets.add_subset_columns(m, seed=1)) != subsets.subsets_checksum(a)
    assert a[list(m.columns)].equals(m)                                                  # старые колонки не тронуты
    assert list(a.columns) == [*m.columns, *subsets.COLUMNS]


def test_validate_catches_broken_nesting_split_group_and_missing_class():
    m = subsets.add_subset_columns(_toy())
    bad = m.copy()
    i = bad.index[bad.split_grouped_f25 == "train"][0]
    bad.loc[i, "split_grouped_f50"] = "unused"                                           # кадр из 25 % выпал из 50 %
    with pytest.raises(AssertionError):
        subsets.validate_subsets(bad)
    bad = m.copy()
    j = bad.index[bad.split_grouped_f25 == "unused"][0]
    bad.loc[j, "split_grouped_f25"] = "train"                                            # группа разделена внутри подвыборки
    with pytest.raises(AssertionError):
        subsets.validate_subsets(bad)
    bad = m.copy()
    bad.loc[bad.split_grouped_f25 == "val", "split_grouped_f25"] = "train"               # val изменился
    with pytest.raises(AssertionError):
        subsets.validate_subsets(bad)
    one_class = m.copy()
    one_class.loc[one_class.cls9 == "c4", "cls9"] = "c0"
    subsets.validate_subsets(subsets.add_subset_columns(one_class))                      # нормально, если класса нет совсем


@pytest.mark.skipif(not (META.exists() and SHA.exists()), reason="сначала: python -m fetal.subsets")
def test_real_subsets_match_frozen_checksum_and_are_valid():
    m = pd.read_csv(META)
    assert subsets.subsets_checksum(m) == SHA.read_text().strip()
    subsets.validate_subsets(m, max_class_dev_pp=0.5)
    rep = subsets.subset_report(m).set_index("доля")
    assert rep.loc["25 %", "мин. кадров в классе"] >= 20                                 # даже у Brain: other
    assert subsets.subsets_checksum(subsets.add_subset_columns(m.drop(columns=list(subsets.COLUMNS)))) == SHA.read_text().strip()


def test_data_efficiency_config_differs_from_ablation_base_only_in_split_col():
    import json
    from fetal.train import load_config
    cfg = Path("configs/data_efficiency.json")
    if not cfg.exists():
        pytest.skip("нет configs/data_efficiency.json")
    base = load_config("configs/ablation.json", "B0")
    for name, col in (("B0_f25", "split_grouped_f25"), ("B0_f50", "split_grouped_f50")):
        c = load_config(cfg, name)
        assert {k for k in c if k != "variant" and c[k] != base[k]} == {"split_col"} and c["split_col"] == col
    assert all(col in pd.read_csv(META, nrows=1) for col in ("split_grouped_f25", "split_grouped_f50")) if META.exists() else True
