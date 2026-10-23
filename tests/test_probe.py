"""Проба «предскажи аппарат»: метрика внутри класса, веса, бутстрэп и сквозной расчёт на синтетике."""
import numpy as np
import pandas as pd
import pytest

from fetal import probe


def test_within_class_balacc_perfect_constant_and_weights():
    cls = np.array(["a"] * 6 + ["b"] * 6)
    y = np.array([0, 0, 1, 1, 2, 2] * 2)
    assert probe.within_class_balacc(cls, y, y, ["a", "b"], 3) == 1.0
    assert probe.within_class_balacc(cls, y, np.zeros(12, int), ["a", "b"], 3) == pytest.approx(1 / 3)
    p = y.copy(); p[0] = 1                                   # одна ошибка: recall класса a, аппарат 0 = 1/2
    base = probe.within_class_balacc(cls, y, p, ["a", "b"], 3)
    assert base == pytest.approx((np.mean([0.5, 1, 1]) + 1) / 2)
    w = np.ones(12); w[1] = 0                                # вес ошибочного объекта-«соседа» нулём: теперь recall 0 = 0
    assert probe.within_class_balacc(cls, y, p, ["a", "b"], 3, w) < base
    assert probe.within_class_balacc(cls, y, y, ["a"], 3) == 1.0                # только выбранные классы


def test_eligible_classes_needs_min_val_for_every_target():
    cls = np.array(["a"] * 30 + ["b"] * 30)
    y = np.array([0] * 10 + [1] * 10 + [2] * 10 + [0] * 10 + [1] * 10 + [2] * 9 + [1])
    assert probe.eligible_classes(cls, y, 3, min_val=10) == ["a"]               # у b цели 2 только 9


def test_cell_weights_remove_class_information():
    cls = np.array(["x"] * 90 + ["y"] * 10 + ["x"] * 10 + ["y"] * 90)
    tgt = np.array([0] * 100 + [1] * 100)
    w = probe.cell_weights(cls, tgt)
    assert w.sum() == pytest.approx(len(w))
    for c in ("x", "y"):                                     # внутри каждого класса суммарные веса целей равны
        s = [w[(cls == c) & (tgt == t)].sum() for t in (0, 1)]
        assert s[0] == pytest.approx(s[1])


def test_boot_ci_pairs_models_and_detects_difference():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600); groups = np.repeat(np.arange(100), 6)
    good = np.where(rng.random(600) < 0.9, y, rng.integers(0, 3, 600)); bad = rng.integers(0, 3, 600)
    r = probe.boot_ci({"A0": bad, "A2": good}, groups, lambda p, w: probe.plain_balacc(y, p, 3, w), n_boot=200, base="A0")
    assert r["A2"]["delta_vs_base"]["ci95"][0] > 0 and r["A0"]["ci95"][0] < 0.45
    assert "delta_vs_base" not in r["A0"]


def _toy_meta(seed=0, n_groups=160, per=12):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per)
    machine = np.repeat(rng.choice(["Aloka", "Voluson E6", "Voluson S10"], n_groups), per)
    # состав классов зависит от аппарата (confound), как в реальных данных
    p = {"Aloka": [0.7, 0.15, 0.15], "Voluson E6": [0.15, 0.7, 0.15], "Voluson S10": [0.15, 0.15, 0.7]}
    cls = np.array([rng.choice(["c0", "c1", "c2"], p=p[m]) for m in machine])
    split = np.where(np.isin(g, rng.permutation(n_groups)[: n_groups // 3]), "val", "train")
    return pd.DataFrame({"machine": machine, "operator": "Op. 1", "cls9": cls, "width": 600, "height": 400, "patient_group": g,
                         "split_grouped": split}), rng


def test_run_probe_separates_real_scanner_signal_from_class_composition():
    meta, rng = _toy_meta()
    codes = meta.machine.map({"Aloka": 0, "Voluson E6": 1, "Voluson S10": 2}).to_numpy()
    signal = np.eye(3)[codes] * 3.0 + rng.normal(size=(len(meta), 3))
    noise = rng.normal(size=(len(meta), 3))
    # A0 содержит сигнал аппарата, A2 — только шум (маскирование «сработало»)
    emb = {"full": np.hstack([signal, noise]), "crop": np.hstack([signal, noise]), "ui": np.hstack([noise, noise]),
           "crop_ui": np.hstack([noise, noise])}
    res = probe.run_probe(meta, emb, "machine", probe.SCANNERS, np.ones(len(meta), bool), 1.0, n_boot=80)
    w, pl = res["within_class_balanced_accuracy"], res["plain_balanced_accuracy"]
    assert w["A0"]["value"] > 0.8 and w["A2"]["value"] < 0.5                                   # сигнал виден только в A0
    assert w["A2"]["delta_vs_base"]["ci95"][1] < 0                                             # и разность значима
    cc = "контроль: только класс (без выравнивания весов)"
    assert pl[cc]["value"] > 0.45 and w[cc]["value"] == pytest.approx(1 / 3, abs=0.08)        # состав классов даёт «обычную» точность, но не внутри класса
    assert set(res["eligible_classes"]) == {"c0", "c1", "c2"} and res["chance"] == pytest.approx(1 / 3)
    assert "A0" in probe.to_markdown({"scanner": res})
