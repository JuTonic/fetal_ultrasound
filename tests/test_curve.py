"""Кривая «качество против размера выборки»: таблица, парные шаги, график (на синтетических прогонах)."""
import json

import numpy as np
import pytest

from fetal import curve, metrics


def _fake(root, variant, seed, y, groups, preds):
    d = root / variant / f"seed{seed}"
    d.mkdir(parents=True)
    machine = np.array(["Aloka", "Voluson E6", "Voluson S10"])[np.arange(len(y)) % 3]
    np.savez_compressed(d / "val_pred.npz", image=np.array([f"i{k}" for k in range(len(y))]), y=y,
                        logits=np.eye(9)[preds] + 0.01, machine=machine, patient_group=groups)
    (d / "metrics.json").write_text(json.dumps(metrics.evaluate(y, preds, machine)))


def _sources(tmp_path):
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(120), 10)
    y = rng.integers(0, 9, len(groups))
    src = {}
    for pct, acc in ((25, 0.55), (50, 0.7), (100, 0.85)):          # качество растёт с долей
        for s in range(3):
            p = np.where(rng.random(len(y)) < acc, y, rng.integers(0, 9, len(y)))
            _fake(tmp_path, f"f{pct}", s, y, groups, p)
        src[pct] = (str(tmp_path), f"f{pct}")
    return src


def test_curve_is_monotone_and_steps_are_significant(tmp_path):
    runs = curve.load(_sources(tmp_path))
    df, steps = curve.build(runs, n_boot=200)
    assert list(df["доля обучающих пациентов, %"]) == [25, 50, 100] and (df["seeds"] == 3).all()
    assert df["macro-F1"].is_monotonic_increasing and (df["macro-F1 lo"] <= df["macro-F1"]).all() and (df["macro-F1"] <= df["macro-F1 hi"]).all()
    assert [(s["от"], s["до"]) for s in steps] == [(25, 50), (50, 100)]
    assert all(s["delta"] > 0.05 and s["ci95"][0] > 0 and all(d > 0 for d in s["per_seed"]) for s in steps)


def test_curve_markdown_and_png(tmp_path):
    runs = curve.load(_sources(tmp_path))
    df, steps = curve.build(runs, n_boot=100)
    md = curve.to_markdown(df, steps)
    assert "25 % → 50 %" in md and "50 % → 100 %" in md and "| 100 % |" in md
    out = tmp_path / "c.png"
    curve.plot(df, out)
    assert out.read_bytes()[:4] == b"\x89PNG" and out.stat().st_size > 5000


def test_curve_refuses_different_val_images(tmp_path):
    src = _sources(tmp_path)
    z = dict(np.load(tmp_path / "f50" / "seed0" / "val_pred.npz"))
    z["image"] = z["image"][::-1].copy()
    np.savez_compressed(tmp_path / "f50" / "seed0" / "val_pred.npz", **z)
    with pytest.raises(AssertionError):
        curve.build(curve.load(src), n_boot=50)
