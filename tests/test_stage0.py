"""Этап 0 абляции: метрики, бутстрэп, детектор сектора, маски интерфейса, варианты области, обучение.
Запуск из корня репозитория:  python -m pytest -q"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw
from sklearn.metrics import balanced_accuracy_score, f1_score

from fetal import bootstrap, metrics, overlay, sector
from fetal.regions import RegionPrep

META = Path("data/processed/metadata.csv")
RAW = Path("data/raw")
CONFIG = Path("configs/ablation.json")
needs_data = pytest.mark.skipif(not (META.exists() and (RAW / "Images").exists()), reason="нужны data/raw и metadata.csv")


# ---------- метрики ----------
def test_metrics_match_sklearn_when_all_classes_present():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 9, 500)
    p = np.where(rng.random(500) < 0.6, y, rng.integers(0, 9, 500))
    r = metrics.evaluate(y, p)
    assert r["macro_f1"] == pytest.approx(f1_score(y, p, average="macro", labels=range(9)))
    assert r["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, p))


def test_metrics_skip_absent_classes_and_report_per_machine():
    y = np.array([0, 0, 1, 1, 2, 2]); p = np.array([0, 1, 1, 1, 2, 0])
    machine = np.array(["A", "A", "A", "B", "B", "B"])
    r = metrics.evaluate(y, p, machine)
    assert r["f1_per_class"]["Fetal thorax"] is not None and r["f1_per_class"]["Brain: other"] is None   # класс без объектов
    assert r["per_machine"]["A"]["classes_present"] == 2 and r["per_machine"]["B"]["classes_present"] == 2
    assert 0 <= r["macro_f1"] <= 1
    sub = metrics.evaluate(y, p, subsets={"first_half": np.arange(6) < 3, "empty": np.zeros(6, bool)})["per_subset"]
    assert sub["first_half"]["n"] == 3 and sub["empty"]["macro_f1"] is None


# ---------- парный бутстрэп по группам ----------
def _toy(n_groups=80, per=12, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per)
    y = rng.integers(0, 9, len(groups))
    base = [np.where(rng.random(len(y)) < 0.6, y, rng.integers(0, 9, len(y))) for _ in range(3)]
    return y, base, groups


def test_bootstrap_no_difference_ci_covers_zero():
    y, base, groups = _toy()
    r = bootstrap.paired_bootstrap(y, base, base, groups, n_boot=500)
    assert r["delta"] == 0 and r["ci95"] == (0.0, 0.0) and r["n_groups"] == 80 and r["n_seeds"] == 3
    assert bootstrap.decision(r) == "различий не видно"


def test_bootstrap_detects_real_improvement_and_not_noise():
    y, base, groups = _toy(n_groups=150)
    rng = np.random.default_rng(1)
    better = [np.where((p != y) & (rng.random(len(y)) < 0.5), y, p) for p in base]      # чинит половину ошибок
    r = bootstrap.paired_bootstrap(y, base, better, groups, n_boot=800, n_comparisons=7)
    assert r["delta"] > 0.05 and r["ci95"][0] > 0 and all(d > 0 for d in r["per_seed_delta"])
    assert "лучше" in bootstrap.decision(r)
    worse = bootstrap.paired_bootstrap(y, better, base, groups, n_boot=800)
    assert worse["ci95"][1] < 0 and "хуже" in bootstrap.decision(worse)


def test_bootstrap_validates_inputs_and_supports_subset():
    y, base, groups = _toy()
    with pytest.raises(ValueError):
        bootstrap.paired_bootstrap(y, base, base[:2], groups)
    sub = np.arange(len(y)) < len(y) // 2
    r = bootstrap.paired_bootstrap(y, base, base, groups, subset=sub, n_boot=50, metric="balanced_accuracy")
    assert r["n_objects"] == sub.sum() and r["n_groups"] == 40


# ---------- детектор сектора ----------
def _fan(size=(300, 200), poly=((60, 20), (240, 20), (280, 190), (20, 190)), text=True):
    im = Image.new("L", size, 0)
    d = ImageDraw.Draw(im)
    d.polygon(poly, fill=90)
    rng = np.random.default_rng(0)
    a = np.asarray(im).astype(np.int16)
    a = np.where(a > 0, np.clip(a + rng.integers(-30, 30, a.shape), 5, 255), 0).astype(np.uint8)
    im = Image.fromarray(a)
    if text:
        ImageDraw.Draw(im).text((250, 4), "MI 1.0", fill=255)        # тонкий текст вне сектора
    return im


def test_detect_finds_fan_and_ignores_thin_text():
    r = sector.detect(_fan())
    assert r["ok"] and 0.45 < r["area"] < 0.85
    assert r["x0"] < 0.15 and r["x1"] > 0.9 and r["y0"] < 0.2 and r["y1"] > 0.9


def test_detect_black_frame_is_not_ok_and_never_crops():
    r = sector.detect(Image.new("L", (200, 120), 0))
    assert not r["ok"] and (r["x0"], r["y0"], r["x1"], r["y1"]) == (0.0, 0.0, 1.0, 1.0)
    img = Image.new("L", (200, 120), 7)
    assert sector.apply_region(img, r, "crop_mask") is img            # запасной вариант: кадр без изменений


def test_apply_region_modes():
    im = _fan(); rec = sector.detect(im)
    c = sector.apply_region(im, rec, "crop")
    assert c.size[0] <= im.size[0] and c.size[1] <= im.size[1] and c.size != im.size
    mk = np.asarray(sector.apply_region(im, rec, "mask"))
    assert mk.shape == (200, 300) and mk[2, 2] == 0 and mk[100, 150] > 0              # угол закрашен, центр цел
    assert sector.apply_region(im, rec, "full") is im
    with pytest.raises(ValueError):
        sector.apply_region(im, rec, "oops")


def test_crop_keeps_nearly_all_brightness_on_fan():
    im = _fan(text=False); rec = sector.detect(im)
    a = np.asarray(im, float); c = np.asarray(sector.apply_region(im, rec, "crop"), float)
    assert c.sum() / a.sum() > 0.97


# ---------- маски интерфейса ----------
def _frames(tmp_path, n=70, size=(160, 110)):
    (tmp_path / "Images").mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        a = np.zeros((size[1], size[0]), np.uint8)
        im = Image.fromarray(a); d = ImageDraw.Draw(im)
        x, y, r = rng.integers(30, 100), rng.integers(30, 80), rng.integers(12, 25)
        d.ellipse((x - r, y - r, x + r, y + r), fill=int(rng.integers(60, 120)))     # «анатомия» движется
        for k in range(4):                                                            # статичный «текст»: тонкие линии
            d.line((120, 8 + 6 * k, 150, 8 + 6 * k), fill=255)
        im.save(tmp_path / "Images" / f"f{i}.png")
        rows.append({"image": f"f{i}", "machine": "M", "width": size[0], "height": size[1], "split_grouped": "train"})
    return pd.DataFrame(rows)


def test_overlay_masks_cover_static_text_not_anatomy(tmp_path):
    meta = _frames(tmp_path)
    masks = overlay.build_masks(meta, tmp_path)
    mk = masks[overlay.fmt_key("M", 160, 110)]
    assert mk[8:30, 120:150].mean() > 0.6               # текст закрыт
    assert mk[:, :100].mean() < 0.02                     # анатомия (левая часть) не закрыта
    out = np.asarray(overlay.apply_overlay(Image.open(tmp_path / "Images" / "f0.png"), mk))
    assert out[8:30, 120:150].max() == 0
    img = Image.new("L", (50, 50), 9)
    assert overlay.apply_overlay(img, mk) is img and overlay.apply_overlay(img, None) is img     # размер не тот / маски нет


def test_overlay_skips_small_formats_and_untrusted_masks(tmp_path):
    meta = _frames(tmp_path, n=overlay.MIN_FRAMES - 1)
    assert overlay.build_masks(meta, tmp_path) == {}                                              # мало кадров


def test_overlay_masks_use_train_only(tmp_path):
    meta = _frames(tmp_path)
    meta["split_grouped"] = "val"
    assert overlay.build_masks(meta, tmp_path) == {}


# ---------- варианты области ----------
def test_region_prep_requires_resources_and_passes_through(tmp_path):
    with pytest.raises(ValueError):
        RegionPrep("crop")
    with pytest.raises(ValueError):
        RegionPrep("ui")
    with pytest.raises(ValueError):
        RegionPrep("nope")
    row = pd.Series({"image": "x", "machine": "M", "width": 300, "height": 200})
    im = _fan()
    assert RegionPrep("full")(im, row).size == im.size
    p = RegionPrep("crop_ui", {"x": {**sector.detect(im)}}, {})            # маски формата нет -> только кроп
    assert p(im, row).size[0] <= 300


# ---------- конфигурация и обучение ----------
@pytest.mark.skipif(not CONFIG.exists(), reason="нет configs/ablation.json")
def test_ablation_config_changes_one_parameter_at_a_time():
    from fetal.train import load_config
    base = load_config(CONFIG, "B0")
    names = list(json.loads(CONFIG.read_text())["variants"])
    assert names[0] == "B0" and len(names) == 8
    for n in names[1:]:
        cfg = load_config(CONFIG, n)
        changed = {k for k in cfg if k != "variant" and cfg[k] != base[k]}
        assert len(changed) == 1, f"{n}: меняет {changed}"
    with pytest.raises(ValueError):
        load_config(CONFIG, "B0", {"aug": "none", "hflip": True})


@needs_data
def test_train_smoke_writes_artifacts_uses_only_val_and_is_idempotent(tmp_path):
    from fetal.train import load_config, run
    cfg = load_config(CONFIG, "B0", {"epochs": 1, "pretrained": False, "num_workers": 0, "size": 64, "batch_size": 16})
    m1 = run(cfg, tmp_path, limit_train=48, limit_val=32, device="cpu", log=lambda *_: None)
    d = tmp_path / "B0" / "seed0"
    assert {f.name for f in d.iterdir()} == {"config.json", "log.csv", "val_pred.npz", "metrics.json", "env.json"}
    z = np.load(d / "val_pred.npz")
    meta = pd.read_csv(META).set_index("image")
    assert (meta.loc[z["image"], "split_grouped"] == "val").all() and z["logits"].shape == (32, 9)
    assert {"macro_f1", "balanced_accuracy", "f1_brain", "per_machine", "per_subset"} <= set(m1)
    assert set(m1["per_subset"]) == {"class_pure_formats", "mixed_formats"}
    stamp = (d / "metrics.json").stat().st_mtime_ns
    m2 = run(cfg, tmp_path, limit_train=48, limit_val=32, device="cpu", log=lambda *_: None)    # повтор пропускается
    assert (d / "metrics.json").stat().st_mtime_ns == stamp and m2 == m1


# ---------- сводка ----------
def _fake_run(root, variant, seed, y, groups, machine, preds):
    d = root / variant / f"seed{seed}"
    d.mkdir(parents=True)
    logits = np.eye(9)[preds] + 0.01
    np.savez_compressed(d / "val_pred.npz", image=np.array([f"i{k}" for k in range(len(y))]), y=y, logits=logits,
                        machine=machine, patient_group=groups)
    (d / "metrics.json").write_text(json.dumps(metrics.evaluate(y, preds, machine)))


def test_report_summarize_pairs_seeds_and_flags_improvement(tmp_path):
    from fetal import report
    y, base, groups = _toy(n_groups=120)
    rng = np.random.default_rng(2)
    better = [np.where((p != y) & (rng.random(len(y)) < 0.6), y, p) for p in base]
    machine = np.where(np.arange(len(y)) % 2 == 0, "Aloka", "Voluson E6")
    for s in range(3):
        _fake_run(tmp_path, "B0", s, y, groups, machine, base[s])
        _fake_run(tmp_path, "A2_ui", s, y, groups, machine, better[s])
    df, md = report.summarize(tmp_path, "B0", n_boot=300)
    row = df[df.variant == "A2_ui"].iloc[0]
    assert "лучше" in row["вывод (по ΔF1)"] and row["групп / seed"] == "120 / 3" and "±" in row["macro_f1"]
    assert "| variant |" in md and list(df.columns)[:3] == ["variant", "seeds", "macro_f1"]


def test_report_refuses_to_pair_runs_on_different_val_images(tmp_path):
    from fetal import report
    y, base, groups = _toy()
    machine = np.array(["Aloka"] * len(y))
    _fake_run(tmp_path, "B0", 0, y, groups, machine, base[0])
    _fake_run(tmp_path, "X", 0, y, groups, machine, base[1])
    z = dict(np.load(tmp_path / "X" / "seed0" / "val_pred.npz"))
    z["image"] = z["image"][::-1].copy()
    np.savez_compressed(tmp_path / "X" / "seed0" / "val_pred.npz", **z)
    with pytest.raises(AssertionError):
        report.summarize(tmp_path, "B0", n_boot=50)
