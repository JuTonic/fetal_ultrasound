"""Контроли шортката без самой картинки: что предсказывается по метаданным, геометрии сектора и формату.

Это справочные линии, а не пороги: результат ниже не доказывает, что модель не использует аппарат,
а выше не доказывает, что использует. Обучение на train, оценка на val (split_grouped).
Запуск:  python -m fetal.controls   (-> results/stage0/controls.json)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from fetal import metrics
from fetal.overlay import fmt_key, load_masks


def _fit_eval(X: pd.DataFrame, y: pd.Series, tr: np.ndarray, va: np.ndarray) -> dict:
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(X[tr], y[tr])
    p = clf.predict(X[va])
    # метрики по классам, присутствующим в val (для аппаратов — по 4 меткам)
    codes, uniq = pd.factorize(pd.concat([y[va], pd.Series(p, index=y[va].index)]))
    yt, yp = codes[:va.sum()], codes[va.sum():]
    k = len(uniq)
    cm = metrics.confusion(yt, yp, k)
    return {"macro_f1": round(metrics.macro_f1(cm), 3), "balanced_accuracy": round(metrics.balanced_accuracy(cm), 3)}


def class_pure_formats(meta: pd.DataFrame, min_frames: int = 50, purity: float = 0.95) -> dict:
    """Какая доля кадров лежит в форматах (аппарат + размер), где один класс >= purity."""
    fmt = [fmt_key(m, w, h) for m, w, h in zip(meta.machine, meta.width, meta.height)]
    g = meta.assign(fmt=fmt).groupby("fmt")
    rows = []
    for k, d in g:
        if len(d) >= min_frames:
            top = d.cls9.value_counts()
            rows.append({"format": k, "n": len(d), "top_class": top.index[0], "share": float(top.iloc[0] / len(d))})
    t = pd.DataFrame(rows, columns=["format", "n", "top_class", "share"])      # пусто, если нет форматов с min_frames
    pure = t[t.share >= purity]
    return {"formats_with_>=min_frames": len(t), "class_pure_formats": len(pure),
            "frames_in_class_pure_formats": int(pure.n.sum()), "share_of_all_frames": round(float(pure.n.sum() / len(meta)), 3),
            "list": pure.sort_values("n", ascending=False).round(3).to_dict("records")}


def pure_format_keys(meta_train: pd.DataFrame, min_frames: int = 30, purity: float = 0.95) -> set[str]:
    """Ключи форматов, где в train один класс составляет >= purity (определяется только по train)."""
    r = class_pure_formats(meta_train, min_frames, purity)
    return {f["format"] for f in r["list"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/processed/metadata.csv")
    ap.add_argument("--sector", default="data/cache/sector.csv")
    ap.add_argument("--overlay", default="data/cache/overlay_masks.npz")
    ap.add_argument("--out", default="results/stage0/controls.json")
    args = ap.parse_args()
    m = pd.read_csv(args.meta)
    s = pd.read_csv(args.sector).drop(columns=["hull", "detector"])
    d = m.merge(s, on="image")
    masks = load_masks(args.overlay)
    d["aspect"] = d.width / d.height
    d["has_ui_mask"] = [fmt_key(a, b, c) in masks for a, b, c in zip(d.machine, d.width, d.height)]
    d["ui_mask_frac"] = [masks[fmt_key(a, b, c)].mean() if fmt_key(a, b, c) in masks else 0.0
                         for a, b, c in zip(d.machine, d.width, d.height)]
    d["box_area"] = (d.x1 - d.x0) * (d.y1 - d.y0)
    tr, va = (d.split_grouped == "train").to_numpy(), (d.split_grouped == "val").to_numpy()
    res = {"chance_macro_f1_classes": round(float(1 / 9), 3), "chance_balanced_acc_machine_4": 0.25,
           "sector_ok_share": round(float(d.sector_ok.mean()), 4), "n_sector_not_ok": int((~d.sector_ok).sum()),
           "ui_mask_coverage_of_frames": round(float(d.has_ui_mask.mean()), 4),
           "box_is_whole_frame_share": round(float((d.box_area >= 0.95).mean()), 3), "class_from": {}, "machine_from": {}}
    feats = {"aspect": ["aspect"], "width_height": ["width", "height"],
             "sector_geometry": ["area", "x0", "y0", "x1", "y1", "aspect"],
             "ui_mask_flag_and_fraction": ["has_ui_mask", "ui_mask_frac"],
             "sector_geometry+aspect+ui_mask": ["area", "x0", "y0", "x1", "y1", "aspect", "has_ui_mask", "ui_mask_frac"]}
    for name, cols in feats.items():
        X = d[cols].astype(float)
        res["class_from"][name] = _fit_eval(X, d["cls9"], tr, va)
        res["machine_from"][name] = _fit_eval(X, d["machine"], tr, va)
    res["class_pure_formats"] = class_pure_formats(d)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "class_pure_formats"}, indent=2, ensure_ascii=False))
    cp = res["class_pure_formats"]
    print(f"\nформаты с >=50 кадров: {cp['formats_with_>=min_frames']}; из них почти чистых по классу: "
          f"{cp['class_pure_formats']} ({cp['frames_in_class_pure_formats']} кадров = {cp['share_of_all_frames']:.1%} всех)")


if __name__ == "__main__":
    main()
