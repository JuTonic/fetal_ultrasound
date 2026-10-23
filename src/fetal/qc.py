"""Контроль качества детектора сектора и fallback: объективная потеря содержимого кропом и отчёт по запасным кадрам.

Запуск:  python -m fetal.qc            (-> results/stage0/qc.json, results/stage0/fallback.md)
         python -m fetal.qc --sheets   (+ results/stage0/sector_manual_check/: листы для просмотра глазами и sample.csv)
Не зависит от torch. Считает по ВСЕМ кадрам (а не по выборке): доля яркости вне бокса кропа, перечень fallback-кадров,
что с ними делает каждый вариант области кадра и верны ли предсказания на fallback-кадрах val у обученных вариантов.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fetal.overlay import apply_overlay, fmt_key, load_masks
from fetal.sector import crop_loss


def _loss(args: tuple) -> float:
    path, rec = args
    return crop_loss(Image.open(path), rec)


def content_loss(meta: pd.DataFrame, sector: pd.DataFrame, raw: Path, workers: int = 8) -> pd.DataFrame:
    """Потеря содержимого кропом для каждого кадра: image, sector_ok, loss."""
    d = meta[["image"]].merge(sector, on="image")
    jobs = [(str(raw / "Images" / f"{r.image}.png"), {"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1}) for r in d.itertuples()]
    with ProcessPoolExecutor(workers) as ex:
        loss = list(ex.map(_loss, jobs, chunksize=64))
    return pd.DataFrame({"image": d["image"], "sector_ok": d["sector_ok"], "loss": loss})


def fallback_table(meta: pd.DataFrame, sector: pd.DataFrame, overlays: dict, raw: Path) -> pd.DataFrame:
    """Fallback-кадры (sector_ok = False) и что с ними делает каждый вариант области кадра."""
    d = meta.merge(sector[["image", "sector_ok", "area", "energy"]], on="image")
    fb = d[~d.sector_ok]
    rows = []
    for r in fb.itertuples():
        im = Image.open(raw / "Images" / f"{r.image}.png").convert("L")
        masked = apply_overlay(im, overlays.get(fmt_key(r.machine, r.width, r.height)))
        rows.append({"image": r.image, "часть": r.split_grouped, "класс": r.cls9, "аппарат": r.machine,
                     "площадь оболочки": round(r.area, 2), "доля яркости в оболочке": round(r.energy, 3),
                     "A1 кроп": "кроп пропущен (как A0)", "A2 маска интерфейса": "маска применена" if not np.array_equal(np.asarray(masked), np.asarray(im)) else "формат без маски (как A0)",
                     "A3": ("кроп и закраска вне оболочки пропущены; " + ("маска интерфейса применена" if not np.array_equal(np.asarray(masked), np.asarray(im)) else "формат без маски"))})
    return pd.DataFrame(rows)


def val_predictions_on(frames: list[str], root: Path) -> pd.DataFrame:
    """Верны ли предсказания обученных вариантов (по seed) на заданных val-кадрах."""
    rows = []
    for f in sorted(Path(root).glob("*/seed*/val_pred.npz")):
        z = np.load(f)
        idx = [int(np.flatnonzero(z["image"] == n)[0]) for n in frames if (z["image"] == n).any()]
        for i in idx:
            rows.append({"вариант": f.parent.parent.name, "seed": int(f.parent.name.removeprefix("seed")), "image": str(z["image"][i]),
                         "верно": bool(z["logits"][i].argmax() == z["y"][i])})
    return pd.DataFrame(rows)


def review_sheets(meta: pd.DataFrame, sector: pd.DataFrame, loss: pd.DataFrame, raw: Path, out_dir: Path,
                  per_cell: int = 5, n_worst: int = 40, seed: int = 0) -> pd.DataFrame:
    """Листы для просмотра глазами: n_worst кадров с наибольшей потерей кропа и per_cell кадров на ячейку «аппарат × класс»
    (только уверенные срабатывания, любые части; выборка воспроизводима). Возвращает таблицу выборки (номер = номер на листе)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ab = {"Voluson E6": "E6", "Voluson S10": "S10", "Aloka": "Alk", "Other": "Oth"}
    d = meta.merge(sector, on="image").merge(loss[["image", "loss"]], on="image")
    d = d[d.sector_ok]

    def sheet(t: pd.DataFrame, fp: Path, title: str, rows: int = 5, cols: int = 8) -> None:
        fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.55 * rows))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (_, r) in zip(axes.ravel(), t.iterrows()):
            im = Image.open(raw / "Images" / f"{r.image}.png").convert("L")
            w, h = im.size
            ax.imshow(im, cmap="gray", vmin=0, vmax=160)
            hl = np.array(json.loads(r.hull)) * [w, h]
            hl = np.vstack([hl, hl[:1]])
            ax.plot(hl[:, 0], hl[:, 1], color="#ff3030", lw=1.1)
            ax.add_patch(plt.Rectangle((r.x0 * w, r.y0 * h), (r.x1 - r.x0) * w, (r.y1 - r.y0) * h, fill=False, ec="#30a0ff", lw=1.2, ls="--"))
            ax.set_title(f"#{r.no} {ab[r.machine]} {r.cls9[:10]} loss={100 * r.loss:.1f}%", fontsize=7.5)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(fp, dpi=55)
        plt.close(fig)

    out_dir.mkdir(parents=True, exist_ok=True)
    worst = d.nlargest(n_worst, "loss").reset_index(drop=True)
    worst["no"], worst["набор"] = worst.index + 1, "наибольшая потеря"
    sheet(worst, out_dir / "worst.png", f"{n_worst} кадров с наибольшей потерей яркости вне бокса кропа (синий пунктир — бокс, красный — оболочка)")
    strat = d.sample(frac=1, random_state=seed).groupby(["machine", "cls9"]).head(per_cell).sort_values(["machine", "cls9"]).reset_index(drop=True)
    strat["no"], strat["набор"] = strat.index + 1, "стратифицированная"
    for k in range(0, len(strat), 40):
        sheet(strat.iloc[k:k + 40], out_dir / f"strat_{k // 40 + 1}.png",
              f"Стратифицированная выборка, кадры {k + 1}-{min(k + 40, len(strat))} (по {per_cell} на ячейку «аппарат × класс»)")
    cols = ["набор", "no", "image", "machine", "cls9", "split_grouped", "area", "energy", "loss"]
    table = pd.concat([worst[cols], strat[cols]], ignore_index=True)
    table.to_csv(out_dir / "sample.csv", index=False)
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/processed/metadata.csv")
    ap.add_argument("--sector", default="data/cache/sector.csv")
    ap.add_argument("--overlay", default="data/cache/overlay_masks.npz")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--ablation", default="results/ablation")
    ap.add_argument("--out", default="results/stage0")
    ap.add_argument("--sheets", action="store_true", help="сохранить листы для просмотра глазами")
    args = ap.parse_args()
    sector_full = pd.read_csv(args.sector)
    meta, sector = pd.read_csv(args.meta), sector_full.drop(columns=["hull"])
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cl = content_loss(meta, sector, raw)
    if args.sheets:
        review_sheets(meta, sector_full, cl, raw, out / "sector_manual_check")
    ok = cl[cl.sector_ok]
    q = lambda p: round(float(100 * np.percentile(ok["loss"], p)), 2)
    worst = ok.nlargest(10, "loss")
    res = {"кадров с уверенным сектором": int(len(ok)),
           "потеря яркости вне бокса кропа, %": {"медиана": q(50), "90-й перцентиль": q(90), "99-й перцентиль": q(99), "максимум": round(float(100 * ok.loss.max()), 2)},
           "кадров с потерей > 5 %": int((ok.loss > 0.05).sum()), "кадров с потерей > 10 %": int((ok.loss > 0.10).sum()),
           "10 худших кадров": [{"image": r.image, "потеря, %": round(100 * r.loss, 2)} for r in worst.itertuples()]}

    ft = fallback_table(meta, sector, load_masks(args.overlay), raw)
    res["fallback"] = {"кадров": int(len(ft)), "в train": int((ft["часть"] == "train").sum()), "в val": int((ft["часть"] == "val").sum()),
                       "в test": int((ft["часть"] == "test").sum()),
                       "A3 всё же применяет маску интерфейса": int(ft["A3"].str.contains("маска интерфейса применена").sum())}
    val_frames = ft.loc[ft["часть"] == "val", "image"].tolist()
    vp = val_predictions_on(val_frames, Path(args.ablation)) if Path(args.ablation).exists() and val_frames else pd.DataFrame()
    if len(vp):
        res["fallback"]["предсказания на fallback-кадрах val"] = {v: f"{int(g['верно'].sum())} из {len(g)} прогонов верно" for v, g in vp.groupby("вариант")}
    (out / "qc.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    md = ["# Fallback и контроль кропа\n",
          f"Fallback-кадры (детектор не уверен в секторе): {len(ft)} из {len(meta)}. Для них кроп и закраска вне оболочки пропускаются; **маска интерфейса, которая определяется по формату "
          "кадра, а не по детектору сектора, всё же применяется** (так реализован A3 и именно так он посчитан в абляции).\n",
          "| " + " | ".join(ft.columns) + " |", "|" + "---|" * len(ft.columns), *["| " + " | ".join(map(str, r)) + " |" for r in ft.itertuples(index=False)],
          "\n## Потеря содержимого кропом (все уверенные кадры)\n", json.dumps(res["потеря яркости вне бокса кропа, %"], ensure_ascii=False),
          f"\nКадров с потерей > 5 %: {res['кадров с потерей > 5 %']}, > 10 %: {res['кадров с потерей > 10 %']} из {len(ok)}.\n"]
    if len(vp):
        md += ["## Предсказания на fallback-кадре(ах) val\n", "| вариант | seed | верно |", "|---|---|---|",
               *[f"| {r.вариант} | {r.seed} | {'да' if r.верно else 'нет'} |" for r in vp.itertuples()]]
    (out / "fallback.md").write_text("\n".join(md) + "\n")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
