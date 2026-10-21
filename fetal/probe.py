"""Проба «предскажи аппарат (и оператора)» по эмбеддингам замороженного ResNet-18 для вариантов A0–A3.

Зачем: если аппарат легко узнаётся по входу модели, у неё есть возможность «схитрить». Проба измеряет, что
аппарат МОЖНО узнать, а не что дообученная модель этим пользуется.

Протокол (согласован заранее):
  * экстрактор — ResNet-18 с весами ImageNet, eval(), без аугментаций, тот же препроцессинг, что у варианта;
    признаки — усреднённый вектор перед классификатором (512-мерный эмбеддинг);
  * линейная голова (логистическая регрессия на стандартизованных признаках) обучается ТОЛЬКО на train,
    оценивается на val (те же группы пациентов, что в split_grouped);
  * цель «аппарат»: три реальных аппарата (Aloka, Voluson E6, Voluson S10); «Other» — смесь, исключён;
  * состав классов: вес объекта в train равен 1 / размер ячейки «аппарат × класс» (класс не подсказывает аппарат);
    главная метрика — balanced accuracy внутри класса, усреднённая по классам, где у КАЖДОГО аппарата в val >= MIN_VAL кадров;
  * оператор: только внутри E6 (Op. 1 против Op. 2); «Other» у оператора — смесь, исключён;
  * контроли: случайный уровень, «только отношение сторон», «только класс» (показывает вклад состава классов);
  * интервалы — бутстрэп по patient_group; сравнение с A0 парное (одни и те же выборки групп).
Запуск:  python -m fetal.probe   (-> results/probe/probe.json, probe.md)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCANNERS = ("Aloka", "Voluson E6", "Voluson S10")
OPERATORS = ("Op. 1", "Op. 2")                      # внутри E6
MIN_VAL = 10
MODES = ("full", "crop", "ui", "crop_ui")             # A0..A3
NAMES = {"full": "A0", "crop": "A1", "ui": "A2", "crop_ui": "A3"}


# ---------- метрика ----------
def eligible_classes(cls: np.ndarray, target: np.ndarray, n_targets: int, min_val: int = MIN_VAL) -> list:
    """Классы, у которых в val для каждого значения цели не меньше min_val кадров."""
    out = []
    for c in np.unique(cls):
        cnt = np.bincount(target[cls == c], minlength=n_targets)
        if (cnt >= min_val).all():
            out.append(c)
    return out


def within_class_balacc(cls, y, p, classes, n_targets: int, w=None) -> float:
    """Balanced accuracy внутри класса (средний recall по значениям цели), усреднённая по классам.
    cls, y, p — массивы одной длины (y, p — коды цели 0..n_targets-1); w — веса объектов (для бутстрэпа)."""
    cls, y, p = np.asarray(cls), np.asarray(y), np.asarray(p)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    ci = {c: i for i, c in enumerate(classes)}
    keep = np.isin(cls, classes)
    cell = np.array([ci[c] for c in cls[keep]]) * n_targets + y[keep]
    den = np.bincount(cell, weights=w[keep], minlength=len(classes) * n_targets)
    hit = p[keep] == y[keep]
    num = np.bincount(cell[hit], weights=w[keep][hit], minlength=len(classes) * n_targets)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(den > 0, num / den, np.nan).reshape(len(classes), n_targets)
    return float(np.nanmean(np.nanmean(rec, axis=1)))


def plain_balacc(y, p, n_targets: int, w=None) -> float:
    y, p = np.asarray(y), np.asarray(p)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    den = np.bincount(y, weights=w, minlength=n_targets)
    hit = p == y
    num = np.bincount(y[hit], weights=w[hit], minlength=n_targets)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nanmean(np.where(den > 0, num / den, np.nan)))


def cell_weights(cls: np.ndarray, target: np.ndarray) -> np.ndarray:
    """1 / размер ячейки (класс, цель), нормировано на среднее 1: класс не подсказывает цель."""
    cell = pd.factorize(pd.Series(cls).astype(str) + "|" + pd.Series(target).astype(str))[0]
    w = 1.0 / np.bincount(cell)[cell]
    return w * len(w) / w.sum()


# ---------- признаки и голова ----------
def extract_embeddings(meta: pd.DataFrame, raw_dir, region, size: int = 224, batch: int = 64, workers: int = 6,
                       device: str = "auto") -> np.ndarray:
    """512-мерные эмбеддинги строк meta (в том же порядке) замороженным ImageNet-ResNet-18."""
    import torch
    import torchvision

    from fetal.datasets import FetalPlanesDataset, make_loader
    dev = torch.device(("mps" if torch.backends.mps.is_available() else "cpu") if device == "auto" else device)
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval().to(dev)
    m = meta.reset_index(drop=True).assign(_part="all")
    ds = FetalPlanesDataset(m, raw_dir, "_part", "all", train=False, size=size, region=region)
    out = []
    with torch.no_grad():
        for x, _ in make_loader(ds, batch, False, workers):
            out.append(net(x.to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def fit_predict(Xtr, ytr, wtr, Xva, C: float = 1.0) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), ytr, sample_weight=wtr)
    return clf.predict(sc.transform(Xva))


def fit_predict_tabular(Xtr, ytr, wtr, Xva) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(Xtr, ytr, sample_weight=wtr).predict(Xva)


# ---------- бутстрэп по группам ----------
def group_weights(groups: np.ndarray, n_boot: int, seed: int = 0):
    """Генератор весов объектов: выборка групп с возвращением (одна и та же для всех сравниваемых моделей)."""
    gi, uniq = pd.factorize(groups)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)))[gi].astype(float)


def boot_ci(scores: dict, groups, metric, n_boot: int = 2000, seed: int = 0, base: str | None = None) -> dict:
    """scores: {имя: предсказания}; metric(pred, w) -> число. Интервалы метрик и разностей с base (парные)."""
    names = list(scores)
    point = {n: metric(scores[n], None) for n in names}
    draws = {n: [] for n in names}
    for w in group_weights(np.asarray(groups), n_boot, seed):
        for n in names:
            draws[n].append(metric(scores[n], w))
    res = {}
    for n in names:
        d = np.array(draws[n])
        res[n] = {"value": point[n], "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        if base and n != base:
            dd = d - np.array(draws[base])
            res[n]["delta_vs_base"] = {"value": point[n] - point[base],
                                      "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
    return res


# ---------- основной расчёт ----------
def run_probe(meta: pd.DataFrame, emb: dict, target_col: str, targets: tuple, subset: np.ndarray, C: float,
              n_boot: int, split_col: str = "split_grouped") -> dict:
    """emb: {режим: эмбеддинги для ВСЕХ строк meta}. subset — булева маска строк meta, на которых строится проба."""
    code = {t: i for i, t in enumerate(targets)}
    sel = subset & meta[target_col].isin(targets).to_numpy()
    tr, va = sel & (meta[split_col] == "train").to_numpy(), sel & (meta[split_col] == "val").to_numpy()
    ytr, yva = meta.loc[tr, target_col].map(code).to_numpy(), meta.loc[va, target_col].map(code).to_numpy()
    ctr, cva = meta.loc[tr, "cls9"].to_numpy(), meta.loc[va, "cls9"].to_numpy()
    wtr = cell_weights(ctr, ytr)
    classes = eligible_classes(cva, yva, len(targets))
    groups = meta.loc[va, "patient_group"].to_numpy()
    nt = len(targets)
    preds = {}
    aspect = (meta.width / meta.height).to_numpy()[:, None]
    for mode in MODES:
        preds[NAMES[mode]] = fit_predict(emb[mode][tr], ytr, wtr, emb[mode][va], C)
    preds["контроль: только отношение сторон"] = fit_predict_tabular(aspect[tr], ytr, wtr, aspect[va])
    onehot = pd.get_dummies(meta["cls9"]).to_numpy(float)
    preds["контроль: только класс (без выравнивания весов)"] = fit_predict_tabular(onehot[tr], ytr, np.ones(len(ytr)), onehot[va])
    preds["контроль: случайно"] = np.random.default_rng(0).integers(0, nt, len(yva))
    within = boot_ci(preds, groups, lambda p, w: within_class_balacc(cva, yva, p, classes, nt, w), n_boot, 0, "A0")
    plain = boot_ci(preds, groups, lambda p, w: plain_balacc(yva, p, nt, w), n_boot, 0, "A0")
    return {"target": target_col, "targets": list(targets), "C": C, "n_train": int(tr.sum()), "n_val": int(va.sum()),
            "val_groups": int(len(np.unique(groups))), "eligible_classes": [str(c) for c in classes],
            "within_class_balanced_accuracy": within, "plain_balanced_accuracy": plain, "chance": 1 / nt}


def to_markdown(res: dict) -> str:
    def block(r: dict, title: str) -> str:
        rows = ["| вход / контроль | внутри класса, bal. acc. [95% ИД] | Δ против A0 [95% ИД] | обычная bal. acc. |", "|---|---|---|---|"]
        for n, v in r["within_class_balanced_accuracy"].items():
            d = v.get("delta_vs_base")
            rows.append(f"| {n} | {100 * v['value']:.1f} % [{100 * v['ci95'][0]:.1f}; {100 * v['ci95'][1]:.1f}] | "
                        + (f"{100 * d['value']:+.1f} п.п. [{100 * d['ci95'][0]:+.1f}; {100 * d['ci95'][1]:+.1f}]" if d else "") + " | "
                        + f"{100 * r['plain_balanced_accuracy'][n]['value']:.1f} % |")
        meta = (f"Цель: {', '.join(r['targets'])}; train {r['n_train']}, val {r['n_val']} кадров из {r['val_groups']} групп; "
                f"классы в оценке: {', '.join(r['eligible_classes'])}; случайный уровень {100 * r['chance']:.1f} %; C={r['C']}.")
        return f"### {title}\n\n{meta}\n\n" + "\n".join(rows) + "\n"
    out = ["# Проба «предскажи аппарат»: замороженные эмбеддинги ResNet-18, линейная голова\n"]
    for k, title in (("scanner", "Аппарат (Aloka / Voluson E6 / Voluson S10)"), ("scanner_C_low", "Аппарат, более сильная регуляризация (робастность)"),
                     ("operator_e6", "Оператор внутри E6 (Op. 1 / Op. 2)")):
        if k in res:
            out.append(block(res[k], title))
    out.append("Проба показывает, что аппарат МОЖНО узнать по входу, а не что дообученная модель этим пользуется. "
               "Интервалы отражают состав val; нижняя граница случайного уровня — контроль «случайно». "
               "Вывод по вариантам делаем по Δ против A0, а не по абсолютному значению.\n")
    return "\n".join(out)


def main() -> None:
    from fetal.overlay import load_masks
    from fetal.regions import RegionPrep
    from fetal.sector import load_sectors
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default="data/processed/metadata.csv")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--sector", default="data/cache/sector.csv")
    ap.add_argument("--overlay", default="data/cache/overlay_masks.npz")
    ap.add_argument("--out", default="results/probe")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    meta = pd.read_csv(args.meta)
    use = meta[meta.split_grouped.isin(["train", "val"]) & meta.machine.isin(SCANNERS)].reset_index(drop=True)
    sectors, overlays = load_sectors(args.sector), load_masks(args.overlay)
    emb = {}
    for mode in MODES:
        f = Path(args.cache) / f"probe_emb_{mode}.npz"
        if f.exists() and (np.load(f, allow_pickle=False)["image"] == use["image"].to_numpy().astype(str)).all():
            emb[mode] = np.load(f)["emb"]
            print(f"{NAMES[mode]}: эмбеддинги из кэша")
            continue
        emb[mode] = extract_embeddings(use, args.raw, RegionPrep(mode, sectors, overlays), workers=args.workers)
        np.savez_compressed(f, image=use["image"].to_numpy().astype(str), emb=emb[mode])
        print(f"{NAMES[mode]}: эмбеддинги {emb[mode].shape} посчитаны")
    allm = np.ones(len(use), bool)
    res = {"scanner": run_probe(use, emb, "machine", SCANNERS, allm, 1.0, args.n_boot),
           "scanner_C_low": run_probe(use, emb, "machine", SCANNERS, allm, 0.05, args.n_boot),
           "operator_e6": run_probe(use, emb, "operator", OPERATORS, (use.machine == "Voluson E6").to_numpy(), 1.0, args.n_boot)}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "probe.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    md = to_markdown(res)
    (Path(args.out) / "probe.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
