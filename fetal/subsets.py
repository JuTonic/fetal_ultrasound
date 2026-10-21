"""Вложенные стратифицированные подвыборки обучающих пациентов (25 / 50 / 100 %) для пункта (e).

Единица выборки — patient_group (пациенты-дубликаты всегда вместе), поэтому ни один пациент не делится между
«взятыми» и «неиспользуемыми» кадрами. Val и test не меняются ни в одной подвыборке.

Алгоритм (без scikit-learn и без генератора случайных чисел, чтобы результат не зависел от версий библиотек):
  1. train делится жадно на 4 фолда: группы идут по убыванию числа кадров, ничьи решает хэш (seed, группа);
     каждая группа попадает в фолд, где она меньше всего ухудшает выровненность долей 9 классов, 4 аппаратов и размера;
  2. порядок фолдов задаёт тот же хэш: 25 % = фолд 0, 50 % = фолды 0–1, 100 % = все (поэтому подвыборки вложены).

Колонки, которые добавляются в таблицу (существующие не меняются):
  train_fold            0..3 для кадров train, -1 для val/test
  split_grouped_f25     как split_grouped, но кадры train вне 25 % подвыборки помечены "unused"
  split_grouped_f50     то же для 50 %; 100 % — это сам split_grouped
Использование: FetalPlanesDataset(meta, "data/raw", "split_grouped_f25", "train", ...) или `split_col` в configs.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 0
N_FOLDS = 4
COLUMNS = ("train_fold", "split_grouped_f25", "split_grouped_f50")
FRACS = {25: 1, 50: 2, 100: 4}                  # доля -> число фолдов
UNUSED = "unused"


def _key(seed: int, name) -> int:
    """Детерминированный «случайный» ключ: не зависит от версии numpy/python."""
    return int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:16], 16)


def assign_folds(meta: pd.DataFrame, split_col: str = "split_grouped", group_col: str = "patient_group",
                 seed: int = SEED, n_folds: int = N_FOLDS) -> pd.Series:
    """train_fold для каждой строки: 0..n_folds-1 у кадров train, -1 у остальных (порядок фолдов = порядок вложенности)."""
    is_tr = (meta[split_col] == "train").to_numpy()
    tr = meta[is_tr]
    groups = pd.Index(sorted(tr[group_col].unique()))
    gi = groups.get_indexer(tr[group_col])
    cls = pd.factorize(tr["cls9"], sort=True)[0]
    mach = pd.factorize(tr["machine"], sort=True)[0]
    G, K, M = len(groups), cls.max() + 1, mach.max() + 1
    Yc = np.zeros((G, K)); np.add.at(Yc, (gi, cls), 1)
    Ym = np.zeros((G, M)); np.add.at(Ym, (gi, mach), 1)
    n = Yc.sum(1)
    Tc, Tm, N = Yc.sum(0), Ym.sum(0), n.sum()
    Fc, Fm, Fn = np.zeros((n_folds, K)), np.zeros((n_folds, M)), np.zeros(n_folds)
    fold_of = np.empty(G, int)
    order = sorted(range(G), key=lambda g: (-n[g], _key(seed, groups[g])))        # крупные группы первыми
    for g in order:
        best, best_cost = 0, np.inf
        for f in range(n_folds):
            # прирост квадрата отклонения долей от 1/n_folds (а не само отклонение после добавления: иначе жадно
            # заполняется один фолд). Больше всего выигрывает фолд с наибольшим «дефицитом» этой группы.
            d = lambda F, y, T: (((F + y) / T - 1 / n_folds) ** 2 - (F / T - 1 / n_folds) ** 2).sum()
            cost = d(Fc[f], Yc[g], Tc) + 0.5 * d(Fm[f], Ym[g], Tm) + 1.0 * d(Fn[f], n[g], N)
            if cost < best_cost - 1e-15:
                best, best_cost = f, cost
        fold_of[g] = best
        Fc[best] += Yc[g]; Fm[best] += Ym[g]; Fn[best] += n[g]
    rank = {f: i for i, f in enumerate(sorted(range(n_folds), key=lambda f: _key(seed, f"fold{f}")))}
    out = np.full(len(meta), -1, dtype=int)
    out[is_tr] = [rank[fold_of[g]] for g in gi]
    return pd.Series(out, index=meta.index, name="train_fold")


def add_subset_columns(meta: pd.DataFrame, split_col: str = "split_grouped", seed: int = SEED) -> pd.DataFrame:
    """Копия таблицы с колонками COLUMNS. Остальные колонки не меняются."""
    m = meta.copy()
    m["train_fold"] = assign_folds(m, split_col, seed=seed)
    for pct, k in FRACS.items():
        if pct == 100:
            continue
        col = f"{split_col}_f{pct}"
        m[col] = np.where((m[split_col] == "train") & (m["train_fold"] >= k), UNUSED, m[split_col])
    return m


def subset_report(meta: pd.DataFrame, split_col: str = "split_grouped", group_col: str = "patient_group") -> pd.DataFrame:
    """Размеры подвыборок и максимальные отклонения долей классов и аппаратов от полного train (п.п.)."""
    tr = meta[meta[split_col] == "train"]
    rows = []
    for pct, k in FRACS.items():
        s = tr[tr["train_fold"] < k]
        r = {"доля": f"{pct} %", "кадров": len(s), "групп": s[group_col].nunique(), "пациентов": s["patient"].nunique()}
        for col, name in (("cls9", "классы"), ("machine", "аппараты")):
            a = s[col].value_counts(normalize=True).reindex(tr[col].unique()).fillna(0)
            b = tr[col].value_counts(normalize=True)
            r[f"макс. откл. {name}, п.п."] = round(float(100 * (a - b.reindex(a.index)).abs().max()), 2)
        r["мин. кадров в классе"] = int(s["cls9"].value_counts().reindex(tr["cls9"].unique()).fillna(0).min())
        rows.append(r)
    return pd.DataFrame(rows)


def validate_subsets(meta: pd.DataFrame, split_col: str = "split_grouped", group_col: str = "patient_group",
                     max_class_dev_pp: float = 3.0) -> None:
    """Вложенность, неизменность val/test, целостность групп, наличие всех классов, баланс долей."""
    assert all(c in meta for c in COLUMNS), f"нет колонок {COLUMNS}"
    base = meta[split_col]
    is_tr = base == "train"
    assert (meta.loc[~is_tr, "train_fold"] == -1).all() and (meta.loc[is_tr, "train_fold"] >= 0).all()
    prev = None
    for pct in (25, 50):
        c = meta[f"{split_col}_f{pct}"]
        assert (c[~is_tr] == base[~is_tr]).all(), f"{pct} %: val/test изменились"
        used = c == "train"
        assert set(c[is_tr].unique()) <= {"train", UNUSED} and (used == (is_tr & (meta.train_fold < FRACS[pct]))).all()
        assert meta.loc[used, group_col].isin(meta.loc[is_tr & ~used, group_col]).sum() == 0, f"{pct} %: группа разделена"
        if prev is not None:
            assert (prev <= used).all(), "подвыборки не вложены: кадр из меньшей подвыборки отсутствует в большей"
        prev = used
    assert (prev <= is_tr).all()
    rep = subset_report(meta, split_col, group_col)
    assert (rep["мин. кадров в классе"] > 0).all(), "в подвыборке нет одного из классов"
    assert (rep["макс. откл. классы, п.п."] <= max_class_dev_pp).all(), f"баланс классов хуже {max_class_dev_pp} п.п."


def subsets_checksum(meta: pd.DataFrame) -> str:
    """SHA-256 соответствия image -> (train_fold, split_grouped_f25, split_grouped_f50): «паспорт» подвыборок."""
    t = meta[["image", *COLUMNS]].sort_values("image")
    return hashlib.sha256(t.to_csv(index=False).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Добавляет колонки подвыборок в metadata.csv (быстро, без пересчёта хэшей картинок)")
    ap.add_argument("--meta", default="data/processed/metadata.csv")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="перезаписать колонки, если они уже есть и отличаются")
    args = ap.parse_args()
    path = Path(args.meta)
    meta = pd.read_csv(path)
    new = add_subset_columns(meta.drop(columns=[c for c in COLUMNS if c in meta]), seed=args.seed)
    validate_subsets(new)
    if all(c in meta for c in COLUMNS):
        same = all((meta[c].astype(str) == new[c].astype(str)).all() for c in COLUMNS)
        if not same and not args.force:
            raise SystemExit("колонки подвыборок уже есть и отличаются от пересчитанных; --force, чтобы перезаписать")
    new.to_csv(path, index=False)
    path.with_name("subsets.sha256").write_text(subsets_checksum(new) + "\n")
    print(subset_report(new).to_string(index=False))
    print("записано:", path, "| контрольная сумма:", path.with_name("subsets.sha256"))


if __name__ == "__main__":
    main()
