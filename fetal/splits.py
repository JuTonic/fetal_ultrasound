"""Разбиения train/val/test: по пациентам (основное), официальное и наивное image-level.

Все функции принимают таблицу из fetal.data (колонки: label, machine, patient,
patient_group, md5_gray, split_official_raw) и возвращают массив строк 'train'/'val'/'test'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

SPLITS = ("train", "val", "test")


def _sgkf_one_fold(y, groups, n_splits: int, seed: int):
    """Первая свёртка StratifiedGroupKFold: (индексы остатка, индексы отложенной части)."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return next(cv.split(np.zeros(len(y)), y, groups))


def grouped_split(meta: pd.DataFrame, seed: int, n_test: int = 7, n_val: int = 6,
                  group_col: str = "patient_group") -> np.ndarray:
    """Разбиение по группам пациентов, стратификация по 9 классам.

    Доли: test = 1/n_test (14.3 %), val = (1 - 1/n_test)/n_val (14.3 %), train — остальное (71.4 %).
    group_col = patient_group, т.е. пациенты-дубликаты всегда попадают в одну часть.
    """
    y, g = meta["label"].to_numpy(), meta[group_col].to_numpy()
    rest, test = _sgkf_one_fold(y, g, n_test, seed)
    tr, va = _sgkf_one_fold(y[rest], g[rest], n_val, seed)
    split = np.full(len(meta), "train", dtype=object)
    split[rest[va]] = "val"
    split[test] = "test"
    return split


def imbalance_score(meta: pd.DataFrame, split: np.ndarray) -> float:
    """Чем меньше, тем лучше: отклонение долей классов и сканеров в val/test от общих
    (макс. абс. отклонение) + отклонение размера части от целевого 1/7."""
    score = 0.0
    for col in ("cls9", "machine"):
        total = meta[col].value_counts(normalize=True)
        for part in ("val", "test"):
            share = meta.loc[split == part, col].value_counts(normalize=True).reindex(total.index).fillna(0)
            score += float((share - total).abs().max())
    for part in ("val", "test"):
        score += abs(float((split == part).mean()) - 1 / 7)
    return score


def best_grouped_split(meta: pd.DataFrame, seeds=range(50)):
    """Перебирает seed разбиения и берёт самый сбалансированный (по классам и сканерам).
    Выбор делается только по метаданным, модель/метрики не используются."""
    best = None
    for s in seeds:
        sp = grouped_split(meta, s)
        sc = imbalance_score(meta, sp)
        if best is None or sc < best[0]:
            best = (sc, s, sp)
    return best[2], best[1], best[0]


def official_split_with_val(meta: pd.DataFrame, seeds=range(50), n_val: int = 10) -> np.ndarray:
    """Официальный сплит (Train=1/0) + валидация = 1/n_val пациентов train
    (в статье Burgos-Artizzu — 10 % train). test не трогаем.
    Seed выбирается по минимальному отклонению долей классов val от train (без модели)."""
    base = meta["split_official_raw"].to_numpy(dtype=object)
    tr = np.flatnonzero(base == "train")
    y, g = meta["label"].to_numpy()[tr], meta["patient_group"].to_numpy()[tr]
    total = meta["cls9"].iloc[tr].value_counts(normalize=True)
    best = None
    for s in seeds:
        _, va = _sgkf_one_fold(y, g, n_val, s)
        share = meta["cls9"].iloc[tr[va]].value_counts(normalize=True).reindex(total.index).fillna(0)
        sc = float((share / total - 1).abs().max())          # относительное отклонение, ловит редкие классы
        if best is None or sc < best[0]:
            best = (sc, va)
    split = base.copy()
    split[tr[best[1]]] = "val"
    return split


def naive_image_split(meta: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Случайное image-level разбиение 5:1:1 со стратификацией по классам. ПРОТЕЧКА по пациентам
    (кадры одного пациента в разных частях) — нужно только как негативный контроль для п. (d)."""
    idx = np.arange(len(meta))
    rest, test = train_test_split(idx, test_size=1 / 7, stratify=meta["label"], random_state=seed)
    tr, va = train_test_split(rest, test_size=1 / 6, stratify=meta["label"].to_numpy()[rest], random_state=seed)
    split = np.full(len(meta), "train", dtype=object)
    split[va] = "val"
    split[test] = "test"
    return split


def check_no_leakage(meta: pd.DataFrame, split_col: str) -> dict:
    """Число пациентов / групп пациентов / побитово одинаковых картинок, встречающихся более
    чем в одной части. Для корректного patient-level разбиения все три значения = 0."""
    res = {}
    for key, col in (("patients", "patient"), ("patient_groups", "patient_group"), ("identical_images", "md5_gray")):
        res[key] = int((meta.groupby(col)[split_col].nunique() > 1).sum())
    return res


def split_summary(meta: pd.DataFrame, split_col: str) -> pd.DataFrame:
    """Размеры частей, число пациентов и доли классов/сканеров (в %)."""
    rows = {}
    for part in SPLITS:
        m = meta[meta[split_col] == part]
        rows[part] = {"images": len(m), "patients": m["patient"].nunique()}
    out = pd.DataFrame(rows).T
    cls = (100 * pd.crosstab(meta["cls9"], meta[split_col], normalize="columns")).round(1)
    mach = (100 * pd.crosstab(meta["machine"], meta[split_col], normalize="columns")).round(1)
    return out, cls[[c for c in SPLITS if c in cls]], mach[[c for c in SPLITS if c in mach]]
