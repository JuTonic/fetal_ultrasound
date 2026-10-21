"""Метрики качества: macro-F1, balanced accuracy, по классам, по аппаратам. Не зависит от torch.

Макро-усреднение идёт по классам, у которых в данных есть хотя бы один объект (support > 0);
классы без объектов (например, редкий класс в маленьком подмножестве аппарата) не учитываются.
На всей val присутствуют все 9 классов, поэтому там результат совпадает с sklearn (macro, 9 меток).
"""
from __future__ import annotations

import numpy as np

from fetal.data import CLASSES9

K = len(CLASSES9)
BRAIN = {c: i for i, c in enumerate(CLASSES9) if c.startswith("Brain")}


def confusion(y: np.ndarray, p: np.ndarray, k: int = K, w: np.ndarray | None = None) -> np.ndarray:
    """cm[i, j] = (взвешенное) число объектов истинного класса i, предсказанных как j."""
    return np.bincount(np.asarray(y) * k + np.asarray(p), weights=w, minlength=k * k).reshape(k, k)


def f1_per_class(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, np.nan)
    return np.where(cm.sum(1) > 0, f1, np.nan)             # нет объектов класса -> nan


def macro_f1(cm: np.ndarray) -> float:
    return float(np.nanmean(f1_per_class(cm)))


def balanced_accuracy(cm: np.ndarray) -> float:
    support = cm.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(support > 0, np.diag(cm) / support, np.nan)
    return float(np.nanmean(rec))


def _subset_scores(y, p, s) -> dict:
    c = confusion(y[s], p[s])
    return {"n": int(s.sum()), "classes_present": int((c.sum(1) > 0).sum()),
            "macro_f1": macro_f1(c) if s.any() else None, "balanced_accuracy": balanced_accuracy(c) if s.any() else None}


def evaluate(y, p, machine=None, subsets: dict | None = None) -> dict:
    """Все метрики одного прогона. machine — массив названий аппаратов той же длины (необязательно);
    subsets — {имя: булева маска} для отдельных срезов (например, форматы, чистые по классу)."""
    y, p = np.asarray(y), np.asarray(p)
    cm = confusion(y, p)
    f1 = f1_per_class(cm)
    out = {"n": int(len(y)), "accuracy": float((y == p).mean()), "macro_f1": macro_f1(cm),
           "balanced_accuracy": balanced_accuracy(cm),
           "f1_per_class": {c: (None if np.isnan(f) else float(f)) for c, f in zip(CLASSES9, f1)},
           "f1_brain": {c: (None if np.isnan(f1[i]) else float(f1[i])) for c, i in BRAIN.items()}}
    if machine is not None:
        machine = np.asarray(machine)
        out["per_machine"] = {}
        for mach in sorted(set(machine)):
            out["per_machine"][mach] = _subset_scores(y, p, machine == mach)
    if subsets:
        out["per_subset"] = {name: _subset_scores(y, p, np.asarray(mask, bool)) for name, mask in subsets.items()}
    return out
