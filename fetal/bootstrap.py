"""Парный бутстрэп по группам пациентов для сравнения вариантов с базовым (B0). Не зависит от torch.

Для каждого повтора выбираются patient_group с возвращением (ОДНА выборка для обеих моделей и всех
seed), считается метрика каждой модели, затем Δ = метрика(вариант) − метрика(B0) на каждом seed и
среднее Δ по seed. Интервал отражает неопределённость СОСТАВА val при уже обученных моделях; случайность
обучения он не учитывает (её показывают std по seed и знак Δ по каждому seed отдельно).
Бутстрэп не создаёт новых независимых наблюдений: в результат всегда входит число групп.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fetal.metrics import K, balanced_accuracy, confusion, macro_f1

METRICS = {"macro_f1": macro_f1, "balanced_accuracy": balanced_accuracy}


def paired_bootstrap(y, base_preds, var_preds, groups, metric: str = "macro_f1", n_boot: int = 5000,
                     seed: int = 0, subset=None, alpha: float = 0.05, n_comparisons: int = 1) -> dict:
    """base_preds, var_preds: списки массивов предсказаний (по одному на seed, порядок seed одинаковый).
    subset: булева маска объектов (например, один аппарат) — бутстрэп только внутри неё.
    n_comparisons: число сравнений с B0; даёт интервал с поправкой Бонферрони (alpha / n_comparisons)."""
    if len(base_preds) != len(var_preds):
        raise ValueError("нужно одинаковое число seed у базового варианта и у сравниваемого")
    f = METRICS[metric]
    y = np.asarray(y)
    sel = np.ones(len(y), bool) if subset is None else np.asarray(subset, bool)
    y = y[sel]
    base = [np.asarray(p)[sel] for p in base_preds]
    var = [np.asarray(p)[sel] for p in var_preds]
    gi, uniq = pd.factorize(np.asarray(groups)[sel])
    G = len(uniq)
    rng = np.random.default_rng(seed)
    point = float(np.mean([f(confusion(y, v)) - f(confusion(y, b)) for b, v in zip(base, var)]))
    per_seed = [float(f(confusion(y, v)) - f(confusion(y, b))) for b, v in zip(base, var)]
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        w = rng.multinomial(G, np.full(G, 1.0 / G))[gi].astype(np.float64)       # выборка групп с возвращением
        deltas[i] = np.mean([f(confusion(y, v, w=w)) - f(confusion(y, b, w=w)) for b, v in zip(base, var)])
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    a2 = alpha / max(n_comparisons, 1)
    lo_adj, hi_adj = np.percentile(deltas, [100 * a2 / 2, 100 * (1 - a2 / 2)])
    return {"metric": metric, "delta": point, "ci95": (float(lo), float(hi)), "ci_adj": (float(lo_adj), float(hi_adj)),
            "n_comparisons": n_comparisons, "per_seed_delta": per_seed, "n_seeds": len(base), "n_groups": int(G),
            "n_objects": int(len(y)), "n_boot": n_boot}


def decision(res: dict, min_delta: float = 0.01) -> str:
    """Правило плана: средняя Δ >= min_delta и нижняя граница 95%-интервала > 0. Строгость — по поправке."""
    if res["delta"] >= min_delta and res["ci95"][0] > 0:
        return "лучше (скорректированный интервал тоже > 0)" if res["ci_adj"][0] > 0 else "лучше (разведочно)"
    if res["delta"] <= -min_delta and res["ci95"][1] < 0:
        return "хуже (скорректированный интервал тоже < 0)" if res["ci_adj"][1] < 0 else "хуже (разведочно)"
    return "различий не видно"
