"""Сводка абляции: среднее ± std по seed и парное сравнение каждого варианта с B0 (бутстрэп по группам).

Запуск:  python -m fetal.report --root results/ablation          (-> results/ablation/summary.md, summary.csv)
Читает results/ablation/<вариант>/seed<k>/{metrics.json, val_pred.npz}. Только val; test не используется.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fetal import bootstrap, metrics

MAIN_MACHINES = ("Aloka", "Voluson E6", "Voluson S10")       # аппарат «Other» — смесь (≈ 60 кадров в val): не показываем в основной таблице


def load_runs(root: Path) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {}
    for m in sorted(Path(root).glob("*/seed*/metrics.json")):
        variant, seed = m.parent.parent.name, int(m.parent.name.removeprefix("seed"))
        z = np.load(m.parent / "val_pred.npz")
        runs.setdefault(variant, {})[seed] = {"metrics": json.loads(m.read_text()), "z": {k: z[k] for k in z.files}}
    return runs


def _ms(values) -> str:
    v = np.asarray(values, float)
    return f"{v.mean():.3f}" if len(v) < 2 else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def _get(m: dict, *path):
    for k in path:
        m = m.get(k) if isinstance(m, dict) else None
        if m is None:
            return None
    return m


def _md_table(df: pd.DataFrame) -> str:
    """Markdown-таблица без внешних зависимостей (pandas.to_markdown требует tabulate)."""
    cols = list(df.columns)
    rows = [[("" if pd.isna(v) else str(v)) for v in r] for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|",
                      *["| " + " | ".join(r) + " |" for r in rows]])


def summarize(root: str | Path, base: str = "B0", n_boot: int = 5000) -> tuple[pd.DataFrame, str]:
    runs = load_runs(Path(root))
    if base not in runs:
        raise SystemExit(f"нет базового варианта {base!r} в {root}")
    others = [v for v in runs if v != base]
    n_cmp = max(len(others), 1)
    rows, lines = [], []
    ref = next(iter(runs[base].values()))["z"]
    for variant in [base, *others]:
        seeds = sorted(runs[variant])
        r = {"variant": variant, "seeds": len(seeds)}
        ms = [runs[variant][s]["metrics"] for s in seeds]
        r["macro_f1"], r["balanced_accuracy"] = _ms([m["macro_f1"] for m in ms]), _ms([m["balanced_accuracy"] for m in ms])
        for mach in MAIN_MACHINES:
            vals = [_get(m, "per_machine", mach, "macro_f1") for m in ms]
            r[f"F1 {mach}"] = _ms([v for v in vals if v is not None]) if any(v is not None for v in vals) else "—"
        for name in ("Brain: TT", "Brain: TC", "Brain: TV", "Brain: other"):
            vals = [_get(m, "f1_brain", name) for m in ms]
            r[f"F1 {name}"] = _ms([v for v in vals if v is not None]) if any(v is not None for v in vals) else "—"
        for sub in ("mixed_formats", "class_pure_formats"):
            vals = [_get(m, "per_subset", sub, "macro_f1") for m in ms]
            r[f"F1 {sub}"] = _ms([v for v in vals if v is not None]) if any(v is not None for v in vals) else "—"
        if variant != base:
            common = sorted(set(seeds) & set(runs[base]))
            if len(common) >= 1:
                zb, zv = [runs[base][s]["z"] for s in common], [runs[variant][s]["z"] for s in common]
                for z in zb + zv:           # парное сравнение корректно только на одних и тех же кадрах в одном порядке
                    assert (z["image"] == ref["image"]).all(), "порядок или состав кадров val различается между прогонами"
                for met, tag in (("macro_f1", "ΔF1"), ("balanced_accuracy", "ΔBA")):
                    res = bootstrap.paired_bootstrap(ref["y"], [z["logits"].argmax(1) for z in zb],
                                                     [z["logits"].argmax(1) for z in zv], ref["patient_group"],
                                                     metric=met, n_boot=n_boot, n_comparisons=n_cmp)
                    r[tag] = f"{100 * res['delta']:+.1f} п.п. [{100 * res['ci95'][0]:+.1f}; {100 * res['ci95'][1]:+.1f}]"
                    if met == "macro_f1":
                        r["вывод (по ΔF1)"] = bootstrap.decision(res)
                        r["знак Δ по seed"] = " ".join(f"{100 * d:+.1f}" for d in res["per_seed_delta"])
                        r["групп / seed"] = f"{res['n_groups']} / {res['n_seeds']}"
        rows.append(r)
    df = pd.DataFrame(rows)
    first = ["variant", "seeds", "macro_f1", "ΔF1", "вывод (по ΔF1)", "знак Δ по seed", "balanced_accuracy", "ΔBA", "групп / seed"]
    df = df[[c for c in first if c in df] + [c for c in df if c not in first]]
    head = (f"# Абляция препроцессинга: сводка (только val)\n\nБаза: **{base}**. Среднее ± std по seed (std при ≥ 2 seed). "
            f"Δ — разность с базой на одних и тех же seed; в скобках 95%-й интервал парного бутстрэпа по `patient_group` "
            f"(неопределённость состава val при обученных моделях; случайность обучения показывают std и знаки Δ по seed). "
            f"Сравнений с базой: {n_cmp}; без поправки на множественность выводы «лучше/хуже» разведочные. "
            f"Правило: Δ ≥ 1 п.п. и интервал вне нуля.\n\n")
    note = ("\n\nF1 по аппаратам считается по классам, присутствующим у аппарата. Малые ячейки (Brain: other — 20 кадров из 16 групп, "
            "S10 — 17 групп) дают широкие интервалы: небольшие различия надёжно оценить нельзя.\n")
    return df, head + _md_table(df) + note


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/ablation")
    ap.add_argument("--base", default="B0")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    df, md = summarize(args.root, args.base, args.n_boot)
    Path(args.root, "summary.md").write_text(md)
    df.to_csv(Path(args.root, "summary.csv"), index=False)
    print(md)


if __name__ == "__main__":
    main()
