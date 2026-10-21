"""Сводка абляции: среднее ± std по seed и парное сравнение каждого варианта с B0 (бутстрэп по группам).

Запуск:  python -m fetal.report --root results/ablation
  -> results/ablation/summary.md, summary.csv           сводка (среднее ± std, Δ с интервалами)
  -> results/ablation/ablation_delta_f1.png             график Δ macro-F1 против базы
  -> results/ablation/decision.json                     выбор по заранее согласованному правилу (применяется кодом)
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
MIN_DELTA = 0.01                                              # правило: Δ >= 1 п.п. и 95%-й интервал выше нуля
LABELS = {"A1_crop": "A1 кроп сектора", "A2_ui": "A2 маска интерфейса", "A3_crop_ui": "A3 кроп + маска",
          "R320": "разрешение 320 (вместо 224)", "HF_on": "отражение по горизонтали", "AUG_none": "без аугментаций",
          "AUG_strong": "сильные аугментации"}


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


def deltas(root: str | Path, base: str = "B0", metric: str = "macro_f1", n_boot: int = 5000) -> pd.DataFrame:
    """Числовые Δ каждого варианта против базы (парный бутстрэп по patient_group, те же seed)."""
    runs = load_runs(Path(root))
    if base not in runs:
        raise SystemExit(f"нет базового варианта {base!r} в {root}")
    others = [v for v in runs if v != base]
    ref = next(iter(runs[base].values()))["z"]
    rows = []
    for v in others:
        common = sorted(set(runs[v]) & set(runs[base]))
        zb, zv = [runs[base][s]["z"] for s in common], [runs[v][s]["z"] for s in common]
        for z in zb + zv:
            assert (z["image"] == ref["image"]).all(), "порядок или состав кадров val различается между прогонами"
        res = bootstrap.paired_bootstrap(ref["y"], [z["logits"].argmax(1) for z in zb], [z["logits"].argmax(1) for z in zv],
                                         ref["patient_group"], metric=metric, n_boot=n_boot, n_comparisons=len(others))
        rows.append({"variant": v, "delta": res["delta"], "lo": res["ci95"][0], "hi": res["ci95"][1],
                     "lo_adj": res["ci_adj"][0], "hi_adj": res["ci_adj"][1], "per_seed": res["per_seed_delta"],
                     "n_groups": res["n_groups"], "decision": bootstrap.decision(res, MIN_DELTA)})
    return pd.DataFrame(rows)


def decide(d: pd.DataFrame, config: str | Path, base: str = "B0") -> dict:
    """Заранее согласованное правило: по каждому параметру берётся вариант, прошедший правило
    (Δ >= 1 п.п. и 95%-й интервал выше нуля), при нескольких — с большей Δ; если ни один — значение базы."""
    spec = json.loads(Path(config).read_text())
    base_cfg = {**spec["base"], **spec["variants"][base]}
    by_param: dict[str, list[tuple[str, object]]] = {}
    for v, ov in spec["variants"].items():
        if v != base and len(ov) == 1:
            (k, val), = ov.items()
            by_param.setdefault(k, []).append((v, val))
    drow = d.set_index("variant")
    chosen, params = dict(base_cfg), {}
    for k, cands in by_param.items():
        ok = [(v, val) for v, val in cands if v in drow.index and drow.loc[v, "decision"].startswith("лучше")]
        if ok:
            v, val = max(ok, key=lambda t: drow.loc[t[0], "delta"])
            chosen[k] = val
            params[k] = {"chosen": val, "from": v, "reason": f"{v} прошёл правило (Δ={100 * drow.loc[v, 'delta']:+.1f} п.п.)"}
        else:
            params[k] = {"chosen": base_cfg[k], "from": base,
                         "reason": "ни один вариант не прошёл правило: значение базы (" + ", ".join(v for v, _ in cands) + ")"}
    changed = {k: v for k, v in chosen.items() if base_cfg.get(k) != v}
    return {"metric": "macro_f1", "rule": f"средняя Δ >= {100 * MIN_DELTA:.0f} п.п. и нижняя граница 95%-го интервала > 0 (парный бутстрэп по patient_group)",
            "n_comparisons": int(len(d)), "base": base, "parameters": params, "final_config": chosen,
            "changed_vs_base": changed, "final_equals_base": not changed,
            "variants": {r.variant: {"delta_pp": round(100 * r.delta, 2), "ci95_pp": [round(100 * r.lo, 2), round(100 * r.hi, 2)],
                                     "ci_adj_pp": [round(100 * r.lo_adj, 2), round(100 * r.hi_adj, 2)],
                                     "per_seed_delta_pp": [round(100 * x, 2) for x in r.per_seed], "decision": r.decision}
                         for r in d.itertuples()}}


def plot_deltas(d: pd.DataFrame, path: str | Path, base: str = "B0") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, INK2, MUTED, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#2a78d6"
    d = d.iloc[::-1].reset_index(drop=True)                                  # первый вариант сверху
    fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(d) + 2.2), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axvspan(-100 * MIN_DELTA, 100 * MIN_DELTA, color="#e9e8e4", zorder=0)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    for i, r in d.iterrows():
        ax.plot([100 * r.lo_adj, 100 * r.hi_adj], [i, i], color=BLUE, lw=1.2, alpha=0.35, zorder=2, solid_capstyle="butt")
        ax.plot([100 * r.lo, 100 * r.hi], [i, i], color=BLUE, lw=4, zorder=3, solid_capstyle="round")
        ax.scatter([100 * x for x in r.per_seed], [i] * len(r.per_seed), s=22, facecolor=SURFACE, edgecolor=INK2, lw=1, zorder=5)
        ax.scatter([100 * r.delta], [i], s=60, color=INK, zorder=6)
        ax.text(1.01, i, f"{100 * r.delta:+.1f}  [{100 * r.lo:+.1f}; {100 * r.hi:+.1f}]".replace(".", ","),
                transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(d)), [LABELS.get(v, v) for v in d.variant], fontsize=10, color=INK)
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_xlabel("Δ macro-F1 против базы, п.п.", fontsize=9.5, color=INK2)
    ax.tick_params(axis="x", colors=INK2, labelsize=9)
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(axis="x", color="#e3e2de", lw=0.6, zorder=0)
    n_seeds = int(d["per_seed"].map(len).max())
    passed = [LABELS.get(v, v) for v, dec in zip(d.variant, d.decision) if dec.startswith(("лучше", "хуже"))]
    sub = ("Ни один вариант не прошёл правило (Δ ≥ 1 п.п. и интервал вне нуля): выбор остаётся за базой" if not passed
           else "Прошли правило (Δ ≥ 1 п.п. и интервал вне нуля): " + ", ".join(passed))
    fig.text(0.02, 0.965, f"Абляция препроцессинга: разность macro-F1 с базой {base} (val, {n_seeds} seed)", fontsize=13, fontweight="bold", color=INK, ha="left")
    fig.text(0.02, 0.918, sub, fontsize=9.5, color=INK2, ha="left")
    fig.text(0.02, 0.025, "● среднее Δ   ━ 95% интервал бутстрэпа по группам пациентов   ─ с поправкой на число сравнений   ○ Δ отдельного seed   "
             "серая полоса: ±1 п.п. (порог)", fontsize=8, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.27, right=0.80, top=0.86, bottom=0.2 if len(d) < 8 else 0.16)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/ablation")
    ap.add_argument("--base", default="B0")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--config", default="configs/ablation.json")
    args = ap.parse_args()
    df, md = summarize(args.root, args.base, args.n_boot)
    Path(args.root, "summary.md").write_text(md)
    df.to_csv(Path(args.root, "summary.csv"), index=False)
    d = deltas(args.root, args.base, "macro_f1", args.n_boot)
    plot_deltas(d, Path(args.root, "ablation_delta_f1.png"), args.base)
    dec = decide(d, args.config, args.base)
    Path(args.root, "decision.json").write_text(json.dumps(dec, indent=2, ensure_ascii=False))
    print(md)
    print("\nВЫБОР:", "финальная конфигурация = база" if dec["final_equals_base"] else f"меняется: {dec['changed_vs_base']}")


if __name__ == "__main__":
    main()
