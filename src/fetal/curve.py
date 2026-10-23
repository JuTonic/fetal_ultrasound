"""Кривая «качество против размера обучающей выборки» (пункт e): таблица, интервалы, график.

Запуск:  python -m fetal.curve   (-> results/data_efficiency/curve.csv, curve.md, curve.png)
Источники: 25 и 50 % — results/data_efficiency/B0_f25|B0_f50 (configs/data_efficiency.json), 100 % — B0 из results/ablation
(тот же конфиг, не пересчитывается). Только val; те же val-кадры у всех долей, поэтому шаги между долями сравниваются парным
бутстрэпом по patient_group. Интервал точки — бутстрэп по группам пациентов для среднего по seed (состав val), std — по seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fetal import bootstrap, metrics

SOURCES = {25: ("results/data_efficiency", "B0_f25"), 50: ("results/data_efficiency", "B0_f50"), 100: ("results/ablation", "B0")}


def load(sources: dict = SOURCES) -> dict[int, dict[int, dict]]:
    out: dict[int, dict[int, dict]] = {}
    for pct, (root, variant) in sources.items():
        for m in sorted(Path(root, variant).glob("seed*/metrics.json")):
            z = np.load(m.parent / "val_pred.npz")
            out.setdefault(pct, {})[int(m.parent.name.removeprefix("seed"))] = {
                "metrics": json.loads(m.read_text()), "z": {k: z[k] for k in z.files}}
    return out


def point_ci(runs: dict[int, dict], metric: str, n_boot: int, seed: int = 0) -> tuple[float, float]:
    """95%-й интервал бутстрэпа по группам для среднего по seed."""
    z0 = next(iter(runs.values()))["z"]
    f = bootstrap.METRICS[metric]
    preds = [r["z"]["logits"].argmax(1) for r in runs.values()]
    y = z0["y"]
    gi, uniq = pd.factorize(z0["patient_group"])
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)))[gi].astype(float)
        vals[b] = np.mean([f(metrics.confusion(y, p, w=w)) for p in preds])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def build(runs: dict[int, dict[int, dict]], n_boot: int = 3000) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    for pct in sorted(runs):
        ms = [r["metrics"] for r in runs[pct].values()]
        row = {"доля обучающих пациентов, %": pct, "seeds": len(ms),
               "групп в train": None, "кадров в train": None}
        for met, tag in (("macro_f1", "macro-F1"), ("balanced_accuracy", "balanced acc.")):
            v = [m[met] for m in ms]
            lo, hi = point_ci(runs[pct], met, n_boot)
            row.update({tag: float(np.mean(v)), f"{tag} std": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
                        f"{tag} lo": lo, f"{tag} hi": hi})
        for name in ("Brain: TT", "Brain: TC", "Brain: TV", "Brain: other"):
            row[f"F1 {name}"] = float(np.mean([m["f1_brain"][name] for m in ms]))
        for mach in ("Aloka", "Voluson E6", "Voluson S10"):
            row[f"F1 {mach}"] = float(np.mean([m["per_machine"][mach]["macro_f1"] for m in ms]))
        rows.append(row)
    steps = []
    pcts = sorted(runs)
    z0 = runs[pcts[0]][min(runs[pcts[0]])]["z"]
    for a, b in zip(pcts[:-1], pcts[1:]):
        common = sorted(set(runs[a]) & set(runs[b]))
        for s in common:
            assert (runs[a][s]["z"]["image"] == z0["image"]).all() and (runs[b][s]["z"]["image"] == z0["image"]).all(), "разный состав val"
        res = bootstrap.paired_bootstrap(z0["y"], [runs[a][s]["z"]["logits"].argmax(1) for s in common],
                                         [runs[b][s]["z"]["logits"].argmax(1) for s in common], z0["patient_group"],
                                         n_boot=n_boot, n_comparisons=len(pcts) - 1)
        steps.append({"от": a, "до": b, "delta": res["delta"], "ci95": res["ci95"], "per_seed": res["per_seed_delta"]})
    return pd.DataFrame(rows), steps


def plot(df: pd.DataFrame, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
    series = [("macro-F1", "#2a78d6"), ("balanced acc.", "#eb6834")]            # слоты 1 и 2 палитры (соседняя пара проверена)
    x = df["доля обучающих пациентов, %"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for (tag, col), dx in zip(series, (-0.9, 0.9)):
        y = df[tag].to_numpy()
        ax.fill_between(x + dx, df[f"{tag} lo"], df[f"{tag} hi"], color=col, alpha=0.12, lw=0, zorder=1)
        ax.errorbar(x + dx, y, yerr=df[f"{tag} std"], color=col, lw=0, elinewidth=1.6, capsize=4, zorder=3)
        ax.plot(x + dx, y, color=col, lw=2, marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4, label=tag)
        for xi, yi in zip(x + dx, y):
            ax.annotate(f"{yi:.3f}".replace(".", ","), (xi, yi), textcoords="offset points", xytext=(0, 11 if tag == "macro-F1" else -17),
                        ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [f"{p} %" for p in x], fontsize=10, color=INK)
    ax.set_xlim(x.min() - 8, x.max() + 8)
    ax.set_xlabel("Доля обучающих пациентов", fontsize=10, color=INK2)
    ax.set_ylabel("Качество на val", fontsize=10, color=INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color="#e3e2de", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    fig.text(0.02, 0.965, "Качество растёт с размером обучающей выборки", fontsize=13, fontweight="bold", color=INK, ha="left")
    fig.text(0.02, 0.918, "ResNet-18 (B0), val, 3 seed: точка — среднее, усы — std по seed, полоса — 95% интервал по группам пациентов",
             fontsize=8.8, color=INK2, ha="left")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.13)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def to_markdown(df: pd.DataFrame, steps: list[dict]) -> str:
    r = lambda v: f"{v:.3f}".replace(".", ",")
    pp = lambda v: f"{100 * v:+.1f}".replace(".", ",")
    rows = ["| Доля обучающих пациентов | macro-F1 (± std по seed) [95% ИД] | balanced acc. | F1 Brain: other | F1 Brain: TV | F1 S10 | F1 E6 | F1 Aloka |", "|---|---|---|---|---|---|---|---|"]
    for _, x in df.iterrows():
        rows.append(f"| {int(x['доля обучающих пациентов, %'])} % | {r(x['macro-F1'])} ± {r(x['macro-F1 std'])} [{r(x['macro-F1 lo'])}; {r(x['macro-F1 hi'])}] | "
                    f"{r(x['balanced acc.'])} ± {r(x['balanced acc. std'])} | {r(x['F1 Brain: other'])} | {r(x['F1 Brain: TV'])} | "
                    f"{r(x['F1 Voluson S10'])} | {r(x['F1 Voluson E6'])} | {r(x['F1 Aloka'])} |")
    st = ["| Шаг | Δ macro-F1 [95% ИД] | Δ по seed, п.п. |", "|---|---|---|"]
    for s in steps:
        st.append(f"| {s['от']} % → {s['до']} % | {pp(s['delta'])} п.п. [{pp(s['ci95'][0])}; {pp(s['ci95'][1])}] | " + " / ".join(pp(d) for d in s["per_seed"]) + " |")
    return "# Качество против размера обучающей выборки (B0, val)\n\n" + "\n".join(rows) + "\n\nПарное сравнение шагов (те же seed и val-кадры, бутстрэп по группам пациентов, интервал с поправкой на число шагов не показан):\n\n" + "\n".join(st) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/data_efficiency")
    ap.add_argument("--n-boot", type=int, default=3000)
    args = ap.parse_args()
    runs = load()
    df, steps = build(runs, args.n_boot)
    out = Path(args.out)
    df.to_csv(out / "curve.csv", index=False)
    md = to_markdown(df, steps)
    (out / "curve.md").write_text(md)
    plot(df, out / "curve.png")
    print(md)


if __name__ == "__main__":
    main()
