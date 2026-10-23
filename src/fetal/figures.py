"""Визуальная проверка препроцессинга: сетки «до -> после» по сканерам и классам, аугментации.

Запуск из корня репозитория:  python -m fetal.figures   (картинки -> results/figures/)
Примеры выбираются случайно с фиксированным seed из train части split_grouped.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from fetal.data import CLASSES9
from fetal.transforms import IMAGENET_MEAN, IMAGENET_STD, AugConfig, preprocess

MACHINES = ["Aloka", "Voluson E6", "Voluson S10", "Other"]


def to_gray(x: np.ndarray) -> np.ndarray:
    """Нормализованный тензор [3, H, W] -> картинка [H, W] в 0..1 (обратная ImageNet-нормализация)."""
    return np.clip(x[0] * IMAGENET_STD[0] + IMAGENET_MEAN[0], 0, 1)


def _blank(ax, text: str = "нет примеров") -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=8, color="gray", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])


def pick_examples(meta: pd.DataFrame, seed: int) -> dict[tuple[str, str], pd.Series]:
    """По одному примеру на пару (сканер, класс); картинки с конфликтом меток не берём."""
    pool = meta[(meta.split_grouped == "train") & ~meta.label_conflict]
    out = {}
    for (mach, cls), g in pool.groupby(["machine", "cls9"]):
        out[(mach, cls)] = g.sample(1, random_state=seed).iloc[0]
    return out


def before_after_by_machine(meta, raw: Path, out_dir: Path, size: int, seed: int) -> list[Path]:
    ex = pick_examples(meta, seed)
    paths = []
    for mach in MACHINES:
        fig, axes = plt.subplots(2, len(CLASSES9), figsize=(2.2 * len(CLASSES9), 4.9))
        for j, cls in enumerate(CLASSES9):
            ax0, ax1 = axes[0, j], axes[1, j]
            ax0.set_title(cls, fontsize=9)
            row = ex.get((mach, cls))
            if row is None:
                _blank(ax0); _blank(ax1)
                continue
            img = Image.open(raw / "Images" / f"{row.image}.png")
            ax0.imshow(img.convert("L"), cmap="gray", vmin=0, vmax=255)
            ax0.set_xlabel(f"{row.width}×{row.height}", fontsize=7)
            ax1.imshow(to_gray(preprocess(img, size)), cmap="gray", vmin=0, vmax=1)
            ax1.set_xlabel(f"{size}×{size}", fontsize=7)
            for ax in (ax0, ax1):
                ax.set_xticks([]); ax.set_yticks([])
        axes[0, 0].set_ylabel("до", fontsize=10); axes[1, 0].set_ylabel("после", fontsize=10)
        fig.suptitle(f"{mach}: исходный кадр (сверху) и вход модели (снизу), letterbox {size}", fontsize=11)
        fig.tight_layout()
        p = out_dir / f"before_after_{mach.replace(' ', '_')}.png"
        fig.savefig(p, dpi=110); plt.close(fig); paths.append(p)
    return paths


def augmentation_examples(meta, raw: Path, out_dir: Path, size: int, seed: int, n: int = 7) -> Path:
    """Один и тот же кадр: без аугментаций, 7 случайных (как на train) и 7 с включённым hflip."""
    row = meta[(meta.split_grouped == "train") & (meta.cls9 == "Brain: TT") & (meta.machine == "Voluson E6")].iloc[3]
    img = Image.open(raw / "Images" / f"{row.image}.png")
    fig, axes = plt.subplots(2, n + 1, figsize=(2.0 * (n + 1), 4.5))
    for r, (cfg, name) in enumerate([(AugConfig(), "по умолчанию"),
                                     (AugConfig(hflip=True), "+ отражение")]):
        axes[r, 0].imshow(to_gray(preprocess(img, size)), cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_title("без аугментаций", fontsize=8)
        rng = random.Random(seed)
        for k in range(n):
            axes[r, k + 1].imshow(to_gray(preprocess(img, size, True, cfg, rng)), cmap="gray", vmin=0, vmax=1)
            axes[r, k + 1].set_title(f"#{k + 1}", fontsize=8)
        fig.text(0.012, 0.70 - 0.40 * r, name, rotation=90, va="center", fontsize=8)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Аугментации одного кадра ({row.machine}, {row.cls9})", fontsize=11)
    fig.tight_layout(rect=(0.03, 0, 1, 0.95))
    p = out_dir / "augmentations.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p


def sizes_overview(meta, out_dir: Path, size: int) -> Path:
    """Размеры и формат кадров по сканерам: сколько занимает картинка на холсте после letterbox."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for mach, c in zip(MACHINES, ["#1f77b4", "#d95f02", "#1b9e77", "#7570b3"]):
        m = meta[meta.machine == mach]
        ax.scatter(m.width, m.height, s=4, alpha=0.35, color=c, label=f"{mach} (n={len(m)})")
    ax.set_xlabel("ширина, px"); ax.set_ylabel("высота, px")
    ax.set_title("Размеры исходных кадров по сканерам")
    ax.legend(markerscale=3, fontsize=8, frameon=False)
    fig.tight_layout()
    p = out_dir / "image_sizes_by_machine.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p


def class_imbalance(meta: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, Path]:
    """Таблица (class_imbalance.csv) и график дисбаланса 9 классов: число кадров, доля, доли в train/val/test split_grouped."""
    SURFACE, INK, INK2, MUTED, BAR = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#2a78d6"
    by = pd.crosstab(meta.cls9, meta.split_grouped).reindex(columns=["train", "val", "test"])
    t = pd.DataFrame({"всего": by.sum(axis=1)})
    t["доля, %"] = (100 * t["всего"] / len(meta)).round(1)
    t = t.join(by).sort_values("всего", ascending=False)
    t["во сколько раз реже самого частого"] = (t["всего"].iloc[0] / t["всего"]).round(1)
    for part in ("train", "val", "test"):
        t[f"доля в {part}, %"] = (100 * t[part] / t[part].sum()).round(1)
    t.index.name = "класс"
    t.to_csv(out_dir / "class_imbalance.csv", encoding="utf-8-sig")
    ratio, even = t["всего"].iloc[0] / t["всего"].iloc[-1], len(meta) / len(t)
    fig, ax = plt.subplots(figsize=(9, 5.0), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(range(len(t)), t["всего"], height=0.55, color=BAR, zorder=3)
    ax.axvline(even, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(even + 40, len(t) - 0.35, f"поровну: {even:,.0f}".replace(",", " "), color=MUTED, fontsize=8.5, va="center")
    for i, (n, pc) in enumerate(zip(t["всего"], t["доля, %"])):
        ax.text(n + 40, i, f"{n:,}".replace(",", " ") + f"  ·  {pc:.1f} %".replace(".", ","), va="center", fontsize=9.5, color=INK,
                zorder=4, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.set_yticks(range(len(t)), t.index, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, t["всего"].max() * 1.22)
    ax.xaxis.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(0.02, 0.955, "Дисбаланс классов в FETAL_PLANES_DB", fontsize=13.5, fontweight="bold", color=INK, ha="left")
    fig.text(0.02, 0.905, f"Самый частый класс ({t.index[0]}) встречается в {ratio:.1f} раза чаще самого редкого ({t.index[-1]});".replace(".", ",")
             + f" всего {len(meta):,} кадров в 9 классах".replace(",", " "), fontsize=9.5, color=INK2, ha="left")
    fig.text(0.02, 0.02, "Доли классов в train, val и test почти одинаковы (разбиение по пациентам со стратификацией), см. таблицу.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.86, bottom=0.08)
    path = out_dir / "class_imbalance.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return t, path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/processed/metadata.csv")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    meta = pd.read_csv(args.meta)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = Path(args.raw)
    paths = before_after_by_machine(meta, raw, out, args.size, args.seed)
    paths += [augmentation_examples(meta, raw, out, args.size, args.seed), sizes_overview(meta, out, args.size),
              class_imbalance(meta, out)[1]]
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
