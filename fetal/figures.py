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
    paths += [augmentation_examples(meta, raw, out, args.size, args.seed), sizes_overview(meta, out, args.size)]
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
