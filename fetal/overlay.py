"""Маски статического интерфейса (текст, значки, шкалы) по формату кадра. Не зависит от torch.

Идея: интерфейс аппарата стоит на одном и том же месте у всех кадров одного формата
(аппарат + ширина×высота), а анатомия двигается. Для каждого формата с достаточным числом кадров
train считаем, как часто в каждом пикселе встречается тонкий яркий штрих (белый top-hat), берём
пиксели с частотой >= FREQ, склеиваем строки текста в блоки и немного расширяем.
Маска строится ТОЛЬКО по кадрам train (val и test не используются, даже без меток).
Формат без маски (мало кадров) возвращается без изменений: так же, как вариант A0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

VERSION = "overlay-v1.1"
STROKE = 25          # порог белого top-hat (0..255): яркий штрих тоньше окна
WIN = 5              # окно opening, px (штрихи тоньше -> top-hat)
FREQ = 0.30          # доля кадров формата со штрихом в пикселе, чтобы считать пиксель интерфейсом
                     # (при 0.10 спекл ткани в «однотипных» форматах давал 40 % кадра; текст стабильно >= 0.3)
MAX_MASK = 0.20      # предохранитель: маска больше этой доли кадра считается ненадёжной, формат идёт без маски
CLOSE = (9, 25)      # склейка строк текста в блок (высота, ширина), px
GROW = 4             # расширение маски, px
MIN_BLOB = 40        # меньшие пятна считаем шумом, px
MIN_FRAMES = 60      # минимум кадров train в формате
MAX_FRAMES = 600     # не больше кадров на формат при расчёте частоты (seed фиксирован)


def fmt_key(machine: str, width: int, height: int) -> str:
    return f"{machine}|{width}x{height}"


def stroke_map(a: np.ndarray) -> np.ndarray:
    """a: uint8 [H, W] -> bool [H, W]: тонкие яркие штрихи (текст, линии, штангенциркули)."""
    a = a.astype(np.float32)
    return (a - ndi.grey_opening(a, size=(WIN, WIN))) > STROKE


def mask_from_freq(freq: np.ndarray) -> np.ndarray:
    m = freq >= FREQ
    m = ndi.binary_closing(m, structure=np.ones(CLOSE, dtype=bool))
    lab, n = ndi.label(m)
    if n:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, np.flatnonzero(sizes >= MIN_BLOB) + 1)
    return ndi.binary_dilation(m, structure=np.ones((2 * GROW + 1, 2 * GROW + 1), dtype=bool))


def build_masks(meta: pd.DataFrame, raw: Path, split_col: str = "split_grouped") -> dict[str, np.ndarray]:
    """{ключ формата: bool-маска [H, W]} по кадрам части train."""
    tr = meta[meta[split_col] == "train"].copy()
    tr["fmt"] = [fmt_key(m, w, h) for m, w, h in zip(tr.machine, tr.width, tr.height)]
    out = {}
    for fmt, g in tr.groupby("fmt"):
        if len(g) < MIN_FRAMES:
            continue
        g = g.sample(min(len(g), MAX_FRAMES), random_state=0)
        acc = None
        for name in g["image"]:
            s = stroke_map(np.asarray(Image.open(raw / "Images" / f"{name}.png").convert("L")))
            acc = s.astype(np.float32) if acc is None else acc + s
        mk = mask_from_freq(acc / len(g))
        if mk.mean() <= MAX_MASK:
            out[fmt] = mk
    return out


def apply_overlay(img: Image.Image, mask: np.ndarray | None) -> Image.Image:
    """img — 'L'. Закрашивает маску чёрным; без маски или при несовпадении размера возвращает img."""
    if mask is None or mask.shape != (img.size[1], img.size[0]):
        return img
    a = np.asarray(img).copy()
    a[mask] = 0
    return Image.fromarray(a, "L")


def save_masks(masks: dict[str, np.ndarray], path: str | Path) -> None:
    np.savez_compressed(path, **{k: v.astype(np.uint8) for k, v in masks.items()})


def load_masks(path: str | Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k].astype(bool) for k in z.files}


def main() -> None:
    ap = argparse.ArgumentParser(description="Считает маски интерфейса по формату (только train)")
    ap.add_argument("--meta", default="data/processed/metadata.csv")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/cache/overlay_masks.npz")
    ap.add_argument("--split-col", default="split_grouped")
    args = ap.parse_args()
    meta = pd.read_csv(args.meta)
    masks = build_masks(meta, Path(args.raw), args.split_col)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_masks(masks, args.out)
    keys = {fmt_key(m, w, h) for m, w, h in zip(meta.machine, meta.width, meta.height)}
    covered = np.mean([fmt_key(m, w, h) in masks for m, w, h in zip(meta.machine, meta.width, meta.height)])
    print(f"{VERSION}: масок {len(masks)} из {len(keys)} форматов; кадров с маской: {covered:.1%} -> {args.out}")
    for k, v in sorted(masks.items()):
        print(f"  {k:28s} закрашено {v.mean():.1%} кадра")


if __name__ == "__main__":
    main()
