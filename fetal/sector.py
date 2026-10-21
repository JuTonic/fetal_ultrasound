"""Поиск УЗ-сектора на кадре для вариантов препроцессинга A1–A3 (кроп / маска вне сектора).

Не зависит от torch. Метод (на кадре, уменьшенном до ширины 256):
  1. сглаженная яркость > THRESH  — низкий порог, чтобы тёмная жидкость внутри сектора не терялась;
  2. morphological opening убирает тонкий текст и линии интерфейса;
  3. берутся все существенные связные области (>= MIN_COMP кадра; тёмная жидкость у края часто режет
     сектор на несколько кусков) и строится их общая выпуклая оболочка (сектор выпуклый);
  4. проверка уверенности: оболочка занимает >= MIN_AREA кадра И содержит >= MIN_ENERGY всей яркости.
     Второе условие ловит экраны с несколькими панелями (допплер, M-режим) и тёмные кадры, где
     найден только яркий фрагмент: кроп такого кадра отрезал бы содержимое.
Кадр с низкой уверенностью (sector_ok=False) НЕ выбрасывается: apply_region возвращает его без
изменений (как вариант A0, полный кадр). Пороги заданы по визуальной проверке train, не по метрикам val.
Итог проверки: оболочка по всем существенным областям безопасна (потеря яркости в среднем 1 %), но у
81–91 % кадров бокс = весь кадр: сектор уже заполняет кадр, поэтому кроп почти ничего не меняет.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

VERSION = "sector-v2.1"
W = 256               # рабочая ширина детектора
THRESH = 2.5          # порог сглаженной яркости (0..255)
OPEN = 5              # размер структурного элемента opening (px на рабочем масштабе)
MIN_COMP = 0.01       # минимальная доля кадра для связной области, чтобы она входила в оболочку
MIN_AREA = 0.25       # доля кадра, которую должна занимать оболочка
MIN_ENERGY = 0.90     # доля суммарной яркости кадра внутри оболочки
BOX_MARGIN = 0.03     # запас вокруг бокса кропа (доля размера кадра)
MASK_PAD = 3          # расширение маски (px на рабочем масштабе)
MAX_VERTS = 32        # вершин оболочки в таблице
MODES = ("full", "crop", "mask", "crop_mask")


def _poly_mask(poly: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """poly в пикселях (x, y), size=(w, h) -> bool[h, w]."""
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).polygon([tuple(map(float, p)) for p in poly], fill=1)
    return np.asarray(m, dtype=bool)


def detect(img: Image.Image) -> dict:
    """-> {ok, reason, area, energy, n_comp, x0,y0,x1,y1 (бокс с запасом, 0..1), hull [[x,y]...] (0..1)}"""
    img = img.convert("L")
    w, h = img.size
    s = np.asarray(img.resize((W, max(8, round(h * W / w))), Image.BILINEAR), dtype=np.float32)
    H = s.shape[0]
    g = ndi.gaussian_filter(s, 1.0)
    mask = ndi.binary_opening(g > THRESH, structure=np.ones((OPEN, OPEN)))
    lab, n = ndi.label(mask)
    full = dict(ok=False, reason="no_component", area=0.0, energy=0.0, n_comp=int(n),
                x0=0.0, y0=0.0, x1=1.0, y1=1.0, hull=[[0, 0], [1, 0], [1, 1], [0, 1]])
    if n == 0:
        return full
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    keep = np.flatnonzero(sizes >= max(30, MIN_COMP * W * H)) + 1
    if len(keep) == 0:
        keep = np.array([int(np.argmax(sizes)) + 1])
    ys, xs = np.nonzero(np.isin(lab, keep))
    if len(xs) < 30:
        return full
    pts = np.stack([xs, ys], 1).astype(np.float64)
    pts = np.concatenate([pts, pts + [1, 0], pts + [0, 1], pts + [1, 1]])      # пиксель = квадрат, не точка
    verts = pts[ConvexHull(pts).vertices]
    hm = _poly_mask(verts, (W, H))
    area = float(hm.mean())
    energy = float(g[hm].sum() / max(g.sum(), 1e-6))
    x0, y0 = verts.min(0)
    x1, y1 = verts.max(0)
    x0, x1 = np.clip([x0 / W - BOX_MARGIN, x1 / W + BOX_MARGIN], 0, 1)
    y0, y1 = np.clip([y0 / H - BOX_MARGIN, y1 / H + BOX_MARGIN], 0, 1)
    idx = np.unique(np.linspace(0, len(verts) - 1, MAX_VERTS).astype(int))
    hull = np.round(verts[idx] / [W, H], 4).tolist()
    ok = area >= MIN_AREA and energy >= MIN_ENERGY
    reason = "ok" if ok else ("small_area" if area < MIN_AREA else "low_energy")
    return dict(ok=bool(ok), reason=reason, area=round(area, 4), energy=round(energy, 4), n_comp=int(n),
                x0=round(float(x0), 4), y0=round(float(y0), 4), x1=round(float(x1), 4), y1=round(float(y1), 4),
                hull=hull)


def apply_region(img: Image.Image, rec: dict | None, mode: str) -> Image.Image:
    """img — градации серого 'L'. mode: full | crop | mask | crop_mask.
    crop: вырезать бокс сектора; mask: закрасить чёрным всё вне оболочки сектора; crop_mask: оба.
    Без записи о секторе или при sector_ok=False кадр возвращается без изменений (запасной вариант)."""
    if mode not in MODES:
        raise ValueError(f"region mode {mode!r} не из {MODES}")
    if mode == "full" or rec is None or not rec["ok"]:
        return img
    w, h = img.size
    if mode in ("mask", "crop_mask"):
        hull = np.asarray(rec["hull"], dtype=np.float64)
        Hs = max(8, round(h * W / w))
        small = Image.fromarray(_poly_mask(hull * [W, Hs], (W, Hs)).astype(np.uint8) * 255)
        small = small.filter(ImageFilter.MaxFilter(2 * MASK_PAD + 1))
        m = small.resize((w, h), Image.NEAREST)
        img = Image.composite(img, Image.new("L", (w, h), 0), m)
    if mode in ("crop", "crop_mask"):
        box = (int(rec["x0"] * w), int(rec["y0"] * h), max(int(rec["x0"] * w) + 1, round(rec["x1"] * w)),
               max(int(rec["y0"] * h) + 1, round(rec["y1"] * h)))
        img = img.crop(box)
    return img


def _detect_path(path: str) -> dict:
    return detect(Image.open(path))


def build_table(meta: pd.DataFrame, raw: Path, workers: int = 8) -> pd.DataFrame:
    paths = [str(raw / "Images" / f"{n}.png") for n in meta["image"]]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_detect_path, paths, chunksize=64))
    t = pd.DataFrame(res)
    t["hull"] = [json.dumps(h, separators=(",", ":")) for h in t["hull"]]
    t.insert(0, "image", meta["image"].to_numpy())
    t.insert(1, "detector", VERSION)
    return t.rename(columns={"ok": "sector_ok"})


def load_sectors(path: str | Path) -> dict[str, dict]:
    """data/cache/sector.csv -> {image: запись для apply_region}."""
    t = pd.read_csv(path)
    out = {}
    for r in t.itertuples(index=False):
        out[r.image] = dict(ok=bool(r.sector_ok), x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1, hull=json.loads(r.hull))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Считает data/processed/sector.csv (сектор, уверенность, оболочка)")
    ap.add_argument("--meta", default="data/processed/metadata.csv")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/cache/sector.csv")
    args = ap.parse_args()
    meta = pd.read_csv(args.meta)
    t = build_table(meta, Path(args.raw))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out, index=False)
    print(f"{VERSION}: {len(t)} кадров, sector_ok = {t.sector_ok.mean():.1%} -> {args.out}")
    print(t.reason.value_counts().to_string())


if __name__ == "__main__":
    main()
