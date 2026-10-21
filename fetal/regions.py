"""Варианты области кадра для абляции препроцессинга (a). Не зависит от torch.

  full     A0  полный кадр (базовый вариант)
  crop     A1  кроп бокса сектора (fetal.sector, консервативный: сектор + запас, потери содержимого ~1 %)
  ui       A2  интерфейс (текст, шкалы) закрашен чёрным по маске формата (fetal.overlay), без кропа
  crop_ui  A3  ui + кроп + всё вне оболочки сектора закрашено

Кадр без записи о секторе / формат без маски проходит без изменений (как A0): ничего не отбрасывается.
Артефакты (data/cache/sector.csv, data/cache/overlay_masks.npz) пересоздаются командами
python -m fetal.sector и python -m fetal.overlay.
"""
from __future__ import annotations

from PIL import Image

from fetal.overlay import apply_overlay, fmt_key
from fetal.sector import apply_region

REGION_MODES = ("full", "crop", "ui", "crop_ui")


class RegionPrep:
    def __init__(self, mode: str = "full", sectors: dict | None = None, overlays: dict | None = None):
        if mode not in REGION_MODES:
            raise ValueError(f"region {mode!r} не из {REGION_MODES}")
        if mode in ("crop", "crop_ui") and sectors is None:
            raise ValueError(f"region={mode!r} требует sectors (fetal.sector.load_sectors)")
        if mode in ("ui", "crop_ui") and overlays is None:
            raise ValueError(f"region={mode!r} требует overlays (fetal.overlay.load_masks)")
        self.mode, self.sectors, self.overlays = mode, sectors, overlays

    def __call__(self, img: Image.Image, row) -> Image.Image:
        """img — исходный кадр; row — строка metadata (image, machine, width, height)."""
        img = img.convert("L")
        if self.mode == "full":
            return img
        if self.mode in ("ui", "crop_ui"):
            img = apply_overlay(img, self.overlays.get(fmt_key(row.machine, row.width, row.height)))
        if self.mode in ("crop", "crop_ui"):
            img = apply_region(img, self.sectors.get(row.image), "crop" if self.mode == "crop" else "crop_mask")
        return img
