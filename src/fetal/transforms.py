"""Фиксированный препроцессинг для всех экспериментов (без torch, только PIL/numpy).

Детерминированная часть (train и eval одинаково):
  1. любой режим (L / RGBA) -> градации серого 'L';
  2. letterbox: длинная сторона -> size, пропорции сохраняются, добивка чёрным до квадрата;
  3. (после аугментаций) разворот в 3 канала и нормализация ImageNet mean/std.
Аугментации (только train, умеренные): аффин (поворот ±10°, масштаб 0.9–1.1, сдвиг ±5 %),
яркость/контраст ±20 %; отражение по горизонтали выключено по умолчанию (допущение, см. docs).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox_square(img: Image.Image, size: int, fill: int = 0) -> Image.Image:
    """Масштабирует с сохранением пропорций (бикубика) и центрирует на квадратном холсте."""
    w, h = img.size
    k = size / max(w, h)
    nw, nh = max(1, round(w * k)), max(1, round(h * k))
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("L", (size, size), fill)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


@dataclass(frozen=True)
class AugConfig:
    degrees: float = 10.0
    scale: tuple[float, float] = (0.9, 1.1)
    translate: float = 0.05          # доля от размера
    brightness: float = 0.2
    contrast: float = 0.2
    hflip: bool = False


def _random_affine(img: Image.Image, cfg: AugConfig, rng: random.Random) -> Image.Image:
    size = img.size[0]
    ang = math.radians(rng.uniform(-cfg.degrees, cfg.degrees))
    sc = rng.uniform(*cfg.scale)
    tx, ty = (rng.uniform(-cfg.translate, cfg.translate) * size for _ in range(2))
    c = size / 2
    cos, sin = math.cos(ang) / sc, math.sin(ang) / sc
    # PIL ожидает обратное отображение (выход -> вход) относительно центра
    a, b, d, e = cos, sin, -sin, cos
    return img.transform(img.size, Image.AFFINE,
                         (a, b, c - a * c - b * c - tx * a - ty * b,
                          d, e, c - d * c - e * c - tx * d - ty * e),
                         resample=Image.BILINEAR, fillcolor=0)


def preprocess(img: Image.Image, size: int = 224, train: bool = False,
               cfg: AugConfig | None = None, rng: random.Random | None = None) -> np.ndarray:
    """PIL-изображение -> float32 [3, size, size], нормализованный ImageNet mean/std."""
    img = letterbox_square(img.convert("L"), size)
    if train:
        if rng is None:      # глобальный random не сидируется torch.manual_seed — воспроизводимость только явная
            raise ValueError("train=True требует rng=random.Random(seed)")
        cfg = cfg or AugConfig()
        img = _random_affine(img, cfg, rng)
        img = ImageEnhance.Brightness(img).enhance(1 + rng.uniform(-cfg.brightness, cfg.brightness))
        img = ImageEnhance.Contrast(img).enhance(1 + rng.uniform(-cfg.contrast, cfg.contrast))
        if cfg.hflip and rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.repeat(x[None], 3, axis=0)                                  # серый -> 3 канала
    return (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
