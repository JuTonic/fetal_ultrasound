"""PyTorch Dataset поверх data/processed/metadata.csv."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from fetal.transforms import AugConfig, preprocess


class FetalPlanesDataset(Dataset):
    """Строки metadata.csv, у которых `split_col == part`.

    Пример: FetalPlanesDataset(meta, "data/raw", "split_grouped", "train", train=True)
    Аугментации включаются флагом train (обычно только для part='train').

    Воспроизводимость: случайность аугментаций берётся из генератора torch, поэтому
    torch.manual_seed(seed) фиксирует её и при num_workers=0, и в воркерах DataLoader
    (там torch сидирует генератор как base_seed + worker_id). Совпадение гарантируется
    при тех же seed, batch_size и num_workers — как у аугментаций torchvision.
    """

    def __init__(self, meta: pd.DataFrame, raw_dir: str | Path, split_col: str, part: str,
                 train: bool = False, size: int = 224, aug: AugConfig | None = None):
        self.df = meta[meta[split_col] == part].reset_index(drop=True)
        self.raw_dir, self.train, self.size, self.aug = Path(raw_dir), train, size, aug

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(self.raw_dir / "Images" / f"{row.image}.png")
        # не глобальный random: torch.manual_seed его не сидирует, и при num_workers=0
        # аугментации не воспроизводились бы
        rng = random.Random(int(torch.randint(2**62, ()))) if self.train else None
        x = preprocess(img, self.size, self.train, self.aug, rng)
        return torch.from_numpy(x), int(row.label)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0, **kw) -> DataLoader:
    """DataLoader с настройками, проверенными на macOS (torch 2.14, Python 3.12).

    С воркерами: persistent_workers (не пересоздавать каждую эпоху) и на macOS — forkserver:
    при дефолтном spawn завершение воркеров занимает ~10–16 с на каждый проход по лоадеру.
    Батчи при том же seed совпадают со spawn. Seed задаётся снаружи: torch.manual_seed(seed).
    """
    if num_workers > 0:
        kw.setdefault("persistent_workers", True)
        if sys.platform == "darwin":
            kw.setdefault("multiprocessing_context", "forkserver")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kw)
