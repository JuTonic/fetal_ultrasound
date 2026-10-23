"""Метаданные FETAL_PLANES_DB, 9-классовая метка и поиск дубликатов.

Не зависит от torch. Источник: Burgos-Artizzu et al., Sci. Rep. 10:10200 (2020),
Zenodo 10.5281/zenodo.3904280.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Порядок = убывание размера класса; индекс в списке = целочисленная метка.
CLASSES9 = [
    "Other", "Fetal thorax", "Brain: TT", "Maternal cervix", "Fetal femur",
    "Brain: TC", "Fetal abdomen", "Brain: TV", "Brain: other",
]
_BRAIN = {
    "Trans-thalamic": "Brain: TT",
    "Trans-cerebellum": "Brain: TC",
    "Trans-ventricular": "Brain: TV",
    "Other": "Brain: other",
}


def load_metadata(raw_dir: str | Path) -> pd.DataFrame:
    """Читает FETAL_PLANES_DB_data.csv и строит 9-классовую метку.

    9 классов = 5 небрейновых плоскостей (включая Other) + Fetal brain,
    разбитый по Brain_plane на TT / TC / TV / other.
    """
    raw_dir = Path(raw_dir)
    df = pd.read_csv(raw_dir / "FETAL_PLANES_DB_data.csv", sep=";")
    df.columns = df.columns.str.strip()
    # is_string_dtype, а не dtype == object: в pandas 3 строки имеют dtype str, и проверка
    # на object молча пропускала бы обрезку пробелов
    df = df.apply(lambda s: s.str.strip() if pd.api.types.is_string_dtype(s) else s)
    is_brain = df["Plane"] == "Fetal brain"
    cls9 = np.where(is_brain, df["Brain_plane"].map(_BRAIN), df["Plane"])
    out = pd.DataFrame({
        "image": df["Image_name"],
        "patient": df["Patient_num"].astype(int),
        "plane": df["Plane"],
        "brain_plane": df["Brain_plane"],
        "operator": df["Operator"],
        "machine": df["US_Machine"],
        "cls9": cls9,
        "split_official_raw": df["Train"].map({1: "train", 0: "test"}),
    })
    unknown = set(out["cls9"].dropna()) - set(CLASSES9)
    if out["cls9"].isna().any() or unknown:
        unknown_brain = sorted(df.loc[is_brain & df["Brain_plane"].map(_BRAIN).isna(), "Brain_plane"].astype(str).unique())
        raise ValueError(f"Unknown plane labels: {unknown}; unknown brain planes: {unknown_brain}")
    out["label"] = out["cls9"].map({c: i for i, c in enumerate(CLASSES9)})
    out["path"] = [str(raw_dir / "Images" / f"{n}.png") for n in out["image"]]
    return out


def _image_features(path: str):
    im = Image.open(path)
    mode, (w, h) = im.mode, im.size
    g = im.convert("L")
    md5 = hashlib.md5(g.tobytes()).hexdigest()
    s = np.asarray(g.resize((9, 8), Image.BILINEAR), dtype=np.int16)
    dhash = (s[:, 1:] > s[:, :-1]).ravel()                      # 64 бита
    thumb = np.asarray(g.resize((32, 32), Image.BILINEAR), dtype=np.uint8).ravel()
    return mode, w, h, md5, dhash, thumb


def add_image_features(meta: pd.DataFrame, workers: int = 8):
    """Добавляет mode/width/height/md5_gray. Возвращает (meta, dhash[N,64], thumb[N,1024])."""
    with ThreadPoolExecutor(workers) as ex:
        feats = list(ex.map(_image_features, meta["path"]))
    meta = meta.copy()
    meta["mode"] = [f[0] for f in feats]
    meta["width"] = [f[1] for f in feats]
    meta["height"] = [f[2] for f in feats]
    meta["md5_gray"] = [f[3] for f in feats]
    dhash = np.stack([f[4] for f in feats]).astype(np.int8)
    thumb = np.stack([f[5] for f in feats])
    return meta, dhash, thumb


def near_duplicate_pairs(dhash: np.ndarray, thumb: np.ndarray,
                         dh_max: int = 4, mad_max: float = 4.0, chunk: int = 500) -> pd.DataFrame:
    """Пары (i<j): расстояние Хэмминга dHash <= dh_max и средняя |Δ| миниатюр 32x32 <= mad_max.

    Включает и побитово одинаковые картинки. Порог мягкий: на Voluson S10 даёт
    ложные срабатывания между разными пациентами, поэтому для связывания
    пациентов используется правило min_pairs (см. link_duplicate_patients).
    """
    n = len(dhash)
    thumb = thumb.astype(np.int16)
    out = []
    for i in range(0, n, chunk):
        hi = min(i + chunk, n)
        d = (dhash[i:hi, None, :] != dhash[None, :, :]).sum(-1)
        r, c = np.nonzero((d <= dh_max) & (np.arange(n)[None, :] > np.arange(i, hi)[:, None]))
        r = r + i
        mad = np.abs(thumb[r] - thumb[c]).mean(1)
        k = mad <= mad_max
        out.append(pd.DataFrame({"i": r[k], "j": c[k], "mad": mad[k]}))
    return pd.concat(out, ignore_index=True)


def link_duplicate_patients(meta: pd.DataFrame, pairs: pd.DataFrame, min_pairs: int = 3):
    """Объединяет пациентов, у которых >= min_pairs почти-дубликатов друг с другом.

    Возвращает (meta с колонкой patient_group, таблица связанных пар пациентов).
    patient_group = минимальный номер пациента в компоненте связности.
    Единичные совпадения (1-2 пары) игнорируются: на практике это разные снимки
    с похожей геометрией экрана.
    """
    pat = meta["patient"].to_numpy()
    p = pd.DataFrame({"a": np.minimum(pat[pairs.i], pat[pairs.j]),
                      "b": np.maximum(pat[pairs.i], pat[pairs.j])})
    p = p[p.a != p.b]
    linked = p.groupby(["a", "b"]).size().rename("n_pairs").reset_index()
    linked = linked[linked.n_pairs >= min_pairs].reset_index(drop=True)

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(linked.a, linked.b):
        ra, rb = find(int(a)), find(int(b))
        parent[max(ra, rb)] = min(ra, rb)
    meta = meta.copy()
    meta["patient_group"] = [find(int(x)) if int(x) in parent else int(x) for x in meta["patient"]]
    return meta, linked


def add_exact_duplicate_flags(meta: pd.DataFrame) -> pd.DataFrame:
    """dup_group: id группы побитово одинаковых (в градациях серого) картинок, -1 если уникальна.
    label_conflict: в группе встречаются разные 9-классовые метки (шум разметки)."""
    meta = meta.copy()
    size = meta.groupby("md5_gray")["image"].transform("size")
    ncls = meta.groupby("md5_gray")["cls9"].transform("nunique")
    codes = pd.factorize(meta["md5_gray"])[0]
    meta["dup_group"] = np.where(size > 1, codes, -1)
    meta["label_conflict"] = (size > 1) & (ncls > 1)
    return meta
