"""Download the original release and verify the frozen dataset contract."""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from fetal import data, splits, subsets
from fetal.files import sha256
from fetal.transfer import download_file

DATA_URL = "https://zenodo.org/records/3904280/files/FETAL_PLANES_ZENODO.zip?download=1"
DATA_MD5 = "2a5fcc2cefb789bcc0f6c1f73e0ea43f"


def download_data() -> None:
    raw = Path("data/raw")
    raw.mkdir(parents=True, exist_ok=True)
    marker = raw / ".complete.json"
    if not marker.exists():
        archive = Path("/downloads/FETAL_PLANES_ZENODO.zip")
        download_file(DATA_URL, archive, checksum=DATA_MD5, algorithm="md5")
        # Locate the metadata and PNGs by name; release folder prefixes may vary.
        with zipfile.ZipFile(archive) as source:
            metadata = [n for n in source.namelist() if Path(n).name == "FETAL_PLANES_DB_data.csv"]
            if len(metadata) != 1:
                raise ValueError("Expected one release metadata CSV")
            with source.open(metadata[0]) as src, (raw / "FETAL_PLANES_DB_data.csv").open("wb") as dst:
                shutil.copyfileobj(src, dst)
            images = raw / "Images"
            images.mkdir(exist_ok=True)
            names = set()
            for name in source.namelist():
                if not name.lower().endswith(".png"):
                    continue
                filename = Path(name).name
                if filename in names:
                    raise ValueError(f"Ambiguous image filename {filename}")
                names.add(filename)
                with source.open(name) as src, (images / filename).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        validate_data()
        marker.write_text(json.dumps({"url": DATA_URL, "md5": DATA_MD5}))
    validate_data()


def validate_data() -> dict:
    path = Path("data/processed/metadata.csv")
    if not path.exists():
        # Rebuilding is deliberately a separate command, never an implicit split change.
        raise FileNotFoundError("Frozen metadata missing; rebuild with python -m fetal.prepare and review hashes")
    meta = pd.read_csv(path)
    splits.validate_splits(meta)
    subsets.validate_subsets(meta)
    for name, actual in (("splits", splits.splits_checksum(meta)), ("subsets", subsets.subsets_checksum(meta))):
        if actual != path.with_name(f"{name}.sha256").read_text().strip():
            raise ValueError(f"Frozen {name} checksum mismatch")
    source = data.load_metadata("data/raw").set_index("image")
    frozen = meta.set_index("image")
    if set(source.index) != set(frozen.index):
        raise ValueError("Release image IDs differ from frozen metadata")
    source = source.loc[frozen.index]
    for column in ("patient", "plane", "brain_plane", "operator", "machine", "label"):
        if not source[column].equals(frozen[column]):
            raise ValueError(f"Release differs from frozen metadata: {column}")
    official_test = source["split_official_raw"].eq("test")
    if not official_test.equals(frozen["split_official"].eq("test")):
        raise ValueError("Official test annotations changed")
    for image in meta["image"]:
        image_path = Path("data/raw/Images") / f"{image}.png"
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing image: {image_path}")
    return {
        "source": DATA_URL, "archive_md5": DATA_MD5, "metadata_sha256": sha256(path),
        "splits_sha256": splits.splits_checksum(meta), "subsets_sha256": subsets.subsets_checksum(meta),
        "images": len(meta), "patient_groups": int(meta.patient_group.nunique()),
        "class_counts": meta.cls9.value_counts().to_dict(),
        "machine_counts": meta.machine.value_counts().to_dict(),
    }
