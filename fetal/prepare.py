"""Строит data/processed/metadata.csv: метки, признаки изображений, дубликаты, три разбиения.

Запуск из корня репозитория:  python -m fetal.prepare
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fetal import data, splits


def changed_columns(old: pd.DataFrame, new: pd.DataFrame) -> dict[str, int]:
    """Что пересборка new меняет в уже записанной таблице old: {колонка: число строк}.
    Строки сопоставляются по имени изображения; добавленные/удалённые/повторяющиеся изображения
    попадают в ключ "image", удалённая колонка считается целиком изменённой,
    новые колонки изменением не считаются."""
    diff = {}
    n_set = len(set(old["image"]) ^ set(new["image"])) + int(new["image"].duplicated().sum())
    if n_set:
        diff["image"] = n_set
    o = old.set_index("image").reindex(new["image"])
    for c in o.columns:
        if c not in new:
            diff[c] = len(old)
            continue
        n = int((o[c].astype(str).to_numpy() != new[c].astype(str).to_numpy()).sum())
        if n:
            diff[c] = n
    return diff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="папка с FETAL_PLANES_DB_data.csv и Images/")
    ap.add_argument("--out", default="data/processed/metadata.csv")
    ap.add_argument("--seeds", type=int, default=50, help="сколько seed перебрать при выборе разбиения")
    ap.add_argument("--min-pairs", type=int, default=3, help="порог связывания пациентов-дубликатов")
    ap.add_argument("--force", action="store_true",
                    help="перезаписать --out, даже если меняются уже записанные колонки (в т.ч. разбиения)")
    args = ap.parse_args()

    meta = data.load_metadata(args.raw)
    meta, dhash, thumb = data.add_image_features(meta)
    meta = data.add_exact_duplicate_flags(meta)
    pairs = data.near_duplicate_pairs(dhash, thumb)
    meta, linked = data.link_duplicate_patients(meta, pairs, args.min_pairs)
    print(f"{len(meta)} изображений, {meta.patient.nunique()} пациентов, "
          f"{meta.patient_group.nunique()} групп пациентов после объединения дубликатов")
    print("Связанные пациенты (дубликаты под разными ID):")
    print(linked.to_string(index=False))

    meta["split_grouped"], seed, score = splits.best_grouped_split(meta, range(args.seeds))
    meta["split_official"] = splits.official_split_with_val(meta)
    meta["split_naive"] = splits.naive_image_split(meta)
    print(f"\nsplit_grouped: seed={seed}, imbalance_score={score:.4f}")

    for col in ("split_grouped", "split_official", "split_naive"):
        print(f"\n=== {col} === утечки:", splits.check_no_leakage(meta, col))
        sizes, cls, mach = splits.split_summary(meta, col)
        print(sizes.to_string()); print(cls.to_string()); print(mach.to_string())

    splits.validate_splits(meta)          # полнота частей и нулевая утечка в grouped и official
    out = Path(args.out)
    table = meta.drop(columns=["path", "split_official_raw"])
    if out.exists() and not args.force:
        diff = changed_columns(pd.read_csv(out), table)
        if diff:
            raise SystemExit(
                f"\n{out} НЕ перезаписан: пересборка меняет колонки {diff} (число строк).\n"
                "На эти колонки (особенно разбиения) опираются эксперименты всей команды. Частая причина — "
                "другая версия scikit-learn: StratifiedGroupKFold(shuffle=True) изменился в 1.8.\n"
                "Запишите в другой файл (--out) или перезапишите осознанно (--force).")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    checksum = out.with_name("splits.sha256")
    checksum.write_text(splits.splits_checksum(table) + "\n")
    print("\nзаписано:", out, "| контрольная сумма разбиений:", checksum)


if __name__ == "__main__":
    main()
