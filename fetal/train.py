"""Обучение одной конфигурации абляции препроцессинга (ResNet-18, ImageNet-веса, дообучение).

Запуск из корня репозитория (конфиги — в файле, результаты — в results/ablation/<вариант>/seed<k>/):
    python -m fetal.train --config configs/ablation.json --variant B0 --seed 0
    python -m fetal.train --config configs/ablation.json --all --seed 0 1 2

Принципы: фиксированный seed; выбор — только по val (части test в этом файле нет вообще);
последняя эпоха без выбора чекпоинта по val; прогон с готовым metrics.json пропускается (--force — заново).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn

from fetal import metrics
from fetal.data import CLASSES9
from fetal.controls import pure_format_keys
from fetal.datasets import FetalPlanesDataset, make_loader
from fetal.overlay import fmt_key
from fetal.regions import RegionPrep
from fetal.transforms import AugConfig

AUG_PRESETS = {
    "moderate": AugConfig(),
    "strong": AugConfig(degrees=20.0, scale=(0.8, 1.2), translate=0.10, brightness=0.4, contrast=0.4),
}
PARTS = ("train", "val")        # test сознательно недоступен: модель и вариант выбираются только по val


def load_config(path: str | Path, variant: str, overrides: dict | None = None) -> dict:
    spec = json.loads(Path(path).read_text())
    if variant not in spec["variants"]:
        raise KeyError(f"нет варианта {variant!r}; есть: {list(spec['variants'])}")
    cfg = {**spec["base"], **spec["variants"][variant], **(overrides or {}), "variant": variant}
    if cfg["aug"] == "none" and cfg["hflip"]:
        raise ValueError("hflip работает только вместе с аугментациями (aug != 'none')")
    return cfg


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_region(cfg: dict) -> RegionPrep:
    from fetal.overlay import load_masks
    from fetal.sector import load_sectors
    mode = cfg["region"]
    sectors = load_sectors(cfg["sector_file"]) if mode in ("crop", "crop_ui") else None
    overlays = load_masks(cfg["overlay_file"]) if mode in ("ui", "crop_ui") else None
    return RegionPrep(mode, sectors, overlays)


def build_model(arch: str, pretrained: bool) -> nn.Module:
    if arch == "resnet18":
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif arch == "resnet50":
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"архитектура {arch!r} не поддерживается этим скриптом")
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES9))
    return m


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def subsample(meta: pd.DataFrame, split_col: str, part: str, n: int | None, seed: int) -> pd.DataFrame:
    """Для быстрых проверок: оставить n случайных кадров части part (остальные части не меняем)."""
    if n is None:
        return meta
    keep = meta[meta[split_col] == part].sample(n=min(n, (meta[split_col] == part).sum()), random_state=seed).index
    drop = meta[(meta[split_col] == part) & ~meta.index.isin(keep)].index
    return meta.drop(index=drop)


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.check_output(["git", *a], stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device) -> np.ndarray:
    model.eval()
    return torch.cat([model(x.to(device)).float().cpu() for x, _ in loader]).numpy()


def run(cfg: dict, out_root: str | Path = "results/ablation", force: bool = False, limit_train: int | None = None,
        limit_val: int | None = None, device: str = "auto", log=print) -> dict:
    out = Path(out_root) / cfg["variant"] / f"seed{cfg['seed']}"
    if (out / "metrics.json").exists() and not force:
        log(f"[{cfg['variant']} seed{cfg['seed']}] уже есть, пропускаю ({out})")
        return json.loads((out / "metrics.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seed"])
    dev = pick_device(device)

    meta = pd.read_csv(cfg["meta_file"])
    sc = cfg["split_col"]
    # подвыборка для быстрых проверок не зависит от seed обучения: все прогоны видят одни и те же кадры
    meta = subsample(subsample(meta, sc, "train", limit_train, 0), sc, "val", limit_val, 0)
    region = build_region(cfg)
    train_aug = cfg["aug"] != "none"
    aug = dataclasses.replace(AUG_PRESETS[cfg["aug"]], hflip=cfg["hflip"]) if train_aug else None
    ds_tr = FetalPlanesDataset(meta, cfg["raw_dir"], sc, "train", train=train_aug, size=cfg["size"], aug=aug, region=region)
    ds_va = FetalPlanesDataset(meta, cfg["raw_dir"], sc, "val", train=False, size=cfg["size"], region=region)
    nw = cfg["num_workers"]
    dl_tr = make_loader(ds_tr, cfg["batch_size"], True, nw, drop_last=True)
    dl_va = make_loader(ds_va, 2 * cfg["batch_size"], False, max(nw // 2, 0))

    model = build_model(cfg["arch"], cfg["pretrained"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_ep = len(dl_tr)
    total, warm = cfg["epochs"] * steps_ep, steps_ep                      # прогрев 1 эпоха, затем косинус до 0
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    y_va = ds_va.df["label"].to_numpy()
    rows = []
    t_all = time.perf_counter()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        t0, loss_sum, n = time.perf_counter(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            n += len(y)
        t_train = time.perf_counter() - t0
        logits = predict(model, dl_va, dev)
        ev = metrics.evaluate(y_va, logits.argmax(1))
        rows.append({"epoch": ep, "train_loss": loss_sum / n, "val_macro_f1": ev["macro_f1"],
                     "val_bal_acc": ev["balanced_accuracy"], "train_img_per_s": n / t_train,
                     "epoch_seconds": time.perf_counter() - t0})
        log(f"[{cfg['variant']} seed{cfg['seed']}] epoch {ep}/{cfg['epochs']}  loss={rows[-1]['train_loss']:.3f}  "
            f"val macro-F1={ev['macro_f1']:.3f}  bal.acc={ev['balanced_accuracy']:.3f}  "
            f"({rows[-1]['train_img_per_s']:.0f} img/s, {rows[-1]['epoch_seconds']:.0f} s)")

    pure = pure_format_keys(meta[meta[sc] == "train"])              # форматы, где класс почти определён форматом (по train)
    is_pure = np.array([fmt_key(a, b, c) in pure for a, b, c in zip(ds_va.df.machine, ds_va.df.width, ds_va.df.height)])
    final = metrics.evaluate(y_va, logits.argmax(1), ds_va.df["machine"].to_numpy(),
                             {"class_pure_formats": is_pure, "mixed_formats": ~is_pure})
    final["n_val_groups"] = int(ds_va.df["patient_group"].nunique())
    pd.DataFrame(rows).to_csv(out / "log.csv", index=False)
    np.savez_compressed(out / "val_pred.npz", image=ds_va.df["image"].to_numpy().astype(str), y=y_va, logits=logits,
                        machine=ds_va.df["machine"].to_numpy().astype(str),
                        patient_group=ds_va.df["patient_group"].to_numpy())
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    (out / "env.json").write_text(json.dumps({
        "torch": torch.__version__, "torchvision": torchvision.__version__, "device": str(dev),
        "python": platform.python_version(), "platform": platform.platform(), "git": git_state(),
        "limit_train": limit_train, "limit_val": limit_val, "total_seconds": time.perf_counter() - t_all,
        "note": "MPS не гарантирует побитовой воспроизводимости при том же seed"}, indent=2))
    (out / "metrics.json").write_text(json.dumps(final, indent=2, ensure_ascii=False))
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/ablation.json")
    ap.add_argument("--variant", action="append", help="имя варианта (можно несколько раз)")
    ap.add_argument("--all", action="store_true", help="все варианты из файла")
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out", default="results/ablation")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit-train", type=int)
    ap.add_argument("--limit-val", type=int)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()
    names = list(json.loads(Path(args.config).read_text())["variants"]) if args.all else (args.variant or [])
    if not names:
        ap.error("укажите --variant или --all")
    for seed in args.seed:                  # сначала все варианты на одном seed: при остановке есть полные пары
        for name in names:
            ov = {"seed": seed}
            if args.epochs:
                ov["epochs"] = args.epochs
            if args.workers is not None:
                ov["num_workers"] = args.workers
            run(load_config(args.config, name, ov), args.out, args.force, args.limit_train, args.limit_val, args.device)


if __name__ == "__main__":
    main()
