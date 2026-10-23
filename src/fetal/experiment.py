"""New benchmark training using the existing datasets, metrics and FP32 recipe."""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from fetal import metrics
from fetal.config import TrainingConfig
from fetal.data import CLASSES9
from fetal.datasets import FetalPlanesDataset, make_loader
from fetal.files import sha256
from fetal.models import TRANSFORM, WEIGHTS, build_model, load_checkpoint
from fetal.runtime import pick_device, predict, set_seed


def checkpoint_contract(job: dict, dataset: dict) -> dict:
    return {
        "arch": job["arch"], "classes": CLASSES9, "transform": TRANSFORM,
        "protocol": job["protocol"], "splits_sha256": dataset["splits_sha256"],
        "subsets_sha256": dataset["subsets_sha256"],
        "metadata_sha256": dataset["metadata_sha256"], "smoke": job["smoke"],
    }


def save_training_state(path: Path, state: dict) -> None:
    """A stopped write leaves the previous epoch intact; temporary files are ignored."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def train_checkpoint(job: dict, dataset: dict, out: Path, device: str) -> Path:
    cfg = TrainingConfig.model_validate(job["config"])
    target_epochs = job.get("target_epochs", cfg.epochs)
    if target_epochs < cfg.epochs:
        raise ValueError("Continuation target must be at least the configured epoch count")
    set_seed(job["seed"])
    dev = pick_device(device)
    meta = pd.read_csv("data/processed/metadata.csv")
    if job["smoke"]:
        meta = meta.groupby(job["protocol"], group_keys=False).head(32)
    train = FetalPlanesDataset(meta, "data/raw", job["protocol"], "train", train=True)
    val = FetalPlanesDataset(meta, "data/raw", job["protocol"], "val")
    # Restart workers every epoch so augmentation/shuffle seeds also survive a restart.
    loader_options = {"persistent_workers": False} if cfg.num_workers else {}
    train_loader = make_loader(train, cfg.batch_size, True, cfg.num_workers, **loader_options)
    val_loader = make_loader(val, cfg.batch_size, False, cfg.num_workers, **loader_options)
    recovery_path = out / "resume.pt"
    expected = {
        **checkpoint_contract(job, dataset), "config": cfg.model_dump(mode="json"), "seed": job["seed"],
        "pretrained": job["pretrained"], "device": device,
        "seed_policy": "job seed + epoch; fresh loader workers every epoch",
    }
    recovery = None
    if recovery_path.exists():
        recovery = torch.load(recovery_path, map_location="cpu", weights_only=True)
        if recovery["identity"] != expected:
            raise ValueError("Recovery checkpoint configuration/protocol differs from this job")
    initialization_source = None
    model = build_model(job["arch"], job["pretrained"] and recovery is None).to(dev)
    if job["pretrained"] and recovery is None:
        url = WEIGHTS[job["arch"]].url
        weight_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(url).name
        initialization_source = {"url": url, "sha256": sha256(weight_path)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = len(train_loader)
    total = cfg.epochs * warm
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warm if step < warm else
        0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1))),
    )
    best = -1.0
    best_checkpoint = None
    rows = []
    completed_epoch = 0
    elapsed_seconds = 0.0
    hardware = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    hardware_history = [{"hardware": hardware, "from_epoch": 1}]
    if recovery is not None:
        model.load_state_dict(recovery["model"])
        optimizer.load_state_dict(recovery["optimizer"])
        scheduler.load_state_dict(recovery["scheduler"])
        best = recovery["best_score"]
        best_checkpoint = recovery["best_checkpoint"]
        rows = recovery["learning_curve"]
        completed_epoch = recovery["completed_epoch"]
        elapsed_seconds = recovery["elapsed_seconds"]
        initialization_source = recovery["initialization_source"]
        hardware_history = recovery["hardware_history"]
        previous_target = recovery.get("target_epochs", cfg.epochs)
        if target_epochs > previous_target:
            # The original cosine schedule ends at zero. Continue the same optimizer
            # state with a fresh, lower-rate cosine phase for the added epochs.
            phase_epochs = target_epochs - completed_epoch
            phase_total = phase_epochs * warm
            phase_lr = cfg.lr * 0.1
            for group in optimizer.param_groups:
                group["lr"] = phase_lr
                group["initial_lr"] = phase_lr
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: (step + 1) / warm if step < warm else
                0.5 * (1 + math.cos(math.pi * (step - warm) / max(phase_total - warm, 1))),
            )
        if hardware_history[-1]["hardware"] != hardware:
            hardware_history.append({"hardware": hardware, "from_epoch": completed_epoch + 1})
        print(f"[{job['id']}] resume after epoch {completed_epoch}/{target_epochs}", flush=True)
        del recovery
    started = time.perf_counter()
    checkpoint = out / "model.pt"
    for epoch in range(completed_epoch + 1, target_epochs + 1):
        set_seed(job["seed"] + epoch)
        model.train()
        loss_sum = 0.0
        epoch_start = time.perf_counter()
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x.to(dev)), y.to(dev))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(y)
        logits = predict(model, val_loader, dev)
        if not np.isfinite(logits).all():
            raise ValueError("Non-finite validation logits")
        score = metrics.evaluate(val.df.label.to_numpy(), logits.argmax(1))["macro_f1"]
        rows.append({"epoch": epoch, "loss": loss_sum / len(train), "val_macro_f1": score,
                     "epoch_seconds": time.perf_counter() - epoch_start})
        print(job["id"], rows[-1], flush=True)
        if score > best:
            best = score
            contract = {
                **checkpoint_contract(job, dataset), "config": cfg.model_dump(mode="json"), "seed": job["seed"],
                "selected_epoch": epoch, "validation_macro_f1": score,
                "total_epochs": target_epochs,
                "selection_policy": "best validation macro-F1; earlier epoch breaks ties",
                "initialization": str(WEIGHTS[job["arch"]]) if job["pretrained"] else "scratch",
                "initialization_source": initialization_source,
                "training_groups": sorted(int(g) for g in train.df.patient_group.unique()),
            }
            best_checkpoint = {
                "contract": contract,
                "state_dict": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
            }
        # Latest model/optimizer and best model are one atomic snapshot, never mismatched files.
        save_training_state(recovery_path, {
            "identity": expected, "completed_epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "target_epochs": target_epochs,
            "best_score": best, "best_checkpoint": best_checkpoint, "learning_curve": rows,
            "initialization_source": initialization_source, "hardware_history": hardware_history,
            "elapsed_seconds": elapsed_seconds + time.perf_counter() - started,
        })
        pd.DataFrame(rows).to_csv(out / "learning_curve.csv", index=False)
    # Recreate exports from the authoritative snapshot, including a stop after the last epoch.
    if best_checkpoint is None:
        raise ValueError("No completed training epoch available")
    best_checkpoint["contract"]["total_epochs"] = target_epochs
    if target_epochs > cfg.epochs:
        best_checkpoint["contract"]["continued_from_epoch"] = cfg.epochs
    best_checkpoint["contract"]["training_hardware"] = hardware_history
    save_training_state(checkpoint, best_checkpoint)
    pd.DataFrame(rows).to_csv(out / "learning_curve.csv", index=False)
    load_checkpoint(checkpoint, checkpoint_contract(job, dataset))
    (out / "training_seconds.txt").write_text(str(elapsed_seconds + time.perf_counter() - started))
    return checkpoint
