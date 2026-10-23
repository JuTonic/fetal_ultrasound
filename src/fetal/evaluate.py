"""Checkpoint predictions, quality metrics and separately reported model cost."""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from fvcore.nn import FlopCountAnalysis

from fetal import metrics
from fetal.config import TrainingConfig
from fetal.data import CLASSES9
from fetal.datasets import FetalPlanesDataset, make_loader
from fetal.experiment import checkpoint_contract
from fetal.files import sha256
from fetal.models import WEIGHTS, load_checkpoint
from fetal.runtime import pick_device, predict


def quality(frame: pd.DataFrame, predicted: np.ndarray) -> dict:
    labels = frame.label.to_numpy()
    result = metrics.evaluate(labels, predicted, frame.machine.to_numpy())
    matrix = metrics.confusion(labels, predicted)
    result["confusion_counts"] = matrix.tolist()
    result["classes_present"] = [CLASSES9[i] for i in np.flatnonzero(matrix.sum(1))]
    result["macro_convention"] = "Average over classes with true support; all nine required for full partitions"
    result["patient_groups"] = int(frame.patient_group.nunique())
    per_class = {}
    for i, name in enumerate(CLASSES9):
        support, predictions = int(matrix[i].sum()), int(matrix[:, i].sum())
        per_class[name] = {
            "precision": float(matrix[i, i] / predictions) if predictions else (0.0 if support else None),
            "recall": float(matrix[i, i] / support) if support else None,
            "f1": result["f1_per_class"][name], "support": support,
        }
    result["per_class"] = per_class
    for machine, values in result["per_machine"].items():
        subset = frame[frame.machine == machine]
        values["patient_groups"] = int(subset.patient_group.nunique())
        values["classes_present"] = sorted(subset.cls9.unique().tolist())
    return result


@torch.no_grad()
def profile(model: torch.nn.Module, device: torch.device) -> dict:
    model.eval()
    x = torch.zeros(1, 3, 224, 224, device=device)
    for _ in range(20):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    times = []
    for _ in range(100):
        start = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - start) * 1000)
    analysis = FlopCountAnalysis(model, x)
    count = int(analysis.total())
    return {
        "status": "completed", "parameters": sum(p.numel() for p in model.parameters()),
        "operations": count, "operation_convention": "fvcore FLOPs; fused multiply-add counted as one operation",
        "unsupported_operations": dict(analysis.unsupported_ops()),
        "input_shape": [1, 3, 224, 224], "precision": "fp32", "device": str(device),
        "latency_ms": {"mean": float(np.mean(times)), "median": float(np.median(times)),
                       "p95": float(np.percentile(times, 95))},
        "timing": "20 warmups; 100 batch-1 forwards on resident tensor; excludes loading/preprocessing",
    }


def evaluate_checkpoint(job: dict, dataset: dict, checkpoint: Path, out: Path, device: str) -> dict:
    cfg = TrainingConfig.model_validate(job["config"])
    dev = pick_device(device)
    model, contract = load_checkpoint(checkpoint, checkpoint_contract(job, dataset))
    meta = pd.read_csv("data/processed/metadata.csv")
    expected_groups = set(meta.loc[meta[job["protocol"]] == "train", "patient_group"].astype(int))
    actual_groups = set(contract["training_groups"])
    if not actual_groups or not actual_groups <= expected_groups:
        raise ValueError("Checkpoint training groups are missing or overlap held-out patients")
    if not job["smoke"] and actual_groups != expected_groups:
        raise ValueError("Checkpoint does not describe the full configured training partition")
    if contract["seed"] != job["seed"] or contract["config"] != cfg.model_dump(mode="json"):
        raise ValueError("Checkpoint seed/configuration differs from requested experiment")
    initialization = contract["initialization"]
    expected_initialization = str(WEIGHTS[job["arch"]]) if job["pretrained"] else "scratch"
    if initialization != expected_initialization:
        raise ValueError("Checkpoint initialization differs from requested experiment")
    if job["smoke"]:
        meta = meta.groupby(job["protocol"], group_keys=False).head(32)
    model.to(dev)
    checkpoint_hash = sha256(checkpoint)
    records = []
    scores = {}
    for part in ("val", "test"):
        ds = FetalPlanesDataset(meta, "data/raw", job["protocol"], part)
        loader = make_loader(ds, cfg.batch_size, False, cfg.num_workers)
        logits = predict(model, loader, dev)
        if logits.shape != (len(ds), 9) or not np.isfinite(logits).all():
            raise ValueError("Expected finite nine-class logits")
        if not job["smoke"] and set(ds.df.label) != set(range(9)):
            raise ValueError(f"Missing class support in full {part} partition")
        predicted = logits.argmax(1)
        scores[part] = quality(ds.df, predicted)
        clean = ~ds.df.label_conflict.to_numpy(dtype=bool)
        unique = ~ds.df.md5_gray.duplicated().to_numpy()
        scores[part]["sensitivity"] = {}
        for name, mask in (("exclude_label_conflicts", clean), ("one_per_exact_hash", unique)):
            if mask.any():
                scores[part]["sensitivity"][name] = quality(ds.df[mask], predicted[mask])
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        for index, row in enumerate(ds.df.to_dict("records")):
            records.append({
                "image": row["image"], "label": row["label"], "prediction": int(predicted[index]),
                "probabilities": probabilities[index].tolist(), "patient_group": row["patient_group"],
                "machine": row["machine"], "operator": row["operator"], "partition": part,
                "protocol": job["protocol"], "seed": job["seed"], "checkpoint_sha256": checkpoint_hash,
            })
        matrix = np.asarray(scores[part]["confusion_counts"])
        normalized = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(10, 8))
        plot = ax.imshow(normalized, vmin=0, vmax=1)
        ax.set(xticks=range(9), yticks=range(9), xticklabels=CLASSES9, yticklabels=CLASSES9,
               xlabel="Predicted class", ylabel="True class", title=f"{job['id']} / {part}")
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
        fig.colorbar(plot, ax=ax)
        fig.tight_layout()
        fig.savefig(out / f"confusion_{part}.png")
        plt.close(fig)
    try:
        cost = profile(model, dev)
    except Exception as error:
        cost = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                "parameters": sum(p.numel() for p in model.parameters())}
    return {"status": "completed", "contract": contract, "checkpoint_sha256": checkpoint_hash,
            "predictions": records, "quality": scores, "cost": cost}
