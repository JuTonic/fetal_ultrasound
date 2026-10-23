"""Explicit ImageNet initializations and nine-class classifier heads."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models

from fetal.data import CLASSES9
from fetal.transfer import download_file

WEIGHTS = {
    "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
    "densenet121": models.DenseNet121_Weights.IMAGENET1K_V1,
    "convnext_tiny": models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
    "swin_t": models.Swin_T_Weights.IMAGENET1K_V1,
}
TRANSFORM = "B0-gray-letterbox224-moderate-no-flip-imagenet-v1"


def download_weights(arch: str) -> Path:
    url = WEIGHTS[arch].url
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(url).name
    return download_file(url, path, checksum=path.stem.rsplit("-", 1)[1])


def build_model(arch: str, pretrained: bool = False) -> nn.Module:
    weights = WEIGHTS[arch] if pretrained else None
    if pretrained:
        download_weights(arch)
    model = models.get_model(arch, weights=weights)
    if arch.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES9))
    elif arch == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES9))
    elif arch == "convnext_tiny":
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(CLASSES9))
    elif arch == "swin_t":
        model.head = nn.Linear(model.head.in_features, len(CLASSES9))
    return model


def load_checkpoint(path: Path, expected: dict) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    contract = checkpoint["contract"]
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"Checkpoint {key}: expected {value!r}, got {contract.get(key)!r}")
    for key in ("config", "seed", "selected_epoch", "initialization", "training_groups"):
        if key not in contract:
            raise ValueError(f"Checkpoint contract missing {key}")
    model = build_model(contract["arch"])
    model.load_state_dict(checkpoint["state_dict"])
    return model, contract
