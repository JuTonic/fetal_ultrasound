"""Plot Grad-CAM for three held-out DenseNet-121 brain examples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from fetal.models import TRANSFORM, build_model
from fetal.transforms import letterbox_square, preprocess


CASES = [
    ("Brain: TT", "Brain: TT"),
    ("Brain: TV", "Brain: TT"),
    ("Brain: other", "Brain: TT"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/figures/gradcam_densenet_brain.png"))
    args = parser.parse_args()

    report = json.loads(Path("results/final.json").read_text())
    run = next(item for item in report["runs"] if item["id"] == "densenet121-split_grouped-imagenet-seed0")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    contract = checkpoint["contract"]
    if contract["transform"] != TRANSFORM or contract["arch"] != "densenet121":
        raise ValueError("Checkpoint does not match the trained DenseNet-121 pipeline")
    model = build_model("densenet121")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    examples = []
    for actual, predicted in CASES:
        matches = [
            row for row in run["predictions"]
            if row["partition"] == "test"
            and report["classes"][row["label"]] == actual
            and report["classes"][row["prediction"]] == predicted
        ]
        # A fixed mid-confidence example makes the display reproducible.
        examples.append(min(matches, key=lambda row: abs(max(row["probabilities"]) - 0.85)))

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for column, row in enumerate(examples):
        path = args.raw_dir / "Images" / f"{row['image']}.png"
        original = Image.open(path)
        gray = np.asarray(letterbox_square(original.convert("L"), 224), dtype=np.float32) / 255
        x = torch.from_numpy(preprocess(original)).unsqueeze(0)
        activation: list[torch.Tensor] = []

        def record(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            output.retain_grad()
            activation.append(output)

        hook = model.features.norm5.register_forward_hook(record)
        logits = model(x)
        predicted = int(logits.argmax(1).item())
        if predicted != row["prediction"]:
            raise ValueError(f"Stored prediction differs for {row['image']}")
        model.zero_grad(set_to_none=True)
        logits[0, predicted].backward()
        feature = activation[0]
        weights = feature.grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feature).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().numpy()
        cam = cam / max(float(cam.max()), 1e-8)
        hook.remove()

        actual_name = report["classes"][row["label"]]
        predicted_name = report["classes"][predicted]
        confidence = max(row["probabilities"])
        axes[0, column].imshow(gray, cmap="gray", vmin=0, vmax=1)
        axes[1, column].imshow(gray, cmap="gray", vmin=0, vmax=1)
        axes[1, column].imshow(cam, cmap="inferno", vmin=0, vmax=1, alpha=0.55)
        axes[0, column].set_title(f"True: {actual_name}\nPredicted: {predicted_name} ({confidence:.2f})", fontsize=11)
        for ax in axes[:, column]:
            ax.axis("off")
    axes[0, 0].set_ylabel("Input", fontsize=11)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=11)
    fig.suptitle("DenseNet-121 Grad-CAM · grouped test images", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.025, "Warm colors show regions contributing to the predicted-class score; this is not a causal test.",
             ha="center", fontsize=10, color="#444444")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, facecolor="white")
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": run["checkpoint_sha256"],
        "target_layer": "features.norm5",
        "target": "predicted-class logit",
        "examples": [{key: row[key] for key in ("image", "label", "prediction", "partition")} for row in examples],
    }, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
