"""Create presentation tables from the final report and training configuration."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NAMES = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_t": "Swin-T",
}
OUTPUT = Path("results/figures")


def save_table(name: str, title: str, subtitle: str, columns: list[str], rows: list[list[str]],
               note: str, widths: list[float]) -> None:
    height = 3.8 + 0.42 * max(0, len(rows) - 4)
    fig, ax = plt.subplots(figsize=(12, height))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    fig.text(0.055, 0.91, title, fontsize=21, fontweight="bold", color="#142c48")
    fig.text(0.055, 0.85, subtitle, fontsize=11, color="#4b5a6a")
    table = ax.table(
        cellText=rows, colLabels=columns, colWidths=widths,
        cellLoc="left", colLoc="left", bbox=[0.04, 0.22, 0.92, 0.58],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(2)
        cell.PAD = 0.13
        if row == 0:
            cell.set_facecolor("#173a60")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#edf3f8" if row % 2 else "#f8fafc")
            if col == 0:
                cell.get_text().set_fontweight("bold")
    fig.text(0.055, 0.12, note, fontsize=9.5, color="#4b5a6a", va="top")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT / f"{name}.png")


def main() -> None:
    report = json.loads(Path("results/final.json").read_text())
    config = json.loads(Path("configs/pipeline.json").read_text())
    training = config["training"]
    by_arch = {
        arch: next(run for run in report["runs"] if run["status"] == "completed"
                   and run["arch"] == arch and run["protocol"] == "split_grouped"
                   and run["pretrained"] and run["seed"] == 0)
        for arch in NAMES
    }

    profile_rows = []
    for arch, label in NAMES.items():
        cost = by_arch[arch]["cost"]
        profile_rows.append([
            label,
            f"{cost['parameters'] / 1e6:.2f} M",
            f"{cost['operations'] / 1e9:.2f} G",
            f"{cost['latency_ms']['median']:.2f} ms",
        ])
    save_table(
        "model_size_and_cost", "Model size and inference cost",
        "Nine-class classifier · 3 × 224 × 224 input · ImageNet seed-0 checkpoint",
        ["Architecture", "Parameters", "FLOPs / image", "Median latency"],
        profile_rows,
        "All timing: RTX 3080 Ti, FP32, batch 1; 20 warm-ups + 100 forwards.\n"
        "FLOPs: fvcore convention (fused multiply-add = one operation). Loading and preprocessing excluded.",
        [0.34, 0.18, 0.22, 0.26],
    )

    coverage_rows = [
        ["ResNet-50", "ImageNet + scratch", "0, 1", "20", "Official split: 6 epochs"],
        ["DenseNet-121", "ImageNet + scratch", "0, 1", "20", "—"],
        ["ConvNeXt-Tiny", "ImageNet", "0", "20", "—"],
        ["Swin-T", "ImageNet", "0", "20", "—"],
    ]
    save_table(
        "model_training_coverage", "Model training coverage",
        "Grouped split · 20 completed epochs per listed configuration",
        ["Architecture", "Initialization", "Seeds", "Epochs", "Additional run"],
        coverage_rows,
        "The official-split ResNet-50 is a separate six-epoch reference run.\n"
        "Best evaluated checkpoint is chosen by validation macro-F1, often before epoch 20.",
        [0.21, 0.28, 0.11, 0.10, 0.30],
    )

    hyper_rows = [
        ["Input", "Grayscale → 3 channels; aspect-preserving 224 × 224 letterbox"],
        ["Loss / optimizer", "Cross-entropy / AdamW"],
        ["Batch / precision", f"{training['batch_size']} / {training['precision'].upper()}"],
        ["Weight decay", f"{training['weight_decay']:.0e}"],
        ["Initial phase", f"{training['epochs']} epochs; learning rate {training['lr']:.0e}"],
        ["Continuation", f"14 more epochs; learning rate {training['lr'] * 0.1:.0e}"],
        ["Schedule", "One-epoch warm-up, then cosine decay in each phase"],
        ["Checkpoint choice", "Highest validation macro-F1; earlier epoch breaks ties"],
    ]
    save_table(
        "training_hyperparameters", "Shared training parameters",
        "Same optimization recipe for the four grouped architectures",
        ["Setting", "Value"], hyper_rows,
        "The continuation restores saved model and AdamW optimizer state after epoch 6.",
        [0.28, 0.72],
    )


if __name__ == "__main__":
    main()
