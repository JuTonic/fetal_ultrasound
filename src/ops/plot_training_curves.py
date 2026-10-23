"""Plot the saved 20-epoch training histories; no model execution is needed."""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ARCHITECTURES = [
    ("resnet50", "ResNet-50"),
    ("densenet121", "DenseNet-121"),
    ("convnext_tiny", "ConvNeXt-Tiny"),
    ("swin_t", "Swin-T"),
]
COLORS = {True: "#1769aa", False: "#d47b16"}


def main() -> None:
    report = json.loads(Path("results/final.json").read_text())
    histories = {}
    for run in report["runs"]:
        if run["status"] != "completed" or run["protocol"] != "split_grouped":
            continue
        continuation_id = run.get("continuation_run_id")
        if continuation_id is None:
            continue
        paths = list(Path("results/distributed").glob(f"*/{continuation_id}/learning_curve.csv"))
        if len(paths) != 1:
            raise ValueError(f"Expected one learning curve for {continuation_id}; found {len(paths)}")
        with paths[0].open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if [int(row["epoch"]) for row in rows] != list(range(1, 21)):
            raise ValueError(f"Incomplete 20-epoch history: {paths[0]}")
        histories[(run["arch"], run["pretrained"], run["seed"])] = rows

    output = Path("results/figures")
    output.mkdir(parents=True, exist_ok=True)
    for field, title, filename in [
        ("loss", "Training loss by epoch", "training_loss_20_epochs"),
        ("val_macro_f1", "Validation macro-F1 by epoch", "validation_macro_f1_20_epochs"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), sharex=True, sharey=True)
        for ax, (arch, name) in zip(axes.flat, ARCHITECTURES):
            for pretrained in (True, False):
                for seed in (0, 1):
                    rows = histories.get((arch, pretrained, seed))
                    if rows is None:
                        continue
                    ax.plot(
                        [int(row["epoch"]) for row in rows],
                        [float(row[field]) for row in rows],
                        color=COLORS[pretrained],
                        linestyle="-" if seed == 0 else "--",
                        linewidth=2,
                    )
            ax.set_title(name, loc="left", fontsize=13, fontweight="bold")
            ax.set_xlim(1, 20)
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            if field == "loss":
                ax.set_yscale("log")
                ax.set_ylim(0.008, 2)
            else:
                ax.set_ylim(0.35, 0.9)
        for ax in axes[-1]:
            ax.set_xlabel("Epoch")
        for ax in axes[:, 0]:
            ax.set_ylabel("Cross-entropy loss" if field == "loss" else "Macro-F1")
        legend = [
            Line2D([0], [0], color=COLORS[True], linewidth=2, label="ImageNet, seed 0"),
            Line2D([0], [0], color=COLORS[True], linestyle="--", linewidth=2, label="ImageNet, seed 1"),
            Line2D([0], [0], color=COLORS[False], linewidth=2, label="Scratch, seed 0"),
            Line2D([0], [0], color=COLORS[False], linestyle="--", linewidth=2, label="Scratch, seed 1"),
        ]
        fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, 0.075))
        fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
        fig.text(0.5, 0.025,
                 "Patient-grouped split · 20 training epochs · ConvNeXt-Tiny and Swin-T have one seed",
                 ha="center", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0.02, 0.11, 0.98, 0.94), h_pad=2)
        fig.savefig(output / f"{filename}.png", dpi=200)
        fig.savefig(output / f"{filename}.svg")
        plt.close(fig)
        print(output / f"{filename}.png")


if __name__ == "__main__":
    main()
