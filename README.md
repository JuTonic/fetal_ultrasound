# Fetal ultrasound plane classification

NeuroML course project. We classify fetal ultrasound frames into nine plane labels and compare ResNet-50, DenseNet-121, ConvNeXt-Tiny, and Swin-T. This is **plane recognition**, not fetal diagnosis.

## Submission links

- **Experiment tracking:** [browse runs, configurations, metrics, learning curves, and figures](docs/tracking/README.md).
- **Written report (open Markdown format):** [report.md](docs/report.md).
- **Presentation notes:** [presentation_script.md](docs/presentation_script.md).

The repository is an original project repository, not a fork. The `pipeline` work is integrated into `main`.

## Main result

On the patient-grouped nine-class test set, ImageNet-initialized DenseNet-121 achieved mean macro-F1 **0.852** and balanced accuracy **0.851** across two seeds. ResNet-50 achieved **0.841** and **0.844**. ConvNeXt-Tiny and Swin-T each have one seed. See the [tracking page](docs/tracking/README.md) for individual runs and the [report](docs/report.md) for limits of these comparisons.

## Setup and run

Requirements: Docker, Bash, Git, and an NVIDIA GPU with driver and NVIDIA Container Toolkit for CUDA training. Run from the repository root:

```bash
./run.sh --mode train                 # configured training and evaluation plan
./run.sh                              # evaluate compatible local checkpoints
./run.sh --mode smoke --device cpu    # small pipeline check
```

The first run downloads the public dataset and required ImageNet weights into local caches. Training and preprocessing run in Docker. Dependencies are locked with `uv.lock`; configuration is in `configs/pipeline.json`. The main trained checkpoints are too large for Git and are not included. A new full training run is therefore needed to reproduce model weights; existing run metrics and histories are published in [experiment tracking](docs/tracking/README.md). Rerunning the same interrupted training command resumes completed epochs and skips completed jobs.

See [pipeline instructions](docs/pipeline.md) for configuration, download recovery, checkpoints, and artifact paths. Final local output is `results/final.json` and `results/final.zip`. The raw dataset, caches, and checkpoints are excluded from Git.

## Project structure

- `src/fetal/`: preprocessing, model training, evaluation, reporting, and configuration code.
- `configs/`: JSON pipeline settings.
- `docs/tracking/`: compact, navigable snapshot of the final experiment records.
- `docs/report.md`: objectives, methods, results, limitations, and next steps.
- `results/figures/`: selected plots and interpretation figures.
- `tests/`: pipeline tests.

## Development

```bash
uv sync --locked --extra cpu
uv run --locked --extra cpu pytest -q
```

For GPU development, use `--extra cuda` in place of `--extra cpu`. The scientific benchmark uses `./run.sh` and Docker.
