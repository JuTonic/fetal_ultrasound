# Experiment tracking

**Snapshot:** 11 completed evaluations from the final report. The [run index](runs.csv) is downloadable as CSV. Open a run below for its configuration, validation/test metrics, confusion counts, and epoch history. The full evaluation output and prediction archive remain local; these pages contain the compact evidence used in the presentation.

| Model | Split | Init. | Seed | Epochs | Selected | Test macro-F1 | Balanced acc. | Artifacts |
|---|---|---|---:|---:|---:|---:|---:|---|
| resnet50 | split_grouped | ImageNet | 0 | 20 | 7 | 0.838 | 0.839 | [config](runs/resnet50-split_grouped-imagenet-seed0/config.json) · [metrics](runs/resnet50-split_grouped-imagenet-seed0/metrics.json) · [epochs](runs/resnet50-split_grouped-imagenet-seed0/history.csv) |
| densenet121 | split_grouped | ImageNet | 0 | 20 | 5 | 0.851 | 0.850 | [config](runs/densenet121-split_grouped-imagenet-seed0/config.json) · [metrics](runs/densenet121-split_grouped-imagenet-seed0/metrics.json) · [epochs](runs/densenet121-split_grouped-imagenet-seed0/history.csv) |
| convnext_tiny | split_grouped | ImageNet | 0 | 20 | 8 | 0.841 | 0.839 | [config](runs/convnext_tiny-split_grouped-imagenet-seed0/config.json) · [metrics](runs/convnext_tiny-split_grouped-imagenet-seed0/metrics.json) · [epochs](runs/convnext_tiny-split_grouped-imagenet-seed0/history.csv) |
| swin_t | split_grouped | ImageNet | 0 | 20 | 10 | 0.819 | 0.814 | [config](runs/swin_t-split_grouped-imagenet-seed0/config.json) · [metrics](runs/swin_t-split_grouped-imagenet-seed0/metrics.json) · [epochs](runs/swin_t-split_grouped-imagenet-seed0/history.csv) |
| resnet50 | split_official | ImageNet | 0 | 6 | 3 | 0.774 | 0.790 | [config](runs/resnet50-split_official-imagenet-seed0/config.json) · [metrics](runs/resnet50-split_official-imagenet-seed0/metrics.json) · [epochs](runs/resnet50-split_official-imagenet-seed0/history.csv) |
| resnet50 | split_grouped | ImageNet | 1 | 20 | 7 | 0.845 | 0.848 | [config](runs/resnet50-split_grouped-imagenet-seed1/config.json) · [metrics](runs/resnet50-split_grouped-imagenet-seed1/metrics.json) · [epochs](runs/resnet50-split_grouped-imagenet-seed1/history.csv) |
| resnet50 | split_grouped | scratch | 0 | 20 | 17 | 0.799 | 0.805 | [config](runs/resnet50-split_grouped-scratch-seed0/config.json) · [metrics](runs/resnet50-split_grouped-scratch-seed0/metrics.json) · [epochs](runs/resnet50-split_grouped-scratch-seed0/history.csv) |
| resnet50 | split_grouped | scratch | 1 | 20 | 19 | 0.806 | 0.804 | [config](runs/resnet50-split_grouped-scratch-seed1/config.json) · [metrics](runs/resnet50-split_grouped-scratch-seed1/metrics.json) · [epochs](runs/resnet50-split_grouped-scratch-seed1/history.csv) |
| densenet121 | split_grouped | ImageNet | 1 | 20 | 14 | 0.852 | 0.853 | [config](runs/densenet121-split_grouped-imagenet-seed1/config.json) · [metrics](runs/densenet121-split_grouped-imagenet-seed1/metrics.json) · [epochs](runs/densenet121-split_grouped-imagenet-seed1/history.csv) |
| densenet121 | split_grouped | scratch | 0 | 20 | 16 | 0.827 | 0.824 | [config](runs/densenet121-split_grouped-scratch-seed0/config.json) · [metrics](runs/densenet121-split_grouped-scratch-seed0/metrics.json) · [epochs](runs/densenet121-split_grouped-scratch-seed0/history.csv) |
| densenet121 | split_grouped | scratch | 1 | 20 | 19 | 0.797 | 0.796 | [config](runs/densenet121-split_grouped-scratch-seed1/config.json) · [metrics](runs/densenet121-split_grouped-scratch-seed1/metrics.json) · [epochs](runs/densenet121-split_grouped-scratch-seed1/history.csv) |

## Visual artifacts

- [Validation macro-F1 by epoch](../../results/figures/validation_macro_f1_20_epochs.png)
- [Training loss by epoch](../../results/figures/training_loss_20_epochs.png)
- [Model size and inference cost](../../results/figures/model_size_and_cost.png)
- [Grad-CAM example](../../results/figures/gradcam_densenet_brain.png)

## Reading the results

Each score belongs to its own split and training budget. The official-split run has six epochs; grouped runs have 20. The selected checkpoint may be earlier than the last epoch. ConvNeXt-Tiny and Swin-T have one seed each. `history.csv` gives training loss and validation macro-F1; test metrics are in `metrics.json`. The disabled team model has no run or score.
