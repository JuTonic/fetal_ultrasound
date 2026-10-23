# Fetal ultrasound plane classification: final report

## Objective

We compared four image classifiers for recognizing nine maternal and fetal ultrasound plane labels. We asked which architecture offers useful classification quality and computational cost, whether ImageNet initialization helps, and how evaluation protocol affects the reported score. Plane recognition is not clinical diagnosis.

## Data and methods

The public FETAL_PLANES_DB dataset contains 12,400 images and 1,792 original patient IDs. Five linked pairs of IDs were joined, leaving 1,787 patient groups. The main split contains 8,851 training, 1,770 validation, and 1,779 test frames. Patient groups do not cross partitions. Exact duplicate frames and conflicting labels were flagged; duplicate images were retained. The largest label, Other, contains 4,213 frames, while Brain: other contains 143.

Each image was converted to grayscale, resized with its longest side at 224 pixels, padded to 224 × 224, replicated into three channels, and normalized with ImageNet statistics. Training images received moderate geometric and intensity augmentation. All four architectures used the same nine-class head and preprocessing: ResNet-50, DenseNet-121, ConvNeXt-Tiny, and Swin-T. Training used cross-entropy, AdamW, batch size eight, FP32, and weight decay 10⁻⁴. Runs began with a six-epoch phase at learning rate 3×10⁻⁴ and were continued to 20 total epochs at 3×10⁻⁵ with warm-up and cosine decay. The best checkpoint was chosen by validation macro-F1, with earlier epochs winning ties.

The four ImageNet-initialized architectures were screened with seed zero. The two finalists, ResNet-50 and DenseNet-121, were additionally trained from random initialization and with a second seed. The test set was used for final reporting. Our primary metric is macro-F1; we also report balanced accuracy, per-class metrics, confusion counts, model size, and batch-one GPU latency. [Individual runs, configurations, metrics, and learning curves](tracking/README.md) are available in the experiment tracker.

## Results

| Model | Initialization | Seeds | Test macro-F1 | Balanced accuracy |
|---|---|---:|---:|---:|
| DenseNet-121 | ImageNet | 2 | 0.852 | 0.851 |
| ResNet-50 | ImageNet | 2 | 0.841 | 0.844 |
| ConvNeXt-Tiny | ImageNet | 1 | 0.841 | 0.839 |
| Swin-T | ImageNet | 1 | 0.819 | 0.814 |
| DenseNet-121 | scratch | 2 | 0.812 | 0.810 |
| ResNet-50 | scratch | 2 | 0.803 | 0.805 |

Values for two-seed models are arithmetic means, rounded to three decimals. Individual values are in the [tracking table](tracking/README.md). ImageNet initialization improved mean test macro-F1 by about 0.040 for DenseNet-121 and 0.039 for ResNet-50. These are post-training comparisons under the same grouped protocol. The best selected DenseNet-121 checkpoint has about 7.0 million parameters; ResNet-50 has about 23.5 million. Measured batch-one FP32 forward time on the same RTX 3080 Ti was about 13.9 ms for DenseNet-121 and 5.8 ms for ResNet-50, excluding preprocessing and data loading.

The rare Brain: other class was difficult: its F1 was 0.485 on 21 test frames for the selected DenseNet-121 seed-zero run. The [epoch curves](../results/figures/validation_macro_f1_20_epochs.png) show why the best validation checkpoint can precede the final epoch. The [Grad-CAM example](../results/figures/gradcam_densenet_brain.png) is illustrative and does not establish that the model relies only on anatomy.

For the matched **six-epoch** ResNet-50 protocol comparison, the grouped split gave test macro-F1 0.847 and balanced accuracy 0.855; the supplied split gave 0.774 and 0.790. The cohorts, training sizes, and scanner mixes also differ, so this gap is not a measured leakage penalty. It should not be compared with the 20-epoch grouped result as though budgets matched.

## Relation to previous work

[Burgos-Artizzu et al. (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7311420/) reported 93.8% top-1 accuracy for DenseNet-169 on six broad plane classes and evaluated brain subplanes separately. [Barrientos et al. (2026)](https://arxiv.org/html/2605.27796) reported 0.9476 macro-F1 for USFM on a six-class experiment using a five-model cross-validation ensemble. Those results cannot be ranked directly against our nine-class macro-F1 because the labels, split, and evaluation procedures differ.

## Limitations and next steps

The data come from one hospital network, and models may use scanner or screen-interface cues. Some labels are rare, exact duplicates and conflicting labels remain in the dataset, and ConvNeXt-Tiny and Swin-T have only one seed each. There was no independent external clinical test. A team-specific model checkpoint was unavailable, so it is not included in the measured comparison.

Next steps are independent evaluation on another scanner or hospital, more seeds for the single-seed architectures, focused review of rare brain classes, and tests that mask or perturb interface cues. The dataset and protocol should be held fixed for any direct comparison with another model.
