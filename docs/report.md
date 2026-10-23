# Fetal ultrasound plane classification: final report

## Objective

We trained four models to classify ultrasound frames into nine maternal and fetal plane labels. We compared their classification scores and computational cost, tested whether ImageNet starting weights helped, and checked how the evaluation split changed the measured score.

## Data and methods

We used the public FETAL_PLANES_DB dataset: 12,400 images with 1,792 original patient IDs. We linked five pairs of IDs that referred to the same patient, leaving 1,787 patient groups. The main split has 8,851 training, 1,770 validation, and 1,779 test frames. All frames from a patient group stay in one partition. We flagged exact duplicates and identical frames with conflicting labels, but kept them in the dataset. The class sizes vary a lot: Other has 4,213 frames, while Brain: other has 143.

We converted every image to grayscale, resized its longest side to 224 pixels, padded it to 224 × 224, copied the grayscale channel three times, and applied ImageNet normalization. Only training images received moderate geometric and intensity augmentation. We used this preprocessing and a nine-class output head for ResNet-50, DenseNet-121, ConvNeXt-Tiny, and Swin-T. All runs used cross-entropy, AdamW, batch size eight, FP32, and weight decay 10⁻⁴. The first six epochs used a learning rate of 3×10⁻⁴. The grouped runs were then continued to 20 total epochs at 3×10⁻⁵. Both phases used warm-up and cosine decay. We saved the checkpoint with the highest validation macro-F1; ties went to the earlier epoch.

We first ran all four architectures with ImageNet weights and seed zero. Based on validation macro-F1, we selected ResNet-50 and DenseNet-121 for another seed and for runs starting from random weights. We used the test set for the final scores. Macro-F1 is the main metric because the classes are imbalanced. We also measured balanced accuracy, per-class scores, confusion counts, model size, and GPU inference time for one image at a time. The [tracking page](tracking/README.md) has the runs, configurations, metrics, and learning curves.

## Results

| Model | Initialization | Seeds | Test macro-F1 | Balanced accuracy |
|---|---|---:|---:|---:|
| DenseNet-121 | ImageNet | 2 | 0.852 | 0.851 |
| ResNet-50 | ImageNet | 2 | 0.841 | 0.844 |
| ConvNeXt-Tiny | ImageNet | 1 | 0.841 | 0.839 |
| Swin-T | ImageNet | 1 | 0.819 | 0.814 |
| DenseNet-121 | scratch | 2 | 0.812 | 0.810 |
| ResNet-50 | scratch | 2 | 0.803 | 0.805 |

For models with two seeds, the table shows the arithmetic mean rounded to three decimals. The [tracking table](tracking/README.md) gives each run separately. ImageNet initialization raised mean test macro-F1 by about 0.040 for DenseNet-121 and 0.039 for ResNet-50. Both versions were trained on the ultrasound data before we measured this difference. DenseNet-121 has about 7.0 million parameters, compared with 23.5 million for ResNet-50. On the same RTX 3080 Ti, a batch-one FP32 forward pass took about 13.9 ms for DenseNet-121 and 5.8 ms for ResNet-50. These times exclude image loading and preprocessing.

Brain: other was the hardest class in the selected DenseNet-121 seed-zero run: F1 was 0.485 on 21 test frames. In the [epoch curves](../results/figures/validation_macro_f1_20_epochs.png), the best validation score can occur before the last epoch. The [Grad-CAM example](../results/figures/gradcam_densenet_brain.png) shows where one prediction drew attention, but one image cannot tell us whether the model usually relies on anatomy.

We also compared two splits using ResNet-50 trained for **six epochs** in each case. The grouped split gave test macro-F1 0.847 and balanced accuracy 0.855; the dataset-supplied split gave 0.774 and 0.790. The test patients, training-set sizes, and scanner mixes differ, so we cannot attribute the score gap to leakage alone. The 20-epoch grouped run has a different training budget and is not part of this comparison.

## Relation to previous work

[Burgos-Artizzu et al. (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7311420/) reported 93.8% top-1 accuracy for DenseNet-169 on six broad plane classes. They tested the brain subplanes separately. [Barrientos et al. (2026)](https://arxiv.org/html/2605.27796) reported 0.9476 macro-F1 for USFM on six classes, using an ensemble of five cross-validation models. Our nine-class macro-F1 is not directly comparable with either number: the labels, data splits, and evaluation procedures differ.

## Limitations and next steps

The data come from one hospital network. The models may learn scanner or screen-interface cues as well as anatomy. Some classes have few images, and the dataset still contains duplicates and conflicting labels. ConvNeXt-Tiny and Swin-T were each run with only one seed. We did not test on an independent external clinical dataset. We also could not include the team-specific model because its checkpoint was unavailable.

We would next test on images from another scanner or hospital, repeat the single-seed runs, and review errors in the rare brain classes. Masking or perturbing screen-interface elements would help check whether the models use those cues. A direct comparison with another model would require the same dataset and evaluation protocol.
