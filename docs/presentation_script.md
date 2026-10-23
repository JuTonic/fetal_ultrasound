# Fetal ultrasound benchmark: speaker script and questions

Use this as a talk track, not text to read word for word. The first 16 slides are the main presentation; slides 17–27 are references and backup material. At a normal pace the main talk is roughly 11–14 minutes. Numbers below come from the completed run report and the preprocessing experiments. Keep the limitations in the spoken presentation.

## Main talk

### Slide 1 — Title

“We benchmarked four neural network architectures for classifying standard fetal ultrasound planes. The practical question is which model is a good starting point when we care about classification quality, the value of ImageNet pretraining, and computation.” Introduce the three team members and their roles if needed.

### Slide 2 — Task and research question

“The dataset has 12,400 frames and 1,787 patient groups after linking five pairs of patient IDs. We classify nine labels. We compare ResNet-50, DenseNet-121, ConvNeXt-Tiny, and Swin-T using one fixed preprocessing and evaluation protocol. The primary metric is macro-F1 because the largest class has about 29 times as many images as the smallest. We also report balanced accuracy, class-wise errors, parameters, and inference latency. We compare ImageNet initialization with training from scratch for the two finalists.”

If asked about relevance: recognizing a standard plane can support a later ultrasound workflow, but this dataset and experiment do not establish clinical utility or safety. The literature slide lists the dataset paper and related transfer-learning work; our scores should not be presented as directly comparable to other papers unless labels and evaluation protocols match.

### Slide 3 — Exploratory data analysis

“The dataset is not just 12,400 independent examples. Five pairs of patient IDs appear linked. There are 1,486 pixel-identical frames in 653 exact-duplicate groups, including 18 identical pairs with conflicting labels. The smallest class, Brain: other, has 143 frames, while Other has 4,213. Scanner and image format are also associated with class. These observations determined our patient-grouped split, our primary metric, and our error checks.”

Do not say the conflicting labels were corrected. They were retained and flagged for expert review. Do not say all duplicates were removed: they were retained and flagged, with no exact-duplicate group crossing the grouped partitions.

### Slide 4 — Evaluation protocol

“We chose the main split by patient group: 8,851 train, 1,770 validation, and 1,779 test images. No known linked patient group crosses parts, and the grouping also keeps identical frames together. The supplied split is a secondary comparison. A naive image-level split would place 1,008 of 1,792 original patient IDs in more than one part, so we did not use it as the main estimate. For the protocol comparison we use matched six-epoch ResNet-50 runs: grouped test macro-F1 0.847 versus supplied test 0.774. This difference is not a causal leakage effect. The two protocols also have different cohorts, training-set sizes, and scanner mixes. Both quality metrics are shown in the results section.”

Distinguish the six-epoch protocol comparison from the 20-epoch architecture results later in the talk.

### Slide 5 — Reasoning about data

“A linear probe can predict scanner identity from frozen image features even when screen text is hidden. That demonstrates scanner information remains in the pixels. It does not prove that our trained plane classifiers use scanner identity. We therefore report scores by scanner and treat shortcut learning as a remaining risk. The frame-size-only probe has weak class performance, but certain Aloka formats are highly class-specific. We should not interpret good aggregate accuracy as robust anatomical recognition without further tests.”

These are preprocessing-team probes, not an attribution analysis of the final DenseNet-121.

### Slide 6 — Preprocessing and ablation

“For every model, we convert to grayscale, resize the longer side to 224 pixels, pad to a 224-by-224 square, replicate to three channels, and use ImageNet normalization. Aspect ratio is preserved. Only training images get moderate rotation, scale, translation, and brightness/contrast augmentation; no horizontal flip. A ResNet-18 validation-only ablation tested alternative crops, text masking, resolution, and augmentation. None met the prespecified improvement rule, so we kept the simple pipeline. The crop-plus-mask variant looked promising, but its uncertainty interval included zero, so we did not call it a confirmed improvement.”

### Slide 7 — Experimental design

“We first screened four architectures, all ImageNet-initialized, with seed zero. We selected two finalists by validation macro-F1, then trained pretrained and scratch versions of each with seeds zero and one. ResNet-50 also serves as the six-epoch protocol comparison. All grouped runs were extended to 20 epochs using saved training state. The test set is used for final reporting after validation-based selection.”

Do not imply every architecture has multiple seeds. ConvNeXt-Tiny and Swin-T each have one ImageNet seed.

### Slide 8 — Training setup and convergence

“All models use a nine-class head, cross-entropy, AdamW, batch size eight, weight decay 10⁻⁴, and FP32. The first six epochs use an initial learning rate of 3×10⁻⁴. The continuation uses 3×10⁻⁵ for 14 further epochs. Each phase has warm-up and cosine decay. We choose the checkpoint with the highest validation macro-F1, breaking ties in favor of the earlier epoch. Docker and a locked `uv` environment make the runs reproducible across the RTX 3080 Ti PC and RTX 4060 laptop.”

### Slide 9 — Optimization and overfitting

“This DenseNet-121 run illustrates why checkpoint selection matters. Training loss keeps falling through epoch 20, but validation macro-F1 reaches its best value at epoch five and later fluctuates. We use that selected checkpoint for test evaluation, rather than automatically using the last epoch. The same recipe across architectures supports a fair comparison, but we did not conduct a systematic per-model hyperparameter search. The divergence between falling training loss and a non-improving validation metric is consistent with overfitting or diminishing returns; the plot alone cannot distinguish every cause.”

### Slide 10 — Quality versus cost

“DenseNet-121 has the best mean grouped-test macro-F1, 0.852 across two pretrained seeds, and the highest mean balanced accuracy, 0.851. Balanced accuracy averages recall over the nine classes, giving each class equal weight. DenseNet-121 is also the smallest head-adjusted model, about 7.0 million parameters. ResNet-50 reaches 0.841 macro-F1 and 0.844 balanced accuracy across two seeds and is faster in batch-one inference: 5.8 versus 13.9 milliseconds on the same RTX 3080 Ti. ConvNeXt-Tiny and Swin-T each have one seed. The latency ranking is measured on the same GPU in FP32 without data loading or preprocessing. DenseNet has fewer parameters and FLOPs but higher measured latency; operation counts alone do not predict wall-clock speed.”

The `±` values are sample standard deviations over just two seeds, not confidence intervals or proof of statistical significance. Parameter counts include the nine-class head. The FLOP convention counts a fused multiply-add as one operation and omits some unsupported operators.

### Slide 11 — Pretraining and protocol effect

“On the two finalists, ImageNet initialization improves mean test macro-F1 by about four percentage points: 0.852 versus 0.812 for DenseNet-121, and 0.841 versus 0.803 for ResNet-50. The paired gain is positive in both tested seeds of both architectures. That is useful evidence for this task, but two seeds do not support a broad claim about all ultrasound datasets. Separately, the matched six-epoch ResNet-50 comparison scores 0.847 macro-F1 and 0.855 balanced accuracy on our grouped test, versus 0.774 and 0.790 on the supplied test. Those gaps can reflect the different cohort, train size, and scanner distribution; they are not measured leakage penalties.”

### Slide 12 — Per-class and scanner results

“Macro-F1 exposes uneven performance. Brain: other is hardest: in the selected DenseNet-121 seed-zero test, its F1 is 0.485 on only 21 examples. Maternal cervix is near 0.994. The brain-class confusion plot shows where TT, TC, TV, and Brain: other mix. The scanner panel shows different within-scanner macro-F1 values, but scanner subsets also differ in classes, operators, and sample sizes. The small ‘Other scanner’ subgroup has only 37 test images, so avoid ranking scanners as if they were controlled experimental conditions.”

### Slide 13 — Model interpretation

“The confusion matrix describes error types. DenseNet-121 misclassifies 13 of 21 Brain: other images; 10 are predicted as TT. It predicts 18 of 86 true TV images as TT. Confidence does not make every prediction safe: among 1,487 test frames with top softmax at least 0.9, 38 are wrong. Softmax scores were not calibrated. The next slide probes which broad image regions influence three of these predictions.”

### Slide 14 — Grad-CAM examples

“These are Grad-CAM maps from the saved DenseNet-121 checkpoint on three held-out brain images. The top row shows each input; the lower row overlays regions associated with the model's predicted TT score. The first TT prediction is correct. In the other two images, the true labels are TV and Brain: other, but the model predicts TT. The maps emphasize a broad head region in all three examples. They are coarse maps from one model and three selected examples; they do not establish which anatomical landmark the model used, nor rule out scanner shortcuts. A controlled occlusion experiment and clinical review would be needed for stronger conclusions.”

### Slide 15 — Hypotheses and limitations

“Our first hypothesis is supported in a limited sense: DenseNet-121 is smaller and has the highest mean macro-F1, but it is slower than ResNet-50 in measured inference. The second is supported for the two finalists and two seeds: pretraining increases macro-F1. The third is observed: the split protocol changes the measured score, but the causes are confounded. Limitations include one hospital network, scanner and interface cues, rare labels, only two finalist seeds, and only illustrative Grad-CAM maps. The team-specific model was unavailable, so this benchmark covers four public backbones. Plane classification is not a clinical validation.”

### Slide 16 — Conclusions and future work

“If quality and model size matter most, DenseNet-121 is the starting point from this experiment. If latency matters more, ResNet-50 is attractive. ImageNet initialization helped both finalists. Evaluation protocol materially affected the observed score, so we should not compare models across different splits or training budgets without care. Next, we would extend Grad-CAM with occlusion and scanner-intervention tests, extend the patient-subset study to finalists, and validate on data from another scanner or hospital.”

Finish with the precise conclusion: these are results for this dataset and evaluation setup, not a general medical-device recommendation.

## Backup slides

- **17 — Literature:** Sources for the dataset and related methods. Avoid claiming that a reported score is directly comparable without matching labels and splits.
- **18 — Supplementary divider:** Move on quickly unless asked.
- **19 — Class imbalance and split balance:** The largest class is about 29.5 times the smallest. Brain: other has 20 validation and 21 test frames, so its F1 is noisy.
- **20 — Preprocessing examples:** Letterboxing preserves anatomy proportions. The sector crop and interface mask were tested in the ablation but not adopted.
- **21 — Data-efficiency pilot:** A separate ResNet-18 pilot on 25%, 50%, and 100% of training patients gave validation macro-F1 of 0.778, 0.815, and 0.846. It used one architecture and one subset draw; do not present it as a finalist result.
- **22 — Team and reproducibility:** Explain role allocation, frozen data checksums, saved checkpoints, per-run logs, and the final prediction/report archive.
- **23–27 — Anatomical references:** Use only if asked about what each plane depicts. These images illustrate anatomy; they are not model explanations or Grad-CAM results.

## Likely questions and direct answers

**Why nine classes?** The project uses the nine-class labels in FETAL_PLANES_DB, including three named brain sub-planes plus Brain: other. Some papers collapse classes, so their F1 scores are not directly comparable.

**Why macro-F1 rather than accuracy?** Other has 4,213 images and Brain: other has 143. Accuracy can look high while the rare class fails. Macro-F1 weights the nine class F1 scores equally. We also report balanced accuracy and per-class F1.

**What is balanced accuracy?** Calculate recall separately for each class—the fraction of its true images correctly recognized—then average the nine recalls. This gives small and large classes equal weight. Macro-F1 also balances classes, but combines precision and recall for each class.

**How did you prevent leakage?** We linked five apparent duplicate patient-ID pairs before grouping, split by the resulting patient group, and kept exact-duplicate groups in a single partition. The grouped split has no known linked-patient overlap. This cannot rule out unidentified near-duplicates or all metadata errors.

**Were repeated images removed?** No. Exact duplicates were retained and flagged. The main grouped test had 1,779 frames. Counting one frame per exact hash leaves 1,644 and changes selected DenseNet-121 seed-zero macro-F1 from 0.8515 to 0.8529. Excluding conflicting-label frames leaves 1,775 and changes it to 0.8527. These small differences do not erase the labeling concern.

**Why not use the official split as primary?** Its training and test sets have a large scanner shift; Voluson S10 is about 0.2% of official training and about 20% of official test. Our grouped split balances known scanner and class shares while preventing known patient overlap. Neither split demonstrates performance at an unseen hospital.

**Does the six-epoch score gap prove leakage?** No. The grouped six-epoch ResNet-50 test macro-F1 is 0.847; the supplied-split score is 0.774. Different patients, train sizes, and scanner mixes are confounded. The 20-epoch grouped score must not be used in that matched-budget comparison.

**Why these four architectures?** They span conventional residual and dense CNNs, a modern CNN, and a hierarchical transformer. The design meets the four-architecture, three-family screen in the project brief while fitting available GPU time.

**Why ImageNet weights for grayscale ultrasound?** The same grayscale frame is replicated into three channels so pretrained three-channel backbones can be used. The benefit was tested rather than assumed: both finalists improved by roughly 0.04 macro-F1 over scratch in the tested seeds. This does not prove that ImageNet is optimal for all ultrasound tasks.

**Why no horizontal flip?** It can mirror on-screen text and was not beneficial in the preprocessing ablation. The fixed pipeline also preserves aspect ratio and avoids cropping anatomy from the main comparison.

**Why AdamW and those learning rates?** They form a simple shared, reproducible recipe. Weight decay adds regularization; the lower-rate continuation lets training proceed from the saved state. We did not run a full optimizer or learning-rate sweep, so we do not claim this is optimal for each architecture.

**Was 20 epochs enough? Was it too much?** Every grouped run reached 20 epochs. The reported checkpoint is selected by validation macro-F1 and may be much earlier. For example, DenseNet seed zero selected epoch five despite further decline in training loss. More training could help with a different schedule, but it is not established by these results.

**Why does DenseNet have fewer parameters but higher latency?** Runtime depends on memory traffic, kernel sizes, operation structure, framework implementation, and GPU behavior, not just parameter or FLOP totals. On the same GPU and timing setup, DenseNet measured 13.9 ms versus 5.8 ms for ResNet-50. These are model-forward timings, not end-to-end clinical workflow latency.

**Are differences statistically significant?** We did not run a formal significance test for the final model ranking. The two-seed mean and sample standard deviation are descriptive; ConvNeXt-Tiny and Swin-T have one seed each. The ranking could change with additional seeds or another dataset.

**Why is Brain: other weak?** It has few examples and is a heterogeneous residual label. In the selected DenseNet run, 13 of 21 test examples are wrong, 10 predicted TT. Label ambiguity and related brain anatomy are plausible explanations, but the current analysis does not isolate the cause.

**Does the scanner probe or Grad-CAM prove shortcut learning in the final model?** No. The probe shows that scanner identity is recoverable from images. The three Grad-CAM examples show broad head-region sensitivity, but do not rule out other cues. A controlled scanner intervention would be needed to test reliance. Scanner-specific scores are descriptive because scanner groups differ in other ways.

**Did you run Grad-CAM?** Yes, for three held-out brain images with the saved DenseNet-121 seed-zero checkpoint, targeting its last spatial feature layer and the predicted-class score. One prediction is correct and two are errors. These maps are illustrative, not an explanation of the entire test set or a causal test. The anatomy reference slides are separate from the Grad-CAM slide.

**Is the model calibrated?** We have not measured calibration. The confidence chart reports error rates among frames above softmax thresholds; those scores cannot be interpreted as calibrated probabilities.

**What is the main recommendation?** Start with DenseNet-121 if grouped-test quality and model size are priorities; consider ResNet-50 if inference latency matters more. Validate the choice on the intended scanner, hospital, and patient population before any deployment.

**What would you do next?** Extend attribution beyond three examples, run controlled occlusion and scanner-intervention tests, investigate ambiguous labels with a clinician, add seeds and uncertainty intervals, test finalist data efficiency, and evaluate a truly external scanner or hospital. These are future experiments, not completed claims.

**Where are the code and artifacts?** The repository contains the Docker/`uv` pipeline, frozen split metadata, and plotting scripts. The results archive contains predictions, metrics, learning curves, and run metadata. The team-specific model weights were unavailable and were not included in the completed benchmark.
