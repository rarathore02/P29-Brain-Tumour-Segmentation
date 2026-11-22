# Brain Tumour Segmentation Using Multi-Modal MRI

A 3D deep learning project for automatic brain tumour segmentation using multi-modal MRI scans from the BraTS 2021 dataset.

## Problem Statement

The objective is to segment brain tumour regions from four MRI modalities:

- T1
- T1ce
- T2
- FLAIR

This project investigates conventional and improved 3D U-Net configurations and evaluates the effect of enhanced inference using test-time augmentation (TTA) and increased sliding-window overlap.

## Dataset

**Dataset:** BraTS 2021 Training Dataset  
**Source:** [BraTS 2021](https://www.synapse.org/Synapse:syn25829067)  
**Subset:** 80 selected patients.

| Split | Patients |
|---|---:|
| Training | 52 |
| Validation | 8 |
| Held-out testing | 20 |
| **Total** | **80** |

The dataset is not included in this repository. Users must obtain it from the official source.

## Methodology

### Model A — Baseline

- MONAI 3D U-Net
- Batch Normalisation
- ReLU activation
- Cross-Entropy loss
- Adam optimiser

### Model B — Improved configuration

- MONAI 3D U-Net
- Instance Normalisation
- LeakyReLU activation
- Dice + Cross-Entropy loss
- Tumour-aware patch sampling
- Spatial and intensity augmentation
- Learning-rate scheduling

### Inference experiments

- **Arm D:** Model B with standard sliding-window inference.
- **Arm E:** Model B with 8-way test-time augmentation and increased sliding-window overlap (0.25 to 0.5).

All experiments use the same patient-level split.

## Held-out Test Results

Evaluation was performed on 20 held-out patients.

| Metric | Model A | Arm D | Arm E |
|---|---:|---:|---:|
| Whole Tumour Dice | 0.8492 | 0.8528 | **0.8612** |
| Tumour Core Dice | 0.6445 | 0.7696 | **0.7876** |
| Mean Dice | 0.7468 | 0.8112 | **0.8244** |

Values represent mean per-patient Dice scores.

Arm E achieved the highest mean Dice among the evaluated configurations. Compared with Model A, its mean Dice improved by approximately 0.0776.

## Ablation Study

Arm D and Arm E use the same Model B weights, allowing an inference-stage comparison.

| Metric | Arm D | Arm E | Improvement |
|---|---:|---:|---:|
| Whole Tumour Dice | 0.8528 | 0.8612 | +0.0084 |
| Tumour Core Dice | 0.7696 | 0.7876 | +0.0180 |
| Mean Dice | 0.8112 | 0.8244 | +0.0132 |

The ablation combines increased sliding-window overlap and 8-way TTA. Therefore, the individual contribution of each inference modification is not isolated.

## Class-wise Analysis and Failure Analysis

The project includes:

- Per-patient Dice evaluation for necrotic core, oedema and enhancing tumour.
- Paired statistical comparisons with bootstrap confidence intervals and Wilcoxon tests.
- Patient-level precision, recall and segmentation-volume analysis.
- Qualitative failure analysis of four selected test patients.
- Uncertainty maps based on predictive entropy.

## Results and Visualisations

The `results/` directory contains consolidated experiment tables, performance comparisons, training curves and failure-analysis visualisations.

## Hardware and Frameworks

- GPU: NVIDIA Tesla T4
- Framework: PyTorch
- Medical imaging framework: MONAI
- Training: 50 epochs per model

## Repository Structure

The repository will contain the original notebook, reusable source scripts, configurations, logs, result tables and figures.

## Reproduction

The dataset and trained checkpoints are not included directly in this repository.

Reproduction instructions, dependency versions, dataset preparation steps and execution scripts will be provided with the final release.

## Credits

- BraTS 2021 dataset and its organisers.
- [MONAI](https://monai.io/)
- [PyTorch](https://pytorch.org/)

All reused implementations and relevant research papers will be credited in the final documentation.

## Random Seeds

The experiments use a fixed seed of 42 where configured. Exact reproducibility settings will be documented with the released implementation.

## AI Declaration

AI tools, including ChatGPT, were used for coding assistance, debugging, conceptual explanations and documentation support. The team is responsible for reviewing, testing and understanding the submitted implementation.

## Contributions

Individual team contributions will be specified in the final submission.

## Limitations

- The dataset subset contains 80 patients and may not represent the full variability of brain tumour cases.
- Results are based on a single fixed patient-level split.
- The inference ablation combines two modifications, so their individual effects remain undetermined.
- Additional external validation is needed to assess generalisation.

## Project Status

Model development, held-out test evaluation, inference ablation, uncertainty analysis and failure analysis have been completed. Repository reproducibility scripts and final documentation are being prepared.
