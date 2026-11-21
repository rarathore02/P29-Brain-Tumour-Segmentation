# Brain Tumour Segmentation using Multi-Modal MRI

A 3D deep learning project for automatic brain tumour segmentation using multi-modal MRI scans from the BraTS 2021 dataset.

## Problem Statement

The objective is to segment brain tumour regions from four MRI modalities:
- T1
- T1ce
- T2
- FLAIR

The project investigates whether an improved 3D U-Net configuration can improve segmentation performance over a baseline 3D U-Net.

## Dataset

**Dataset:** BraTS 2021 Training Dataset  
**Source:** https://www.synapse.org/Synapse:syn25829067  
**Subset:** 80 selected patients.

| Split | Patients |
|---|---:|
| Training | 52 |
| Validation | 8 |
| Held-out testing | 20 |
| Total | 80 |

The dataset is not included in this repository. Users must obtain the dataset through its official source and prepare the selected subset using the provided scripts.

## Methodology

Two 3D U-Net configurations are investigated.

**Model A — Baseline**
- MONAI 3D U-Net
- Batch Normalisation
- ReLU activation
- Cross-Entropy loss

**Model B — Improved configuration**
- 3D U-Net
- Instance Normalisation
- LeakyReLU activation
- Dice + Cross-Entropy loss
- Tumour-aware patch sampling
- Spatial and intensity augmentation

Both configurations use the same patient-level data split.

## Validation Results

| Metric | Model A | Model B |
|---|---:|---:|
| Whole Tumour Dice | 0.8650 | 0.8804 |
| Tumour Core Dice | 0.7514 | 0.8591 |
| Mean Dice | 0.8082 | 0.8697 |

These are validation results on 8 patients, not held-out test results.

## Hardware and Runtime

- GPU: NVIDIA Tesla T4
- Framework: PyTorch
- Medical imaging framework: MONAI
- Model A training: approximately 96.54 minutes
- Model B training: approximately 160.41 minutes

## Repository Structure

```text
P29-Brain-Tumour-Segmentation/
├── README.md
├── requirements.txt
├── run_all.sh
├── configs/
├── src/
├── experiments/
└── results/
    ├── tables/
    ├── figures/
    └── logs/
```

## Reproduction

The reproduction pipeline is being organised.

The intended execution command is:

```bash
bash run_all.sh
```

Dataset preparation instructions, configuration files and execution requirements will be documented before the final release.

## Credits and References

- BraTS 2021 dataset and its organisers.
- MONAI framework: https://monai.io/
- PyTorch: https://pytorch.org/

All additional reused implementations, papers and pretrained resources will be credited in the final version.

## Random Seeds

The final experiment configurations and random seeds will be documented in the released configuration files and logs.

## AI Declaration

AI tools, including ChatGPT, were used for coding assistance, debugging, conceptual explanations and documentation support. The team is responsible for reviewing, testing and understanding the submitted implementation and results.

## Contributions

Individual team contributions will be specified in the final submission.

## Project Status

Experimental implementation and validation completed for two 3D U-Net configurations. Held-out test evaluation, ablation experiments, uncertainty analysis and final documentation are in progress.
