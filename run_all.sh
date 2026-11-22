
#!/usr/bin/env bash

set -euo pipefail

# BraTS archive must be supplied by the user.
if [[ -z "${BRATS_ARCHIVE:-}" ]]; then
    echo "ERROR: Set BRATS_ARCHIVE to your BraTS 2021 archive."
    echo 'Example: export BRATS_ARCHIVE="/path/to/BraTS2021_Training_Data.tar"'
    exit 1
fi

if [[ ! -f "$BRATS_ARCHIVE" ]]; then
    echo "ERROR: BraTS archive not found: $BRATS_ARCHIVE"
    exit 1
fi

SEED=42
EPOCHS="${EPOCHS:-50}"

echo "========================================"
echo "P29 BRAIN TUMOUR SEGMENTATION PIPELINE"
echo "========================================"

echo "[1/7] Preparing dataset..."
python scripts/prepare_data.py \
    --archive "$BRATS_ARCHIVE" \
    --output-dir data \
    --seed "$SEED" \
    --num-patients 80

MANIFEST="data/manifests/patient_manifest.csv"

echo "[2/7] Preprocessing MRI volumes..."
python scripts/preprocess_data.py \
    --manifest "$MANIFEST" \
    --output-dir data/preprocessed

echo "[3/7] Training Model A..."
python scripts/train.py \
    --manifest "$MANIFEST" \
    --data-dir data/preprocessed \
    --model A \
    --epochs "$EPOCHS" \
    --output-dir checkpoints

echo "[4/7] Training Model B..."
python scripts/train.py \
    --manifest "$MANIFEST" \
    --data-dir data/preprocessed \
    --model B \
    --epochs "$EPOCHS" \
    --output-dir checkpoints

echo "[5/7] Evaluating Model A..."
python scripts/evaluate.py \
    --manifest "$MANIFEST" \
    --data-dir data/preprocessed \
    --checkpoint checkpoints/model_A/best.pth \
    --model A \
    --output-dir results/model_a_test

echo "[6/7] Evaluating Model B and running Arm E..."
python scripts/evaluate.py \
    --manifest "$MANIFEST" \
    --data-dir data/preprocessed \
    --checkpoint checkpoints/model_B/best.pth \
    --model B \
    --output-dir results/arm_d_test

python scripts/inference_tta.py \
    --manifest "$MANIFEST" \
    --data-dir data/preprocessed \
    --checkpoint checkpoints/model_B/best.pth \
    --split test \
    --overlap 0.5 \
    --output-dir results/arm_e_test

echo "[7/7] Creating main comparison table..."
python scripts/compare_results.py \
    --model-a results/model_a_test/test_patient_results.csv \
    --arm-d results/arm_d_test/test_patient_results.csv \
    --arm-e results/arm_e_test/arm_e_patient_results.csv \
    --output results/final_tables/main_test_results.csv

echo "========================================"
echo "PIPELINE COMPLETED"
echo "========================================"

