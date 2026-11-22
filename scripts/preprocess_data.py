
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


MODALITIES = ["t1", "t1ce", "t2", "flair"]
LABEL_MAP = {0: 0, 1: 1, 2: 2, 4: 3}

EXPECTED_SPLIT_COUNTS = {
    "train": 52,
    "validation": 8,
    "test": 20
}


def load_volume(path):
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def normalize_volume(volume):
    mask = volume != 0

    if not np.any(mask):
        return np.zeros_like(volume, dtype=np.float32)

    values = volume[mask]
    mean = values.mean()
    std = values.std()

    normalized = np.zeros_like(volume, dtype=np.float32)

    if std > 0:
        normalized[mask] = (values - mean) / std

    return normalized


def preprocess_patient(row, output_dir):
    patient_id = row["patient_id"]
    split = row["split"]

    modality_volumes = []

    for modality in MODALITIES:
        volume = load_volume(row[modality])
        modality_volumes.append(volume)

    shapes = [volume.shape for volume in modality_volumes]

    if len(set(shapes)) != 1:
        raise ValueError(
            f"Modality shape mismatch for {patient_id}: {shapes}"
        )

    image = np.stack(
        [normalize_volume(volume) for volume in modality_volumes],
        axis=0
    )

    segmentation = load_volume(row["seg"]).astype(np.uint8)

    if segmentation.shape != image.shape[1:]:
        raise ValueError(
            f"Image/label shape mismatch for {patient_id}"
        )

    unique_labels = set(np.unique(segmentation).tolist())

    if not unique_labels.issubset(LABEL_MAP.keys()):
        raise ValueError(
            f"Unexpected labels for {patient_id}: {unique_labels}"
        )

    label = np.zeros_like(segmentation, dtype=np.uint8)

    for original, remapped in LABEL_MAP.items():
        label[segmentation == original] = remapped

    patient_dir = output_dir / split
    patient_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        patient_dir / f"{patient_id}_image.npy",
        image.astype(np.float16)
    )

    np.save(
        patient_dir / f"{patient_id}_label.npy",
        label
    )

    return {
        "patient_id": patient_id,
        "split": split,
        "image_shape": str(image.shape),
        "label_shape": str(label.shape),
        "image_finite": bool(np.isfinite(image).all()),
        "labels": str(sorted(np.unique(label).tolist()))
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to patient_manifest.csv"
    )

    parser.add_argument(
        "--output-dir",
        default="data/preprocessed",
        help="Directory for preprocessed arrays"
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = pd.read_csv(manifest_path)

    # Validate the patient split distribution.
    actual_counts = manifest["split"].value_counts().to_dict()

    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"Unexpected split distribution: {actual_counts}. "
            f"Expected: {EXPECTED_SPLIT_COUNTS}"
        )

    # Ensure every patient appears only once.
    if manifest["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient IDs found in manifest.")

    required_columns = [
        "patient_id", "split", "t1", "t1ce",
        "t2", "flair", "seg"
    ]

    missing = [
        column for column in required_columns
        if column not in manifest.columns
    ]

    if missing:
        raise ValueError(f"Missing manifest columns: {missing}")

    print(f"Patients to preprocess: {len(manifest)}")
    print(f"Output directory: {output_dir}")
    print(f"Split distribution: {actual_counts}")

    summaries = []

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Preprocessing"
    ):
        summaries.append(
            preprocess_patient(row, output_dir)
        )

    summary_df = pd.DataFrame(summaries)

    summary_dir = Path("logs")
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(
        summary_dir / "preprocessing_summary.csv",
        index=False
    )

    print("\nPreprocessing completed.")
    print(summary_df.groupby("split").size())
    print(f"Summary saved to {summary_dir / 'preprocessing_summary.csv'}")


if __name__ == "__main__":
    main()
