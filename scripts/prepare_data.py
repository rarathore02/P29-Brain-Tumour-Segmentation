
#!/usr/bin/env python3
"""
Prepare a reproducible 80-patient BraTS 2021 subset.

Steps:
1. Index the BraTS archive.
2. Identify patients with all five required NIfTI files.
3. Select 80 patients using a fixed random seed.
4. Create a patient-level 52/8/20 train/validation/test split.
5. Extract the selected patients.
6. Validate the extracted NIfTI files.
7. Save the archive index and patient manifest.

The dataset itself is not included in this repository.
"""

import argparse
import random
import re
import shutil
import tarfile
from pathlib import Path

import nibabel as nib
import pandas as pd
from tqdm import tqdm


SEED = 42
NUM_PATIENTS = 80

SPLIT_COUNTS = {
    "train": 52,
    "validation": 8,
    "test": 20,
}

MODALITIES = ["t1", "t1ce", "t2", "flair", "seg"]

FILE_PATTERN = re.compile(
    r"(BraTS2021_[A-Za-z0-9]+)_(t1ce|flair|t1|t2|seg)\.nii\.gz$",
    re.IGNORECASE,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Index, select, split and extract a BraTS 2021 subset."
    )

    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Path to BraTS2021_Training_Data.tar",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where extracted data and manifests will be stored.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducible patient selection.",
    )

    parser.add_argument(
        "--num-patients",
        type=int,
        default=NUM_PATIENTS,
        help="Number of complete patients to select.",
    )

    return parser.parse_args()


def index_archive(archive):
    """Find all matching BraTS modality files in the archive."""

    print("\nIndexing BraTS archive...")

    records = []

    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()

        for member in tqdm(members, desc="Scanning archive"):
            if not member.isfile():
                continue

            match = FILE_PATTERN.search(Path(member.name).name)

            if match is None:
                continue

            patient_id = match.group(1)
            modality = match.group(2).lower()

            records.append({
                "patient_id": patient_id,
                "modality": modality,
                "archive_path": member.name,
                "file_size": member.size,
            })

    if not records:
        raise RuntimeError(
            "No BraTS files found. Check the archive and filename format."
        )

    archive_index = pd.DataFrame(records)

    print(f"Indexed files: {len(archive_index)}")
    print(f"Patients found: {archive_index['patient_id'].nunique()}")

    return archive_index


def find_complete_patients(archive_index):
    """Keep patients containing all five required modalities."""

    complete = []

    for patient_id, group in archive_index.groupby("patient_id"):
        available = set(group["modality"])

        if all(modality in available for modality in MODALITIES):
            complete.append(patient_id)

    complete.sort()

    print(f"Complete patients: {len(complete)}")

    return complete


def select_patients(complete_patients, num_patients, seed):
    """
    Reproduce the notebook's selection procedure:
    shuffle complete patients, select the first N, sort them,
    then shuffle the selected subset again for splitting.
    """

    if len(complete_patients) < num_patients:
        raise ValueError(
            f"Only {len(complete_patients)} complete patients found; "
            f"{num_patients} required."
        )

    rng = random.Random(seed)

    candidates = complete_patients.copy()
    rng.shuffle(candidates)

    selected = sorted(candidates[:num_patients])

    rng.shuffle(selected)

    return selected


def create_manifest(selected_patients, archive_index, output_dir):
    """Create the fixed patient-level train/validation/test split."""

    train_end = SPLIT_COUNTS["train"]
    val_end = train_end + SPLIT_COUNTS["validation"]

    if len(selected_patients) != sum(SPLIT_COUNTS.values()):
        raise ValueError(
            "The selected patient count must match the configured "
            "52/8/20 split."
        )

    split_assignments = {}

    for patient_id in selected_patients[:train_end]:
        split_assignments[patient_id] = "train"

    for patient_id in selected_patients[train_end:val_end]:
        split_assignments[patient_id] = "validation"

    for patient_id in selected_patients[val_end:]:
        split_assignments[patient_id] = "test"

    selected_index = archive_index[
        archive_index["patient_id"].isin(selected_patients)
    ]

    manifest = []

    for patient_id in selected_patients:
        patient_files = selected_index[
            selected_index["patient_id"] == patient_id
        ]

        row = {
            "patient_id": patient_id,
            "split": split_assignments[patient_id],
        }

        for modality in MODALITIES:
            matches = patient_files[
                patient_files["modality"] == modality
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"{patient_id}: expected exactly one {modality} file, "
                    f"found {len(matches)}."
                )

            row[modality] = str(
                output_dir / "extracted" / matches.iloc[0]["archive_path"]
            )

        manifest.append(row)

    manifest_df = pd.DataFrame(manifest)

    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / "patient_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print("\nSplit distribution:")
    print(manifest_df["split"].value_counts())

    print(f"\nManifest saved: {manifest_path}")

    return manifest_df, selected_index


def extract_selected_files(archive, selected_index, output_dir):
    """Extract only the selected patients' NIfTI files."""

    extraction_dir = output_dir / "extracted"
    extraction_dir.mkdir(parents=True, exist_ok=True)

    print("\nExtracting selected patient files...")

    with tarfile.open(archive, "r:*") as tar:
        for row in tqdm(
            selected_index.itertuples(index=False),
            total=len(selected_index),
            desc="Extracting",
        ):
            archive_path = row.archive_path

            destination = extraction_dir / archive_path

            # Ensure the archive member cannot write outside extraction_dir.
            destination_resolved = destination.resolve()
            extraction_resolved = extraction_dir.resolve()

            if (
                destination_resolved != extraction_resolved
                and extraction_resolved not in destination_resolved.parents
            ):
                raise ValueError(f"Unsafe archive path: {archive_path}")

            destination.parent.mkdir(parents=True, exist_ok=True)

            if destination.exists():
                continue

            source = tar.extractfile(archive_path)

            if source is None:
                raise RuntimeError(
                    f"Could not read archive member: {archive_path}"
                )

            with source, destination.open("wb") as output_file:
                shutil.copyfileobj(source, output_file)

    print(f"Extracted files into: {extraction_dir}")


def validate_manifest(manifest):
    """Check that all selected NIfTI files exist and are readable."""

    print("\nValidating extracted NIfTI files...")

    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc="Validating",
    ):
        for modality in MODALITIES:
            path = Path(getattr(row, modality))

            if not path.exists():
                raise FileNotFoundError(path)

            image = nib.load(str(path))

            if len(image.shape) != 3:
                raise ValueError(
                    f"Expected a 3D volume: {path}, got {image.shape}"
                )

    print("Validation successful: all selected files are readable.")


def main():
    args = parse_arguments()

    archive = args.archive.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not archive.is_file():
        raise FileNotFoundError(f"BraTS archive not found: {archive}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BraTS 2021 DATA PREPARATION")
    print("=" * 60)
    print(f"Archive: {archive}")
    print(f"Output: {output_dir}")
    print(f"Seed: {args.seed}")
    print(f"Patients to select: {args.num_patients}")

    archive_index = index_archive(archive)

    index_dir = output_dir / "manifests"
    index_dir.mkdir(parents=True, exist_ok=True)

    archive_index.to_csv(
        index_dir / "archive_index.csv",
        index=False,
    )

    complete_patients = find_complete_patients(archive_index)

    selected_patients = select_patients(
        complete_patients,
        args.num_patients,
        args.seed,
    )

    manifest, selected_index = create_manifest(
        selected_patients,
        archive_index,
        output_dir,
    )

    extract_selected_files(
        archive,
        selected_index,
        output_dir,
    )

    validate_manifest(manifest)

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED")
    print("=" * 60)
    print(f"Selected patients: {len(manifest)}")
    print(f"Extracted NIfTI files: {len(selected_index)}")
    print(f"Train: {(manifest['split'] == 'train').sum()}")
    print(f"Validation: {(manifest['split'] == 'validation').sum()}")
    print(f"Test: {(manifest['split'] == 'test').sum()}")
    print(f"Manifest: {index_dir / 'patient_manifest.csv'}")


if __name__ == "__main__":
    main()

