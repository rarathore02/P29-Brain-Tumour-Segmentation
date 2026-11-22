
import argparse
from pathlib import Path

import pandas as pd


EXPECTED_PATIENTS = 20

REQUIRED_COLUMNS = [
    "patient_id",
    "WT_Dice",
    "TC_Dice",
    "Mean_Dice",
]


def load_results(path, model_name):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_csv(path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    if len(df) != EXPECTED_PATIENTS:
        raise ValueError(
            f"{model_name}: Expected {EXPECTED_PATIENTS} patients, "
            f"found {len(df)}."
        )

    if df["patient_id"].duplicated().any():
        raise ValueError(f"{model_name}: Duplicate patient IDs found.")

    metrics = ["WT_Dice", "TC_Dice", "Mean_Dice"]

    if df[metrics].isna().any().any():
        raise ValueError(f"{model_name}: Missing Dice metric values found.")

    if not df[metrics].apply(
        lambda col: col.between(0, 1).all()
    ).all():
        raise ValueError(f"{model_name}: Dice values outside [0, 1].")

    result = {
        "Model": model_name,
        "Patients": len(df),
    }

    for metric in metrics:
        result[f"{metric}_mean"] = df[metric].mean()
        result[f"{metric}_std"] = df[metric].std(ddof=1)

    if "inference_time_seconds" in df.columns:
        result["Inference_time_seconds_mean"] = (
            df["inference_time_seconds"].mean()
        )
    else:
        result["Inference_time_seconds_mean"] = float("nan")

    return result, set(df["patient_id"].astype(str))


def main():
    parser = argparse.ArgumentParser(
        description="Create the main BraTS test comparison table."
    )

    parser.add_argument("--model-a", required=True)
    parser.add_argument("--arm-d", required=True)
    parser.add_argument("--arm-e", required=True)

    parser.add_argument(
        "--output",
        default="results/final_tables/main_test_results.csv"
    )

    args = parser.parse_args()

    model_a, ids_a = load_results(args.model_a, "Model A")
    arm_d, ids_d = load_results(args.arm_d, "Arm D")
    arm_e, ids_e = load_results(args.arm_e, "Arm E")

    if not (ids_a == ids_d == ids_e):
        raise ValueError(
            "Patient IDs do not match across Model A, Arm D and Arm E."
        )

    result_df = pd.DataFrame([model_a, arm_d, arm_e])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(output, index=False)

    print("\nMain test comparison:")
    print(result_df.to_string(index=False))

    print(f"\nPatient consistency check: PASSED")
    print(f"Patients evaluated: {EXPECTED_PATIENTS}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
