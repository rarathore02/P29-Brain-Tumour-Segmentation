
import argparse
from pathlib import Path

import pandas as pd


def load_results(path, model_name):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_csv(path)

    required = ["WT_Dice", "TC_Dice", "Mean_Dice"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    result = {
        "Model": model_name,
        "Patients": len(df),
    }

    for metric in required:
        result[f"{metric}_mean"] = df[metric].mean()
        result[f"{metric}_std"] = df[metric].std()

    if "inference_time_seconds" in df.columns:
        result["Inference_time_seconds_mean"] = (
            df["inference_time_seconds"].mean()
        )
    else:
        result["Inference_time_seconds_mean"] = float("nan")

    return result


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

    comparisons = [
        load_results(args.model_a, "Model A"),
        load_results(args.arm_d, "Arm D"),
        load_results(args.arm_e, "Arm E"),
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame(comparisons)

    result_df.to_csv(output, index=False)

    print("\nMain test comparison:")
    print(result_df.to_string(index=False))

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
