
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference


PATCH_SIZE = 96


class TestDataset(Dataset):
    def __init__(self, manifest, data_dir):
        self.data_dir = Path(data_dir)

        self.rows = manifest[
            manifest["split"] == "test"
        ].reset_index(drop=True)

        if len(self.rows) != 20:
            raise ValueError(
                f"Expected 20 test patients, found {len(self.rows)}"
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        patient_id = row["patient_id"]

        folder = self.data_dir / "test"

        image = np.load(
            folder / f"{patient_id}_image.npy"
        ).astype(np.float32)

        label = np.load(
            folder / f"{patient_id}_label.npy"
        ).astype(np.int64)

        original_shape = label.shape

        _, depth, height, width = image.shape

        pad_d = max(0, PATCH_SIZE - depth)
        pad_h = max(0, PATCH_SIZE - height)
        pad_w = max(0, PATCH_SIZE - width)

        if pad_d or pad_h or pad_w:
            image = np.pad(
                image,
                ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                mode="constant"
            )

            label = np.pad(
                label,
                ((0, pad_d), (0, pad_h), (0, pad_w)),
                mode="constant"
            )

        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(label).long(),
            patient_id,
            original_shape
        )


def build_model(model_type):
    if model_type == "A":
        norm = "BATCH"
        activation = "RELU"
    else:
        norm = "INSTANCE"
        activation = ("LEAKYRELU", {"negative_slope": 0.01})

    return UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=1,
        norm=norm,
        act=activation
    )


def dice_score(prediction, target):
    prediction = prediction.bool()
    target = target.bool()

    intersection = (prediction & target).sum().item()
    denominator = prediction.sum().item() + target.sum().item()

    if denominator == 0:
        return 1.0

    return 2.0 * intersection / denominator


def evaluate_patient(prediction, target):
    pred_wt = prediction > 0
    true_wt = target > 0

    pred_tc = (prediction == 1) | (prediction == 3)
    true_tc = (target == 1) | (target == 3)

    wt = dice_score(pred_wt, true_wt)
    tc = dice_score(pred_tc, true_tc)

    return {
        "WT_Dice": wt,
        "TC_Dice": tc,
        "Mean_Dice": (wt + tc) / 2
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", default="data/preprocessed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=["A", "B"], required=True)
    parser.add_argument(
        "--output-dir", default="results/test_evaluation"
    )
    parser.add_argument("--overlap", type=float, default=0.25)

    args = parser.parse_args()

    if not 0 <= args.overlap < 1:
        raise ValueError("Overlap must be between 0 and 1.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Model:", args.model)
    print("Checkpoint:", args.checkpoint)

    manifest = pd.read_csv(args.manifest)

    dataset = TestDataset(manifest, args.data_dir)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = build_model(args.model).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = []

    with torch.inference_mode():
        for images, labels, patient_ids, original_shapes in loader:
            patient_id = patient_ids[0]

            # DataLoader batches each dimension into a one-element tensor.
            # Convert them to ordinary Python integers before slicing.
            original_shape = tuple(
                int(shape[0].item()) for shape in original_shapes
            )

            start = time.time()

            outputs = sliding_window_inference(
                inputs=images,
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=model,
                overlap=args.overlap,
                mode="gaussian",
                sw_device=device,
                device="cpu"
            )

            prediction = outputs.argmax(dim=1)[0]
            target = labels[0]

            d, h, w = original_shape

            prediction = prediction[:d, :h, :w]
            target = target[:d, :h, :w]

            metrics = evaluate_patient(prediction, target)

            elapsed = time.time() - start

            results.append({
                "patient_id": patient_id,
                **metrics,
                "inference_time_seconds": elapsed
            })

            print(
                f"{patient_id} | "
                f"WT: {metrics['WT_Dice']:.4f} | "
                f"TC: {metrics['TC_Dice']:.4f} | "
                f"Mean: {metrics['Mean_Dice']:.4f} | "
                f"Time: {elapsed:.2f}s"
            )

    results_df = pd.DataFrame(results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        output_dir / "test_patient_results.csv",
        index=False
    )

    metric_columns = [
        "WT_Dice",
        "TC_Dice",
        "Mean_Dice",
        "inference_time_seconds"
    ]

    summary = results_df[metric_columns].agg(
        ["mean", "std"]
    ).T

    summary.to_csv(output_dir / "test_summary.csv")

    print("\nTest evaluation completed.")
    print(summary)
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
