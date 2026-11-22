
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference


SEED = 42
ROI_SIZE = (96, 96, 96)
TTA_AXES = (1, 2, 3)


class BrainTumourTestDataset(Dataset):
    def __init__(self, manifest, data_dir, split):
        self.data_dir = Path(data_dir)
        self.rows = manifest[
            manifest["split"] == split
        ].reset_index(drop=True)

        self.split = split

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        patient_id = row["patient_id"]

        folder = self.data_dir / self.split

        image = np.load(
            folder / f"{patient_id}_image.npy"
        ).astype(np.float32)

        label = np.load(
            folder / f"{patient_id}_label.npy"
        ).astype(np.int64)

        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(label).long(),
            patient_id
        )


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model():
    return UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=1,
        norm="INSTANCE",
        act=("LEAKYRELU", {"negative_slope": 0.01})
    )


def dice_score(prediction, target):
    prediction = prediction.bool()
    target = target.bool()

    intersection = (prediction & target).sum().item()
    denominator = prediction.sum().item() + target.sum().item()

    if denominator == 0:
        return 1.0

    return 2.0 * intersection / denominator


def calculate_metrics(prediction, target):
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


def predict_tta(model, image, device, overlap):
    """
    Eight combinations of flipping the three spatial axes.
    Input shape: (1, 4, D, H, W).
    """
    probability_sum = None

    for flip_d in (False, True):
        for flip_h in (False, True):
            for flip_w in (False, True):

                transformed = image

                axes = []

                if flip_d:
                    axes.append(2)
                if flip_h:
                    axes.append(3)
                if flip_w:
                    axes.append(4)

                if axes:
                    transformed = torch.flip(
                        transformed, dims=axes
                    )

                probabilities = sliding_window_inference(
                    inputs=transformed,
                    roi_size=ROI_SIZE,
                    sw_batch_size=1,
                    predictor=lambda x: torch.softmax(
                        model(x), dim=1
                    ),
                    overlap=overlap,
                    mode="gaussian",
                    sw_device=device,
                    device="cpu"
                )

                if axes:
                    probabilities = torch.flip(
                        probabilities, dims=axes
                    )

                if probability_sum is None:
                    probability_sum = probabilities
                else:
                    probability_sum += probabilities

    return probability_sum / 8.0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--data-dir", default="data/preprocessed"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split", choices=["validation", "test"],
        default="test"
    )
    parser.add_argument(
        "--output-dir", default="results/arm_e_test"
    )
    parser.add_argument("--overlap", type=float, default=0.5)

    args = parser.parse_args()

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Inference: 8-way TTA")
    print("Overlap:", args.overlap)
    print("Split:", args.split)

    manifest = pd.read_csv(args.manifest)

    dataset = BrainTumourTestDataset(
        manifest, args.data_dir, args.split
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = build_model().to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = Path(args.output_dir)
    prediction_dir = output_dir / "predictions"
    uncertainty_dir = output_dir / "uncertainty_maps"

    prediction_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_dir.mkdir(parents=True, exist_ok=True)

    patient_results = []

    with torch.inference_mode():
        for images, labels, patient_ids in loader:
            patient_id = patient_ids[0]
            target = labels[0]

            start = time.time()

            probabilities = predict_tta(
                model, images, device, args.overlap
            )

            prediction = probabilities.argmax(dim=1)[0]

            elapsed = time.time() - start

            metrics = calculate_metrics(
                prediction, target
            )

            # Predictive entropy, averaged over the 4 classes.
            entropy = -torch.sum(
                probabilities.clamp_min(1e-8)
                * probabilities.clamp_min(1e-8).log(),
                dim=1
            )[0]

            np.save(
                prediction_dir / f"{patient_id}_prediction.npy",
                prediction.numpy().astype(np.uint8)
            )

            np.save(
                uncertainty_dir / f"{patient_id}_entropy.npy",
                entropy.numpy().astype(np.float32)
            )

            patient_results.append({
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

    results_df = pd.DataFrame(patient_results)

    results_df.to_csv(
        output_dir / "arm_e_patient_results.csv",
        index=False
    )

    summary = results_df[
        ["WT_Dice", "TC_Dice", "Mean_Dice",
         "inference_time_seconds"]
    ].agg(["mean", "std"]).T

    summary.to_csv(
        output_dir / "arm_e_summary.csv"
    )

    print("\nTTA inference completed.")
    print(summary)
    print("Saved in:", output_dir)


if __name__ == "__main__":
    main()
