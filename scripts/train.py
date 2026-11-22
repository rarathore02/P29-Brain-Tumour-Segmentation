
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference


SEED = 42
PATCH_SIZE = 96
BATCH_SIZE = 2
PATCHES_PER_PATIENT = 2
LEARNING_RATE = 3e-4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_volume(image, label):
    _, depth, height, width = image.shape

    pad_d = max(0, PATCH_SIZE - depth)
    pad_h = max(0, PATCH_SIZE - height)
    pad_w = max(0, PATCH_SIZE - width)

    if pad_d or pad_h or pad_w:
        image = np.pad(
            image,
            (
                (0, 0),
                (0, pad_d),
                (0, pad_h),
                (0, pad_w)
            ),
            mode="constant"
        )

        label = np.pad(
            label,
            (
                (0, pad_d),
                (0, pad_h),
                (0, pad_w)
            ),
            mode="constant"
        )

    return image, label


class BrainTumourDataset(Dataset):
    def __init__(self, manifest, data_dir, split, model_type):
        self.data_dir = Path(data_dir)
        self.split = split
        self.model_type = model_type

        self.patients = manifest[
            manifest["split"] == split
        ]["patient_id"].tolist()

    def __len__(self):
        if self.split == "train":
            return len(self.patients) * PATCHES_PER_PATIENT

        return len(self.patients)

    def load_patient(self, patient_id):
        folder = self.data_dir / self.split

        image = np.load(
            folder / f"{patient_id}_image.npy"
        ).astype(np.float32)

        label = np.load(
            folder / f"{patient_id}_label.npy"
        ).astype(np.int64)

        return pad_volume(image, label)

    def sample_patch(self, image, label):
        _, depth, height, width = image.shape
        dimensions = (depth, height, width)

        if random.random() < 2 / 3 and np.any(label > 0):
            tumour_voxels = np.argwhere(label > 0)

            center = tumour_voxels[
                random.randrange(len(tumour_voxels))
            ]

            starts = [
                max(
                    0,
                    min(
                        int(center[i]) - PATCH_SIZE // 2,
                        dimensions[i] - PATCH_SIZE
                    )
                )
                for i in range(3)
            ]

        else:
            starts = [
                random.randint(0, dimensions[i] - PATCH_SIZE)
                for i in range(3)
            ]

        d, h, w = starts

        image_patch = image[
            :, d:d + PATCH_SIZE,
            h:h + PATCH_SIZE,
            w:w + PATCH_SIZE
        ]

        label_patch = label[
            d:d + PATCH_SIZE,
            h:h + PATCH_SIZE,
            w:w + PATCH_SIZE
        ]

        return image_patch, label_patch

    def augment(self, image, label):
        for axis in (1, 2, 3):
            if random.random() < 0.5:
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis - 1).copy()

        if random.random() < 0.5:
            image = (
                image * random.uniform(0.9, 1.1)
                + random.uniform(-0.1, 0.1)
            )

        if random.random() < 0.3:
            noise = np.random.normal(
                0, 0.05, image.shape
            ).astype(np.float32)

            image = image + noise

        return image, label

    def __getitem__(self, index):
        if self.split == "train":
            patient_index = index // PATCHES_PER_PATIENT
        else:
            patient_index = index

        patient_id = self.patients[patient_index]
        image, label = self.load_patient(patient_id)

        if self.split == "train":
            image, label = self.sample_patch(image, label)

            if self.model_type == "B":
                image, label = self.augment(image, label)

        return (
            torch.from_numpy(image.copy()).float(),
            torch.from_numpy(label.copy()).long()
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


def validate(model, loader, device):
    model.eval()

    wt_scores = []
    tc_scores = []

    with torch.inference_mode():
        for images, labels in loader:
            original_shape = labels.shape[-3:]

            outputs = sliding_window_inference(
                inputs=images,
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
                mode="gaussian",
                sw_device=device,
                device="cpu"
            )

            # Remove padding before calculating metrics.
            outputs = outputs[
                :,
                :,
                :original_shape[0],
                :original_shape[1],
                :original_shape[2]
            ]

            predictions = outputs.argmax(dim=1)[0]
            target = labels[0]

            pred_wt = predictions > 0
            true_wt = target > 0

            pred_tc = (
                (predictions == 1) | (predictions == 3)
            )

            true_tc = (
                (target == 1) | (target == 3)
            )

            wt_scores.append(dice_score(pred_wt, true_wt))
            tc_scores.append(dice_score(pred_tc, true_tc))

    wt_mean = float(np.mean(wt_scores))
    tc_mean = float(np.mean(tc_scores))

    return wt_mean, tc_mean, (wt_mean + tc_mean) / 2


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", default="data/preprocessed")
    parser.add_argument("--model", choices=["A", "B"], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-dir", default="checkpoints")

    args = parser.parse_args()

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Model:", args.model)
    print("Random seed:", SEED)

    manifest = pd.read_csv(args.manifest)

    train_dataset = BrainTumourDataset(
        manifest, args.data_dir, "train", args.model
    )

    val_dataset = BrainTumourDataset(
        manifest, args.data_dir, "validation", args.model
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = build_model(args.model).to(device)

    if args.model == "A":
        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )

        scheduler = None

    else:
        criterion = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=1.0,
            lambda_ce=1.0
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-5
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

    output_dir = Path(args.output_dir) / f"model_{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_mean_dice = -1.0

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            if args.model == "B":
                labels = labels.unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        wt_dice, tc_dice, mean_dice = validate(
            model, val_loader, device
        )

        if scheduler is not None:
            scheduler.step(mean_dice)

        epoch_time = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch + 1,
            "train_loss": average_loss,
            "val_WT_Dice": wt_dice,
            "val_TC_Dice": tc_dice,
            "val_mean_Dice": mean_dice,
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time
        }

        history.append(record)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Loss: {average_loss:.4f} | "
            f"WT: {wt_dice:.4f} | "
            f"TC: {tc_dice:.4f} | "
            f"Mean: {mean_dice:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": average_loss,
            "val_WT_Dice": wt_dice,
            "val_TC_Dice": tc_dice,
            "val_mean_Dice": mean_dice,
            "history": history,
            "seed": SEED,
            "model_type": args.model
        }

        torch.save(checkpoint, output_dir / "latest.pth")

        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            torch.save(checkpoint, output_dir / "best.pth")
            print("Best checkpoint updated.")

        pd.DataFrame(history).to_csv(
            output_dir / "training_history.csv",
            index=False
        )

    print("\nTraining completed.")
    print("Best validation mean Dice:", best_mean_dice)
    print("Checkpoints saved in:", output_dir)


if __name__ == "__main__":
    main()
