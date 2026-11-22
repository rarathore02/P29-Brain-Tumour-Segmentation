import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from monai.networks.nets import UNet
from monai.losses import DiceCELoss


SEED = 42
PATCH_SIZE = 96
BATCH_SIZE = 2
PATCHES_PER_PATIENT = 2
EPOCHS = 50
LEARNING_RATE = 3e-4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BrainTumourDataset(Dataset):
    def __init__(self, manifest, data_dir, split, model_type):
        self.data_dir = Path(data_dir)
        self.model_type = model_type

        self.patients = manifest[
            manifest["split"] == split
        ]["patient_id"].tolist()

        self.split = split

    def __len__(self):
        if self.split == "train":
            return len(self.patients) * PATCHES_PER_PATIENT
        return len(self.patients)

    def load_patient(self, patient_id):
        image_path = (
            self.data_dir / self.split /
            f"{patient_id}_image.npy"
        )

        label_path = (
            self.data_dir / self.split /
            f"{patient_id}_label.npy"
        )

        image = np.load(image_path).astype(np.float32)
        label = np.load(label_path).astype(np.int64)

        return image, label

    def sample_patch(self, image, label):
        _, depth, height, width = image.shape

        for _ in range(20):
            if random.random() < 2 / 3 and np.any(label > 0):
                tumour_voxels = np.argwhere(label > 0)
                center = tumour_voxels[
                    random.randrange(len(tumour_voxels))
                ]

                starts = [
                    max(0, min(int(center[i]) - PATCH_SIZE // 2,
                               [depth, height, width][i] - PATCH_SIZE))
                    for i in range(3)
                ]
            else:
                starts = [
                    random.randint(0, max(0, dim - PATCH_SIZE))
                    for dim in (depth, height, width)
                ]

            d, h, w = starts

            image_patch = image[
                :, d:d+PATCH_SIZE,
                h:h+PATCH_SIZE,
                w:w+PATCH_SIZE
            ]

            label_patch = label[
                d:d+PATCH_SIZE,
                h:h+PATCH_SIZE,
                w:w+PATCH_SIZE
            ]

            if image_patch.shape[1:] == (
                PATCH_SIZE, PATCH_SIZE, PATCH_SIZE
            ):
                return image_patch, label_patch

        raise RuntimeError("Could not sample a valid patch")

    def augment(self, image, label):
        for axis in (1, 2, 3):
            if random.random() < 0.5:
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis - 1).copy()

        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            shift = random.uniform(-0.1, 0.1)
            image = image * scale + shift

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", default="data/preprocessed")
    parser.add_argument("--model", choices=["A", "B"], required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--output-dir", default="checkpoints")

    args = parser.parse_args()

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Model:", args.model)
    print("Seed:", SEED)

    manifest = pd.read_csv(args.manifest)

    train_dataset = BrainTumourDataset(
        manifest, args.data_dir, "train", args.model
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = build_model(args.model).to(device)

    if args.model == "A":
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE
        )
    else:
        criterion = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-5
        )

    output_dir = Path(args.output_dir) / f"model_{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            if args.model == "A":
                labels = labels

            else:
                labels = labels.unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start

        history.append({
            "epoch": epoch + 1,
            "train_loss": average_loss,
            "epoch_time_seconds": epoch_time
        })

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Loss: {average_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": average_loss,
            "seed": SEED,
            "model_type": args.model
        }

        torch.save(checkpoint, output_dir / "latest.pth")

        pd.DataFrame(history).to_csv(
            output_dir / "training_history.csv",
            index=False
        )

    torch.save(
        checkpoint,
        output_dir / "final.pth"
    )

    print("\nTraining completed.")
    print("Checkpoints:", output_dir)


if __name__ == "__main__":
    main()
