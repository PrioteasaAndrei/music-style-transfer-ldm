import torch
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from io import BytesIO
from config import config
import numpy as np
import sys
from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms


sys.path.append(str(Path(__file__).parent.parent))


class SpectrogramDataset(Dataset):

    def __init__(self, config):

        self.data_dir = config["data_dir"]
        self.file_name = config["file_name"]
        # Load DataFrame from .parquet file
        self.df = pd.read_parquet(os.path.join(self.data_dir, self.file_name))
        self.df.drop(columns=["title", "chunk_id"], inplace=True)
        self.df.rename(columns={"instrument": "label"}, inplace=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Convert raw image bytes to a PIL image in grayscale
        image = Image.open(BytesIO(row["spectogram"])).convert("L")
        image = image.crop((0, 0, 128, 128))
        # Convert PIL image to tensor with values in [0,1]
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension
        img_label = (image_tensor, row["label"])
        return img_label


def prepare_dataset_old(config):
    dataset = SpectrogramDataset(config)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    return train_loader, test_loader


def prepare_dataset(config):

    transform = transforms.Compose(
        [
            # add crop from 130 to 128
            # ! If the chunk size is different, this needs to be changed
            transforms.Lambda(lambda x: x.crop((0, 0, 128, 128))),  # Crop to 128x128
            transforms.Grayscale(),  # Needed because ImageFolder by default converts to RGB -> convert back
            transforms.ToTensor(),  # automatically normalizes [0,255] to [0,1]
        ]
    )

    dataset = ImageFolder(root="processed_images", transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    return train_loader, test_loader


if __name__ == "__main__":

    dataset = SpectrogramDataset(config)
    print(len(dataset))
    train_loader, test_loader = prepare_dataset(config)
    print(len(train_loader))
    print(len(test_loader))

    a = 0

    for batch in train_loader:
        a += 1
        if a > 10:
            break
        print(batch[0].shape)
        print(batch[1])

        print("--------------------------------")
