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

    # TODO: Remove num_workers=0, so far it doesnt work with multiprocessing
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    return train_loader, test_loader
