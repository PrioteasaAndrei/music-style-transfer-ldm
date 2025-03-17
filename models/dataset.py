import random
import torch
import os
from os import PathLike
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import csv
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from io import BytesIO
from config import config
import numpy as np
import sys
from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import (
    IMG_EXTENSIONS,
    has_file_allowed_extension,
    find_classes,
    make_dataset as original_make_dataset,
)
from torchvision import transforms, datasets


sys.path.append(str(Path(__file__).parent.parent))


class SpectrogramDataset(Dataset):

    def __init__(self, config):
        super(SpectrogramDataset, self).__init__()
        self.image_dir_path = config["processed_spectograms_dataset_folderpath"]
        self.data = datasets.ImageFolder(root=self.image_dir_path, transform=self._get_transform())

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y

    def __len__(self):
        return len(self.data)

    def _get_transform(self):
        """
        Define the transformations to be applied to the images.
        :return: Transformations
        """
        return transforms.Compose(
            [
                # add crop from 130 to 128
                # ! If the chunk size is different, this needs to be changed
                transforms.Lambda(lambda x: x.crop((0, 0, 128, 128))),  # Crop to 128x128
                transforms.Grayscale(),  # Needed because ImageFolder by default converts to RGB -> convert back
                transforms.ToTensor(),  # Automatically normalizes [0,255] to [0,1]
            ]
        )


# class MultiFolderSpectrogramPairDataset(Dataset):
#     def __init__(self, root_folder, num_pairs, transform=None):
#         """
#         Args:
#             root_folder (str): Path to the root directory containing subfolders for each label.
#             num_pairs (int): Number of (sample, sample) pairs to generate.
#             transform: Transformation to apply to the images.
#         """
#         self.num_pairs = num_pairs
#         self.transform = transform if transform is not None else self._get_transform()

#         # Get the list of subfolder paths
#         self.subfolders = [
#             os.path.join(root_folder, folder)
#             for folder in os.listdir(root_folder)
#             if os.path.isdir(os.path.join(root_folder, folder))
#         ]

#         # Create a separate ImageFolder dataset for each subfolder.
#         # We'll use the subfolder name as the label.
#         self.datasets = {}
#         for folder in self.subfolders:
#             label = os.path.basename(folder)
#             self.datasets[label] = datasets.ImageFolder(root=folder, transform=self.transform)

#         self.labels = list(self.datasets.keys())

#         if len(self.labels) < 2:
#             raise ValueError("Need at least two classes to form pairs with different labels.")

#     def __len__(self):
#         return self.num_pairs

#     def __getitem__(self, idx):
#         # Randomly select two distinct labels
#         label1, label2 = random.sample(self.labels, 2)
#         dataset1 = self.datasets[label1]
#         dataset2 = self.datasets[label2]

#         # Randomly sample one image from each dataset.
#         img1, _ = dataset1[random.randint(0, len(dataset1) - 1)]
#         img2, _ = dataset2[random.randint(0, len(dataset2) - 1)]

#         return (img1, label1), (img2, label2)

#     def _get_transform(self):
#         """
#         Define the transformations to be applied to the images.
#         :return: Transformations
#         """
#         return transforms.Compose(
#             [
#                 # add crop from 130 to 128
#                 # ! If the chunk size is different, this needs to be changed
#                 transforms.Lambda(lambda x: x.crop((0, 0, 128, 128))),  # Crop to 128x128
#                 transforms.Grayscale(),  # Needed because ImageFolder by default converts to RGB -> convert back
#                 transforms.ToTensor(),  # Automatically normalizes [0,255] to [0,1]
#             ]
#         )


class ImageFolderNoSubdirs(ImageFolder):
    """
    Custom ImageFolder that, if no subdirectories are found in a given directory,
    treats the directory itself as a single class.
    """

    def find_classes(self, directory):
        # List directories inside the given directory.
        subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        if len(subdirs) == 0:
            # If no subdirectories, use the directory name as the class.
            class_name = os.path.basename(os.path.normpath(directory))
            return [class_name], {class_name: 0}
        else:
            # Otherwise, follow the normal ImageFolder behavior.
            return super().find_classes(directory)

    @staticmethod
    def make_dataset(
        directory: Union[str, PathLike],
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of the form (path_to_sample, class).

        Modified so that if the directory itself is the class folder
        (i.e. its basename equals the target class) then files are searched in that directory.
        """

        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_idx' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            # If the current directory's basename equals the target class,
            # assume the images are stored directly in 'directory'
            if os.path.basename(os.path.normpath(directory)) == target_class:
                target_dir = directory
            else:
                target_dir = os.path.join(directory, target_class)
            # If target_dir does not exist, and we are in the case, use directory.
            if not os.path.isdir(target_dir):
                if os.path.basename(os.path.normpath(directory)) == target_class:
                    target_dir = directory
                else:
                    continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = (path, class_index)
                        instances.append(item)
                        available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes and not allow_empty:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += (
                    f"Supported extensions are: "
                    f"{extensions if isinstance(extensions, str) else ', '.join(extensions)}"
                )
            raise FileNotFoundError(msg)

        return instances


class SpectrogramPairDataset(Dataset):
    def __init__(self, root_folder, pairing_file, transform=None):
        """
        Args:
            root_folder (str): Root directory with subfolders (each for one label).
            pairing_file (str): Path to the CSV file with predetermined pairings.
            transform: Transformations to apply to each image.
        """
        self.root_folder = root_folder
        self.pairing_file = pairing_file
        self.transform = transform if transform is not None else self._get_transform()

        # Load the precomputed pairs from the CSV file.
        # Each row in the CSV should contain: label1, idx1, label2, idx2
        self.pairs = []
        with open(self.pairing_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                # Convert index strings to integers.
                self.pairs.append((row[0], int(row[1]), row[2], int(row[3])))

        # Build a dictionary of ImageFolder datasets keyed by label.
        self.datasets = {}
        # Sorting the subfolders ensures deterministic order.
        for folder in sorted(os.listdir(root_folder)):
            folder_path = os.path.join(root_folder, folder)
            if os.path.isdir(folder_path):
                # Assume the folder name is the label.
                self.datasets[folder] = ImageFolderNoSubdirs(root=folder_path, transform=self.transform)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        # Use the index to get the predetermined pairing.
        label1, idx1, label2, idx2 = self.pairs[index]
        img1, _ = self.datasets[label1][idx1]
        img2, _ = self.datasets[label2][idx2]
        return (img1, label1), (img2, label2)

    @classmethod
    def _get_transform(cls):
        """
        Define the transformations to be applied to the images.
        :return: Transformations
        """
        return transforms.Compose(
            [
                # add crop from 130 to 128
                # ! If the chunk size is different, this needs to be changed
                transforms.Lambda(lambda x: x.crop((0, 0, 128, 128))),  # Crop to 128x128
                transforms.Grayscale(),  # Needed because ImageFolder by default converts to RGB -> convert back
                transforms.ToTensor(),  # Automatically normalizes [0,255] to [0,1]
            ]
        )

    @classmethod
    def generate_pairings(cls, root_folder, output_file_path="spectrogram_pair_dataset_pairings.csv", num_pairs=15000):
        """
        Generates a CSV file containing the predetermined pairings.

        Args:
            root_folder (str): Root directory with subfolders for each label.
            output_file (str): Path where the CSV file will be saved.
            num_pairs (int): Number of pairs to generate.
        """
        # List of labels (i.e. subfolder names) sorted deterministically.
        labels = sorted(
            [folder for folder in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, folder))]
        )

        if len(labels) < 2:
            raise ValueError("Need at least two classes to form pairs.")

        # Create ImageFolder datasets for each label.
        datasets_dict = {}
        for label in labels:
            folder_path = os.path.join(root_folder, label)
            datasets_dict[label] = ImageFolderNoSubdirs(root=folder_path, transform=cls._get_transform())

        pairs = []
        # We precompute the pairs and save them as a file. Like this the future sampling is deterministic.
        rng = np.random.RandomState(42)
        for _ in range(num_pairs):
            # Randomly select two distinct labels.
            label1, label2 = rng.choice(labels, size=2, replace=False)
            ds1, ds2 = datasets_dict[label1], datasets_dict[label2]
            # Randomly select indices within each dataset.
            idx1 = rng.randint(0, len(ds1))
            idx2 = rng.randint(0, len(ds2))
            pairs.append((label1, idx1, label2, idx2))

        # Write the pairs to a CSV file.
        with open(output_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            for pair in pairs:
                writer.writerow(pair)
        print(f"Pairings saved to {output_file_path}")


def prepare_dataset(config):
    dataset = SpectrogramDataset(config)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    return train_loader, test_loader


def prepare_dataset_other(config):

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
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    return train_loader, test_loader


if __name__ == "__main__":

    print("Testing SpectrogramDataset\n")
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

    print("Testing SpectrogramPairDataset\n")
    SpectrogramPairDataset.generate_pairings(
        config["processed_spectograms_dataset_folderpath"], config["pairing_file_path"]
    )
    dataset = SpectrogramPairDataset(config["processed_spectograms_dataset_folderpath"], config["pairing_file_path"])
    print(len(dataset))

    print(dataset[0])
    print(dataset[1])
    
