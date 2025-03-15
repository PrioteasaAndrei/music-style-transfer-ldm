import torch
import os
from torch.utils.data import Dataset, DataLoader

# Additional imports for SpectrogramDatasetWithTransform 
import pandas as pd
from io import BytesIO
from PIL import Image
import torchvision.transforms as T

class SpectrogramDatasetWithTransform(Dataset):
    # TODO: Implement dataset
    '''
    15 000 spectrograms with labels
    no augumantation
    no transformations
    normalized to 0-1 (not 0 255)
    > 3 instruments
    255 x 255
    '''
    
    def __init__(self, data_dir: str = "../downloads", file_name: str = "processed_dataset.parquet"):
        """Read the dataset from an .parquet file into a DataFrame

        :param data_dir: Path to the directory containing the .parquet file
        :param file_name: Name of the .parquet file
        """
        self.data_dir = data_dir
        self.file_name = file_name
        # Load DataFrame from .parquet file
        self.df = pd.read_parquet(os.path.join(self.data_dir, self.file_name))
        self.df.drop(columns=["title", "chunk_id"], inplace=True)
        self.df.rename(columns={"instrument": "label"}, inplace=True)
        self.transform = T.ToTensor()  # convert PIL image to tensor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Convert raw image bytes to a PIL image in grayscale
        image = Image.open(BytesIO(row['spectogram'])).convert('L')
        # Convert image to tensor with values in [0,1]
        image_tensor = self.transform(image)
        img_label = (image_tensor, row['label'])
        return img_label



class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000):
        """
        Initialize dummy spectrogram dataset with random images
        Args:
            num_samples: Number of random spectrograms to generate
        """
        self.num_samples = num_samples
        # Generate random spectrograms between -1 and 1 to match tanh output
        self.data = torch.rand((num_samples, 1, 256, 256)) * 2 - 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            spectrogram: Random spectrogram of shape [1, 256, 256] 
                        with values in [-1, 1]
        """
        return self.data[idx]
    


def prepare_dataset(config):
    dataset = SpectrogramDataset()

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    return train_loader, test_loader
