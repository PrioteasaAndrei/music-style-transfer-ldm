import torch
import os
from torch.utils.data import Dataset, DataLoader

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
