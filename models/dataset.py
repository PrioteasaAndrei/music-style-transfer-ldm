import torch
import os
from torch.utils.data import Dataset, DataLoader

class SpectrogramDataset(torch.utils.data.Dataset):
    # TODO: Implement dataset
    '''
    15 000 spectrograms with labels
    no augumantation
    no transformations
    normalized to 0-1 (not 0 255)
    > 3 instruments
    '''