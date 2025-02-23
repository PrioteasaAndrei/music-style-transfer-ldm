import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
from huggingface_hub import login
from utils.env_utils import load_local_env, get_env_variable
from typing import Literal


def authenticate():
    """
    Authenticate with Hugging Face using the HF_TOKEN from the .env.local file.
    """
    load_local_env()
    login(get_env_variable("HF_TOKEN"))


def download_hf_dataset(dataset_name, split: Literal["train", "validation", "test"], **kwargs):
    """
    Download a specific split from an HF dataset.
    :param dataset_name: The dataset repository name.
    :param split: The dataset split to download ("train" or "test").
    :param kwargs: Additional keyword arguments for load_dataset.
    :return: The downloaded dataset split.
    """
    return load_dataset(dataset_name, split=split, **kwargs)


def construct_hf_dataset():
    """
    Download dataset and filter it on the fly.
    :return: The filtered dataset containing piano samples.
    """

    # dataset = load_dataset("benjamin-paine/free-music-archive-large", split="train", streaming=True)
    dataset = load_dataset("benjamin-paine/free-music-archive-small", split="train", streaming=True)

    total_samples = 100
    num_samples = 0
    
    for sample in dataset:
        # Filter samples
        if "piano" in sample["tags"]: # type: ignore
            # can only contain 1 tag
            if len(sample["tags"]) == 1: # type: ignore
                print(sample)
                num_samples += 1
        
        # If we have enough samples, break
        if num_samples == total_samples:
            break


if __name__ == "__main__":
    # Example usage:
    authenticate()
    ds = construct_hf_dataset()
    print(ds)
