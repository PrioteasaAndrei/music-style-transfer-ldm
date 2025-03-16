"""
File to create a tabular style dataset of the spectograms (images) and their corresponding labels.
"""

from io import BytesIO
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import os
import numpy as np
import pandas as pd
from PIL import Image

# import parquet as pq
import matplotlib.pyplot as plt  # added import for image saving
from data.audio_processor import AudioPreprocessor


def build_dataset_df(save_to_file: bool = True, save_path: str = "downloads/processed_dataset.parquet"):
    """
    Build a dataset DataFrame from the processed dataset
    :return: DataFrame with columns ['spectogram', 'instrument', 'title', 'chunk_id']
    """
    CHUNK_SIZE = 3  # seconds
    # 1800 is 30 minutes (lowest duration in current videofiles -> all instruments have same duration then)
    MAX_DURATION = 1800  # maximum duration per file in seconds, None for no limit
    new_dataset = pd.DataFrame(columns=["spectogram", "instrument", "title", "chunk_id"])
    AudioPreprocessor = AudioPreprocessor()

    # 1) read in all mp3 files from the audio directory
    mp3_dir = Path("downloads")
    mp3_files = list(mp3_dir.rglob("*.mp3"))

    for mp3_file in mp3_files:
        print(f"Processing file: {mp3_file}")  # print file being processed
        # Load the audio file
        audio, sr = AudioPreprocessor.load_audio(mp3_file)
        # Trim silence from front and back
        audio = AudioPreprocessor.trim_silence(audio)
        # Split the audio into chunks
        chunk_size = int(CHUNK_SIZE * sr)  # get #samples in 5 seconds
        # Iterate over the audio in chunks
        for i in range(0, len(audio), chunk_size):
            # Check if we have reached the maximum duration for the file
            if MAX_DURATION is not None and (i / sr) >= MAX_DURATION:
                break
            chunk = audio[i : i + chunk_size]
            # Pad with zeros if the chunk is too short
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode="constant")

            # Extract features
            spectogram = AudioPreprocessor.get_mel_spectogram(chunk, sr, n_mels=128)

            #  Just testing
            image = AudioPreprocessor.mel_spectogram_to_grayscale_image(spectogram)
            image = AudioPreprocessor.get_raw_image_bytes(image)

            # Add to the dataset using pd.concat instead of deprecated append
            new_row = pd.DataFrame(
                [{"spectogram": image, "instrument": mp3_file.parent.name, "title": mp3_file.stem, "chunk_id": i}]
            )
            new_dataset = pd.concat([new_dataset, new_row], ignore_index=True)
            # break
        print(f"Finished processing for file: {mp3_file}")  # print saving info
        # break
    if save_to_file:
        new_dataset.to_parquet(save_path)
        print(f"Saved dataset to '{save_path}'")

    return new_dataset


def build_dataset_folder_structure(
    mp3_dir="downloads", output_root="processed_images", chunk_size_sec=3, max_duration=1800, n_mels=128
):
    """
    Process audio files in the mp3_dir, generate spectrogram images,
    and save them into folders (named after instrument labels) under output_root.

    :param mp3_dir: Directory containing the audio files.
    :param output_root: Root directory to save processed spectrogram images.
    :param chunk_size_sec: Duration of each audio chunk in seconds.
    :param max_duration: Maximum duration to process per file (in seconds).
    """
    ap = AudioPreprocessor()
    mp3_dir = Path(mp3_dir)
    mp3_files = list(mp3_dir.rglob("*.mp3"))

    for mp3_file in mp3_files:
        # Use the parent directory's name as the instrument label.
        instrument = mp3_file.parent.name
        instrument_dir = Path(output_root) / instrument
        instrument_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing file: {mp3_file}")
        # Load and preprocess audio.
        audio, sr = ap.load_audio(mp3_file)
        audio = ap.trim_silence(audio)

        # Calculate the number of samples per chunk.
        chunk_size = int(chunk_size_sec * sr)

        for chunk_idx, i in enumerate(range(0, len(audio), chunk_size)):
            if max_duration is not None and (i / sr) >= max_duration:
                break
            chunk = audio[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode="constant")

            spectrogram = ap.get_mel_spectogram(chunk, sr, n_mels=n_mels)
            image_pil = ap.mel_spectogram_to_grayscale_image(spectrogram)

            filename = f"{mp3_file.stem}_chunk{chunk_idx}.png"
            image_path = instrument_dir / filename
            image_pil.save(image_path)
            print(f"Saved image: {image_path}")
        print(f"Finished processing file: {mp3_file}")


if __name__ == "__main__":
    build_dataset_folder_structure()
