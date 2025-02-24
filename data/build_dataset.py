'''
File to create a tabular style dataset of the spectograms (images) and their corresponding labels.
'''
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import os
import numpy as np
import pandas as pd
# import parquet as pq
import librosa
import matplotlib.pyplot as plt  # added import for image saving

from data.audio_processor import AudioPreprocessor


CHUNK_SIZE = 5  # seconds
new_dataset = pd.DataFrame(columns=['spectogram', 'instrument', 'title', 'chunk_id'])
AudioPreprocessor = AudioPreprocessor()


# 1) read in all mp3 files from the audio directory
mp3_dir = Path('downloads')
mp3_files = list(mp3_dir.rglob('*.mp3'))

for mp3_file in mp3_files:
    print(f"Processing file: {mp3_file}")  # print file being processed
    # Load the audio file
    audio, sr = AudioPreprocessor.load_audio(mp3_file)
    # Trim silence from front and back
    audio = AudioPreprocessor.trim_silence(audio)
    # Split the audio into chunks
    chunk_size = int(CHUNK_SIZE * sr) # get #samples in 5 seconds
    # Iterate over the audio in chunks
    for i in range(0, len(audio), chunk_size):
        # print(f"  Processing chunk starting at sample {i}")  # print chunk info
        chunk = audio[i:i + chunk_size]
        # 5) Extract features
        spectogram = AudioPreprocessor.extract_features(chunk, sr)
        # 6) Add to the dataset using pd.concat instead of deprecated append
        new_row = pd.DataFrame([{
            'spectogram': spectogram, 
            'instrument': mp3_file.parent.name,
            'title': mp3_file.stem, 
            'chunk_id': i
        }])
        new_dataset = pd.concat([new_dataset, new_row], ignore_index=True)
        break
    break
new_dataset.to_pickle('data/dataset/processed_dataset.pkl')
# new_dataset.to_parquet('data/dataset/processed_dataset.parquet')
# print(f"Saved dataset for file: {mp3_file}")  # print saving info
