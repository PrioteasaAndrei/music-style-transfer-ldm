'''
File to create a tabular style dataset of the spectograms (images) and their corresponding labels.
'''
from io import BytesIO
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
# 1800 is 30 minutes (lowest duration in current videofiles -> all instruments have same duration then)
MAX_DURATION = 1800  # maximum duration per file in seconds, None for no limit
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
        # Check if we have reached the maximum duration for the file
        if MAX_DURATION is not None and (i / sr) >= MAX_DURATION:
            break
        chunk = audio[i:i + chunk_size]
        # Pad with zeros if the chunk is too short
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            
        # Extract features
        spectogram = AudioPreprocessor.get_mel_spectogram(chunk, sr)
        
        #  Just testing
        image = AudioPreprocessor.mel_spectogram_to_grayscale_image(spectogram)
        image = AudioPreprocessor.get_raw_image_bytes(image)
        
        # Add to the dataset using pd.concat instead of deprecated append
        new_row = pd.DataFrame([{
            'spectogram': image, 
            'instrument': mp3_file.parent.name,
            'title': mp3_file.stem, 
            'chunk_id': i
        }])
        new_dataset = pd.concat([new_dataset, new_row], ignore_index=True)
        # break
    print(f"Finished processing for file: {mp3_file}")  # print saving info
    # break

new_dataset.to_parquet('downloads/processed_dataset.parquet')
print("Saved dataset to 'downloads/processed_dataset.parquet'")