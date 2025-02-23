import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import yt_dlp as ytdlp
from typing import List

class AudioDownloader:
    def __init__(self, output_path='downloads', codec='mp3'):
        """
        Initialize AudioDownloader.  
        :param output_path: Directory to save downloaded audio.  
        :param codec: Audio codec to convert to (e.g., 'mp3').  
        """
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.codec = codec

    def download_audio(self, youtube_url: str, filename=None) -> str:
        """
        Downloads the audio stream from the provided YouTube URL using youtube-dlp.  
        Documentation: https://github.com/yt-dlp/yt-dlp?tab=readme-ov-file#format-selection  
        :param youtube_url: URL of the YouTube video.  
        :param filename: Desired filename (with extension). If None, uses video's title.  
        :return: Path to the downloaded audio file.  
        """
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(self.output_path, '%(title)s.%(ext)s') if filename is None 
                         else os.path.join(self.output_path, filename),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': self.codec,
                'preferredquality': '192',
            }],
        }
        with ytdlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            if filename is None:
                filename = os.path.join(self.output_path, f"{info.get('title', 'audio')}.{self.codec}")
            return filename

    def download_from_file(self, filepath: str) -> List[str]:
        """
        Downloads audio from multiple YouTube URLs stored in a text file.
        :param filepath: Path to text file containing YouTube URLs (one per line)
        :return: List of paths to downloaded audio files
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"URL file not found: {filepath}")
            
        downloaded_files = []
        failed_urls = []
        
        with open(filepath, 'r') as file:
            for line_num, url in enumerate(file, 1):
                url = url.strip()
                if not url or url.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                try:
                    audio_path = self.download_audio(url)
                    downloaded_files.append(audio_path)
                    print(f"Successfully downloaded: {url} -> {audio_path}")
                except Exception as e:
                    failed_urls.append((url, str(e)))
                    print(f"Failed to download {url}: {str(e)}")
        
        if failed_urls:
            print("\nFailed downloads:")
            for url, error in failed_urls:
                print(f"- {url}: {error}")
        
        return downloaded_files

# Example usage in main:
def main():
    downloader = AudioDownloader(output_path='downloads', codec='mp3')
    
    # Single URL download
    # youtube_url = 'https://www.youtube.com/watch?v=hQncT4Hswhw'
    # audio_file = downloader.download_audio(youtube_url)
    
    # Multiple URLs from file
    downloaded_files = downloader.download_from_file('youtube_urls.txt')
    print(f"\nTotal files downloaded: {len(downloaded_files)}")