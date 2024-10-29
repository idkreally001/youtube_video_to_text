# YouTube Audio Processing and Transcription

This project provides a Python script that allows users to download audio from a YouTube video, transcribe the audio using Whisper AI, and clean the transcription for better readability. The entire process is automated, making it easy to obtain text from audio content.

## Requirements

- Python 3.6+
- yt-dlp
- whisper
- transformers
- soundfile
- numpy

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Install the required packages:
    ```bash
    pip install yt-dlp whisper transformers soundfile numpy
    ```

## Usage

1. Run the script:
    ```bash
    python your_script_name.py
    ```

2. Enter the YouTube URL when prompted.

## Script Overview

- **download_audio(youtube_url)**: Downloads audio from a YouTube video.
- **check_audio_quality(audio_path, threshold=0.01)**: Checks the audio quality.
- **split_long_audio(audio_path, max_duration=600)**: Splits long audio files into smaller segments.
- **transcribe_audio_with_eta(audio_path)**: Transcribes the audio using Whisper.
- **clean_text(text)**: Cleans the transcript text.
- **process_youtube_video(youtube_url)**: Main function to process the YouTube video.

## Helper Functions

- **retry(func, max_attempts=3, wait_time=5, *args, **kwargs)**: Retries a function if it fails.
- **sanitize_filename(filename)**: Sanitizes the filename by removing/replacing special characters.
- **is_valid_youtube_url(url)**: Validates the YouTube URL.
## License
This project is licensed under the MIT License.



