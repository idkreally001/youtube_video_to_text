import os
import time
import yt_dlp
import whisper
from transformers import pipeline
import warnings
import re
import math
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")  # Suppress minor warnings for clarity

# Create audio folder relative to the script location
audio_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
os.makedirs(audio_folder, exist_ok=True)

# Helper Function: Retry mechanism
def retry(func, max_attempts=3, wait_time=5, *args, **kwargs):
    attempt = 0
    while attempt < max_attempts:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            attempt += 1
            if attempt < max_attempts:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("All attempts failed.")
                return None

# Helper Function: Sanitize filename by removing/replacing special characters
def sanitize_filename(filename):
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)  # removes problematic characters
    filename = filename.replace("：", "")  # specifically remove the full-width colon
    return filename

# Helper Function: Check audio quality (basic volume threshold)
def check_audio_quality(audio_path, threshold=0.01):
    data, _ = sf.read(audio_path)
    volume = np.sqrt(np.mean(data**2))
    if volume < threshold:
        print("Warning: Audio quality appears low. This may impact transcription accuracy.")
    else:
        print("Audio quality check passed.")

# Helper Function: Validate YouTube URL
def is_valid_youtube_url(url):
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    )
    return re.match(youtube_regex, url) is not None

# Step 1: Download audio from YouTube with sanitized filename
def download_audio(youtube_url):
    try:
        print("Downloading audio from YouTube...")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(audio_folder, 'audio.wav'),  # Use a fixed filename
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            print("Audio downloaded and saved as:", os.path.join(audio_folder, 'audio.wav'))
            return os.path.join(audio_folder, 'audio.wav'), info['id']
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None, None

# Step 2: Check audio duration and optionally split if too long
def split_long_audio(audio_path, max_duration=600):
    data, sample_rate = sf.read(audio_path)
    duration = len(data) / sample_rate
    if duration > max_duration:
        print(f"Warning: Audio duration ({duration / 60:.2f} minutes) exceeds {max_duration / 60} minutes.")
        print("Splitting into smaller segments for processing...")
        segments = []
        num_segments = math.ceil(duration / max_duration)
        for i in range(num_segments):
            start = int(i * max_duration * sample_rate)
            end = int((i + 1) * max_duration * sample_rate)
            segment_path = f"{audio_path}_segment_{i + 1}.wav"
            sf.write(segment_path, data[start:end], sample_rate)
            segments.append(segment_path)
        return segments
    else:
        return [audio_path]

# Step 3: Transcribe audio with Whisper and show estimated time
def transcribe_audio_with_eta(audio_path):
    print("Transcribing audio... This may take a few minutes.")
    model = whisper.load_model("medium").to("cuda")  # Ensure GPU usage
    start_time = time.time()
    # Transcribe
    result = model.transcribe(audio_path)
    end_time = time.time()
    actual_time = end_time - start_time
    print(f"Transcription completed in {round(actual_time, 2)} seconds.")
    return result["text"]

# Step 4: Clean text
def clean_text(text):
    try:
        print("Cleaning up the transcript text...")
        cleaner = pipeline("text2text-generation", model="t5-small", device=0)  # Specify device for GPU usage
        clean_text = cleaner(f"Fix grammar and punctuation: {text}", max_length=500)
        print("Text cleaning completed!")
        return clean_text[0]['generated_text']
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return text

# Main function to process YouTube video
def process_youtube_video(youtube_url):
    start_time = time.time()
    # Step 1: Validate YouTube URL
    if not is_valid_youtube_url(youtube_url):
        print("Error: Invalid YouTube URL. Please provide a valid link.")
        return
    # Step 2: Download audio
    audio_file, video_id = retry(download_audio, 3, 5, youtube_url)
    if audio_file is None:
        print("Exiting process due to download error.")
        return
    # Wait for user confirmation to proceed to next step
    input("Audio downloaded successfully. Press Enter to proceed to audio quality check...")
    # Step 3: Check audio quality and duration
    check_audio_quality(audio_file)
    audio_segments = split_long_audio(audio_file)
    # Wait for user confirmation to proceed to transcription
    input("Audio quality checked. Press Enter to proceed to transcription...")
    # Step 4: Transcribe each audio segment
    full_transcript = ""
    for segment in audio_segments:
        transcript = retry(transcribe_audio_with_eta, 3, 5, segment)
        if transcript is None:
            print("Exiting process due to transcription error.")
            return
        full_transcript += transcript + "\n"
    # Wait for user confirmation to proceed to text cleaning
    input("Transcription completed. Press Enter to proceed to text cleaning...")
    # Step 5: Clean transcript text
    cleaned_text = retry(clean_text, 3, 5, full_transcript)
    # Save the cleaned transcript
    try:
        transcript_file = os.path.join(audio_folder, 'transcript.txt')
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(full_transcript)
        print(f"Transcript saved as '{transcript_file}'")
    except Exception as e:
        print(f"Error saving transcript file: {e}")

    # Save cleaned transcript
    try:
        cleaned_transcript_file = os.path.join(audio_folder, 'cleaned_transcript.txt')
        with open(cleaned_transcript_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        print(f"Cleaned transcript saved as '{cleaned_transcript_file}'")
    except Exception as e:
        print(f"Error saving cleaned transcript file: {e}")

    # Display total time taken
    end_time = time.time()
    print(f"Total time taken: {round(end_time - start_time, 2)} seconds")

# Example usage:
youtube_url = input("Enter the YouTube URL: ")
process_youtube_video(youtube_url)
