import os
import subprocess

import torch
import torchaudio

from src.v2.config import Config


def mp4_to_wav(mp4_path, output_folder=None):
    """
    Convert an MP4 video to a sanitized WAV audio file using FFmpeg.

    This function performs the conversion and sanitization in a single step,
    ensuring the output is in the ideal format for Whisper:
    - Sample Rate: 16000 Hz
    - Channels: 1 (Mono)
    - Codec: pcm_s16le (Signed 16-bit PCM)

    Args:
        mp4_path (str): The path to the input MP4 video file.
        output_folder (str, optional): The folder to save the WAV file. 
                                       Defaults to the current directory.

    Returns:
        str: The path to the newly created sanitized WAV file.
    """
    if not os.path.exists(mp4_path):
        raise FileNotFoundError(f"Input file not found: {mp4_path}")

    if output_folder is None:
        output_folder = os.getcwd()
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(mp4_path))[0]
    wav_path = os.path.join(output_folder, f"vid.wav")

    print(f"Converting and sanitizing {mp4_path} -> {wav_path}...")

    # --- This is the core FFmpeg command ---
    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-i', mp4_path,  # Input file
        '-vn',  # No video (strip video stream)
        '-ar', '16000',  # Audio rate: 16kHz
        '-ac', '1',  # Audio channels: 1 (mono)
        '-c:a', 'pcm_s16le',  # Audio codec: PCM signed 16-bit little-endian
        wav_path
    ]
    # ----------------------------------------

    try:
        # Run the command
        subprocess.run(
            command,
            check=True,         # Raise an error if ffmpeg fails
            capture_output=True,# Capture stdout and stderr
            text=True           # Decode output as text
        )
        print("Conversion and sanitization completed successfully.")
        return wav_path
    except FileNotFoundError:
        print("\n--- FFmpeg Error ---")
        print("`ffmpeg` command not found. Please ensure FFmpeg is installed")
        print("and that its 'bin' directory is in your system's PATH.")
        print("Download from: https://ffmpeg.org/download.html")
        raise
    except subprocess.CalledProcessError as e:
        # If ffmpeg returns a non-zero exit code, print its error output
        print("\n--- FFmpeg Error ---")
        print("FFmpeg failed to execute. Here is the error:")
        print(e.stderr)
        raise


def audio_setup(wav_path):
    """

    :param wav_path:
    :return: audio_samples
    """
    waveform, _ = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    audio_samples = waveform.squeeze(0).numpy().astype('float32')
    return audio_samples

# This function is still useful for slicing with pydub
def extract_audio_segment(audio, start_ms, end_ms, temp_path="temp_segment.wav"):
    """Extract a segment from AudioSegment and save as WAV."""
    segment = audio[start_ms:end_ms]
    segment.export(temp_path, format="wav")
    return temp_path