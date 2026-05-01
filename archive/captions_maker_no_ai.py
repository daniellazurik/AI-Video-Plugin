# test_transcriber_captions.py (with Smart Caption Grouping)
import torch
import torchaudio
import os
import time

# Import your upgraded, powerful transcriber class
from src.v1.transcriber import FasterWhisperTranscriber
# We need the 'Word' type for type hinting
from faster_whisper.transcribe import Word
from typing import List

print("--- Starting Transcriber Test with Smart Caption Generation ---")

# --- Configuration ---
WAV_PATH = "D:\\via\\mp3s\\vid3_sanitized.wav"
MODEL_DOWNLOAD_PATH = "D:\\downloaded_models"
MODEL_NAME = "ivrit-ai/whisper-large-v3-ct2"

# --- NEW: More advanced configuration for SRT Caption Generation ---
CAPTIONS_CONFIG = {
    "output_srt_file": f"D:\\via\\output\\captions.srt",
    # We now use min/max words to create flexible-length captions
    "min_words_per_line": 2,
    "max_words_per_line": 4  # A hard limit to prevent overly long lines
}


# --------------------------------------------------------------------

# Helper Function to Format Timestamps for SRT
def format_srt_timestamp(seconds: float) -> str:
    """Converts a float number of seconds to the SRT format HH:MM:SS,ms"""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


# --- NEW: THE SMART CAPTION GENERATION FUNCTION ---
def generate_srt_from_words(all_words: List[Word], config: dict) -> str:
    """
    Generates an SRT file content by grouping words into logical lines
    based on punctuation and word counts.
    """
    if not all_words:
        return ""

    srt_content = []
    caption_index = 1
    current_line_words = []

    # Define characters that mark a good end of a caption
    sentence_ending_punctuations = ".?!,"

    for i, word_obj in enumerate(all_words):
        current_line_words.append(word_obj)
        word_text = word_obj.word

        # Check if we should end the current line
        # Condition 1: The line has reached the hard word limit
        is_hard_limit_reached = len(current_line_words) >= config['max_words_per_line']
        # Condition 2: The word has punctuation AND we've met the minimum word count
        is_natural_break = (word_text.strip()[-1] in sentence_ending_punctuations) and \
                           (len(current_line_words) >= config['min_words_per_line'])
        # Condition 3: It's the very last word of the entire transcript
        is_last_word = (i == len(all_words) - 1)

        if is_hard_limit_reached or is_natural_break or is_last_word:
            start_time = format_srt_timestamp(current_line_words[0].start)
            end_time = format_srt_timestamp(current_line_words[-1].end)
            text = " ".join(w.word for w in current_line_words).strip()

            # Create the SRT block
            srt_block = f"{caption_index}\n{start_time} --> {end_time}\n{text}\n"
            srt_content.append(srt_block)

            # Reset for the next line
            caption_index += 1
            current_line_words = []

    return "\n".join(srt_content)


# ----------------------------------------------------

# --- Main Execution ---
if not os.path.exists(WAV_PATH):
    print(f"FATAL ERROR: Sanitized audio file not found at: {WAV_PATH}")
    exit()

# 1. Initialize Transcriber
transcriber = FasterWhisperTranscriber(model_name=MODEL_NAME, model_cache_dir=MODEL_DOWNLOAD_PATH)

# 2. Load Audio
print(f"Loading audio file: {WAV_PATH}")
waveform, sample_rate = torchaudio.load(WAV_PATH)
if waveform.shape[0] > 1:  # Ensure mono
    waveform = torch.mean(waveform, dim=0, keepdim=True)
audio_samples = waveform.squeeze(0).numpy().astype('float32')

# 3. Transcribe with Word Timestamps
start_time = time.time()
all_words = transcriber.transcribe_with_words(audio_samples)
transcribe_time = time.time() - start_time

# 4. Process and Generate Smart SRT
print("\n--- Transcription Complete ---")
print(f"Found {len(all_words)} words in {transcribe_time:.2f} seconds.")
print(f"Generating smart SRT file: {CAPTIONS_CONFIG['output_srt_file']}")

# Use our new smart function
srt_output = generate_srt_from_words(all_words, CAPTIONS_CONFIG)

# 5. Save the SRT File
with open(CAPTIONS_CONFIG['output_srt_file'], 'w', encoding='utf-8') as f:
    f.write(srt_output)

print("\n--- PROCESSING COMPLETE ---")
print(f"Caption file saved to: {os.path.abspath(CAPTIONS_CONFIG['output_srt_file'])}")