# master_caption_pipeline.py (Final Version with Two-Pass Chain of Thought)
import torch
import torchaudio
import os
import re
import time
from tqdm import tqdm  # For the progress bar
from src.v1.transcriber import FasterWhisperTranscriber
from faster_whisper.transcribe import Word
from typing import List
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import gc

print("--- Starting Production AI Caption Generation Pipeline ---")

# --- Configuration ---
WAV_PATH = "D:\\via\\mp3s\\vid3_sanitized.wav"
MODEL_DOWNLOAD_PATH = "D:\\downloaded_models"
LLM_MODEL_DIR = "D:\\via\\models"
OUTPUT_SRT_FILE = "D:\\via\\output\\captions_final_CoT.srt"  # CoT for Chain of Thought

# --- Model Configuration ---
TRANSCRIPTION_MODEL_NAME = "ivrit-ai/whisper-large-v3-ct2"
LLM_REPO_ID = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
LLM_FILENAME = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# --- Chunking Configuration ---
WORD_CHUNK_SIZE = 250  # Reduced slightly for the more complex two-pass prompt


def format_srt_timestamp(seconds: float) -> str:
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


# --- STAGE 3: AI Caption Grouping (Two-Pass Chain of Thought) ---
class CaptionerAI:
    def __init__(self, model_path: str):
        print("Initializing Advanced Captioner AI (Llama 3.1)...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LLM model file not found at: {model_path}.")
        self.llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=8192, verbose=False, chat_format="llama-3",
                         n_batch=512)
        print("Captioner AI initialized for GPU.")

    def group_text_into_captions(self, sanitized_text_chunk: str) -> List[str]:
        # --- PASS 1: The "Linguist" - Identify logical phrases ---
        pass1_messages = [
            {
                "role": "system",
                "content": """You are a Hebrew linguist. Your only job is to analyze raw text and identify the complete, natural spoken phrases. Insert a pipe character (`|`) after each complete phrase. Do not worry about subtitle length, only linguistic correctness."""
            },
            {
                "role": "user",
                "content": f'Analyze and segment the following text with `|` markers:\n\n"{sanitized_text_chunk}"'
            }
        ]

        pass1_response = self.llm.create_chat_completion(
            messages=pass1_messages,
            temperature=0.0,
            max_tokens=None
        )
        text_with_phrases = pass1_response['choices'][0]['message']['content'].strip()

        # --- PASS 2: The "Editor" - Format the pre-identified phrases ---
        pass2_messages = [
            {
                "role": "system",
                "content": """You are a subtitle formatter. You will receive text pre-segmented into logical phrases using `|`. Your job is to format these phrases into short, 2-3 word lines for Instagram.
- You can split a long phrase into multiple lines.
- You MUST NOT combine text from different sides of a `|` marker.
- Your output must ONLY be the final formatted Hebrew lines."""
            },
            {
                "role": "user",
                "content": f'Format the following segmented text into short subtitle lines:\n\n"{text_with_phrases}"'
            }
        ]

        pass2_response = self.llm.create_chat_completion(
            messages=pass2_messages,
            temperature=0.2,
            max_tokens=None
        )

        result_text = pass2_response['choices'][0]['message']['content']
        return [line.strip() for line in result_text.split('\n') if line.strip()]


def assemble_final_srt(all_words: List[Word], refined_lines: List[str]) -> str:
    print("Assembling final SRT with corrected timings...")
    srt_content, caption_index, word_idx = [], 1, 0
    for line_text in refined_lines:
        num_words_in_line = len(line_text.split())
        if num_words_in_line == 0: continue
        if word_idx + num_words_in_line > len(all_words):
            print(f"FATAL ERROR: LLM output word count mismatch. Halting assembly.")
            break
        word_objects_for_line = all_words[word_idx: word_idx + num_words_in_line]
        word_idx += num_words_in_line
        accurate_text = " ".join(re.sub(r'[,.?!]', '', w.word) for w in word_objects_for_line).strip()
        start_time = format_srt_timestamp(word_objects_for_line[0].start)
        end_time = format_srt_timestamp(word_objects_for_line[-1].end)
        srt_block = f"{caption_index}\n{start_time} --> {end_time}\n{accurate_text}\n"
        srt_content.append(srt_block)
        caption_index += 1
    print("Final SRT assembled.")
    return "\n".join(srt_content)


if __name__ == "__main__":
    total_start_time = time.time()

    os.makedirs(os.path.dirname(OUTPUT_SRT_FILE), exist_ok=True)
    if not os.path.exists(WAV_PATH):
        print(f"FATAL ERROR: Audio file not found at: {WAV_PATH}")
        exit()

    print("\n--- Stage 1: Audio Transcription ---")
    stage1_start = time.time()
    transcriber = FasterWhisperTranscriber(model_name=TRANSCRIPTION_MODEL_NAME, model_cache_dir=MODEL_DOWNLOAD_PATH)
    waveform, _ = torchaudio.load(WAV_PATH)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    audio_samples = waveform.squeeze(0).numpy().astype('float32')
    all_words = transcriber.transcribe_with_words(audio_samples)
    stage1_end = time.time()
    print(f"Transcription complete. Found {len(all_words)} words. (Took {stage1_end - stage1_start:.2f} seconds)")

    print("\nUnloading transcription model to free VRAM for LLM...")
    del transcriber
    gc.collect()
    torch.cuda.empty_cache()
    print("VRAM freed.")

    print("\n--- Stage 2 & 3: LLM Caption Grouping (Two-Pass Method) ---")
    stage3_start = time.time()
    llm_model_path = os.path.join(LLM_MODEL_DIR, LLM_FILENAME)
    if not os.path.exists(llm_model_path):
        print(f"LLM model not found. Downloading the faster ~5 GB Llama 3.1 model...")
        hf_hub_download(
            repo_id=LLM_REPO_ID, filename=LLM_FILENAME,
            local_dir=LLM_MODEL_DIR, local_dir_use_symlinks=False
        )
    else:
        print(f"LLM model found locally.")

    captioner = CaptionerAI(model_path=llm_model_path)
    sanitized_words_list = [re.sub(r'[,.?!]', '', word.word).strip() for word in all_words]
    all_refined_lines = []

    for i in tqdm(range(0, len(sanitized_words_list), WORD_CHUNK_SIZE), desc="Processing Text Chunks"):
        word_chunk = sanitized_words_list[i:i + WORD_CHUNK_SIZE]
        sanitized_text_chunk = " ".join(word_chunk)

        refined_batch_lines = captioner.group_text_into_captions(sanitized_text_chunk)

        if refined_batch_lines:
            all_refined_lines.extend(refined_batch_lines)

    stage3_end = time.time()
    print(f"LLM processing complete. (Took {stage3_end - stage3_start:.2f} seconds)")

    print("\n--- Stage 4: Final SRT Assembly ---")
    stage4_start = time.time()
    final_srt_output = assemble_final_srt(all_words, all_refined_lines)
    stage4_end = time.time()
    print(f"Assembly complete. (Took {stage4_end - stage4_start:.2f} seconds)")

    with open(OUTPUT_SRT_FILE, 'w', encoding='utf-8') as f:
        f.write(final_srt_output)

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    minutes, seconds = divmod(total_duration, 60)

    print("\n--- PIPELINE COMPLETE ---")
    print(f"Refined social media captions saved to: {os.path.abspath(OUTPUT_SRT_FILE)}")
    print(f"Total processing time: {int(minutes)} minutes and {seconds:.2f} seconds.")