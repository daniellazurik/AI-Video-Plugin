# src/transcriber.py
import torch
import numpy as np
from typing import List
from tqdm import tqdm
from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Word


class FasterWhisperTranscriber:
    """
    A robust transcriber class that loads models and forces GPU usage if available.
    """

    def __init__(self, model_name="large-v3", model_cache_dir="D:\\downloaded_models"):
        if WhisperModel is None:
            raise ImportError("faster-whisper is not installed.")


        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 2. Use a faster compute type (float16) if on GPU.
        compute_type = "float16" if self.device == "cuda" else "auto"
        # ----------------------

        print(f"FasterWhisperTranscriber using device: {self.device} with compute_type: {compute_type}")
        print(f"Loading model: {model_name}...")

        model_path_or_name = model_name
        if "/" in model_name:
            print(f"Detected Hugging Face model. Ensuring it is downloaded...")
            model_path_or_name = snapshot_download(
                repo_id=model_name,
                cache_dir=model_cache_dir,
                local_dir_use_symlinks=False
            )
        else:
            print(f"Detected official OpenAI model.")

        try:
            # 3. Pass the determined device and compute_type to the model.
            self.model = WhisperModel(
                model_path_or_name,
                device=self.device,
                compute_type=compute_type,
                download_root=model_cache_dir if "/" not in model_name else None
            )
        except Exception as e:
            print(f"FATAL: Failed to load model '{model_name}': {e}")
            raise
        print("Faster-Whisper model loaded successfully.")

    def transcribe_with_words(self, audio_segment: np.ndarray) -> List[Word]:
        """
        Transcribes audio and returns a list of words with timestamps.
        """
        if Word is None:
            raise ImportError("faster-whisper is not installed correctly to provide Word objects.")

        segments, info = self.model.transcribe(
            audio_segment,
            language="he",
            beam_size=5,
            vad_filter=False,
            word_timestamps=True
        )

        all_words = []
        with tqdm(total=round(info.duration, 2), unit="s", desc="Transcription Progress") as pbar:
            for segment in segments:
                if segment.words:
                    all_words.extend(segment.words)
                pbar.update(segment.end - pbar.n)

        return all_words