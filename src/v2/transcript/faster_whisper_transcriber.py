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

    def __init__(self, model_name, device, compute_type, model_cache_dir="D:\\via\\downloaded_models"):
        if WhisperModel is None:
            raise ImportError("faster-whisper is not installed.")
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type

        # ----------------------

        print(f"FasterWhisperTranscriber using device: {self.device} with compute_type: {self.compute_type}")
        print(f"Loading model: {self.model_name}...")

        model_path_or_name = self.model_name
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

    def transcribe(self, audio_segment: np.ndarray, **transcribe_args) -> List[Word]:
        """
        Transcribes audio and returns a list of words with timestamps.
        """
        if Word is None:
            raise ImportError("faster-whisper is not installed correctly to provide Word objects.")

        default_args = {
            "language": "he",
            "beam_size": 5,
            "vad_filter": False,
            "word_timestamps": True,
        }
        merged_args = {**default_args, **transcribe_args}

        segments, info = self.model.transcribe(
            audio_segment,
            **merged_args
        )

        all_words = []
        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        with tqdm(total=round(info.duration, 2), unit="s", desc="Transcription Progress") as pbar:
            for segment in segments:
                if segment.words:
                    all_words.extend(segment.words)
                pbar.update(segment.end - pbar.n)

        return all_words
