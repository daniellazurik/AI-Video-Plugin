import json

from tqdm import tqdm

from .faster_whisper_transcriber import FasterWhisperTranscriber
from .transcript_utils import is_gpu_available
from ..config import Config
from ..utils import save_words_to_srt


# VAD_FILTER = Voice Activity Detection filter
class TranscripterModule:
    def __init__(self, device="cpu", compute_type="auto", use_gpu=False):
        self.device = device
        self.compute_type = compute_type
        self.use_gpu = use_gpu

        if self.use_gpu and is_gpu_available():
            self.device = "cuda"
            self.compute_type = "float16"

    def run_transcription(self, wav_path, model_name, output_json_path, **transcribe_args):
        print("--- Starting Transcription Process () ---")

        model = FasterWhisperTranscriber(model_name, device=self.device, compute_type=self.compute_type)

        segments, info = model.transcribe(wav_path, **transcribe_args)
        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        timestamps = []
        with tqdm(total=round(info.duration, 2), desc="Transcription Progress", unit="s") as pbar:
            for segment in segments:
                for word in segment.words: timestamps.append({'start': word.start, 'end': word.end})
                pbar.update(segment.end - pbar.n)
        print(f"Timestamp detection complete. Found {len(timestamps)} spoken words.")
        with open(output_json_path, 'w') as f:
            json.dump(timestamps, f)
        print("--- Transcription Process Finished ---")

    def run_transcription_preset_large_v3(self, wav_path, captions_output=Config.get_captions_output_path(),
                                          **transcribe_args):
        print("--- Starting Transcription Process (large_v3) ---")
        model = FasterWhisperTranscriber("ivrit-ai/whisper-large-v3-ct2", device=self.device,
                                         compute_type=self.compute_type)
        all_words = model.transcribe(wav_path, vad_filter=True, **transcribe_args)
        print(all_words)
        print("--- Transcription Process Finished ---")
        save_words_to_srt(all_words, captions_output)
        return all_words
