import os
import torchaudio
import torch
from pyannote.audio import Pipeline

from src.v2.diarization.diarization_utils import save_diarization_to_json
from src.v2.utils import is_gpu_available

class DiarizationModule:
    def __init__(self, device=torch.device("cpu"), compute_type="auto", use_gpu=False):
        self.diarization_pipeline = None
        print("Loading AI model for diarization...")
        self.device = device
        self.compute_type = compute_type
        self.use_gpu = use_gpu

        if self.use_gpu and is_gpu_available():
            self.device = torch.device("cuda")
            self.compute_type = "float16"

    def run_diarization(self):
        pass

    def run_diarization_preset_diarization_3_1(self, audio_path, output_path):

        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token="TOKEN_HERE"
            ).to(self.device)
            print("Diarization model loaded successfully.")
        except Exception as e:
            print(f"Error loading diarization model: {e}")
            raise

        save_diarization_to_json(self.diarize_audio(audio_path), output_path=output_path)

    def diarize_audio(self, audio_path):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            diarization = self.diarization_pipeline({
                "waveform": waveform,
                "sample_rate": sample_rate
            })
            return diarization

        except Exception as e:
            print(f"Diarization failed: {e}")
            raise
