# src/diarization.py (Now with Hyperparameter Tuning)
import torch
from typing import Dict, List
from pyannote.audio import Pipeline
import torchaudio


class SpeakerDiarizer:
    def __init__(self, hf_token: str):
        print("Loading AI model for diarization...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Diarization using device: {self.device}")
        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote.audio.pipelines.SpeakerDiarization",
                use_auth_token=hf_token
            ).to(self.device)
            print("Diarization model loaded successfully.")
        except Exception as e:
            print(f"Error loading diarization model: {e}")
            raise

    def diarize_chunk(self, waveform_chunk: torch.Tensor, sample_rate: int,
                      min_speakers: int = None, max_speakers: int = None,
                      # --- NEW: Add a parameter for custom settings ---
                      hyper_params: dict = None) -> List[Dict]:
        """Runs speaker diarization on a single audio chunk with optional hyperparameters."""
        waveform_chunk = waveform_chunk.to(self.device)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(original_freq=sample_rate, new_freq=16000).to(self.device)
            waveform_chunk = resampler(waveform_chunk)

        # --- NEW: Pass the hyperparameters to the pipeline ---
        # The pipeline will use these settings instead of its defaults.
        diarization = self.diarization_pipeline(
            {"waveform": waveform_chunk, "sample_rate": 16000},
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            # This is where we inject our custom tuning
            **hyper_params if hyper_params else {}
        )

        segments = [
            {"start": round(turn.start, 2), "end": round(turn.end, 2), "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        return segments