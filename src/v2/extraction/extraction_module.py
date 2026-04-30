from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from src.v2.extraction.extraction_utils import analyze_transcript
from src.v2.utils import is_gpu_available


class ExtractionModule:
    def __init__(self, device=torch.device("cpu"), compute_type="auto", use_gpu=False):
        self.device = device
        self.compute_type = compute_type
        self.use_gpu = use_gpu

        if self.use_gpu and is_gpu_available():
            self.device = torch.device("cuda")
            self.compute_type = torch.float16

    def run_extraction(self, wav_path, model_name, output_json_path, **transcribe_args):
        pass

    def run_transcription_preset_qwen(self, final_transcript_path):
        # 1. Load the model and tokenizer
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Small & fast. Use "7B" for better Hebrew.

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.compute_type,
            device_map=self.device
        )
        with open(final_transcript_path, 'r' ,encoding='utf-8') as f:
            analyze_transcript(f.read(), model, tokenizer, self.device)
