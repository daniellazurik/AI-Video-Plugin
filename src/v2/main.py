import os
import sys

from config import Config
from src.v2.diarization.diarization_utils import update_captions
from src.v2.extraction.extraction_module import ExtractionModule
from src.v2.transcript.transcripter_module import TranscripterModule
from src.v2.transcript.audio_utils import audio_setup, mp4_to_wav
from src.v2.diarization.diarization_module import DiarizationModule
import time

from src.v2.utils import apply_srt


def transcribe(audio_path=Config.get_video_file_path(), captions_output=Config.Files.CAPTIONS_OUTPUT_FILE, ):
    audio_samples = audio_setup(audio_path)
    transcribe_m = TranscripterModule(use_gpu=Config.Performance.USE_GPU)
    transcribe_m.run_transcription_preset_large_v3(audio_samples, captions_output)


def diarization(audio_path=Config.get_audio_file_path(), diarization_output=Config.get_diarization_output_path()):
    diarization_m = DiarizationModule(use_gpu=Config.Performance.USE_GPU)
    diarization_m.run_diarization_preset_diarization_3_1(audio_path, diarization_output)


def extract_text_clips(captions_path=Config.Files.CAPTIONS_OUTPUT_FILE_FINAL):
    extraction_m = ExtractionModule(use_gpu=Config.Performance.USE_GPU)
    extraction_m.run_transcription_preset_qwen(captions_path)



def main():

    total_start_time = time.time()
    #video_path = Config.get_video_file_path()
    #wav_path = mp4_to_wav(video_path,output_folder=Config.Files.WAV_BASE_PATH)
    #diarization()
    #transcribe(Config.get_audio_file_path())
    #update_captions(Config.get_captions_output_path(), Config.get_diarization_output_path())
    print("DONE")
    extract_text_clips()
    #apply_srt(Config.get_video_file_path(),Config.Files.CAPTIONS_OUTPUT_FILE_FINAL,Config.Files.VIDEO_FILE_OUTPUT_PATH)


if __name__ == '__main__':
    main()
