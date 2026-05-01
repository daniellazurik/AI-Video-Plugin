# main.py (FINAL VERSION WITH AUDIO NORMALIZATION)
import os
import numpy as np
from pydub import AudioSegment
import torch

from src.v1.diarization import SpeakerDiarizer
from src.v1.transcriber import FasterWhisperTranscriber
from src.v1.utils import format_time, ensure_folder
from src.v1.audio_utils import mp4_to_wav


def merge_segments(segments, max_pause_s=1.5):
    if not segments: return []
    merged = []
    current_segment = segments[0].copy()
    for i in range(1, len(segments)):
        next_segment = segments[i]
        if (next_segment['speaker'] == current_segment['speaker'] and
                (next_segment['start'] - current_segment['end']) < max_pause_s):
            current_segment['end'] = next_segment['end']
        else:
            merged.append(current_segment)
            current_segment = next_segment.copy()
    merged.append(current_segment)
    return merged


# --- Configuration ---
TRANSCRIPTS_FOLDER = "transcripts"
ensure_folder(TRANSCRIPTS_FOLDER)
HF_TOKEN = os.environ.get("HF_TOKEN", "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")
file_name = "vid1"
mp4_path = f"D:\\via\\vids\\{file_name}.mp4"
wav_folder = "D:\\via\\mp3s"
wav_path = os.path.join(wav_folder, f"{file_name}.wav")
MIN_SEGMENT_DURATION = 1.0

if __name__ == "__main__":
    if not os.path.exists(wav_path):
        wav_path = mp4_to_wav(mp4_path, wav_folder)
    else:
        print(f"Using existing WAV file: {wav_path}")

    print("\nInitializing models...")
    try:
        transcriber = FasterWhisperTranscriber()
        diarizer = SpeakerDiarizer(HF_TOKEN)
        print("All models initialized successfully.")
    except Exception as e:
        print(f"Fatal error initializing models: {e}")
        exit()

    print("Loading and preparing audio data...")
    audio = AudioSegment.from_file(wav_path)


    print("Normalizing audio volume...")
    target_dBFS = -1.0
    change_in_dBFS = target_dBFS - audio.dBFS
    audio = audio.apply_gain(change_in_dBFS)
    # --------------------------------------------------

    audio = audio.set_frame_rate(16000).set_channels(1)

    audio_samples_np = np.array(audio.get_array_of_samples()).astype(np.float32) / np.iinfo(audio.sample_width * 8).max
    audio_samples_torch = torch.from_numpy(audio_samples_np).unsqueeze(0)
    print("Audio loaded and prepared successfully.")

    diarization_segments = diarizer.diarize(audio_samples_torch, 16000)

    print(f"\n[DEBUG] Raw diarization returned {len(diarization_segments)} segments.")
    if diarization_segments: print(f"[DEBUG] First 5 raw segments: {diarization_segments[:5]}")

    merged_diarization_segments = merge_segments(diarization_segments)
    print(f"[DEBUG] After merging, we have {len(merged_diarization_segments)} segments.")
    if merged_diarization_segments: print(f"[DEBUG] First 5 merged segments: {merged_diarization_segments[:5]}")

    valid_segments_metadata = []
    audio_segments_to_transcribe = []
    for segment_data in merged_diarization_segments:
        start_s, end_s = segment_data['start'], segment_data['end']
        if (end_s - start_s) < MIN_SEGMENT_DURATION:
            continue
        start_sample = int(start_s * 16000)
        end_sample = int(end_s * 16000)
        segment_samples = audio_samples_np[start_sample:end_sample]
        valid_segments_metadata.append(segment_data)
        audio_segments_to_transcribe.append(segment_samples)

    print(f"[DEBUG] After duration filtering, we have {len(valid_segments_metadata)} segments to transcribe.")

    transcript_lines = []
    if audio_segments_to_transcribe:
        print(f"Transcribing {len(valid_segments_metadata)} valid segments...")
        try:
            all_transcriptions = transcriber.transcribe_batch(audio_segments_to_transcribe)
            for i, segment_data in enumerate(valid_segments_metadata):
                transcription_text = all_transcriptions[i]
                if not transcription_text or not transcription_text.strip(): continue
                timestamp = format_time(segment_data['start'])
                speaker = segment_data['speaker']
                transcript_lines.append(f"#{speaker};{timestamp};{transcription_text}")
        except Exception as e:
            print(f"An error occurred during batch transcription: {e}")
    else:
        print("No valid audio segments found to transcribe after filtering.")

    base_name = os.path.splitext(os.path.basename(mp4_path))[0]
    save_path = os.path.join(TRANSCRIPTS_FOLDER, f"{base_name}_transcript.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(transcript_lines))

    print(f"\n--- Processing Complete ---")
    print(f"Final transcript contains {len(transcript_lines)} lines.")
    print(f"Result saved to: {save_path}")