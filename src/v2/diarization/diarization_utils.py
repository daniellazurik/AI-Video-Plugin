import json
import os
import re

from src.v2.config import Config


def save_diarization_to_json(diarization, output_path=Config.Files.DIARIZATION_OUTPUT_PATH):
    data = []

    # 1. Handle Pyannote 3.1+ DiarizeOutput dataclass
    # This object wraps speaker_diarization, embeddings, etc.
    if hasattr(diarization, 'speaker_diarization'):
        diarization = diarization.speaker_diarization

    # 2. Process the Annotation object (Standard Pyannote behavior)
    if hasattr(diarization, 'itertracks'):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            data.append({
                "speaker": speaker,
                "start": float(turn.start),
                "end": float(turn.end)
            })

    # 3. Fallback for lists (if data was already processed elsewhere)
    elif isinstance(diarization, list):
        data = diarization

    # 4. Fallback for other potential wrapper objects (.segments)
    elif hasattr(diarization, 'segments'):
        for segment in diarization.segments:
            data.append({
                "speaker": getattr(segment, "speaker", "UNKNOWN"),
                "start": float(segment.start),
                "end": float(segment.end)
            })

    else:
        raise TypeError(f"Could not parse diarization output of type {type(diarization)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved diarization to {output_path}")

def time_to_seconds(t):
    h, m, s = t.split(':')
    s, ms = s.split(',')
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

def get_best_speaker(start, end, speakers):
    best_speaker = "UNKNOWN"
    max_overlap = 0

    for seg in speakers:
        overlap_start = max(start, seg["start"])
        overlap_end = min(end, seg["end"])
        overlap = max(0, overlap_end - overlap_start)

        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = seg["speaker"]

    return best_speaker

def insert_speakers_to_srt(srt_text, speakers):
    blocks = re.split(r"\n\s*\n", srt_text.strip())  # Split into blocks
    new_blocks = []

    for block in blocks:
        lines = block.split("\n")
        if len(lines) < 3:
            continue

        index = lines[0]
        timing = lines[1]
        text = "\n".join(lines[2:])

        # Extract start and end time from the timing line
        start_str, end_str = timing.split(" --> ")
        start = time_to_seconds(start_str)
        end = time_to_seconds(end_str)

        # Get the best speaker for the time range
        speaker = get_best_speaker(start, end, speakers)

        # Insert speaker label into the text
        new_text = f"{speaker}: {text}"
        new_block = f"{index}\n{timing}\n{new_text}"

        new_blocks.append(new_block)

    return "\n\n".join(new_blocks)


def update_captions(captions_path, diarization_path, output_path=Config.Files.CAPTIONS_OUTPUT_FILE_FINAL):
    try:
        # Check if input files exist
        if not os.path.exists(captions_path):
            print(f"Error: Captions file '{captions_path}' does not exist.")
            return

        if not os.path.exists(diarization_path):
            print(f"Error: Diarization file '{diarization_path}' does not exist.")
            return

        # Read captions from .srt file
        with open(captions_path, "r", encoding="utf-8") as captions_file:
            captions = captions_file.read()

        # Read diarization data from JSON file
        with open(diarization_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        # Insert speakers into the captions
        updated_captions = insert_speakers_to_srt(captions, data)

        # Write the updated captions to the output file
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(updated_captions)

        print(f"Updated captions have been saved to {output_path}")

    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {diarization_path}. Check the file format.")

    except FileNotFoundError as e:
        print(f"Error: {e.strerror} - {e.filename}")

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
