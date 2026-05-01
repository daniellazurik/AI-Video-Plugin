import sys
import json
import subprocess
import time
import os
from tqdm import tqdm
import multiprocessing

# --- HIGH-SPEED CONFIGURATION ---
file_name = "vid1"
VIDEO_FILE = f"D:\\via\\vids\\{file_name}.mp4"
OUTPUT_FILE = f"D:\\via\\vids\\{file_name}_edited_final_fast_ai.mp4"
TIMESTAMP_FILE = f"D:\\via\\vids\\{file_name}_timestamps.json"
CONCAT_LIST_FILE = f"D:\\via\\vids\\{file_name}_concat_list.txt"

# --- AI & TRANSCRIPTION SETTINGS ---
# Use the fastest possible model and compute type
WHISPER_MODEL = "tiny"
COMPUTE_TYPE = "int8"
BEAM_SIZE = 1
USE_VAD_FILTER = True

# --- VIDEO & PARALLEL PROCESSING SETTINGS ---
MAX_SILENCE_BETWEEN_WORDS = 0.3
FFMPEG_LOG_LEVEL = "warning"
# Increase workers if you have a strong CPU (8+ cores). 4 is a good number.
NUM_WORKERS = 4
# A slightly larger batch size can be more efficient
BATCH_SIZE = 75
# Use a faster preset now that the AI is faster
ENCODER_PRESET = "p5"  # "p5" is a great balance of speed/quality. "p4" is even faster.
CONSTANT_QUALITY = "23"
AUDIO_BITRATE = "192k"
# Assign fewer threads since we're I/O bound. Let the OS handle it.
CPU_THREADS = 2


# ==============================================================================
# WORKER FUNCTION (OPTIMIZED FOR FAST SEEKING)
# ==============================================================================
def process_batch(task_args):
    """
    This function is executed by each worker in the pool.
    It now uses the '-ss' flag before '-i' for massive speedup.
    """
    batch_index, batch_segments, video_path, temp_output_path = task_args

    # We only need to read a small portion of the source video for each batch
    first_start_time = batch_segments[0]['start']
    last_end_time = batch_segments[-1]['end']

    # Adjust the timestamps to be relative to the new, shorter input clip
    adjusted_segments = [{
        'start': seg['start'] - first_start_time,
        'end': seg['end'] - first_start_time
    } for seg in batch_segments]

    filter_complex = []
    video_parts, audio_parts = [], []
    for j, segment in enumerate(adjusted_segments):
        start, end = segment['start'], segment['end']
        filter_complex.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{j}]")
        video_parts.append(f"[v{j}]")
        filter_complex.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{j}]")
        audio_parts.append(f"[a{j}]")

    filter_complex.append(f"{''.join(video_parts)}concat=n={len(adjusted_segments)}:v=1:a=0[outv]")
    filter_complex.append(f"{''.join(audio_parts)}concat=n={len(adjusted_segments)}:v=0:a=1[outa]")

    # --- CRITICAL CHANGE FOR SPEED ---
    # The '-ss' (seek) flag is placed BEFORE '-i' (input).
    # This makes FFmpeg jump almost instantly to the start of the segment.
    ffmpeg_command = [
        'ffmpeg', '-loglevel', FFMPEG_LOG_LEVEL,
        '-ss', str(first_start_time),  # Fast seek to the start time
        '-i', video_path,
        '-to', str(last_end_time - first_start_time),  # Process only up to the end time
        '-filter_complex', ";".join(filter_complex),
        '-map', '[outv]', '-map', '[outa]',
        '-c:v', 'h264_nvenc', '-preset', ENCODER_PRESET, '-cq', CONSTANT_QUALITY,
        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', AUDIO_BITRATE,
        '-threads', str(CPU_THREADS), '-y', temp_output_path
    ]

    result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed on batch {batch_index}:\n{result.stderr}")

    return temp_output_path


# ==============================================================================
# SCRIPT MODE 1: TRANSCRIBER (Unchanged, but will be much faster with new config)
# ==============================================================================
def run_transcription(video_path, model_name, output_json_path):
    from faster_whisper import WhisperModel
    print("--- Starting Transcription Process (Optimized for Speed) ---")
    model = WhisperModel(model_name, device="cuda", compute_type=COMPUTE_TYPE)
    segments, info = model.transcribe(video_path, word_timestamps=True, beam_size=BEAM_SIZE, vad_filter=USE_VAD_FILTER)
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


# ==============================================================================
# SCRIPT MODE 2: VIDEO EDITOR (Unchanged, but calls new worker)
# ==============================================================================
def run_editing_reencode_batched(video_path, input_json_path, output_video_path):
    temp_files = []
    try:
        print("\n--- Starting Editing Process (Parallel Batch Re-encoding) ---")
        with open(input_json_path, 'r') as f:
            word_timestamps = json.load(f)
        if not word_timestamps:
            print("The timestamp file is empty. Exiting.");
            sys.exit(1)

        print(f"Merging {len(word_timestamps)} word timestamps...")
        merged_segments = []
        current_start, current_end = word_timestamps[0]['start'], word_timestamps[0]['end']
        for i in range(1, len(word_timestamps)):
            if (word_timestamps[i]['start'] - current_end) <= MAX_SILENCE_BETWEEN_WORDS:
                current_end = word_timestamps[i]['end']
            else:
                merged_segments.append({'start': current_start, 'end': current_end})
                current_start, current_end = word_timestamps[i]['start'], word_timestamps[i]['end']
        merged_segments.append({'start': current_start, 'end': current_end})
        print(f"Reduced to {len(merged_segments)} speech segments.")

        tasks = []
        num_batches = (len(merged_segments) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(num_batches):
            batch_start, batch_end = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
            batch_segments = merged_segments[batch_start:batch_end]
            if not batch_segments: continue
            temp_output_path = f"D:\\via\\vids\\temp_part_{i}.mp4"
            temp_files.append(temp_output_path)
            tasks.append((i, batch_segments, video_path, temp_output_path))

        print(f"Starting parallel processing of {num_batches} batches with {NUM_WORKERS} workers.")

        pbar = tqdm(total=len(tasks), desc="Processing Batches")

        def update_pbar(*args):
            pbar.update()

        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            results = []
            for task in tasks:
                results.append(pool.apply_async(process_batch, args=(task,), callback=update_pbar))

            pool.close()
            pool.join()

            # Check for errors in the results
            for res in results:
                res.get()  # This will re-raise any exception from the worker

        pbar.close()

        print(f"\nAll batches processed. Concatenating temporary files...")
        temp_files.append(CONCAT_LIST_FILE)
        with open(CONCAT_LIST_FILE, 'w') as f:
            for i in range(num_batches):
                path = f"D:\\via\\vids\\temp_part_{i}.mp4"
                if os.path.exists(path):
                    safe_path = path.replace("\\", "/").replace("'", "'\\''")
                    f.write(f"file '{safe_path}'\n")

        ffmpeg_concat_command = [
            'ffmpeg', '-loglevel', FFMPEG_LOG_LEVEL,
            '-f', 'concat', '-safe', '0', '-i', CONCAT_LIST_FILE,
            '-c', 'copy', '-y', output_video_path
        ]

        subprocess.run(ffmpeg_concat_command, check=True)

        print(f"\nFinal video saved as {output_video_path}")
        print("--- Editing Process Finished ---")

    except Exception as e:
        print(f"\n\n!!!!!!!!!!!!!! AN ERROR OCCURRED: {type(e).__name__} !!!!!!!!!!!!!!")
        print(f"ERROR DETAILS: {e}")
        sys.exit(1)
    finally:
        print("\nCleaning up temporary files...")
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)


# ==============================================================================
# MAIN CONTROLLER
# ==============================================================================
if __name__ == '__main__':
    multiprocessing.freeze_support()

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "transcribe":
            run_transcription(VIDEO_FILE, WHISPER_MODEL, TIMESTAMP_FILE)
        elif mode == "edit":
            run_editing_reencode_batched(VIDEO_FILE, TIMESTAMP_FILE, OUTPUT_FILE)
    else:
        start_time = time.time()
        skip_transcription = False
        if os.path.exists(TIMESTAMP_FILE):
            try:
                with open(TIMESTAMP_FILE, 'r') as f:
                    data = json.load(f)
                if not data:
                    print(f"Found an empty timestamp file '{TIMESTAMP_FILE}'. It will be recreated.")
                else:
                    reuse = input(f"Found existing timestamp file with {len(data)} entries. Reuse it? (y/n): ").lower()
                    if reuse in ['y', 'yes']:
                        skip_transcription = True
                    else:
                        print("User chose not to reuse the file. It will be recreated.")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Found a corrupt timestamp file: {e}. It will be recreated.")

        if not skip_transcription:
            print(">>> Starting Step 1: Transcription (Ultra-Fast Mode) <<<")
            p = subprocess.run([sys.executable, __file__, "transcribe"])
            if p.returncode != 0 and not os.path.exists(TIMESTAMP_FILE):
                print("\nERROR: Transcription failed! Aborting.");
                sys.exit(1)
            elif p.returncode != 0:
                print("\nWARNING: Transcription exited with non-zero code, but file was created. Proceeding...")
            else:
                print("\n>>> Transcription successful.")

        if os.path.exists(TIMESTAMP_FILE):
            print(">>> Starting Step 2: Editing (Optimized Parallel Mode) <<<")
            p = subprocess.run([sys.executable, __file__, "edit"])
            if p.returncode != 0:
                print("\n--- MAIN SCRIPT: The editing subprocess failed. ---")
            else:
                print("\n>>> All steps completed successfully! <<<")
        else:
            print("\nERROR: Timestamp file not found, cannot proceed to editing.")

        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds.")