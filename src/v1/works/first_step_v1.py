import sys
import json
import subprocess
import time
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# CONFIG
# =========================
class Config:
    FILE_NAME = "vid1"

    BASE_PATH = "/vids"
    VIDEO_FILE = f"{BASE_PATH}\\{FILE_NAME}.mp4"
    OUTPUT_FILE = f"{BASE_PATH}\\{FILE_NAME}_final.mp4"
    TIMESTAMP_FILE = f"{BASE_PATH}\\{FILE_NAME}_timestamps.json"
    CONCAT_LIST_FILE = f"{BASE_PATH}\\concat.txt"

    MAX_SILENCE = 0.3
    BATCH_SIZE = 50

    # Performance
    MAX_WORKERS = 4  # 🔥 parallelism (tune this)
    USE_GPU = True

    # Encoding
    ENCODER_PRESET = "p7"
    CONSTANT_QUALITY = "23"
    AUDIO_BITRATE = "192k"
    CPU_THREADS = 8
    FFMPEG_LOG_LEVEL = "warning"


# =========================
# UTILS
# =========================
def run_command(cmd):
    subprocess.run(cmd, check=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def cleanup(files):
    for f in files:
        if os.path.exists(f):
            os.remove(f)


# =========================
# SEGMENT LOGIC
# =========================
def merge_segments(words):
    if not words:
        return []

    merged = []
    s, e = words[0]["start"], words[0]["end"]

    for w in words[1:]:
        if w["start"] - e <= Config.MAX_SILENCE:
            e = w["end"]
        else:
            merged.append({"start": s, "end": e})
            s, e = w["start"], w["end"]

    merged.append({"start": s, "end": e})
    return merged


def split_batches(segments):
    for i in range(0, len(segments), Config.BATCH_SIZE):
        yield segments[i:i + Config.BATCH_SIZE]


# =========================
# FFMPEG
# =========================
def get_video_encoder():
    return "h264_nvenc" if Config.USE_GPU else "libx264"


def build_filter(segments):
    filters = []
    v_parts, a_parts = [], []

    for i, seg in enumerate(segments):
        s, e = seg["start"], seg["end"]

        filters.append(f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[v{i}]")
        filters.append(f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{i}]")

        v_parts.append(f"[v{i}]")
        a_parts.append(f"[a{i}]")

    filters.append(f"{''.join(v_parts)}concat=n={len(segments)}:v=1:a=0[outv]")
    filters.append(f"{''.join(a_parts)}concat=n={len(segments)}:v=0:a=1[outa]")

    return ";".join(filters)


def process_batch(batch_id, video_path, segments):
    output = f"{Config.BASE_PATH}\\temp_{batch_id}.mp4"

    cmd = [
        "ffmpeg", "-loglevel", Config.FFMPEG_LOG_LEVEL,
        "-i", video_path,
        "-filter_complex", build_filter(segments),
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", get_video_encoder(),
        "-preset", Config.ENCODER_PRESET,
        "-cq", Config.CONSTANT_QUALITY,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", Config.AUDIO_BITRATE,
        "-threads", str(Config.CPU_THREADS),
        "-y", output
    ]

    print(f"[Batch {batch_id}] Processing...")
    run_command(cmd)

    return output


def concat_files(files):
    with open(Config.CONCAT_LIST_FILE, "w") as f:
        for p in files:
            safe = p.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe}'\n")

    cmd = [
        "ffmpeg", "-loglevel", Config.FFMPEG_LOG_LEVEL,
        "-f", "concat", "-safe", "0",
        "-i", Config.CONCAT_LIST_FILE,
        "-c", "copy",
        "-y", Config.OUTPUT_FILE
    ]

    run_command(cmd)


# =========================
# PARALLEL PIPELINE
# =========================
def process_all_batches(video_path, segments):
    batches = list(split_batches(segments))
    results = []

    print(f"Running {len(batches)} batches with {Config.MAX_WORKERS} workers")

    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_batch, i, video_path, batch): i
            for i, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            batch_id = futures[future]
            try:
                result = future.result()
                results.append((batch_id, result))
            except Exception as e:
                print(f"Batch {batch_id} failed: {e}")
                raise

    # 🔥 keep correct order
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]


# =========================
# MAIN EDIT FLOW
# =========================
def edit():
    temp_files = []

    try:
        words = load_json(Config.TIMESTAMP_FILE)
        segments = merge_segments(words)

        print(f"Merged → {len(segments)} segments")

        temp_files = process_all_batches(Config.VIDEO_FILE, segments)

        print("Concatenating...")
        concat_files(temp_files)

        print("Done!")

    finally:
        cleanup(temp_files + [Config.CONCAT_LIST_FILE])


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    start = time.time()

    if not os.path.exists(Config.TIMESTAMP_FILE):
        print("Missing timestamps.json")
        sys.exit(1)

    edit()

    print(f"Finished in {time.time() - start:.2f}s")