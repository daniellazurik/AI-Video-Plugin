import os
import subprocess
import shutil
import torch


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)

    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def ensure_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)

def save_words_to_srt(
    words,
    output_path,
    group_words=False,
    max_words_per_line=6,
    max_gap=0.6
):
    """
    Convert Word list to SRT.

    Args:
        words: list of Word objects
        output_path: output .srt file
        group_words: if False → each word is its own subtitle
        max_words_per_line: grouping limit
        max_gap: silence threshold for grouping
    """



    subtitles = []

    # 🔥 MODE 1: word-by-word
    if not group_words:
        for w in words:
            subtitles.append((
                float(w.start),
                float(w.end),
                w.word.strip()
            ))

    # 🟢 MODE 2: grouped
    else:
        current_chunk = []
        start_time = None

        for i, w in enumerate(words):
            word_text = w.word.strip()
            w_start = float(w.start)
            w_end = float(w.end)

            if start_time is None:
                start_time = w_start

            if current_chunk:
                prev_word = words[i - 1]
                gap = w_start - float(prev_word.end)

                if gap > max_gap or len(current_chunk) >= max_words_per_line:
                    subtitles.append((
                        start_time,
                        float(prev_word.end),
                        " ".join(current_chunk)
                    ))
                    current_chunk = []
                    start_time = w_start

            current_chunk.append(word_text)

        if current_chunk:
            subtitles.append((
                start_time,
                float(words[-1].end),
                " ".join(current_chunk)
            ))

    # ️ Write file
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(subtitles, 1):
            f.write(f"{i}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{text}\n\n")


def delete_all_files_in_folder(folder_path):
    """
    Deletes all files in the specified folder.
    Subdirectories within the folder will remain untouched.
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path): # Check if it's a file, not a directory
                os.remove(file_path)
                print(f"Deleted: {filename}")
        except OSError as e:
            print(f"Error deleting {filename}: {e}")
    print(f"All files in '{folder_path}' have been processed.")



def apply_srt(video, srt, output):
    temp_dir = "D:/via/temp"
    os.makedirs(temp_dir, exist_ok=True)

    video_path = os.path.join(temp_dir, "vid.mp4")
    srt_path = os.path.join(temp_dir, "caps.srt")

    shutil.copy(video, video_path)
    shutil.copy(srt, srt_path)

    cmd = [
        "ffmpeg",
        "-i", "vid.mp4",
        "-vf", "subtitles=caps.srt",
        "-c:v", "h264_nvenc",
        "-preset", "fast",
        "-c:a", "copy",
        f"{output}\\new_video.mp4"
    ]

    subprocess.run(cmd, cwd=temp_dir, check=True)


def is_gpu_available():
    return torch.cuda.is_available()