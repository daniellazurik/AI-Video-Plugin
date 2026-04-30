from moviepy import VideoFileClip


def extract_clip(input_video, start_sec, end_sec, output_name):
    with VideoFileClip(input_video) as video:
        new_clip = video.subclip(start_sec, end_sec)
        new_clip.write_videofile(output_name, codec="libx264", audio_codec="aac")