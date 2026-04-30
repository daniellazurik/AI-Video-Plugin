class Config:
    class Files:
        VIDEO_FILE_NAME = "vid3"
        VIDEO_FILE_OUTPUT_PATH = "D:\\via\\output\\video"
        VIDEO_BASE_PATH = "D:\\via\\vids"
        CAPTIONS_FILE_NAME = "captions.srt"
        CAPTIONS_BASE_PATH = "D:\\via\\output\\captions"
        CAPTIONS_OUTPUT_FILE = "D:\\via\\output\\captions\\captions.srt"
        CAPTIONS_OUTPUT_FILE_FINAL = "D:\\via\\output\\captions\\captions_dt.srt"
        DIARIZATION_OUTPUT_PATH = "D:\\via\\output\\diarization"
        DIARIZATION_FILE_NAME = "diarization"
        WAV_FILE_NAME = "vid"
        WAV_BASE_PATH = "D:\\via\\output\\audio"
        MODEL_DOWNLOAD_PATH = "D:\\downloaded_models"

    class Processing:
        MAX_SILENCE = 0.3
        BATCH_SIZE = 50

    class Performance:
        MAX_WORKERS = 4
        USE_GPU = True

    class Encoding:
        ENCODER_PRESET = "p7"
        CONSTANT_QUALITY = "23"
        AUDIO_BITRATE = "192k"
        CPU_THREADS = 8
        FFMPEG_LOG_LEVEL = "warning"

    class Ai:
        MODEL_NAME = "ivrit-ai/whisper-large-v3-ct2"
        MIN_WORDS_PER_LINE = 2
        MAX_WORDS_PER_LINE = 4

    @staticmethod
    def get_video_file_path():
        return f"{Config.Files.VIDEO_BASE_PATH}\\{Config.Files.VIDEO_FILE_NAME}.mp4"
    @staticmethod
    def get_captions_output_path():
        return f"{Config.Files.CAPTIONS_BASE_PATH}\\captions.srt"
    @staticmethod
    def get_audio_file_path():
        return f"{Config.Files.WAV_BASE_PATH}\\{Config.Files.WAV_FILE_NAME}.wav"
    @staticmethod
    def get_video_output_path():
        return f"{Config.Files.VIDEO_FILE_OUTPUT_PATH}\\{Config.Files.VIDEO_FILE_NAME}.mp4"
    @staticmethod
    def get_diarization_output_path():
        return f"{Config.Files.DIARIZATION_OUTPUT_PATH}\\{Config.Files.DIARIZATION_FILE_NAME}.json"