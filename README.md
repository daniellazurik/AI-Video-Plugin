# Standalone AI Video Subtitler & Diarizer (Premiere Pro Plugin Backend POC)

A functional proof-of-concept (POC) Python application designed to automate video transcription, subtitle generation, and speaker identification (diarization). 

Currently operating as a **standalone software**, this project is being developed as the foundational backend for a future Adobe Premiere Pro extension. The architecture is specifically engineered to handle the complexities of Right-to-Left (RTL) languages like Hebrew, ensuring correct text formatting before it reaches the editing timeline.

## 💡 Core Philosophy: Free & Local Experimentation
A major driving force behind this project is the exclusive use of **local Large Language Models (LLMs)**. By running quantized models locally on consumer hardware, this project:
*   Completely eliminates cloud API dependency and recurring costs.
*   Allows for free, unlimited experimentation with open-weight models.
*   Ensures 100% data privacy for processed video and audio files.

## ✨ Key Features
*   **Local Inference Engine:** Executes locally-hosted models using hardware-accelerated Python scripts.
*   **Speaker Diarization:** Processes audio to distinguish between multiple speakers and attributes dialogue correctly.
*   **Bilingual & RTL Engineering:** Custom logic to handle Hebrew's Right-to-Left formatting and contextual nuances.
*   **Hardware Diagnostics:** Includes scripts to verify CUDA allocation and ensure the local GPU is correctly utilized.

## 🛠 Tech Stack & Models
*   **Core Logic:** Python 3.x
*   **Local LLMs Tested (GGUF Format):** 
    *   `Meta-Llama-3.1-8B-Instruct` (Q4_K_M & Q8_0)
    *   `Mistral-7B-Instruct-v0.2`
    *   `Phi-3-mini-4k-instruct-q4`
*   **Hugging Face Audio Models:** `pyannote/speaker-diarization-3.1,
ivrit-ai/whisper-large-v3-ct2`

## 📁 Repository Structure
The project has evolved through trial and error, and the repository is structured to reflect this iterative process:

*   `src/v2/`: **The active, working codebase.** Contains the refined architecture and current processing scripts.
*   `archive/v1/`: Legacy code and initial experimental scripts (kept for reference and to document the learning process).
*   `output/`: Directory where the system automatically routes generated data:
    *   `output/diarization/`: Raw speaker identification metadata (e.g., `diarization.json`).
    *   `output/captions/`: Formatted subtitle files (e.g., `captions_dt.srt`).
    *   `output/video/`: Final muxed video outputs for testing.

## 📈 Current Status & Performance
 * **Status:** Standalone Work in Progress (WIP). The system currently processes video and outputs subtitle files independently. The next major development phase will focus on leveraging AI to automatically identify and extract interesting, highlight-worthy clips directly from the generated and modified `.srt` files. * 
* **Performance:** Processing times are highly dependent on the individual machine's CPU and GPU capabilities. On average, for a high-performance PC, the system requires approximately 20 seconds of processing time per 1 minute of video. 


