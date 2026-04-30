import torch
import gc
import os
import io
from contextlib import redirect_stdout

# --- Configuration ---
# We use a tiny model for a quick, low-impact test.
# This avoids large downloads just for a verification check.
TINY_WHISPER_MODEL = "Systran/faster-whisper-tiny-en"
MODEL_CACHE_DIR = "D:\\downloaded_models"


def check_pytorch_cuda():
    """Checks if PyTorch can see the CUDA device."""
    print("--- 1. Verifying PyTorch and CUDA ---")
    is_available = torch.cuda.is_available()
    if is_available:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ SUCCESS: PyTorch found {gpu_count} CUDA device(s).")
        print(f"   - Device 0: {gpu_name}")
    else:
        print("❌ FAILED: PyTorch could NOT find any CUDA devices.")
        print("   - Please ensure PyTorch was installed with CUDA support.")
    print("-" * 35)
    return is_available


def check_faster_whisper():
    """Checks if faster-whisper can load a model onto the GPU."""
    print("--- 2. Verifying faster-whisper (CTranslate2) ---")
    try:
        from faster_whisper import WhisperModel
        print("Import successful. Attempting to load a tiny model on GPU...")

        model = WhisperModel(
            TINY_WHISPER_MODEL,
            device="cuda",
            compute_type="float16",
            download_root=MODEL_CACHE_DIR
        )

        print("✅ SUCCESS: faster-whisper successfully loaded a model onto the GPU.")

        # Clean up to release VRAM
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print("   - Model unloaded and VRAM released.")

    except ImportError:
        print("❌ FAILED: The 'faster_whisper' library is not installed.")
    except Exception as e:
        print(f"❌ FAILED: An error occurred while loading the faster-whisper model on the GPU.")
        print(f"   - Error details: {e}")
        print("   - This often means the ctranslate2 backend is not CUDA-enabled.")
    print("-" * 35)


def check_llama_cpp_python():
    """
    Checks if llama-cpp-python was built with CUDA (cuBLAS) support.
    It does this by capturing the library's initialization output.
    """
    print("--- 3. Verifying llama-cpp-python ---")
    try:
        from llama_cpp import Llama

        # We capture the C-level stdout from the library to check for CUDA messages.
        f = io.StringIO()
        with redirect_stdout(f):
            # We initialize with a fake path. We expect a file error, but we just
            # want to see the library's startup messages before the error occurs.
            try:
                Llama(model_path="D:\\fake_model_path.gguf", verbose=True)
            except Exception:
                # We ignore the actual error (e.g., file not found) because we don't care about it.
                pass

        output = f.getvalue()

        # The key indicator of a successful CUDA build is this line.
        if "ggml_init_cublas: found 1 CUDA devices" in output or "ggml_cuda_init: found 1 CUDA devices" in output:
            print("✅ SUCCESS: llama-cpp-python was built with CUDA (cuBLAS) support.")
            print("   - The library is ready for GPU offloading.")
        else:
            print("❌ FAILED: llama-cpp-python was NOT built with CUDA support.")
            print("   - The library is running in CPU-only mode.")
            print("   - You must reinstall it from source with the CUDA flag enabled.")

    except ImportError:
        print("❌ FAILED: The 'llama_cpp' library is not installed.")
    except Exception as e:
        print(f"❌ FAILED: An unexpected error occurred while testing llama-cpp-python.")
        print(f"   - Error details: {e}")
    print("-" * 35)


if __name__ == "__main__":
    print("\nStarting GPU Verification Script...")
    print("===================================\n")

    if check_pytorch_cuda():
        # Only test the other libraries if the base CUDA setup is working in PyTorch.
        check_faster_whisper()
        check_llama_cpp_python()
    else:
        print("Skipping further checks as the fundamental PyTorch CUDA setup has failed.")

    print("\nVerification complete.\n")