"""
CLI Audio Cleaning & Background Noise Separation Tool for Pocket-TTS Enhanced
-------------------------------------------------------------------------
Usage:
    python audio_cleaner.py <input_audio_path> [output_audio_path]
"""

import sys
from src.studio.audio.cleaner import (
    clean_audio_file,
    load_audio,
    resample_audio,
    highpass_filter,
    spectral_denoise,
    noise_gate,
    normalize_peak
)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_cleaner.py <input_audio_path> [output_audio_path]")
        sys.exit(1)
        
    inp = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) > 2 else None
    clean_audio_file(inp, outp)
