"""
Audio cleaning & DSP processing package.
"""

from .cleaner import (
    load_audio,
    resample_audio,
    highpass_filter,
    spectral_denoise,
    noise_gate,
    normalize_peak,
    trim_silence,
    apply_broadcast_mastering,
    clean_audio_file,
    process_audio_numpy
)

__all__ = [
    "load_audio",
    "resample_audio",
    "highpass_filter",
    "spectral_denoise",
    "noise_gate",
    "normalize_peak",
    "trim_silence",
    "apply_broadcast_mastering",
    "clean_audio_file",
    "process_audio_numpy"
]
