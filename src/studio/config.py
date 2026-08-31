"""
Configuration constants and defaults for Pocket-TTS Enhanced Studio.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLEANED_VOICE_CACHE = str(BASE_DIR / "cached_cleaned_voice.wav")
VOICES_DIR = BASE_DIR / "voices"
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# Audio Constants
SAMPLE_RATE = 24000  # Pocket-TTS Mimi native sampling rate (Hz)
HIGH_PASS_CUTOFF = 80.0  # Hz
NOISE_GATE_THRESHOLD = -35.0  # dB
TARGET_PEAK_NORM = 0.95

# Model Defaults
DEFAULT_LANGUAGE = "english"
DEFAULT_TEMP = 0.22
DEFAULT_LSD_STEPS = 2
DEFAULT_NOISE_CLAMP = 1.5

PRESET_VOICES = ["alba", "ann", "azelma", "george", "jean", "paul", "estelle"]

SUPPORTED_LANGUAGES = {
    "English (Default)": "english",
    "French (Français 24L)": "french_24l",
    "German (Deutsch 24L)": "german_24l",
    "Spanish (Español 24L)": "spanish_24l",
    "Italian (Italiano 24L)": "italian_24l",
    "Portuguese (Português 24L)": "portuguese_24l"
}

