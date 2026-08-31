"""
TTS Engine, Model Manager, and Script Parser Package.
"""

from .model import ModelManager
from .parser import parse_script_pauses, apply_prosody_nuances, load_script_text
from .generator import generate_speech

__all__ = [
    "ModelManager",
    "parse_script_pauses",
    "apply_prosody_nuances",
    "load_script_text",
    "generate_speech"
]
