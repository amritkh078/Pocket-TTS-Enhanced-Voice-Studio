"""
Speech generator pipeline for Pocket-TTS Enhanced Studio.
"""

import os
import numpy as np
import torch
import gradio as gr
from .model import ModelManager
from .parser import apply_prosody_nuances, parse_script_pauses
from ..config import CLEANED_VOICE_CACHE
from ..audio import apply_broadcast_mastering

def generate_speech(
    model_manager: ModelManager,
    text: str,
    voice_input,
    voice_preset: str,
    saved_voice_choice: str,
    temp: float,
    lsd_steps: int,
    noise_clamp: float,
    micro_breaths: bool,
    context_warmup: bool,
    mastering_cb: bool = True
):
    """Executes end-to-end speech generation pipeline."""
    if not text or not text.strip():
        raise gr.Error("Please enter or upload a script to generate speech.")

    tts_model = model_manager.model
    tts_model.temp = float(temp)
    tts_model.lsd_decode_steps = int(lsd_steps)
    tts_model.noise_clamp = float(noise_clamp) if noise_clamp > 0 else None

    # Pre-process text prosody nuances
    processed_text = apply_prosody_nuances(text, micro_breaths=micro_breaths, context_warmup=context_warmup)

    # Voice state extraction priority:
    # 1. Saved Custom Voice Profile (.safetensors)
    # 2. Live Recording / Uploaded prompt
    # 3. Built-in Preset Voice
    if saved_voice_choice and saved_voice_choice != "None" and os.path.exists(saved_voice_choice):
        if not tts_model.has_voice_cloning:
            raise gr.Error("Voice cloning weights not unlocked yet!")
        voice_state = tts_model.get_state_for_audio_prompt(saved_voice_choice, truncate=True)
    elif voice_input is not None:
        if not tts_model.has_voice_cloning:
            raise gr.Error(
                "Voice cloning model weights not unlocked yet! Accept terms at "
                "https://huggingface.co/kyutai/pocket-tts and paste your HF Token in the '🔑 Unlock Voice Cloning' box above."
            )
        if os.path.exists(CLEANED_VOICE_CACHE):
            voice_state = tts_model.get_state_for_audio_prompt(CLEANED_VOICE_CACHE, truncate=True)
        else:
            raise gr.Error("No cleaned voice sample found. Please record or upload a voice sample first.")
    else:
        voice_state = tts_model.get_state_for_audio_prompt(voice_preset)

    # Parse pauses and generate audio segments
    parsed_items = parse_script_pauses(processed_text)
    audio_chunks = []

    for item_type, content in parsed_items:
        if item_type == "text":
            chunk = tts_model.generate_audio(model_state=voice_state, text_to_generate=content)
            if chunk.device != torch.device("cpu"):
                chunk = chunk.cpu()
            audio_chunks.append(chunk.numpy())
        elif item_type == "pause":
            duration_sec = float(content)
            num_samples = int(tts_model.sample_rate * duration_sec)
            silent_chunk = np.zeros(num_samples, dtype=np.float32)
            audio_chunks.append(silent_chunk)

    if not audio_chunks:
        raise gr.Error("No valid text found in script.")

    final_audio = np.concatenate(audio_chunks)

    # Optional Studio Broadcast Mastering Filter
    if mastering_cb:
        final_audio = apply_broadcast_mastering(final_audio, sr=tts_model.sample_rate)

    return tts_model.sample_rate, final_audio
