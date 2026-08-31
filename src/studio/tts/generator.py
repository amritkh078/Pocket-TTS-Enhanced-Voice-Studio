"""
Speech generator pipeline for Pocket-TTS Enhanced Studio.
"""

import os
import numpy as np
import torch
import gradio as gr
from .model import ModelManager
from .parser import apply_prosody_nuances, parse_script_tokens
from ..config import CLEANED_VOICE_CACHE, EMOTION_PRESETS
from ..audio import apply_broadcast_mastering, apply_vocal_character

def get_emotion_params(emotion_key: str, base_temp: float, base_lsd: int, base_noise: float):
    """Resolves Flow-LM hyperparameter modulation for a given emotion style key."""
    if not emotion_key:
        return base_temp, base_lsd, base_noise, 1.0
    
    clean_key = emotion_key.lower().strip()
    for name, params in EMOTION_PRESETS.items():
        if clean_key in name.lower() or name.lower().startswith(clean_key):
            return params["temp"], params["lsd_steps"], params["noise_clamp"], params.get("pause_mult", 1.0)
            
    return base_temp, base_lsd, base_noise, 1.0

def generate_speech(
    model_manager: ModelManager,
    text: str,
    voice_input,
    voice_preset: str,
    saved_voice_choice: str,
    emotion_style: str,
    vocal_character: str = "Neutral / Balanced",
    bass_boost: float = 0.0,
    temp: float = 0.22,
    lsd_steps: int = 2,
    noise_clamp: float = 1.5,
    micro_breaths: bool = True,
    context_warmup: bool = True,
    mastering_cb: bool = True
):
    """Executes end-to-end speech generation pipeline with emotion and character conditioning."""
    if not text or not text.strip():
        raise gr.Error("Please enter or upload a script to generate speech.")

    tts_model = model_manager.model

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

    # Parse inline emotion and pause tags
    parsed_items = parse_script_tokens(processed_text, default_emotion=emotion_style)
    audio_chunks = []

    for item_type, content, active_emotion in parsed_items:
        # Modulate Flow-LM parameters per active segment emotion
        e_temp, e_lsd, e_noise, pause_mult = get_emotion_params(active_emotion, temp, lsd_steps, noise_clamp)
        tts_model.temp = float(e_temp)
        tts_model.lsd_decode_steps = int(e_lsd)
        tts_model.noise_clamp = float(e_noise) if e_noise > 0 else None

        if item_type == "text":
            chunk = tts_model.generate_audio(model_state=voice_state, text_to_generate=content)
            if chunk.device != torch.device("cpu"):
                chunk = chunk.cpu()
            audio_chunks.append(chunk.numpy())
        elif item_type == "pause":
            duration_sec = float(content) * pause_mult
            num_samples = int(tts_model.sample_rate * duration_sec)
            silent_chunk = np.zeros(num_samples, dtype=np.float32)
            audio_chunks.append(silent_chunk)

    if not audio_chunks:
        raise gr.Error("No valid text found in script.")

    final_audio = np.concatenate(audio_chunks)

    # Apply Vocal Character (Heavy Bass / Resonance EQ / Brightness)
    final_audio = apply_vocal_character(
        final_audio,
        sr=tts_model.sample_rate,
        character_style=vocal_character,
        bass_boost_db=bass_boost
    )

    # Optional Studio Broadcast Mastering Filter
    if mastering_cb:
        final_audio = apply_broadcast_mastering(final_audio, sr=tts_model.sample_rate)

    return tts_model.sample_rate, final_audio
