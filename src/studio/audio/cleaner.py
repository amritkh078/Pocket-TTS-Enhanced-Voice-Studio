"""
Audio Cleaning & Background Noise Separation Tool for Pocket-TTS Enhanced Studio
---------------------------------------------------------------------------------
Provides functions for:
1. Converting audio to Mono and 24kHz (native Pocket-TTS Mimi rate).
2. Applying an 80Hz High-Pass Filter to eliminate low-frequency hum & mic pops.
3. Applying Spectral Subtraction & Noise Gating to separate background noise from voice.
4. Normalizing peak amplitude to 0.95 for optimal voice cloning.
"""

import os
import wave
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from ..config import SAMPLE_RATE, HIGH_PASS_CUTOFF, NOISE_GATE_THRESHOLD, TARGET_PEAK_NORM

def load_audio(file_path: str):
    """Loads audio file and returns (sample_rate, numpy_float32_array)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    sr, data = wavfile.read(file_path)
    
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    return sr, data

def resample_audio(data: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resamples audio array to target sample rate (24kHz for Pocket-TTS)."""
    if orig_sr == target_sr:
        return data
    num_output_samples = int(round(len(data) * target_sr / orig_sr))
    resampled = signal.resample(data, num_output_samples)
    return resampled.astype(np.float32)

def highpass_filter(data: np.ndarray, sr: int = SAMPLE_RATE, cutoff: float = HIGH_PASS_CUTOFF) -> np.ndarray:
    """Cuts low-frequency AC hum, wind, and mic rumble below cutoff frequency (80 Hz)."""
    sos = signal.butter(4, cutoff, btype='highpass', fs=sr, output='sos')
    return signal.sosfilt(sos, data)

def spectral_denoise(data: np.ndarray, sr: int = SAMPLE_RATE, noise_reduction_factor: float = 0.75) -> np.ndarray:
    """
    Performs Spectral Subtraction Denoising:
    1. Estimates noise profile from lowest-energy initial frames.
    2. Subtracts noise magnitude spectrum in STFT domain.
    3. Reconstructs audio with Inverse STFT.
    """
    nperseg = int(sr * 0.032)  # 32ms window
    noverlap = nperseg // 2
    
    f, t, Zxx = signal.stft(data, fs=sr, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    frame_energies = np.sum(magnitude ** 2, axis=0)
    quietest_frame_indices = np.argsort(frame_energies)[:max(3, int(len(frame_energies) * 0.1))]
    noise_profile = np.mean(magnitude[:, quietest_frame_indices], axis=1, keepdims=True)
    
    subtracted_mag = magnitude - (noise_reduction_factor * noise_profile)
    cleaned_mag = np.maximum(subtracted_mag, 0.05 * magnitude)
    
    Zxx_cleaned = cleaned_mag * np.exp(1j * phase)
    _, cleaned_audio = signal.istft(Zxx_cleaned, fs=sr, nperseg=nperseg, noverlap=noverlap)
    
    if len(cleaned_audio) > len(data):
        cleaned_audio = cleaned_audio[:len(data)]
    elif len(cleaned_audio) < len(data):
        cleaned_audio = np.pad(cleaned_audio, (0, len(data) - len(cleaned_audio)))
        
    return cleaned_audio.astype(np.float32)

def noise_gate(data: np.ndarray, sr: int = SAMPLE_RATE, threshold_db: float = NOISE_GATE_THRESHOLD) -> np.ndarray:
    """Applies a dynamic noise gate to attenuate background noise during silent gaps."""
    threshold = 10 ** (threshold_db / 20.0)
    frame_size = int(sr * 0.02)
    gated_data = data.copy()
    
    for i in range(0, len(data), frame_size):
        frame = data[i:i + frame_size]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
        if rms < threshold:
            gated_data[i:i + frame_size] *= 0.1
            
    return gated_data

def normalize_peak(data: np.ndarray, target_peak: float = TARGET_PEAK_NORM) -> np.ndarray:
    """Peak normalizes audio signal."""
    max_peak = np.max(np.abs(data))
    if max_peak > 0:
        data = data * (target_peak / max_peak)
    return data

def trim_silence(data: np.ndarray, sr: int = SAMPLE_RATE, threshold_db: float = -40.0) -> np.ndarray:
    """Trims leading and trailing silence below threshold_db so speaker prompt starts on speech."""
    threshold = 10 ** (threshold_db / 20.0)
    abs_data = np.abs(data)
    non_silent_indices = np.where(abs_data > threshold)[0]
    if len(non_silent_indices) > 0:
        start_idx = max(0, non_silent_indices[0] - int(sr * 0.05))  # 50ms padding
        end_idx = min(len(data), non_silent_indices[-1] + int(sr * 0.05))
        return data[start_idx:end_idx]
    return data

def apply_broadcast_mastering(data: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Applies professional studio mastering chain to 24kHz generated speech PCM:
    1. Subsonic High-Pass Filter at 60Hz (cuts DC offset and low rumble).
    2. Presence EQ boost (+2.5 dB peak filter at 4kHz) for consonant crispness.
    3. Soft Dynamic Peak Compression & Limiting.
    """
    # 1. 60Hz Subsonic HP Filter
    sos_hp = signal.butter(4, 60.0, btype='highpass', fs=sr, output='sos')
    hp_data = signal.sosfilt(sos_hp, data)

    # 2. Presence EQ Boost (+2.5dB at 4000Hz)
    fc = 4000.0
    Q = 1.0
    gain_db = 2.5
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * Q)
    
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0/a0, b1/a0, b2/a0], dtype=np.float32)
    a = np.array([1.0, a1/a0, a2/a0], dtype=np.float32)
    eq_data = signal.lfilter(b, a, hp_data)

    # 3. Soft Dynamic Peak Compression & Limiting
    rms = np.sqrt(np.mean(eq_data**2) + 1e-12)
    if rms > 0:
        target_rms = 0.15
        scaling = min(1.5, target_rms / rms)
        eq_data = eq_data * scaling
    
    mastered = np.clip(eq_data, -0.98, 0.98)
    return mastered.astype(np.float32)

def apply_vocal_character(
    data: np.ndarray,
    sr: int = SAMPLE_RATE,
    character_style: str = "Neutral / Balanced",
    bass_boost_db: float = 0.0
) -> np.ndarray:
    """
    Applies vocal timbre and character equalization:
    - Low-shelf biquad filter for bass boost (150Hz)
    - High-shelf biquad filter for presence (5000Hz)
    """
    if data is None or len(data) == 0:
        return data

    total_bass = float(bass_boost_db)
    if "Heavy" in character_style or "Deep" in character_style:
        total_bass += 5.0
    elif "Radio" in character_style or "Trailer" in character_style:
        total_bass += 6.5
    elif "Warm" in character_style:
        total_bass += 3.5

    presence_boost = 0.0
    if "Radio" in character_style or "Trailer" in character_style:
        presence_boost += 3.0
    elif "Bright" in character_style or "Crisp" in character_style:
        presence_boost += 4.5

    # 1. Low Shelf Bass Filter (fc = 150Hz)
    if abs(total_bass) > 0.1:
        fc = 150.0
        A = 10 ** (total_bass / 40.0)
        w0 = 2 * np.pi * fc / sr
        alpha = np.sin(w0) / (2.0 * 0.707)
        
        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

        b = np.array([b0/a0, b1/a0, b2/a0], dtype=np.float32)
        a = np.array([1.0, a1/a0, a2/a0], dtype=np.float32)
        data = signal.lfilter(b, a, data)

    # 2. High Shelf Presence Filter (fc = 5000Hz)
    if abs(presence_boost) > 0.1:
        fc2 = 5000.0
        A2 = 10 ** (presence_boost / 40.0)
        w02 = 2 * np.pi * fc2 / sr
        alpha2 = np.sin(w02) / (2.0 * 0.707)

        b0 = A2 * ((A2 + 1) + (A2 - 1) * np.cos(w02) + 2 * np.sqrt(A2) * alpha2)
        b1 = -2 * A2 * ((A2 - 1) + (A2 + 1) * np.cos(w02))
        b2 = A2 * ((A2 + 1) + (A2 - 1) * np.cos(w02) - 2 * np.sqrt(A2) * alpha2)
        a0 = (A2 + 1) - (A2 - 1) * np.cos(w02) + 2 * np.sqrt(A2) * alpha2
        a1 = 2 * ((A2 - 1) - (A2 + 1) * np.cos(w02))
        a2 = (A2 + 1) - (A2 - 1) * np.cos(w02) - 2 * np.sqrt(A2) * alpha2

        b = np.array([b0/a0, b1/a0, b2/a0], dtype=np.float32)
        a = np.array([1.0, a1/a0, a2/a0], dtype=np.float32)
        data = signal.lfilter(b, a, data)

    return np.clip(data, -0.98, 0.98).astype(np.float32)

def process_audio_numpy(audio_data: np.ndarray, sr: int) -> np.ndarray:
    """Runs full cleaning pipeline on in-memory numpy audio array."""
    # Convert integer formats to float32 [-1.0, 1.0]
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_data = audio_data.astype(np.float32)

    # Convert Stereo to Mono
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    data_24k = resample_audio(audio_data, sr, target_sr=SAMPLE_RATE)
    trimmed = trim_silence(data_24k, sr=SAMPLE_RATE, threshold_db=-40.0)
    filtered = highpass_filter(trimmed, sr=SAMPLE_RATE, cutoff=HIGH_PASS_CUTOFF)
    denoised = spectral_denoise(filtered, sr=SAMPLE_RATE, noise_reduction_factor=0.75)
    gated = noise_gate(denoised, sr=SAMPLE_RATE, threshold_db=NOISE_GATE_THRESHOLD)
    cleaned = normalize_peak(gated, target_peak=TARGET_PEAK_NORM)
    return cleaned

def clean_audio_file(input_path: str, output_path: str = None) -> str:
    """Main cleaning pipeline for an audio file."""
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_cleaned.wav"
        
    print(f"Loading '{input_path}'...")
    sr, raw_data = load_audio(input_path)
    
    print(f"Resampling from {sr}Hz to {SAMPLE_RATE}Hz (Pocket-TTS Mimi native rate)...")
    cleaned_data = process_audio_numpy(raw_data, sr)
    
    int16_data = (cleaned_data * 32767.0).clip(-32768, 32767).astype(np.int16)
    wavfile.write(output_path, SAMPLE_RATE, int16_data)
    
    print(f"[SUCCESS] Cleaned audio successfully saved to: '{output_path}'")
    return output_path
