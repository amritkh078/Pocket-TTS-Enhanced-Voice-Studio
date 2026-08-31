# Enhancements for Near-Real Voice Cloning Quality in Pocket-TTS

To achieve ultra-realistic, natural, and near-human cloned voice speech using Pocket-TTS, enhancements must be made across **Audio Input Cleaning**, **Multi-Sample Latent Ensembling**, **Model Parameter Tuning**, **Text Prosody Control**, and **Post-Processing Filters**.

---

## 1. 🧽 Audio Input Pre-Processing (Clean Input = Real Output)

Pocket-TTS reproduces the exact acoustic environment of the input audio prompt. If the input voice sample has room echo, background noise, or low bitrates, the model will faithfully clone those flaws.

### Enhancements:
* **Background Noise & Reverb Removal:** Pass reference audio through noise suppression tools before encoding.
* **Loudness Normalization (LUFS):** Normalize prompt audio to **-23 LUFS** (or RMS peak -1 dB) so Mimi encoder receives balanced dynamic range.
* **Silence Trimming:** Trim long leading/trailing silences so the voice prompt starts immediately on speech.
* **Optimal Length:** Use **10 to 20 seconds** of clean, expressive, continuous monologue without background music or overlapping voices.

---

## 2. 🎭 Multi-Sample Latent Ensembling (Multi-Prompt Averaging)

Instead of relying on a single 5-second audio snippet, capture **2–3 different audio clips** of the target speaker (e.g. one calm clip, one energetic clip, one narrative clip).

### How to Implement:
Extract speaker latents from each clip, compute the average latent embedding, and build a **Master Cloned Voice Embedding**:

```python
import torch
from pocket_tts.models.tts_model import TTSModel, export_model_state

def create_master_voice_embedding(tts_model: TTSModel, audio_paths: list[str], output_safetensors: str):
    states = [tts_model.get_state_for_audio_prompt(path, truncate=True) for path in audio_paths]
    
    master_state = {}
    first_state = states[0]
    
    for module_name in first_state:
        master_state[module_name] = {}
        for key in first_state[module_name]:
            tensors = [s[module_name][key] for s in states if key in s[module_name]]
            if len(tensors) > 0 and tensors[0].is_floating_point():
                master_state[module_name][key] = torch.stack(tensors).mean(dim=0)
            else:
                master_state[module_name][key] = first_state[module_name][key]
                
    export_model_state(master_state, output_safetensors)
    print(f"Master voice embedding saved to {output_safetensors}")
```

---

## 3. 🎛️ Flow-LM Generation Parameter Tuning

Tuning generation hyper-parameters drastically improves pitch stability and eliminates robotic artifacts:

| Parameter | Recommended Value | Impact on Quality |
| :--- | :--- | :--- |
| **`temp` (Temperature)** | `0.20` - `0.30` | Lower temperature (e.g. 0.22) makes vocal timbre tighter and eliminates raspy voice breaks. |
| **`lsd_decode_steps`** | `2` or `3` | Slightly higher LSD steps refine acoustic trajectory details. |
| **`noise_clamp`** | `1.5` - `2.0` | Prevents extreme noise sampling during latent generation. |
| **`pad_with_spaces`** | `True` | Pads short prompts with 8 leading spaces to let the context warm up naturally. |

---

## 4. ✍️ Text Formatting, Script Upload & Prosody Injection

Pocket-TTS infers prosody (intonation, pitch shifts, stress) directly from text formatting, punctuation, and explicit pause tags:

* **Script Upload:** Upload `.txt`, `.md`, or `.srt` scripts directly into the studio prompt box.
* **Flexible Inline Pause Tags:** Insert explicit silence gaps anywhere in script text using any of these syntaxes:
  - `[5 sec]`, `[5s]`, `[2.5 sec]`
  - `[500ms]`, `[250 ms]`
  - `[pause 5 sec]`, `[pause 500ms]`, `[silence 3 sec]`
* **Natural Punctuation Micro-Breaths:** Automatically converts commas `,` into 120ms soft micro-breaths, ellipses `...` into 400ms pauses, and semicolons `;` into 200ms pauses.
* **Capitalization for Stress:** Capitalize key words or syllables for vocal emphasis.
* **Short Chunking:** Keep generated chunks under 15–20 words per sentence.

---

## 5. 🎚️ Audio Post-Processing (Broadcast Polish Filter)

Passing generated 24kHz raw PCM output through a light audio mastering chain makes cloned speech sound like a professional studio recording:

1. **High-Pass Filter (Cut below 60 Hz):** Eliminates low-frequency thumps/booms.
2. **Presence EQ Boost (+2dB around 3kHz - 6kHz):** Adds clarity to consonant sounds.
3. **Soft Dynamic Compression:** Evens out loud and quiet words.
