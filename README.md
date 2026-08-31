
# 🎙️ Kyutai Pocket-TTS (Enhanced Edition)

An enhanced Text-to-Speech (TTS) studio built upon **Kyutai Pocket-TTS**. It includes **automated background noise separation**, **in-browser microphone voice cloning**, **custom inline pause tags**, and **GPU/CPU optimization**.

---
title: Pocket-TTS Enhanced Voice Studio
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.28.3
app_file: app.py
pinned: false
license: mit
tags:
  - text-to-speech
  - tts
  - voice-cloning
  - audio-cleaning
  - kyutai
---

> [!NOTE]
> **Attribution & Credits:**
> This repository is an enhanced spinoff built on top of **[Kyutai's Pocket-TTS](https://github.com/kyutai-labs/pocket-tts)** created by [Kyutai Labs](https://kyutai.org).
> 
> * **Original Model:** [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts)
> * **Tech Report:** [Kyutai Blog](https://kyutai.org/blog/2026-01-13-pocket-tts)
> * **Original Codebase:** [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts)
> * **License:** MIT License

---

## ⚡ Enhanced Features
* **🌍 Dynamic Multi-Language Switcher:** Seamlessly switch between English, French (`french_24l`), German (`german_24l`), Spanish (`spanish_24l`), Italian (`italian_24l`), and Portuguese (`portuguese_24l`) checkpoints directly from the UI dropdown.
* **📄 Script File Upload:** Upload `.txt`, `.md`, or `.srt` script files directly into the UI text prompt area.
* **⏸️ Flexible Inline Pause Control:** Insert precise silence gaps in text using tags like `[5 sec]`, `[5s]`, `[500ms]`, `[pause 2.5s]`, or `[silence 3 sec]` with zero model inference cost during silences.
* **🧽 Automatic Background Noise Separation:** Integrated spectral subtraction, high-pass filtering (80Hz), and dynamic RMS noise gating to produce pristine voice clones even from noisy microphone recordings.
* **🎙️ Live Microphone Cloning:** Record 5-10 seconds of audio directly in the browser and instantly create a custom voice preset.
* **🎭 TTS Model Nuance Controls:** Fine-tune expression using **LSD Decode Steps** (acoustic trajectory refinement), **Noise Clamp** (prevents sampling noise spikes), **Natural Punctuation Micro-Breaths** (adds subtle breath gaps at commas/ellipses), and **Context Warm-up** (prevents abrupt sentence starts).
* **🚀 Hardware Acceleration:** Supports CUDA execution on NVIDIA GPUs (e.g., GTX 1050 Ti) and dynamic int8 quantization on CPU.

---

## 📁 Repository Structure
```
pocket-tts-enhanced/
├── app.py                         # Lightweight Studio Entry Point
├── audio_cleaner.py               # Standalone Audio Denoising CLI Tool
├── src/
│   └── studio/                    # Core Modular Studio Package
│       ├── config.py              # System Settings & Constants
│       ├── audio/                 # DSP Audio Denoising & Resampling
│       │   └── cleaner.py
│       ├── tts/                   # Engine Manager, Pause Parser, Speech Generator
│       │   ├── model.py
│       │   ├── parser.py
│       │   └── generator.py
│       └── ui/                    # Gradio Web Interface
│           └── layout.py
├── requirements.txt               # Dependencies
├── .gitignore                     # Git Exclusions
├── README.md                      # Hugging Face Card & Documentation
├── voice_cloning_enhancements.md  # Detailed Voice Cloning Quality Guide
└── hardware_features.md           # Hardware Optimization Guide
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Hugging Face Model Access (In-UI Authentication)
Preset built-in voices work out of the box. Full **custom voice cloning weights** are gated on Hugging Face and can be unlocked directly inside the web studio UI:

1. Visit **[huggingface.co/kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts)** and click **"Agree and accept conditions"**.
2. Copy a User Access Token from **[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)**.
3. Launch the studio (`python app.py`) and open **`http://localhost:7860`**.
4. Expand the **🔑 Unlock Voice Cloning** box at the top, paste your token (`hf_...`), and click **Authenticate & Download Weights**.

The studio will automatically download and unlock voice cloning weights in your browser session without needing terminal or CLI setup.

---

### 3. Run the Web Studio
```bash
python app.py
```
Open **`http://localhost:7860`** in your browser.

---

### 4. Clean Standalone Audio File via CLI
```bash
python audio_cleaner.py <input_audio.wav> [output_cleaned.wav]
```

---

## 📜 License
Licensed under the [MIT License](LICENSE) (matching Kyutai Pocket-TTS).
