"""
Model Manager module for loading, GPU acceleration, and HF authentication.
"""

import os
import torch
from pocket_tts import TTSModel
from ..config import DEFAULT_LANGUAGE

class ModelManager:
    """Singleton/Wrapper manager for Kyutai Pocket-TTS Model."""

    def __init__(self, language: str = DEFAULT_LANGUAGE):
        self.language = language
        self.model = None
        self._load_initial_model()

    def _load_initial_model(self):
        print(f"[INFO] Loading Kyutai Pocket-TTS Model ({self.language})...")
        hf_token_env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token_env:
            print("[INFO] HF_TOKEN detected in environment.")

        self.model = TTSModel.load_model(language=self.language)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print("[INFO] CUDA Acceleration Enabled.")

        print(f"[INFO] Model Voice Cloning Status: {self.model.has_voice_cloning}")

    def authenticate_token(self, token_input: str):
        """Authenticates Hugging Face token dynamically and reloads weights."""
        token = token_input.strip()
        if not token:
            return "❌ Please enter a valid Hugging Face Access Token (starting with hf_...).", "⚠️ Voice Cloning Locked"

        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

        print("[INFO] Attempting model reload with provided HF_TOKEN...")
        try:
            new_model = TTSModel.load_model(language=self.language)
            if torch.cuda.is_available():
                new_model = new_model.to("cuda")
            self.model = new_model

            if self.model.has_voice_cloning:
                print("[INFO] Voice cloning model weights successfully downloaded!")
                return (
                    "✅ **Success!** Voice cloning model weights downloaded & unlocked. You can now clone voices!",
                    "✅ Voice Cloning Active & Unlocked!"
                )
            else:
                return (
                    "❌ **Authentication incomplete:** The token was set, but voice cloning weights could not be downloaded.\n"
                    "Did you click 'Agree and accept conditions' on https://huggingface.co/kyutai/pocket-tts ?",
                    "⚠️ Voice Cloning Locked"
                )
        except Exception as e:
            return f"❌ Error downloading weights: {str(e)}", "⚠️ Voice Cloning Locked"

    def switch_language(self, language_display_name: str) -> str:
        """Dynamically switches TTS model language checkpoint."""
        from ..config import SUPPORTED_LANGUAGES
        target_lang = SUPPORTED_LANGUAGES.get(language_display_name, "english")

        if self.language == target_lang:
            return f"[INFO] Model language is already set to **{language_display_name}**."

        print(f"[INFO] Switching model language from '{self.language}' to '{target_lang}'...")
        try:
            new_model = TTSModel.load_model(language=target_lang)
            if torch.cuda.is_available():
                new_model = new_model.to("cuda")
            self.model = new_model
            self.language = target_lang
            print(f"[INFO] Model language successfully switched to '{target_lang}'.")
            return f"**[Language Switched]** Active model: **{language_display_name}** (`{target_lang}`)"
        except Exception as e:
            return f"[ERROR] Error switching model language to '{target_lang}': {str(e)}"

    def create_master_voice_embedding(self, audio_paths: list[str], output_name: str) -> str:
        """
        Extracts speaker latents from multiple audio prompt clips, computes average master latent,
        and saves as a reusable .safetensors profile in voices/ directory.
        """
        import re
        from pocket_tts.models.tts_model import export_model_state
        from ..config import VOICES_DIR

        valid_paths = [p for p in audio_paths if p and os.path.exists(p)]
        if not valid_paths:
            raise ValueError("No valid audio files provided for ensembling.")

        clean_name = re.sub(r"[^\w\-]", "_", output_name.strip())
        if not clean_name:
            clean_name = "custom_master_voice"
        output_file = VOICES_DIR / f"{clean_name}.safetensors"

        print(f"[INFO] Computing master voice latent average from {len(valid_paths)} audio prompt clips...")
        states = [self.model.get_state_for_audio_prompt(path, truncate=True) for path in valid_paths]

        master_state = {}
        first_state = states[0]

        for module_name in first_state:
            master_state[module_name] = {}
            for key in first_state[module_name]:
                tensors = [s[module_name][key] for s in states if module_name in s and key in s[module_name]]
                if len(tensors) > 0 and tensors[0].is_floating_point():
                    master_state[module_name][key] = torch.stack(tensors).mean(dim=0)
                else:
                    master_state[module_name][key] = first_state[module_name][key]

        export_model_state(master_state, str(output_file))
        print(f"[INFO] Master voice profile saved to '{output_file}'.")
        return str(output_file)

    def get_saved_voices(self) -> list[str]:
        """Returns list of saved .safetensors voice profile paths in voices/ directory."""
        from ..config import VOICES_DIR
        if not VOICES_DIR.exists():
            return []
        return [str(f.resolve()) for f in VOICES_DIR.glob("*.safetensors")]

    @property
    def has_voice_cloning(self) -> bool:
        return self.model.has_voice_cloning if self.model else False

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate if self.model else 24000


