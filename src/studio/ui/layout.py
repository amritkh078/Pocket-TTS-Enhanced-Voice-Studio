import os
import scipy.io.wavfile as wavfile
import gradio as gr
from ..config import (
    PRESET_VOICES,
    SUPPORTED_LANGUAGES,
    EMOTION_PRESETS,
    VOCAL_CHARACTERS,
    CLEANED_VOICE_CACHE,
    DEFAULT_TEMP,
    DEFAULT_LSD_STEPS,
    DEFAULT_NOISE_CLAMP
)
from ..audio import process_audio_numpy
from ..tts import ModelManager, load_script_text, generate_speech

def create_studio_app(model_manager: ModelManager = None) -> gr.Blocks:
    """Builds and returns the full Gradio Blocks application."""
    if model_manager is None:
        model_manager = ModelManager()

    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")

    def handle_auto_clean(voice_input):
        """Auto-cleans uploaded or recorded voice sample."""
        if voice_input is None:
            return None, "No voice sample loaded."
        sr, raw_data = voice_input
        print("[INFO] Auto-cleaning loaded audio sample...")
        cleaned_audio = process_audio_numpy(raw_data, sr)
        
        int16_data = (cleaned_audio * 32767.0).clip(-32768, 32767).astype(wavfile.np.int16)
        wavfile.write(CLEANED_VOICE_CACHE, 24000, int16_data)
        
        return (24000, int16_data), "[SUCCESS] Audio automatically cleaned & background noise separated!"

    def handle_speech_generation(
        text, voice_input, voice_preset, saved_voice, emotion_style, character_style, bass_boost, temp, lsd_steps, noise_clamp, micro_breaths, context_warmup, mastering
    ):
        return generate_speech(
            model_manager=model_manager,
            text=text,
            voice_input=voice_input,
            voice_preset=voice_preset,
            saved_voice_choice=saved_voice,
            emotion_style=emotion_style,
            vocal_character=character_style,
            bass_boost=bass_boost,
            temp=temp,
            lsd_steps=lsd_steps,
            noise_clamp=noise_clamp,
            micro_breaths=micro_breaths,
            context_warmup=context_warmup,
            mastering_cb=mastering
        )

    def handle_create_ensemble(clip1, clip2, clip3, voice_name):
        paths = [p for p in [clip1, clip2, clip3] if p]
        if not paths:
            return "❌ Please upload or record at least 1 reference clip.", gr.update()
        try:
            saved_path = model_manager.create_master_voice_embedding(paths, voice_name)
            voices = ["None"] + model_manager.get_saved_voices()
            return f"✅ **Success!** Master Voice Profile created & saved: `{saved_path}`", gr.update(choices=voices, value=saved_path)
        except Exception as e:
            return f"❌ Error creating master voice: {str(e)}", gr.update()

    def handle_refresh_voices():
        voices = ["None"] + model_manager.get_saved_voices()
        return gr.update(choices=voices)

    def make_emotion_trigger(emotion_name, tag_val):
        def _handler(current_text):
            curr = (current_text or "").rstrip()
            new_text = f"{curr} [emotion: {tag_val}] " if curr else f"[emotion: {tag_val}] "
            return new_text, emotion_name
        return _handler

    with gr.Blocks(theme=theme, title="Pocket-TTS Voice Studio") as demo:
        gr.Markdown(
            """
            # 🎙️ Pocket-TTS Enhanced Voice Studio
            ### Powered by **[Kyutai Pocket-TTS](https://github.com/kyutai-labs/pocket-tts)**
            """
        )

        with gr.Row():
            lang_dropdown = gr.Dropdown(
                label="🌍 Select Model Language Checkpoint",
                choices=list(SUPPORTED_LANGUAGES.keys()),
                value="English (Default)",
                scale=2
            )
            emotion_dropdown = gr.Dropdown(
                label="🎭 Primary Vocal Emotion & Style",
                choices=list(EMOTION_PRESETS.keys()),
                value="Neutral / Natural",
                scale=2
            )
            character_dropdown = gr.Dropdown(
                label="🎙️ Vocal Character Timbre (Bass & Resonance)",
                choices=list(VOCAL_CHARACTERS.keys()),
                value="Neutral / Balanced",
                scale=2
            )
        lang_status = gr.Markdown("🌍 Active Model Language: **English (Default)** (`english`)")

        # Hugging Face Auth Box
        with gr.Accordion("🔑 Unlock Voice Cloning (Hugging Face Auth)", open=not model_manager.has_voice_cloning):
            gr.Markdown(
                "The full voice-cloning model weights are gated on Hugging Face.\n\n"
                "**Quick 2-Step Unlock:**\n"
                "1. Click **'Agree and accept conditions'** at [huggingface.co/kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts)\n"
                "2. Copy a User Access Token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and paste it below:"
            )
            with gr.Row():
                hf_token_input = gr.Textbox(
                    label="Hugging Face Access Token (HF_TOKEN)",
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    type="password",
                    scale=3
                )
                unlock_btn = gr.Button("🔑 Authenticate & Download Weights", variant="primary", scale=1)
            auth_status = gr.Markdown(
                "✅ **Voice Cloning Active & Unlocked!**" if model_manager.has_voice_cloning else "⚠️ **Voice Cloning Currently Locked**"
            )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("📄 Upload Script File (.txt / .md / .srt)", open=False):
                    with gr.Row():
                        script_file_input = gr.File(
                            label="Upload Script File",
                            file_types=[".txt", ".md", ".srt"],
                            file_count="single",
                            scale=3
                        )
                        clear_script_btn = gr.Button("🗑️ Clear Script", variant="secondary", scale=1)

                gr.Markdown("**⚡ Quick Emotion Triggers (Click button to insert tag & set active style):**")
                with gr.Row():
                    btn_happy = gr.Button("😊 Happy", variant="secondary", size="sm")
                    btn_sad = gr.Button("😢 Sad", variant="secondary", size="sm")
                    btn_angry = gr.Button("😠 Angry", variant="secondary", size="sm")
                    btn_whisper = gr.Button("🤫 Whisper", variant="secondary", size="sm")
                    btn_dramatic = gr.Button("🎭 Dramatic", variant="secondary", size="sm")
                    btn_neutral = gr.Button("😐 Neutral", variant="secondary", size="sm")

                text_input = gr.Textbox(
                    label="Script / Text Prompt (Supports inline [emotion: happy] & [5 sec] tags)",
                    placeholder="Hello! [emotion: happy] I am so excited! [5 sec] [emotion: sad] But now I am sad.",
                    lines=5,
                    value="Hello world! [emotion: happy] I am so thrilled to try emotional voice cloning! [5 sec] [emotion: dramatic] What comes next... will amaze you."
                )

                with gr.Tab("🎙️ Live Voice Recording"):
                    mic_input = gr.Audio(
                        label="Record or Upload Voice Sample (5-10 secs)",
                        sources=["microphone", "upload"],
                        type="numpy"
                    )
                    clean_status = gr.Markdown("⏳ *Record or upload an audio file above. It will be automatically cleaned when loaded!*")
                    cleaned_preview = gr.Audio(label="✨ Auto-Cleaned Voice Preview", type="numpy", interactive=False)

                with gr.Tab("💾 Saved Custom Voice Profiles"):
                    with gr.Row():
                        saved_voice_dropdown = gr.Dropdown(
                            label="Select Saved Master Voice (.safetensors)",
                            choices=["None"] + model_manager.get_saved_voices(),
                            value="None",
                            scale=3
                        )
                        refresh_voices_btn = gr.Button("🔄 Refresh List", variant="secondary", scale=1)

                with gr.Tab("🎭 Multi-Sample Ensemble & Master Voice Creator"):
                    gr.Markdown("Combine 1 to 3 audio clips of a target speaker to compute an averaged **Master Voice Profile**:")
                    with gr.Row():
                        ens_clip1 = gr.Audio(label="Clip 1 (e.g. Calm)", type="filepath")
                        ens_clip2 = gr.Audio(label="Clip 2 (e.g. Energetic)", type="filepath")
                        ens_clip3 = gr.Audio(label="Clip 3 (e.g. Monologue)", type="filepath")
                    with gr.Row():
                        voice_name_input = gr.Textbox(
                            label="Voice Profile Name",
                            placeholder="e.g. Master_Narrator",
                            value="Master_Voice_1",
                            scale=3
                        )
                        create_ensemble_btn = gr.Button("💾 Build & Save Master Profile", variant="primary", scale=1)
                    ensemble_status = gr.Markdown("⏳ *Upload audio clips above and click Build & Save!*")

                with gr.Tab("🎭 Preset Built-in Voices"):
                    preset_dropdown = gr.Dropdown(
                        label="Select Built-in Voice",
                        choices=PRESET_VOICES,
                        value="alba"
                    )

                with gr.Accordion("⚙️ Advanced Voice Quality & Model Nuances", open=False):
                    with gr.Row():
                        bass_boost_slider = gr.Slider(
                            minimum=-6.0,
                            maximum=8.0,
                            value=0.0,
                            step=0.5,
                            label="🔊 Low-End Bass Resonance Boost (dB)",
                            scale=1
                        )
                    with gr.Row():
                        temp_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.5,
                            value=DEFAULT_TEMP,
                            step=0.01,
                            label="Base Temperature (0.22 recommended)",
                            scale=1
                        )
                        lsd_slider = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=DEFAULT_LSD_STEPS,
                            step=1,
                            label="Base LSD Decode Steps (Acoustic detail refinement)",
                            scale=1
                        )
                    with gr.Row():
                        noise_clamp_slider = gr.Slider(
                            minimum=0.5,
                            maximum=3.0,
                            value=DEFAULT_NOISE_CLAMP,
                            step=0.1,
                            label="Base Noise Clamp (Limits sampling noise spikes)",
                            scale=1
                        )
                    with gr.Row():
                        micro_breaths_cb = gr.Checkbox(
                            value=True,
                            label="Natural Punctuation Micro-Breaths (Adds 120ms pauses at commas/ellipses)"
                        )
                        context_warmup_cb = gr.Checkbox(
                            value=True,
                            label="Context Warm-up (Pads short sentence starts)"
                        )
                    with gr.Row():
                        mastering_cb = gr.Checkbox(
                            value=True,
                            label="✨ Apply Studio Broadcast Mastering Filter (60Hz Cut + 4kHz Presence EQ + Soft Limiter)"
                        )

                generate_btn = gr.Button("🚀 Generate Speech with Cloned Voice", variant="primary", size="lg")

            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Generated Output Audio", type="numpy", autoplay=True)

        # Event Bindings
        lang_dropdown.change(
            fn=model_manager.switch_language,
            inputs=[lang_dropdown],
            outputs=[lang_status]
        )

        script_file_input.change(
            fn=load_script_text,
            inputs=[script_file_input],
            outputs=[text_input]
        )

        clear_script_btn.click(
            fn=lambda: "",
            inputs=[],
            outputs=[text_input]
        )

        unlock_btn.click(
            fn=model_manager.authenticate_token,
            inputs=[hf_token_input],
            outputs=[auth_status, auth_status]
        )

        mic_input.change(
            fn=handle_auto_clean,
            inputs=[mic_input],
            outputs=[cleaned_preview, clean_status]
        )

        refresh_voices_btn.click(
            fn=handle_refresh_voices,
            inputs=[],
            outputs=[saved_voice_dropdown]
        )

        create_ensemble_btn.click(
            fn=handle_create_ensemble,
            inputs=[ens_clip1, ens_clip2, ens_clip3, voice_name_input],
            outputs=[ensemble_status, saved_voice_dropdown]
        )

        # Quick Emotion Trigger Buttons
        btn_happy.click(fn=make_emotion_trigger("Happy / Energetic (😊)", "happy"), inputs=[text_input], outputs=[text_input, emotion_dropdown])
        btn_sad.click(fn=make_emotion_trigger("Sad / Melancholic (😢)", "sad"), inputs=[text_input], outputs=[text_input, emotion_dropdown])
        btn_angry.click(fn=make_emotion_trigger("Angry / Intense (😠)", "angry"), inputs=[text_input], outputs=[text_input, emotion_dropdown])
        btn_whisper.click(fn=make_emotion_trigger("Whisper / Gentle (🤫)", "whisper"), inputs=[text_input], outputs=[text_input, emotion_dropdown])
        btn_dramatic.click(fn=make_emotion_trigger("Dramatic / Storyteller (🎭)", "dramatic"), inputs=[text_input], outputs=[text_input, emotion_dropdown])
        btn_neutral.click(fn=make_emotion_trigger("Neutral / Natural", "neutral"), inputs=[text_input], outputs=[text_input, emotion_dropdown])

        generate_btn.click(
            fn=handle_speech_generation,
            inputs=[
                text_input,
                mic_input,
                preset_dropdown,
                saved_voice_dropdown,
                emotion_dropdown,
                character_dropdown,
                bass_boost_slider,
                temp_slider,
                lsd_slider,
                noise_clamp_slider,
                micro_breaths_cb,
                context_warmup_cb,
                mastering_cb
            ],
            outputs=[audio_output]
        )

    return demo

