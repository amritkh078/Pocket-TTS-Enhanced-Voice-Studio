"""
Main Entry Point for Pocket-TTS Enhanced Voice Studio
---------------------------------------------------
Run with: python app.py
"""

from src.studio.ui.layout import create_studio_app

if __name__ == "__main__":
    demo = create_studio_app()
    demo.launch()
