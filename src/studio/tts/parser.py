"""
Script text & pause tag parser module.
"""

import re
from typing import List, Tuple, Union

# Regex matching [5 sec], [5s], [500ms], [pause 2.5s], [silence 3 sec], [2.5]
PAUSE_TAG_REGEX = r"(?i)\[(?:pause|silence)?\s*([\d\.]+)\s*(s|sec|secs|second|seconds|ms)?\]"

# Regex matching [emotion: happy], [style: sad], [emotion: angry], etc.
EMOTION_TAG_REGEX = r"(?i)\[(?:emotion|style)\s*:\s*([\w\s/😊😢😠🤫🎭\-]+)\]"

def load_script_text(file_obj) -> str:
    """Reads uploaded text/markdown/srt script file."""
    if file_obj is None:
        return ""
    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def apply_prosody_nuances(text: str, micro_breaths: bool = True, context_warmup: bool = True) -> str:
    """Applies punctuation micro-breaths and leading context space padding."""
    if context_warmup and not text.startswith(" "):
        text = "        " + text

    if micro_breaths:
        text = re.sub(r',\s*(?!\s*\[)', ', [pause 120ms] ', text)
        text = re.sub(r'\.\.\.\s*(?!\s*\[)', '... [pause 400ms] ', text)
        text = re.sub(r';\s*(?!\s*\[)', '; [pause 200ms] ', text)

    return text

def parse_script_tokens(text: str, default_emotion: str = "Neutral / Natural") -> List[Tuple[str, Union[str, float], str]]:
    """
    Parses script text into sequential items:
    ('text', text_segment, active_emotion) or ('pause', pause_duration_sec, active_emotion)
    """
    emotion_chunks = re.split(EMOTION_TAG_REGEX, text)
    current_emotion = default_emotion
    items = []

    i = 0
    while i < len(emotion_chunks):
        sub_text = emotion_chunks[i]
        if sub_text:
            pause_tokens = re.split(PAUSE_TAG_REGEX, sub_text)
            j = 0
            while j < len(pause_tokens):
                seg = pause_tokens[j].strip()
                if seg:
                    items.append(("text", seg, current_emotion))
                if j + 2 < len(pause_tokens):
                    val_str, unit_str = pause_tokens[j+1], pause_tokens[j+2]
                    if val_str is not None:
                        val = float(val_str)
                        unit = (unit_str or "s").lower()
                        duration_sec = val / 1000.0 if unit == "ms" else val
                        items.append(("pause", duration_sec, current_emotion))
                    j += 3
                else:
                    j += 1
        if i + 1 < len(emotion_chunks):
            current_emotion = emotion_chunks[i+1].strip()
            i += 2
        else:
            i += 1

    return items

