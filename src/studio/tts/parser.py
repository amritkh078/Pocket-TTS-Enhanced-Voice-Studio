"""
Script text & pause tag parser module.
"""

import re
from typing import List, Tuple, Union

# Regex matching [5 sec], [5s], [500ms], [pause 2.5s], [silence 3 sec], [2.5]
PAUSE_TAG_REGEX = r"(?i)\[(?:pause|silence)?\s*([\d\.]+)\s*(s|sec|secs|second|seconds|ms)?\]"

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

def parse_script_pauses(text: str) -> List[Tuple[str, Union[str, float]]]:
    """
    Parses script text into sequential list of ('text', text_segment) or ('pause', pause_duration_sec).
    """
    tokens = re.split(PAUSE_TAG_REGEX, text)
    parsed_items = []
    i = 0

    while i < len(tokens):
        segment_text = tokens[i].strip()
        if segment_text:
            parsed_items.append(("text", segment_text))

        if i + 2 < len(tokens):
            val_str = tokens[i+1]
            unit_str = tokens[i+2]
            if val_str is not None:
                val = float(val_str)
                unit = (unit_str or "s").lower()
                duration_sec = val / 1000.0 if unit == "ms" else val
                parsed_items.append(("pause", duration_sec))
            i += 3
        else:
            i += 1

    return parsed_items
