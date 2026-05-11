"""
core/transcribe.py
──────────────────
Whisper-based audio transcription.

Produces a flat list of word-level timestamps used by the subtitle pipeline.
"""

from pathlib import Path

import whisper

from config import WHISPER_MODEL


# ── Types ──────────────────────────────────────────────────────────────────────

WordEntry = dict  # {"word": str, "start": float, "end": float}


# ── Public API ─────────────────────────────────────────────────────────────────

def transcribe_words(audio_path: Path) -> list[WordEntry]:
    """
    Transcribe *audio_path* with Whisper and return a flat list of words,
    each with "word", "start", and "end" keys (seconds).

    Raises RuntimeError if Whisper returns no word-level timestamps
    (try a larger model via WHISPER_MODEL in config.py).
    """
    print(f"   Transcribing: {audio_path.name}  (model={WHISPER_MODEL})")

    model  = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language="en",
    )

    words = _extract_words(result)

    if not words:
        raise RuntimeError(
            "Whisper returned no word-level timestamps. "
            "Try a larger WHISPER_MODEL (e.g. 'small') in config.py."
        )

    preview = " ".join(w["word"] for w in words)
    print(f'   Detected {len(words)} words → "{preview[:80]}..."')
    return words


# ── Private helpers ────────────────────────────────────────────────────────────

def _extract_words(result: dict) -> list[WordEntry]:
    """Flatten Whisper segment/word structure into a simple word list."""
    words = []
    for segment in result["segments"]:
        for wd in segment.get("words", []):
            text = wd["word"].strip()
            if text:
                words.append({
                    "word":  text,
                    "start": float(wd["start"]),
                    "end":   float(wd["end"]),
                })
    return words