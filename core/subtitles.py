"""
core/subtitles.py
─────────────────
Word-by-word subtitle rendering.

Each word is drawn onto a transparent RGBA frame and wrapped in a
MoviePy ImageClip that starts at the word's Whisper timestamp.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip

from config import OUT_W, OUT_H, SUBTITLE_Y_FRAC, SUBTITLE_HEX, FONT_SIZE_RATIO
from core.font import find_or_download_font, load_font
from utils import hex_to_rgb

if TYPE_CHECKING:
    from core.transcribe import WordEntry


# ── Public API ─────────────────────────────────────────────────────────────────

def build_subtitle_clips(
    words: list["WordEntry"],
    frame_w: int = OUT_W,
    frame_h: int = OUT_H,
) -> list[ImageClip]:
    """
    Convert a list of word entries (from transcribe_words) into a list of
    transparent ImageClips, each timed to its word's start/end timestamp.
    """
    font = _load_subtitle_font()
    text_rgb = hex_to_rgb(SUBTITLE_HEX)

    _log_subtitle_config(font)

    clips = [
        _word_to_clip(wd, frame_w, frame_h, font, text_rgb)
        for wd in words
    ]
    print(f"   Built {len(clips)} subtitle clips")
    return clips


# ── Private helpers ────────────────────────────────────────────────────────────

def _load_subtitle_font() -> ImageFont.FreeTypeFont:
    """Resolve and load the subtitle font at the configured size."""
    font_path = find_or_download_font()
    font_size = max(int(OUT_W * FONT_SIZE_RATIO), 36)
    return load_font(font_path, font_size)


def _log_subtitle_config(font: ImageFont.FreeTypeFont) -> None:
    """Print subtitle rendering settings for debugging."""
    font_size = max(int(OUT_W * FONT_SIZE_RATIO), 36)
    text_rgb  = hex_to_rgb(SUBTITLE_HEX)
    print(f"   Font size  : {font_size} px")
    print(f"   Text color : {SUBTITLE_HEX}  → RGB{text_rgb}")
    print(f"   Y position : {SUBTITLE_Y_FRAC * 100:.0f}% from top")


def _word_to_clip(
    wd: "WordEntry",
    frame_w: int,
    frame_h: int,
    font: ImageFont.FreeTypeFont,
    text_rgb: tuple[int, int, int],
) -> ImageClip:
    """Render a single word entry as a timed transparent ImageClip."""
    duration = max(wd["end"] - wd["start"], 0.05)
    frame    = _render_word_frame(wd["word"], frame_w, frame_h, font, text_rgb)
    return (
        ImageClip(frame, transparent=True, duration=duration)
        .with_start(wd["start"])
    )


def _render_word_frame(
    word: str,
    frame_w: int,
    frame_h: int,
    font: ImageFont.FreeTypeFont,
    text_rgb: tuple[int, int, int],
) -> np.ndarray:
    """Draw *word* centred horizontally at SUBTITLE_Y_FRAC on a transparent frame."""
    img  = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), word, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]
    x    = (frame_w - tw) / 2
    y    = frame_h * SUBTITLE_Y_FRAC - th / 2

    draw.text((x, y), word, font=font, fill=(*text_rgb, 255))
    return np.array(img)