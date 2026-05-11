"""
config.py
─────────
Single source of truth for every path and tunable constant.
Change values here; nothing else needs to be touched.
"""

from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Asset directories ──────────────────────────────────────────────────────────
INTRO_SCENE_DIR = ROOT / "intro_scene"
VIDEOS_DIR      = INTRO_SCENE_DIR / "videos"
AUDIOS_DIR      = INTRO_SCENE_DIR / "audios"
DIFFICULTY_DIR  = ROOT / "difficulty"
SFX_DIR         = ROOT / "sfx"
OUTPUT_DIR      = ROOT / "output"
FONTS_DIR       = ROOT / "fonts"

# Create output dirs on import so callers never have to think about it
OUTPUT_DIR.mkdir(exist_ok=True)
FONTS_DIR.mkdir(exist_ok=True)

# ── Output dimensions (9:16 portrait for Instagram Reels) ─────────────────────
OUT_W = 1080
OUT_H = 1920

# ── Subtitle ───────────────────────────────────────────────────────────────────
SUBTITLE_HEX    = "#385E4F"   # brand green
SUBTITLE_Y_FRAC = 0.72        # 0 = top, 1 = bottom
FONT_SIZE_RATIO = 0.065       # relative to OUT_W (~70 px at 1080 wide)

# ── Whisper transcription ──────────────────────────────────────────────────────
# Options: "tiny" | "base" | "small" | "medium" | "large"
WHISPER_MODEL = "base"

# ── Difficulty scene ───────────────────────────────────────────────────────────
DIFFICULTY_DURATION = 3.0
VALID_DIFFICULTIES  = ("easy", "medium", "hard")

# ── Video export ───────────────────────────────────────────────────────────────
OUTPUT_FPS    = 30
OUTPUT_CODEC  = "libx264"
OUTPUT_PRESET = "fast"        # ultrafast / fast / medium / slow