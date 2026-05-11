#!/usr/bin/env python3
"""
Instagram Reel Generator
─────────────────────────
Usage:
    python main.py <difficulty>
    difficulty: easy | medium | hard

Pipeline:
    1. intro_scene  — random video + random audio + word-synced subtitles
    2. difficulty   — difficulty/<difficulty>.png held for 3 s over sfx/riser.mp3

Directory layout:
    intro_scene/
        audios/   1.mp3, 2.mp3, …
        videos/   1.mp4, 2.mp4, …
    difficulty/
        easy.png
        medium.png
        hard.png
    sfx/
        riser.mp3
    output/           (created automatically)
    fonts/            (Inter-Black auto-downloaded here)
"""

import os
import io
import sys
import random
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import whisper

from moviepy import (
    VideoFileClip,
    AudioFileClip,
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

# ── Encoding fix for Windows terminals ────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).parent
INTRO_SCENE     = ROOT / "intro_scene"
VIDEOS_DIR      = INTRO_SCENE / "videos"
AUDIOS_DIR      = INTRO_SCENE / "audios"
DIFFICULTY_DIR  = ROOT / "difficulty"
SFX_DIR         = ROOT / "sfx"
OUTPUT_DIR      = ROOT / "output"
FONTS_DIR       = ROOT / "fonts"

OUTPUT_DIR.mkdir(exist_ok=True)
FONTS_DIR.mkdir(exist_ok=True)

# ── Output dimensions (9:16 portrait) ─────────────────────────────────────────
OUT_W = 1080
OUT_H = 1920

# ── Subtitle config ────────────────────────────────────────────────────────────
SUBTITLE_HEX    = "#385E4F"
SUBTITLE_Y_FRAC = 0.72
FONT_SIZE_RATIO = 0.065

# ── Whisper ────────────────────────────────────────────────────────────────────
WHISPER_MODEL = "base"

# ── Difficulty scene ───────────────────────────────────────────────────────────
DIFFICULTY_DURATION = 3.0
VALID_DIFFICULTIES  = ("easy", "medium", "hard")

# ── Video export ───────────────────────────────────────────────────────────────
OUTPUT_FPS    = 30
OUTPUT_CODEC  = "libx264"
OUTPUT_PRESET = "fast"


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def print_step(emoji: str, msg: str) -> None:
    print(f"\n{emoji}  {msg}")


def pick_random_file(directory: Path, extensions: list[str]) -> Path:
    files = [f for ext in extensions for f in directory.glob(f"*{ext}")]
    if not files:
        raise FileNotFoundError(
            f"No files with extensions {extensions} found in: {directory}"
        )
    chosen = random.choice(sorted(files))
    print(f"   -> Selected: {chosen.name}")
    return chosen


# ══════════════════════════════════════════════════════════════════════════════
# 9:16 crop (for video clips)
# ══════════════════════════════════════════════════════════════════════════════

def force_9_16(clip: VideoFileClip, target_w: int = OUT_W, target_h: int = OUT_H) -> VideoFileClip:
    """Centre-crop + scale the clip to exactly target_w x target_h."""
    src_w, src_h = clip.w, clip.h
    target_ratio = target_w / target_h
    src_ratio    = src_w   / src_h

    scale = target_h / src_h if src_ratio > target_ratio else target_w / src_w

    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    resized = clip.resized((new_w, new_h))
    x1 = (new_w - target_w) // 2
    y1 = (new_h - target_h) // 2
    return resized.cropped(x1=x1, y1=y1, x2=x1 + target_w, y2=y1 + target_h)


# ══════════════════════════════════════════════════════════════════════════════
# Image cover-crop to fill 9:16 (for PNG images)
# ══════════════════════════════════════════════════════════════════════════════

def image_cover_crop(img: Image.Image, target_w: int = OUT_W, target_h: int = OUT_H) -> Image.Image:
    """
    Scale the PIL image so it COVERS the full target canvas (no black bars),
    then centre-crop to exactly target_w x target_h.
    Same logic as CSS `object-fit: cover`.
    """
    src_w, src_h = img.size
    target_ratio = target_w / target_h
    src_ratio    = src_w   / src_h

    # Scale up so the image covers the entire canvas in both dimensions
    if src_ratio > target_ratio:
        # Image is wider → fit height, crop sides
        scale = target_h / src_h
    else:
        # Image is taller / narrower → fit width, crop top/bottom
        scale = target_w / src_w

    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Centre-crop
    left = (new_w - target_w) // 2
    top  = (new_h - target_h) // 2
    img  = img.crop((left, top, left + target_w, top + target_h))

    return img


# ══════════════════════════════════════════════════════════════════════════════
# Font
# ══════════════════════════════════════════════════════════════════════════════

def _try_system_font() -> Path | None:
    candidates = [
        FONTS_DIR / "Inter-Black.ttf",
        FONTS_DIR / "Inter-Black.otf",
        Path("/usr/share/fonts/truetype/inter/Inter-Black.ttf"),
        Path("/usr/local/share/fonts/Inter-Black.ttf"),
        Path(os.path.expanduser("~/Library/Fonts/Inter-Black.ttf")),
        Path("C:/Windows/Fonts/Inter-Black.ttf"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def find_or_download_font() -> Path | None:
    found = _try_system_font()
    if found:
        print(f"   Font found: {found}")
        return found

    print("   Inter-Black not found locally — downloading from GitHub...")
    zip_url = "https://github.com/rsms/inter/releases/download/v4.0/Inter-4.0.zip"
    try:
        with urllib.request.urlopen(zip_url, timeout=30) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            ttf_candidates = [
                n for n in z.namelist()
                if "Inter-Black" in n and n.endswith((".ttf", ".otf"))
                and "Variable" not in n
            ]
            if not ttf_candidates:
                raise FileNotFoundError("Inter-Black not inside the zip.")
            ttf_name = next(
                (n for n in ttf_candidates if n.endswith(".ttf")),
                ttf_candidates[0],
            )
            dest = FONTS_DIR / Path(ttf_name).name
            dest.write_bytes(z.read(ttf_name))
        print(f"   Downloaded -> {dest}")
        return dest
    except Exception as exc:
        print(f"   Font download failed: {exc}")
        print("   Falling back to Pillow built-in font.")
        print("   TIP: Manually place Inter-Black.ttf in the fonts/ folder.")
        return None


def _load_pil_font(font_path: Path | None, size: int) -> ImageFont.FreeTypeFont:
    if font_path:
        try:
            return ImageFont.truetype(str(font_path), size)
        except Exception as e:
            print(f"   Could not load font ({e}); using default.")
    return ImageFont.load_default(size=max(size, 10))


# ══════════════════════════════════════════════════════════════════════════════
# Transcription
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_words(audio_path: Path) -> list[dict]:
    print_step("🎙", f"Transcribing: {audio_path.name}  (model={WHISPER_MODEL})")
    model  = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language="en",
    )

    words = []
    for segment in result["segments"]:
        for wd in segment.get("words", []):
            w = wd["word"].strip()
            if w:
                words.append({
                    "word":  w,
                    "start": float(wd["start"]),
                    "end":   float(wd["end"]),
                })

    if not words:
        raise RuntimeError(
            "Whisper returned no word-level timestamps. "
            "Try WHISPER_MODEL='small' for better results."
        )

    full_text = " ".join(d["word"] for d in words)
    print(f'   Detected {len(words)} words -> "{full_text[:80]}..."')
    return words


# ══════════════════════════════════════════════════════════════════════════════
# Subtitle helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_word_frame(
    word:     str,
    frame_w:  int,
    frame_h:  int,
    font:     ImageFont.FreeTypeFont,
    text_rgb: tuple[int, int, int],
) -> np.ndarray:
    img  = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), word, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]
    x    = (frame_w - tw) / 2
    y    = frame_h * SUBTITLE_Y_FRAC - th / 2

    draw.text((x, y), word, font=font, fill=(*text_rgb, 255))
    return np.array(img)


def build_subtitle_clips(
    words:    list[dict],
    frame_w:  int,
    frame_h:  int,
    font:     ImageFont.FreeTypeFont,
    text_rgb: tuple[int, int, int],
) -> list[ImageClip]:
    clips = []
    for wd in words:
        dur   = max(wd["end"] - wd["start"], 0.05)
        frame = make_word_frame(wd["word"], frame_w, frame_h, font, text_rgb)
        clip  = (
            ImageClip(frame, transparent=True, duration=dur)
            .with_start(wd["start"])
        )
        clips.append(clip)
    return clips


# ══════════════════════════════════════════════════════════════════════════════
# Video assembly helpers
# ══════════════════════════════════════════════════════════════════════════════

def loop_clip_to(video: VideoFileClip, duration: float) -> VideoFileClip:
    if video.duration >= duration:
        return video.subclipped(0, duration)
    loops = int(duration / video.duration) + 1
    return concatenate_videoclips([video] * loops).subclipped(0, duration)


# ══════════════════════════════════════════════════════════════════════════════
# Scene builders
# ══════════════════════════════════════════════════════════════════════════════

def build_intro_scene(font_path: Path | None) -> VideoFileClip:
    """Returns a fully composited intro clip (video + subtitles + audio)."""

    print_step("📹", "Picking video...")
    video_path = pick_random_file(VIDEOS_DIR, [".mp4", ".mov", ".avi", ".mkv"])

    print_step("🎵", "Picking audio...")
    audio_path = pick_random_file(AUDIOS_DIR, [".mp3", ".wav", ".m4a", ".aac"])

    words = transcribe_words(audio_path)

    print_step("🎬", f"Loading & reframing video to 9:16 ({OUT_W}x{OUT_H})...")
    video_clip = VideoFileClip(str(video_path))
    print(f"   Original size  : {video_clip.w}x{video_clip.h}")
    video_clip = force_9_16(video_clip)
    print(f"   After 9:16 crop: {video_clip.w}x{video_clip.h}")

    audio_clip     = AudioFileClip(str(audio_path))
    audio_duration = audio_clip.duration
    print(f"   Audio duration : {audio_duration:.2f} s")
    print(f"   Video duration : {video_clip.duration:.2f} s  "
          f"(loops x{max(1, int(audio_duration / video_clip.duration) + 1)})")

    looped_video = loop_clip_to(video_clip, audio_duration).with_audio(audio_clip)

    print_step("📝", "Rendering word-by-word subtitles...")
    font_size = max(int(OUT_W * FONT_SIZE_RATIO), 36)
    font      = _load_pil_font(font_path, font_size)
    text_rgb  = hex_to_rgb(SUBTITLE_HEX)
    print(f"   Font size  : {font_size} px")
    print(f"   Text color : {SUBTITLE_HEX}  -> RGB{text_rgb}")
    print(f"   Y position : {SUBTITLE_Y_FRAC * 100:.0f}% from top")

    sub_clips = build_subtitle_clips(words, OUT_W, OUT_H, font, text_rgb)
    print(f"   Built {len(sub_clips)} word clips")

    print_step("🎞", "Compositing intro layers...")
    intro = CompositeVideoClip(
        [looped_video, *sub_clips],
        size=(OUT_W, OUT_H),
    ).with_duration(audio_duration)

    return intro


def build_difficulty_scene(difficulty: str) -> VideoFileClip:
    """
    Returns a DIFFICULTY_DURATION-second clip:
      - difficulty/<difficulty>.png cover-cropped to fill 1080x1920 (no black bars)
      - sfx/riser.mp3 as audio (trimmed to 3 s if longer)
    """
    print_step("🏷", f"Building difficulty scene: {difficulty.upper()}...")

    # ── Image: cover-crop to fill full screen ──────────────────────
    img_path = DIFFICULTY_DIR / f"{difficulty}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Difficulty image not found: {img_path}")

    src = Image.open(img_path).convert("RGB")
    print(f"   Image source   : {img_path.name}  (original {src.width}x{src.height})")

    filled = image_cover_crop(src, OUT_W, OUT_H)
    print(f"   After fill crop: {filled.width}x{filled.height}  (full screen, no black bars)")

    # Convert to numpy RGB array for MoviePy
    frame_array = np.array(filled)
    img_clip = ImageClip(frame_array, duration=DIFFICULTY_DURATION)

    # ── Audio: riser SFX ───────────────────────────────────────────
    riser_path = SFX_DIR / "riser.mp3"
    if not riser_path.exists():
        raise FileNotFoundError(f"Riser SFX not found: {riser_path}")

    riser_audio = AudioFileClip(str(riser_path))
    print(f"   Riser SFX      : {riser_path.name}  (duration {riser_audio.duration:.2f} s)")

    # Trim riser if it exceeds the scene duration
    if riser_audio.duration > DIFFICULTY_DURATION:
        riser_audio = riser_audio.subclipped(0, DIFFICULTY_DURATION)
        print(f"   Riser trimmed  : to {DIFFICULTY_DURATION:.2f} s")

    # Set audio on the image clip — explicitly set clip duration to match
    difficulty_clip = img_clip.with_audio(riser_audio).with_duration(DIFFICULTY_DURATION)

    return difficulty_clip


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Parse argument ─────────────────────────────────────────────
    if len(sys.argv) < 2:
        print("Usage: python main.py <difficulty>")
        print(f"  difficulty: {' | '.join(VALID_DIFFICULTIES)}")
        sys.exit(1)

    difficulty = sys.argv[1].strip().lower()
    if difficulty not in VALID_DIFFICULTIES:
        print(f"Error: '{difficulty}' is not a valid difficulty.")
        print(f"  Choose from: {', '.join(VALID_DIFFICULTIES)}")
        sys.exit(1)

    print("=" * 62)
    print(f"  Instagram Reel Generator  |  difficulty: {difficulty.upper()}")
    print("=" * 62)

    # ── Font (shared by intro scene) ───────────────────────────────
    print_step("🔤", "Loading Inter-Black font...")
    font_path = find_or_download_font()

    # ── Scene 1: Intro ─────────────────────────────────────────────
    print_step("🎬", "=== INTRO SCENE ===")
    intro_clip = build_intro_scene(font_path)

    # ── Scene 2: Difficulty ────────────────────────────────────────
    print_step("🏁", "=== DIFFICULTY SCENE ===")
    difficulty_clip = build_difficulty_scene(difficulty)

    # ── Stitch together ────────────────────────────────────────────
    print_step("🔗", "Stitching intro + difficulty scenes...")
    final = concatenate_videoclips([intro_clip, difficulty_clip], method="compose")
    print(f"   Total duration : {final.duration:.2f} s")

    # ── Export ─────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / f"reel_{difficulty}.mp4"
    print_step("💾", f"Exporting -> {out_path.relative_to(ROOT)}")

    final.write_videofile(
        str(out_path),
        fps=OUTPUT_FPS,
        codec=OUTPUT_CODEC,
        audio_codec="aac",
        preset=OUTPUT_PRESET,
        threads=os.cpu_count() or 4,
        logger="bar",
    )

    print(f"\n Done!  Output saved to: {out_path}")


if __name__ == "__main__":
    main()