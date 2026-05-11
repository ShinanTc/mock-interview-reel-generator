#!/usr/bin/env python3
"""
Instagram Reel Generator
-------------------------
Usage:
    python main.py <difficulty>
    difficulty: easy | medium | hard

Pipeline:
    1. intro_scene  -- random video + random audio + word-synced subtitles
    2. difficulty   -- difficulty/<difficulty>.png held for 3 s over sfx/riser.mp3

Directory layout:
    intro_scene/
        audios/   1.mp3, 2.mp3, ...
        videos/   1.mp4, 2.mp4, ...
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
    concatenate_audioclips,
)

# -- Encoding fix for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# -- Paths
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

# -- Output dimensions (9:16 portrait)
OUT_W = 1080
OUT_H = 1920

# -- Subtitle config
SUBTITLE_HEX    = "#385E4F"
SUBTITLE_Y_FRAC = 0.72
FONT_SIZE_RATIO = 0.065

# -- Whisper
WHISPER_MODEL = "base"

# -- Difficulty scene
DIFFICULTY_DURATION = 3.0
VALID_DIFFICULTIES  = ("easy", "medium", "hard")

# -- Video export
OUTPUT_FPS    = 30
OUTPUT_CODEC  = "libx264"
OUTPUT_PRESET = "fast"


# =============================================================================
# Shared utilities
# =============================================================================

def hex_to_rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def print_step(emoji, msg):
    print(f"\n{emoji}  {msg}")


def pick_random_file(directory, extensions):
    files = [f for ext in extensions for f in directory.glob(f"*{ext}")]
    if not files:
        raise FileNotFoundError(
            f"No files with extensions {extensions} found in: {directory}"
        )
    chosen = random.choice(sorted(files))
    print(f"   -> Selected: {chosen.name}")
    return chosen


# =============================================================================
# 9:16 crop for video clips
# =============================================================================

def force_9_16(clip, target_w=OUT_W, target_h=OUT_H):
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


# =============================================================================
# Cover-crop PIL image to fill 9:16 (no black bars)
# =============================================================================

def image_cover_crop(img, target_w=OUT_W, target_h=OUT_H):
    src_w, src_h = img.size
    target_ratio = target_w / target_h
    src_ratio    = src_w   / src_h
    scale = target_h / src_h if src_ratio > target_ratio else target_w / src_w
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    img  = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top  = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


# =============================================================================
# Font
# =============================================================================

def _try_system_font():
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


def find_or_download_font():
    found = _try_system_font()
    if found:
        print(f"   Font found: {found}")
        return found

    print("   Inter-Black not found locally -- downloading from GitHub...")
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


def _load_pil_font(font_path, size):
    if font_path:
        try:
            return ImageFont.truetype(str(font_path), size)
        except Exception as e:
            print(f"   Could not load font ({e}); using default.")
    return ImageFont.load_default(size=max(size, 10))


# =============================================================================
# Transcription
# =============================================================================

def transcribe_words(audio_path):
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


# =============================================================================
# Subtitle helpers
# =============================================================================

def make_word_frame(word, frame_w, frame_h, font, text_rgb):
    img  = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), word, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]
    x    = (frame_w - tw) / 2
    y    = frame_h * SUBTITLE_Y_FRAC - th / 2
    draw.text((x, y), word, font=font, fill=(*text_rgb, 255))
    return np.array(img)


def build_subtitle_clips(words, frame_w, frame_h, font, text_rgb):
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


# =============================================================================
# Video assembly helpers
# =============================================================================

def loop_clip_to(video, duration):
    if video.duration >= duration:
        return video.subclipped(0, duration)
    loops = int(duration / video.duration) + 1
    return concatenate_videoclips([video] * loops).subclipped(0, duration)


# =============================================================================
# Scene builders — each returns (silent_video_clip, audio_clip)
# Audio is kept separate so main() can concatenate both tracks independently,
# which is the only reliable way to preserve all audio in MoviePy 2.x.
# =============================================================================

def build_intro_scene(font_path):
    """Returns (silent CompositeVideoClip, AudioFileClip) for the intro."""

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

    # Silent looped video
    looped_silent = loop_clip_to(video_clip.without_audio(), audio_duration)

    print_step("📝", "Rendering word-by-word subtitles...")
    font_size = max(int(OUT_W * FONT_SIZE_RATIO), 36)
    font      = _load_pil_font(font_path, font_size)
    text_rgb  = hex_to_rgb(SUBTITLE_HEX)
    print(f"   Font size  : {font_size} px")
    print(f"   Text color : {SUBTITLE_HEX}  -> RGB{text_rgb}")
    print(f"   Y position : {SUBTITLE_Y_FRAC * 100:.0f}% from top")

    sub_clips = build_subtitle_clips(words, OUT_W, OUT_H, font, text_rgb)
    print(f"   Built {len(sub_clips)} word clips")

    print_step("🎞", "Compositing intro layers (silent)...")
    intro_silent = CompositeVideoClip(
        [looped_silent, *sub_clips],
        size=(OUT_W, OUT_H),
    ).with_duration(audio_duration)

    return intro_silent, audio_clip


def build_difficulty_scene(difficulty):
    """Returns (silent CompositeVideoClip, AudioFileClip) for the difficulty screen."""

    print_step("🏷", f"Building difficulty scene: {difficulty.upper()}...")

    img_path = DIFFICULTY_DIR / f"{difficulty}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Difficulty image not found: {img_path}")

    src = Image.open(img_path).convert("RGB")
    print(f"   Image source   : {img_path.name}  (original {src.width}x{src.height})")

    filled = image_cover_crop(src, OUT_W, OUT_H)
    print(f"   After fill crop: {filled.width}x{filled.height}  (full screen, no black bars)")

    # Load riser first so we can derive actual scene duration from it.
    # Avoids crash when riser is shorter than DIFFICULTY_DURATION.
    riser_path = SFX_DIR / "riser.mp3"
    if not riser_path.exists():
        raise FileNotFoundError(f"Riser SFX not found: {riser_path}")

    riser_audio = AudioFileClip(str(riser_path))
    print(f"   Riser SFX      : {riser_path.name}  ({riser_audio.duration:.2f} s)")

    # Scene duration = riser length, capped at DIFFICULTY_DURATION max
    scene_duration = min(riser_audio.duration, DIFFICULTY_DURATION)
    if riser_audio.duration > DIFFICULTY_DURATION:
        riser_audio = riser_audio.subclipped(0, DIFFICULTY_DURATION)
        print(f"   Riser trimmed  : to {DIFFICULTY_DURATION:.2f} s")
    print(f"   Scene duration : {scene_duration:.2f} s")

    frame_array = np.array(filled)
    diff_silent = CompositeVideoClip(
        [ImageClip(frame_array, duration=scene_duration)],
        size=(OUT_W, OUT_H),
    ).with_duration(scene_duration)

    return diff_silent, riser_audio


# =============================================================================
# Entry point
# =============================================================================

def main():
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

    print_step("🔤", "Loading Inter-Black font...")
    font_path = find_or_download_font()

    print_step("🎬", "=== INTRO SCENE ===")
    intro_silent, intro_audio = build_intro_scene(font_path)

    print_step("🏁", "=== DIFFICULTY SCENE ===")
    diff_silent, diff_audio = build_difficulty_scene(difficulty)

    # -------------------------------------------------------------------------
    # Stitch video and audio tracks independently, then recombine.
    #
    # Why: MoviePy 2.x does not reliably propagate audio on ImageClip /
    # CompositeVideoClip through concatenate_videoclips. The fix is to
    # concatenate the silent video tracks together, concatenate the audio
    # tracks together, then attach the combined audio to the combined video
    # in a single final step.
    # -------------------------------------------------------------------------
    print_step("🔗", "Stitching scenes...")

    final_video = concatenate_videoclips(
        [intro_silent, diff_silent], method="compose"
    )
    final_audio = concatenate_audioclips([intro_audio, diff_audio])

    final = final_video.with_audio(final_audio)
    print(f"   Total duration : {final.duration:.2f} s")

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