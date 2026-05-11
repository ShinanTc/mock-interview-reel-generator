"""
core/scenes.py
──────────────
High-level scene builders.

Each function returns a (silent_video_clip, audio_clip) tuple so that
main() can concatenate video and audio tracks independently before
recombining.  This is necessary because MoviePy 2.x does not reliably
propagate audio through CompositeVideoClip / ImageClip chains.

Scenes
------
build_intro_scene      – random video + random audio + word-synced subtitles
build_difficulty_scene – difficulty PNG held for ≤ DIFFICULTY_DURATION seconds
                         over sfx/riser.mp3
"""

import numpy as np
from PIL import Image
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
)

from config import (
    AUDIOS_DIR,
    DIFFICULTY_DIR,
    DIFFICULTY_DURATION,
    OUT_H,
    OUT_W,
    SFX_DIR,
    VIDEOS_DIR,
)
from core.transcribe import transcribe_words
from core.subtitles import build_subtitle_clips
from core.video import force_9_16, image_cover_crop, loop_clip_to
from utils import pick_random_file, print_step


# ── Public API ─────────────────────────────────────────────────────────────────

def build_intro_scene() -> tuple[CompositeVideoClip, AudioFileClip]:
    """
    Build the intro scene.

    Steps:
      1. Pick a random video and audio file.
      2. Transcribe the audio with Whisper.
      3. Reframe the video to 9:16 and loop it to match audio length.
      4. Render word-by-word subtitle clips.
      5. Composite everything into a single silent clip.

    Returns
    -------
    (silent_composite, audio_clip)
    """
    print_step("📹", "Picking intro video...")
    video_path = pick_random_file(VIDEOS_DIR, [".mp4", ".mov", ".avi", ".mkv"])

    print_step("🎵", "Picking intro audio...")
    audio_path = pick_random_file(AUDIOS_DIR, [".mp3", ".wav", ".m4a", ".aac"])

    print_step("🎙", "Transcribing audio for subtitles...")
    words = transcribe_words(audio_path)

    print_step("🎬", f"Reframing video to 9:16 ({OUT_W}×{OUT_H})...")
    video_clip = _load_and_reframe_video(video_path)

    audio_clip = AudioFileClip(str(audio_path))
    print(f"   Audio duration : {audio_clip.duration:.2f} s")

    looped_silent = loop_clip_to(video_clip.without_audio(), audio_clip.duration)

    print_step("📝", "Rendering word-by-word subtitles...")
    subtitle_clips = build_subtitle_clips(words)

    print_step("🎞", "Compositing intro layers...")
    intro_silent = CompositeVideoClip(
        [looped_silent, *subtitle_clips],
        size=(OUT_W, OUT_H),
    ).with_duration(audio_clip.duration)

    return intro_silent, audio_clip


def build_difficulty_scene(difficulty: str) -> tuple[CompositeVideoClip, AudioFileClip]:
    """
    Build the difficulty reveal screen.

    Steps:
      1. Load and cover-crop the difficulty PNG.
      2. Load sfx/riser.mp3; trim it to at most DIFFICULTY_DURATION seconds.
      3. Hold the static image for the riser's (trimmed) duration.

    Returns
    -------
    (silent_image_clip, riser_audio_clip)
    """
    print_step("🏷", f"Building difficulty scene: {difficulty.upper()}...")

    filled_image = _load_difficulty_image(difficulty)
    riser_audio, scene_duration = _load_riser_audio()

    frame_array = np.array(filled_image)
    diff_silent = CompositeVideoClip(
        [ImageClip(frame_array, duration=scene_duration)],
        size=(OUT_W, OUT_H),
    ).with_duration(scene_duration)

    return diff_silent, riser_audio


# ── Private helpers ────────────────────────────────────────────────────────────

def _load_and_reframe_video(video_path) -> VideoFileClip:
    """Load a video file and reframe it to 9:16."""
    clip = VideoFileClip(str(video_path))
    print(f"   Original size  : {clip.w}×{clip.h}")
    clip = force_9_16(clip)
    print(f"   After 9:16 crop: {clip.w}×{clip.h}")
    return clip


def _load_difficulty_image(difficulty: str) -> Image.Image:
    """Load the difficulty PNG and cover-crop it to fill the frame."""
    img_path = DIFFICULTY_DIR / f"{difficulty}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Difficulty image not found: {img_path}")

    src = Image.open(img_path).convert("RGB")
    print(f"   Image source   : {img_path.name}  ({src.width}×{src.height})")

    filled = image_cover_crop(src)
    print(f"   After fill crop: {filled.width}×{filled.height}")
    return filled


def _load_riser_audio() -> tuple[AudioFileClip, float]:
    """
    Load sfx/riser.mp3, trim it to DIFFICULTY_DURATION if needed, and
    return (trimmed_clip, actual_scene_duration).
    """
    riser_path = SFX_DIR / "riser.mp3"
    if not riser_path.exists():
        raise FileNotFoundError(f"Riser SFX not found: {riser_path}")

    audio = AudioFileClip(str(riser_path))
    print(f"   Riser SFX      : {riser_path.name}  ({audio.duration:.2f} s)")

    scene_duration = min(audio.duration, DIFFICULTY_DURATION)

    if audio.duration > DIFFICULTY_DURATION:
        audio = audio.subclipped(0, DIFFICULTY_DURATION)
        print(f"   Riser trimmed  : to {DIFFICULTY_DURATION:.2f} s")

    print(f"   Scene duration : {scene_duration:.2f} s")
    return audio, scene_duration