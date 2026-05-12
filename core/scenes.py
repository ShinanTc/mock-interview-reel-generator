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
build_question_scene   – question PNG scaled to 9:16, slides in from the right
                         on top of a frozen difficulty background, with a woosh
                         SFX synced to the slide animation
"""

import numpy as np
from pathlib import Path
from PIL import Image
from moviepy import (
    AudioArrayClip,
    AudioFileClip,
    CompositeAudioClip,       # ← added for woosh mix
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


def build_question_scene(
    image_path: Path,
    bg_clip: CompositeVideoClip,
    duration: float = 10,
    slide_duration: float = 0.45,
    fps: int = 30,
) -> tuple:
    """
    Load a question image and animate it sliding in from the right edge,
    composited on top of a frozen still from the difficulty scene.

    The question image is scaled to fill the full 9:16 frame (OUT_W × OUT_H)
    so there are no empty bars or black borders around it.

    The slide animation starts fully off-screen (x = +OUT_W) and eases into
    its resting position (x = 0) over `slide_duration` seconds, then holds
    still for the remainder of `duration`.

    A woosh SFX (sfx/woosh.mp3) plays in sync with the slide-in and is mixed
    on top of a silent bed that fills the entire scene duration.  If the woosh
    file is longer than `slide_duration` it is trimmed; if it is shorter it
    plays naturally and is padded with silence.

    Args:
        image_path:     Path to the PNG/JPG inside the questions/ folder.
        bg_clip:        The difficulty clip — its last frame is frozen and used
                        as the background so the question feels like it slides
                        in ON TOP of the difficulty image.
        duration:       Total display time of the question slide (seconds).
        slide_duration: How long the slide-in animation takes (seconds).
        fps:            Frame rate — should match OUTPUT_FPS from config.

    Returns:
        (video_clip, audio_clip) — audio contains the woosh mixed over silence.
    """
    AUDIO_FPS = 44_100

    # ── 1. Freeze last frame of the difficulty clip as the background ──────────
    last_frame_t = max(0.0, bg_clip.duration - 1 / fps)
    last_frame = bg_clip.get_frame(last_frame_t)          # numpy (H, W, 3)
    bg = ImageClip(last_frame, duration=duration)
    print(f"   BG frozen at   : t={last_frame_t:.3f} s  ({bg.w}×{bg.h})")

    # ── 2. Load question image and scale to full 9:16 frame ───────────────────
    img = ImageClip(str(image_path), duration=duration).resized((OUT_W, OUT_H))
    print(f"   Question image : {img.w}×{img.h}  (scaled to frame)")

    # ── 3. Animate position: slide in from the RIGHT edge ─────────────────────
    def _slide_position(t: float):
        if t >= slide_duration:
            return (0, 0)
        eased = 1 - (1 - t / slide_duration) ** 2        # ease-out quadratic
        return (int(OUT_W * (1 - eased)), 0)              # OUT_W → 0

    animated = img.with_position(_slide_position)

    # ── 4. Composite: frozen difficulty bg + animated question on top ──────────
    video = CompositeVideoClip([bg, animated], size=(OUT_W, OUT_H)).with_fps(fps)

    # ── 5. Build audio: silent bed + woosh synced to the slide-in ─────────────
    #
    #   Strategy
    #   --------
    #   a) Create a full-duration silence array (the "bed").
    #   b) Load woosh.mp3; trim it to slide_duration so it never bleeds past
    #      the animation window (trim can be skipped — it still sounds great
    #      if the woosh is shorter than slide_duration).
    #   c) Set the woosh to start at t=0 via .with_start(0).
    #   d) Mix with CompositeAudioClip; MoviePy sums the two tracks sample-by-
    #      sample, giving a natural blend without clipping artefacts.

    # Silent bed — full scene length
    n_samples = int(duration * AUDIO_FPS)
    silence_arr = np.zeros((n_samples, 2), dtype=np.float32)
    silence_bed = AudioArrayClip(silence_arr, fps=AUDIO_FPS)

    # Woosh SFX
    woosh_path = SFX_DIR / "woosh.mp3"
    if woosh_path.exists():
        print_step("🔊", f"Loading woosh SFX → {woosh_path.name}")
        woosh = AudioFileClip(str(woosh_path))
        print(f"   Woosh duration : {woosh.duration:.2f} s  "
              f"(slide window: {slide_duration:.2f} s)")

        # Trim if the woosh is longer than the slide animation so it doesn't
        # bleed into the "question is just sitting there" part of the scene.
        if woosh.duration > slide_duration:
            woosh = woosh.subclipped(0, slide_duration)
            print(f"   Woosh trimmed  : to {slide_duration:.2f} s")

        # Start at t=0 — perfectly in sync with the first frame of the slide.
        woosh = woosh.with_start(0)

        # Mix: silence provides the full-duration container; woosh layers on top.
        scene_audio = CompositeAudioClip([silence_bed, woosh]).with_duration(duration)
        print(f"   Scene audio    : woosh mixed over {duration:.2f} s silence")
    else:
        # Graceful fallback — pipeline continues without the SFX.
        print(f"   ⚠️  Woosh SFX not found at {woosh_path} — using silence.")
        scene_audio = silence_bed

    return video, scene_audio


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