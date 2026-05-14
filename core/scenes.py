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
                         on top of a frozen difficulty background.
                         A random comment audio from comments/ is attached and
                         transcribed; its words are rendered as word-by-word
                         subtitles in the same style as the intro.
                         The timer SFX is ducked while the comment plays,
                         then restored to full volume once it finishes.
                         An animated countdown clock is composited in the
                         bottom-right corner for the full question duration.
"""

import math
import numpy as np
from pathlib import Path
from PIL import Image
from moviepy import (
    AudioArrayClip,
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_audioclips,
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
from core.clock import build_countdown_clip
from core.transcribe import transcribe_words
from core.subtitles import build_subtitle_clips
from core.video import force_9_16, image_cover_crop, loop_clip_to
from utils import pick_random_file, print_step


# Volume multiplier applied to the timer SFX while the comment audio is
# playing.  0.15 = 15 % of original volume — audible but not distracting.
_TIMER_DUCK_VOLUME = 0.15

# Countdown clock appearance
_CLOCK_SIZE   = 210    # diameter in pixels
_CLOCK_MARGIN = 40     # gap from the frame edge (bottom-right corner)


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
    comments_dir: Path,
    duration: float = 10,
    slide_duration: float = 0.45,
    fps: int = 30,
) -> tuple:
    """
    Load a question image and animate it sliding in from the right edge,
    composited on top of a frozen still from the difficulty scene.

    A random comment audio is picked from `comments_dir`, played over the
    scene, and transcribed so its words appear as word-by-word subtitles in
    the same Inter-Black style used throughout the project.

    During comment playback the timer SFX is ducked to _TIMER_DUCK_VOLUME so
    the voice stays intelligible; once the comment ends the timer returns to
    full volume for the remainder of the scene.

    An animated countdown clock (green → amber → red sweeping arc) is
    composited in the bottom-right corner.  It counts down from `duration`
    seconds and disappears when the scene ends.

    Audio layers (all mixed via CompositeAudioClip)
    -----------------------------------------------
    [silence bed]  0 ──────────────────────────────────── duration
    [woosh]        0 ── slide_duration
    [timer ducked] slide_duration ── slide_duration + comment_hold  (15 % vol)
    [timer normal]                   slide_duration + comment_hold ── duration
    [comment]      0 ── comment.duration

    Args:
        image_path:     Path to the PNG/JPG inside the questions/ folder.
        bg_clip:        The difficulty clip — its last frame is frozen and used
                        as the background so the question feels like it slides
                        in ON TOP of the difficulty image.
        comments_dir:   Folder containing numbered comment mp3 files
                        (e.g. comments/1.mp3, comments/2.mp3 …).
        duration:       Total display time of the question slide (seconds).
        slide_duration: How long the slide-in animation takes (seconds).
        fps:            Frame rate — should match OUTPUT_FPS from config.

    Returns:
        (video_clip, audio_clip)
    """
    AUDIO_FPS    = 44_100
    CLOCK_SIZE   = _CLOCK_SIZE
    CLOCK_MARGIN = _CLOCK_MARGIN

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

    # ── 4. Pick & transcribe comment audio ────────────────────────────────────
    comment_audio, comment_words = _load_comment(comments_dir)

    # ── 5. Build word-by-word subtitle clips for the comment ──────────────────
    #
    #   build_subtitle_clips() expects word timestamps relative to t=0 of the
    #   clip being composited — which is t=0 of the question scene — so the
    #   raw transcription timestamps map directly without any offset.
    print_step("📝", "Rendering comment subtitles...")
    comment_subtitle_clips = build_subtitle_clips(comment_words)
    print(f"   Subtitle clips : {len(comment_subtitle_clips)} word(s)")

    # ── 6. Build animated countdown clock ─────────────────────────────────────
    print_step("⏰", "Building countdown clock...")
    clock_clip = build_countdown_clip(duration=duration, size=CLOCK_SIZE, fps=fps)

    # Position: horizontally centered, vertically centered in the top quarter.
    #
    #   Full frame    : 0 ──────────────────────────── OUT_H
    #   Top half      : 0 ──────────── OUT_H / 2
    #   Top quarter   : 0 ── OUT_H / 4
    #   Zone centre   : OUT_H / 8
    #   Clock top edge: zone_centre - CLOCK_SIZE // 2
    clock_x = (OUT_W - CLOCK_SIZE) // 2
    # clock_y = OUT_H // 8 - CLOCK_SIZE // 2
    clock_y = OUT_H // 8 - CLOCK_SIZE // 2 + 40
    clock_clip = clock_clip.with_position((clock_x, clock_y))
    print(f"   Clock position : ({clock_x}, {clock_y})  "
          f"[top-quarter centre, horizontally centred in {OUT_W}x{OUT_H} frame]")

    # ── 7. Composite: frozen bg → animated question → clock → subtitles ───────
    video = CompositeVideoClip(
        [bg, animated, clock_clip, *comment_subtitle_clips],
        size=(OUT_W, OUT_H),
    ).with_fps(fps)

    # ── 8. Build audio ─────────────────────────────────────────────────────────
    #
    #   The timer SFX starts at `slide_duration` (scene time) and loops until
    #   the end.  While the comment audio is playing we split it into two
    #   segments and duck the first one.
    #
    #   comment_hold: how far into the timer's own timeline the comment
    #                 overlaps (timer starts at slide_duration in scene time;
    #                 comment starts at t=0 in scene time).
    #
    #       scene time:  0 ────── slide_duration ─────────────────────── duration
    #       comment:     |── comment.duration ──|
    #       timer:                |── ducked ───|─── normal ───────────|

    hold_duration = duration - slide_duration

    # How many seconds of the TIMER (starting at slide_duration) are covered
    # by the comment audio.
    comment_hold = max(0.0, min(comment_audio.duration - slide_duration,
                                hold_duration))

    # Silent bed — full scene length
    n_samples   = int(duration * AUDIO_FPS)
    silence_arr = np.zeros((n_samples, 2), dtype=np.float32)
    silence_bed = AudioArrayClip(silence_arr, fps=AUDIO_FPS)

    audio_layers = [silence_bed]

    # ── Woosh SFX ─────────────────────────────────────────────────────────────
    woosh_path = SFX_DIR / "woosh.mp3"
    if woosh_path.exists():
        print_step("🔊", f"Loading woosh SFX → {woosh_path.name}")
        woosh = AudioFileClip(str(woosh_path))
        print(f"   Woosh duration : {woosh.duration:.2f} s  "
              f"(slide window: {slide_duration:.2f} s)")
        if woosh.duration > slide_duration:
            woosh = woosh.subclipped(0, slide_duration)
            print(f"   Woosh trimmed  : to {slide_duration:.2f} s")
        audio_layers.append(woosh.with_start(0))
    else:
        print(f"   ⚠️  Woosh SFX not found at {woosh_path} — skipping.")

    # ── Timer SFX with comment-aware ducking ───────────────────────────────────
    #
    #   concatenate_audioclips() returns a CompositeAudioClip which has no
    #   multiply_volume method.  Instead we bake the looped timer into a raw
    #   numpy array and apply the volume envelope with plain multiplication —
    #   no clip-level volume API required.
    #
    #       timer array index 0  →  scene time slide_duration
    #       samples [0 : duck_end_sample]  →  _TIMER_DUCK_VOLUME
    #       samples [duck_end_sample : ]   →  1.0  (unchanged)
    #
    timer_path = SFX_DIR / "timer.mp3"
    if timer_path.exists():
        print_step("⏱ ", f"Loading timer SFX → {timer_path.name}")
        timer_raw = AudioFileClip(str(timer_path))
        print(f"   Timer duration : {timer_raw.duration:.2f} s  "
              f"(hold window: {hold_duration:.2f} s)")

        timer_looped = _loop_audio_to(timer_raw, hold_duration)

        # Render to numpy so we can manipulate samples directly.
        timer_arr = timer_looped.to_soundarray(fps=AUDIO_FPS).astype(np.float32)

        if comment_hold > 0:
            duck_end_sample = min(int(comment_hold * AUDIO_FPS), len(timer_arr))
            timer_arr[:duck_end_sample] *= _TIMER_DUCK_VOLUME
            print(f"   Timer ducked   : {comment_hold:.2f} s  "
                  f"({int(_TIMER_DUCK_VOLUME * 100)}% vol)  "
                  f"→ full volume for remaining "
                  f"{hold_duration - comment_hold:.2f} s")
        else:
            print(f"   Timer normal   : {hold_duration:.2f} s  "
                  f"(comment ends before timer begins — no ducking)")

        timer_final = AudioArrayClip(timer_arr, fps=AUDIO_FPS)
        audio_layers.append(timer_final.with_start(slide_duration))
    else:
        print(f"   ⚠️  Timer SFX not found at {timer_path} — skipping.")

    # ── Comment audio layer ────────────────────────────────────────────────────
    # Clamp to scene duration so it never extends the clip unintentionally.
    comment_clamped = (
        comment_audio.subclipped(0, min(comment_audio.duration, duration))
        .with_start(0)
    )
    audio_layers.append(comment_clamped)
    print(f"   Comment audio  : {comment_clamped.duration:.2f} s  "
          f"starting at t=0 s")

    scene_audio = CompositeAudioClip(audio_layers).with_duration(duration)
    print(f"   Scene audio    : {len(audio_layers)} layer(s) mixed over "
          f"{duration:.2f} s")

    return video, scene_audio


# ── Private helpers ────────────────────────────────────────────────────────────


def _load_comment(comments_dir: Path) -> tuple[AudioFileClip, list]:
    """
    Pick a random mp3 from `comments_dir`, transcribe it, and return
    (AudioFileClip, words).

    Raises FileNotFoundError if the folder is missing or empty.
    """
    if not comments_dir.exists():
        raise FileNotFoundError(
            f"The comments/ folder does not exist.\n"
            f"Expected it at: {comments_dir.resolve()}\n"
            f"Create it and drop numbered mp3 files inside (1.mp3, 2.mp3, …)."
        )

    print_step("💬", "Picking random comment audio...")
    comment_path = pick_random_file(comments_dir, [".mp3", ".wav", ".m4a", ".aac"])
    print(f"   Comment file   : {comment_path.name}")

    print_step("🎙", "Transcribing comment audio for subtitles...")
    words = transcribe_words(comment_path)
    print(f"   Words found    : {len(words)}")

    audio = AudioFileClip(str(comment_path))
    print(f"   Comment length : {audio.duration:.2f} s")

    return audio, words


def _loop_audio_to(clip: AudioFileClip, target_duration: float) -> AudioFileClip:
    """
    Repeat `clip` end-to-end until it reaches `target_duration`, then trim.

    Uses concatenate_audioclips so the loop boundary is sample-accurate and
    there are no pitch/speed artefacts from MoviePy's built-in looping helpers.
    """
    repeats = math.ceil(target_duration / clip.duration)
    looped  = concatenate_audioclips([clip] * repeats)
    return looped.subclipped(0, target_duration)


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