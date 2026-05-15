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
                         on top of a frozen background clip.
                         When comments_dir is provided (Question 1), a random
                         comment audio is attached and transcribed; its words are
                         rendered as word-by-word subtitles.  The timer SFX is
                         ducked while the comment plays.
                         When comments_dir is None (Questions 2 & 3), no comment
                         audio is used and the timer runs at full volume throughout.
                         An animated countdown clock is composited in the upper
                         centre for the full question duration.
build_review_scene     – question_review/question_review.mp4 looped to match a
                         caller-supplied transition audio (audio_path: Path).
                         The audio is transcribed and rendered as word-by-word
                         subtitles dead-centre horizontally, 30 % from the top
                         of the 9:16 frame.
build_outro_scene      – random video from outro_scene/videos/ + random audio
                         from outro_scene/audios/ + word-synced subtitles.
                         Mirrors build_intro_scene exactly with different paths.
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

# How many seconds to wait after the difficulty image appears before the
# comment audio (and its subtitles) begin playing.  This gives the
# question slide-in animation room to breathe before the voice kicks in.
COMMENT_DELAY = 2.0    # seconds


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
    comments_dir: Path | None = None,
    duration: float = 10,
    slide_duration: float = 0.45,
    fps: int = 30,
) -> tuple:
    """
    Load a question image and animate it sliding in from the right edge,
    composited on top of a frozen still from the previous scene.

    Behaviour depends on whether `comments_dir` is supplied:

    With comments_dir (Question 1)
    ──────────────────────────────
    A random comment mp3 is picked, transcribed, and played starting at
    COMMENT_DELAY seconds into the scene.  Word-by-word subtitles follow
    the voice.  The timer SFX is ducked to _TIMER_DUCK_VOLUME while the
    comment overlaps it.

    Without comments_dir (Questions 2 & 3)
    ───────────────────────────────────────
    No comment audio is loaded.  The timer SFX plays at full volume for the
    entire hold window.  No comment subtitles are rendered.  The question
    image still slides in at the same speed and the countdown clock runs
    identically.

    Audio layers timeline  (comments_dir provided)
    ─────────────────────
    scene time:  0 ────────────────────────────────────────────────── duration
    woosh:       0 ── slide_duration
    timer:             slide_duration ─────────────────────────────── duration
      ducked:                          COMMENT_DELAY ──────────────── COMMENT_DELAY + comment.dur
      normal:          slide_duration ─ COMMENT_DELAY   and   COMMENT_DELAY+comment.dur ─ duration
    comment:                           COMMENT_DELAY ──── COMMENT_DELAY + comment.dur

    Args:
        image_path:     Path to the PNG/JPG inside the questions/ folder.
        bg_clip:        The previous scene clip — its last frame is frozen and
                        used as the background so the question slides in ON TOP.
        comments_dir:   Folder containing numbered comment mp3 files
                        (e.g. comments/1.mp3).  Pass None to skip comment audio.
        duration:       Total display time of the question slide (seconds).
        slide_duration: How long the slide-in animation takes (seconds).
        fps:            Frame rate — should match OUTPUT_FPS from config.

    Returns:
        (video_clip, audio_clip)
    """
    AUDIO_FPS    = 44_100
    CLOCK_SIZE   = _CLOCK_SIZE
    CLOCK_MARGIN = _CLOCK_MARGIN

    use_comment = comments_dir is not None

    # ── 1. Freeze last frame of the background clip ────────────────────────────
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

    # ── 4. Pick & transcribe comment audio (Q1 only) ──────────────────────────
    if use_comment:
        comment_audio, comment_words = _load_comment(comments_dir)

        # Shift Whisper timestamps by COMMENT_DELAY so subtitles sync with
        # the delayed playback position in scene time.
        print_step("📝", "Rendering comment subtitles (delayed by "
                         f"{COMMENT_DELAY:.1f} s)...")
        delayed_comment_words = [
            {**w, "start": w["start"] + COMMENT_DELAY, "end": w["end"] + COMMENT_DELAY}
            for w in comment_words
        ]
        comment_subtitle_clips = build_subtitle_clips(delayed_comment_words)
        print(f"   Subtitle clips : {len(comment_subtitle_clips)} word(s)  "
              f"(shifted +{COMMENT_DELAY:.1f} s)")
    else:
        comment_audio          = None
        comment_subtitle_clips = []
        print_step("💬", "No comment audio — skipping comment & subtitles.")

    # ── 5. Build animated countdown clock ─────────────────────────────────────
    print_step("⏰", "Building countdown clock...")
    clock_clip = build_countdown_clip(duration=duration, size=CLOCK_SIZE, fps=fps)

    clock_x = (OUT_W - CLOCK_SIZE) // 2
    clock_y = OUT_H // 8 - CLOCK_SIZE // 2 + 40
    clock_clip = clock_clip.with_position((clock_x, clock_y))
    print(f"   Clock position : ({clock_x}, {clock_y})  "
          f"[top-quarter centre, horizontally centred in {OUT_W}x{OUT_H} frame]")

    # ── 6. Composite: frozen bg → animated question → clock → subtitles ───────
    video = CompositeVideoClip(
        [bg, animated, clock_clip, *comment_subtitle_clips],
        size=(OUT_W, OUT_H),
    ).with_fps(fps)

    # ── 7. Build audio ─────────────────────────────────────────────────────────

    hold_duration = duration - slide_duration          # timer array length

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

    # ── Timer SFX (with optional comment-aware ducking) ────────────────────────
    timer_path = SFX_DIR / "timer.mp3"
    if timer_path.exists():
        print_step("⏱ ", f"Loading timer SFX → {timer_path.name}")
        timer_raw = AudioFileClip(str(timer_path))
        print(f"   Timer duration : {timer_raw.duration:.2f} s  "
              f"(hold window: {hold_duration:.2f} s)")

        timer_looped = _loop_audio_to(timer_raw, hold_duration)
        timer_arr    = timer_looped.to_soundarray(fps=AUDIO_FPS).astype(np.float32)

        if use_comment and comment_audio is not None:
            # Compute ducking window in scene time, then map to timer timeline.
            duck_start_scene = max(slide_duration, COMMENT_DELAY)
            duck_end_scene   = min(COMMENT_DELAY + comment_audio.duration, duration)
            duck_start_timer = max(0.0, duck_start_scene - slide_duration)
            duck_end_timer   = max(0.0, duck_end_scene   - slide_duration)
            comment_hold     = duck_end_timer - duck_start_timer

            if comment_hold > 0:
                duck_start_sample = int(duck_start_timer * AUDIO_FPS)
                duck_end_sample   = min(int(duck_end_timer * AUDIO_FPS), len(timer_arr))
                timer_arr[duck_start_sample:duck_end_sample] *= _TIMER_DUCK_VOLUME
                print(f"   Timer ducked   : {comment_hold:.2f} s  "
                      f"(timer t={duck_start_timer:.2f}–{duck_end_timer:.2f} s, "
                      f"{int(_TIMER_DUCK_VOLUME * 100)}% vol)")
                print(f"   Timer normal   : before t={duck_start_timer:.2f} s and "
                      f"after t={duck_end_timer:.2f} s in timer timeline")
            else:
                print(f"   Timer normal   : {hold_duration:.2f} s  "
                      f"(comment does not overlap timer — no ducking)")
        else:
            print(f"   Timer normal   : {hold_duration:.2f} s  "
                  f"(no comment audio — full volume throughout)")

        timer_final = AudioArrayClip(timer_arr, fps=AUDIO_FPS)
        audio_layers.append(timer_final.with_start(slide_duration))
    else:
        print(f"   ⚠️  Timer SFX not found at {timer_path} — skipping.")

    # ── Comment audio layer (Q1 only, delayed by COMMENT_DELAY) ───────────────
    if use_comment and comment_audio is not None:
        max_comment_dur = max(0.0, duration - COMMENT_DELAY)
        comment_clamped = (
            comment_audio.subclipped(0, min(comment_audio.duration, max_comment_dur))
            .with_start(COMMENT_DELAY)
        )
        audio_layers.append(comment_clamped)
        print(f"   Comment audio  : {comment_clamped.duration:.2f} s  "
              f"starting at t={COMMENT_DELAY:.1f} s  "
              f"(delayed {COMMENT_DELAY:.1f} s)")

    scene_audio = CompositeAudioClip(audio_layers).with_duration(duration)
    print(f"   Scene audio    : {len(audio_layers)} layer(s) mixed over "
          f"{duration:.2f} s")

    return video, scene_audio


def build_review_scene(
    review_video_path: Path,
    audio_path: Path,
    fps: int = 30,
) -> tuple:
    """
    Build the question-review scene.

    The review video (`question_review/question_review.mp4`) is reframed to
    9:16 and looped to match the duration of the supplied `audio_path`.

    The caller (pipeline.py) is responsible for picking `audio_path` from the
    TransitionPool — this keeps the no-repeat logic in one place and makes
    the scene builder stateless.

    The transition audio is transcribed with Whisper and rendered as
    word-by-word subtitles horizontally centred, with the top edge placed
    30 % down from the top of the 9:16 frame (subtitle_y = int(OUT_H * 0.30)).

    Args:
        review_video_path : Path to question_review/question_review.mp4.
        audio_path        : Pre-picked transition audio file (from TransitionPool).
        fps               : Frame rate — should match OUTPUT_FPS from config.

    Returns:
        (silent_composite_clip, audio_clip)
    """
    # ── 1. Transcribe the supplied transition audio ────────────────────────────
    print_step("🎵", f"Loading transition audio → {audio_path.name}")
    print_step("🎙", "Transcribing transition audio for subtitles...")
    words = transcribe_words(audio_path)
    print(f"   Words found    : {len(words)}")

    audio_clip     = AudioFileClip(str(audio_path))
    scene_duration = audio_clip.duration
    print(f"   Audio duration : {scene_duration:.2f} s")

    # ── 2. Load the review video, reframe to 9:16, loop to audio length ────────
    print_step("📹", f"Loading review video → {review_video_path.name}")
    if not review_video_path.exists():
        raise FileNotFoundError(
            f"Review video not found.\n"
            f"Expected : {review_video_path.resolve()}\n"
            f"Create the file or check the path in pipeline.py."
        )

    raw_video = VideoFileClip(str(review_video_path))
    print(f"   Original size  : {raw_video.w}×{raw_video.h}")

    reframed = force_9_16(raw_video)
    print(f"   After 9:16 crop: {reframed.w}×{reframed.h}")

    looped_silent = loop_clip_to(reframed.without_audio(), scene_duration)
    print(f"   Video looped to: {looped_silent.duration:.2f} s")

    # ── 3. Build word-by-word subtitle clips, centred in the upper half ────────
    #
    #   Vertical layout
    #   ───────────────
    #   Subtitle top edge is placed 30 % down from the top of the frame:
    #
    #       subtitle_y = int(OUT_H * 0.30)
    #
    #   Horizontal position uses explicit pixel maths so MoviePy 2.x centres
    #   the clip reliably: (OUT_W - clip.w) // 2
    print_step("📝", "Rendering word-by-word subtitles (30% from top, centred)...")
    raw_subtitle_clips = build_subtitle_clips(words)

    subtitle_y = int(OUT_H * 0.30)   # 30 % from the top of the frame

    subtitle_clips = [
        clip.with_position(((OUT_W - clip.w) // 2, subtitle_y))
        for clip in raw_subtitle_clips
    ]
    print(f"   Subtitle clips : {len(subtitle_clips)} word(s)  "
          f"→ (centred_x, {subtitle_y})  "
          f"[30% from top, frame={OUT_W}×{OUT_H}]")

    # ── 4. Composite: looped video + repositioned subtitles ───────────────────
    print_step("🎞", "Compositing review scene layers...")
    review_silent = CompositeVideoClip(
        [looped_silent, *subtitle_clips],
        size=(OUT_W, OUT_H),
    ).with_duration(scene_duration).with_fps(fps)

    return review_silent, audio_clip


def build_outro_scene() -> tuple[CompositeVideoClip, AudioFileClip]:
    """
    Build the outro scene.

    Mirrors build_intro_scene exactly, but draws media from a dedicated
    outro_scene/ folder:

        outro_scene/
            videos/   1.mp4          (fixed video — always picked)
            audios/   1.mp3, 2.mp3 … (random pick, same as intro)

    Steps:
      1. Pick the video from outro_scene/videos/.
      2. Pick a random audio file from outro_scene/audios/.
      3. Transcribe the audio with Whisper.
      4. Reframe the video to 9:16 and loop it to match audio length.
      5. Render word-by-word subtitle clips.
      6. Composite everything into a single silent clip.

    Returns
    -------
    (silent_composite, audio_clip)
    """
    OUTRO_DIR        = Path(__file__).parent.parent / "outro_scene"
    outro_videos_dir = OUTRO_DIR / "videos"
    outro_audios_dir = OUTRO_DIR / "audios"

    print_step("📹", "Picking outro video...")
    video_path = pick_random_file(outro_videos_dir, [".mp4", ".mov", ".avi", ".mkv"])

    print_step("🎵", "Picking outro audio...")
    audio_path = pick_random_file(outro_audios_dir, [".mp3", ".wav", ".m4a", ".aac"])

    print_step("🎙", "Transcribing outro audio for subtitles...")
    words = transcribe_words(audio_path)

    print_step("🎬", f"Reframing outro video to 9:16 ({OUT_W}×{OUT_H})...")
    video_clip = _load_and_reframe_video(video_path)

    audio_clip = AudioFileClip(str(audio_path))
    print(f"   Audio duration : {audio_clip.duration:.2f} s")

    looped_silent = loop_clip_to(video_clip.without_audio(), audio_clip.duration)

    print_step("📝", "Rendering outro word-by-word subtitles...")
    subtitle_clips = build_subtitle_clips(words)

    print_step("🎞", "Compositing outro layers...")
    outro_silent = CompositeVideoClip(
        [looped_silent, *subtitle_clips],
        size=(OUT_W, OUT_H),
    ).with_duration(audio_clip.duration)

    return outro_silent, audio_clip


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