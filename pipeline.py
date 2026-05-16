"""
pipeline.py
───────────
Orchestrates the full reel-generation pipeline.

    run(difficulty, video_index, question_pool, transition_pool)
        0.  Build thumbnail scene     (single-frame PNG — acts as video thumbnail)
        1.  Build intro scene         (video + audio + subtitles)
        2.  Build difficulty scene    (PNG + riser SFX)
        3.  Build question 1 scene    (PNG slide-in on top of difficulty +
                                       random comment audio + comment subtitles)
        4.  Build review 1 scene      (question_review.mp4 looped to a random
                                       transition audio + upper-quarter subtitles)
        5.  Build question 2 scene    (PNG slide-in, full 9:16, no comment audio)
        6.  Build review 2 scene      (question_review.mp4 looped to a different
                                       random transition audio + upper-quarter subtitles)
        7.  Build question 3 scene    (PNG slide-in, full 9:16, no comment audio,
                                       no review follows)
        8.  Build outro scene         (video + random audio + subtitles, from
                                       outro_scene/videos/ and outro_scene/audios/)
        9.  Stitch all scenes         thumbnail → intro → difficulty → q1 → review1 →
                                       q2 → review2 → q3 → outro
        10. Export to output/<difficulty>/reel_<difficulty>_<index>.mp4

Pool ownership
--------------
QuestionImagePool and TransitionPool are created in main.py (once per
difficulty) and passed in here.  This guarantees that no question image and
no transition audio is reused across any of the 4 videos generated for a
given difficulty.

Question images live in per-difficulty sub-folders:
    questions/
        easy/    1.png … 12.png   (need at least 12 for 4 videos × 3 picks)
        medium/  1.png … 12.png
        hard/    1.png … 12.png

Thumbnail images live in:
    thumbnails/
        easy.png
        medium.png
        hard.png

The thumbnail is inserted as the very first frame of the final video
(duration = 1/fps seconds).  It is invisible during normal playback but
most social platforms use it as the static preview image shown before the
viewer presses play.
"""

import os
import random
from pathlib import Path

import numpy as np
from moviepy import (
    AudioArrayClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_audioclips,
    concatenate_videoclips,
)

from config import OUTPUT_DIR, OUTPUT_FPS, OUTPUT_CODEC, OUTPUT_PRESET, OUT_W, OUT_H
from core.scenes import (
    build_intro_scene,
    build_difficulty_scene,
    build_question_scene,
    build_review_scene,
    build_outro_scene,
)
from utils import print_step

# ── Asset directories ─────────────────────────────────────────────────────────

# Base folder for question images; actual images are in sub-folders per difficulty.
#   questions/easy/1.png   questions/medium/2.png   questions/hard/3.png
QUESTIONS_DIR = Path(__file__).parent / "questions"

# Thumbnail images shown as video preview (one PNG per difficulty).
#   thumbnails/easy.png   thumbnails/medium.png   thumbnails/hard.png
THUMBNAILS_DIR = Path(__file__).parent / "thumbnails"

# Folder that holds comment audio files (1.mp3, 2.mp3, …).
COMMENTS_DIR = Path(__file__).parent / "comments"

# The fixed review video shown after the question timer expires.
REVIEW_VIDEO_PATH = Path(__file__).parent / "question_review" / "question_review.mp4"

# Folder that holds the transition audio files used in the review scene
# (1.mp3, 2.mp3, … 55.mp3).
TRANSITIONS_DIR = Path(__file__).parent / "transition"


# ── Pool helpers ──────────────────────────────────────────────────────────────


class QuestionImagePool:
    """
    Manages a pool of numbered question images (1.png, 2.png, …) so that
    each image is picked at most once across its lifetime.

    Instantiated in main.py ONCE per difficulty and shared across all videos
    for that difficulty, ensuring no image is reused across videos.

    The pool is initialised from the difficulty-specific sub-folder:
        questions/<difficulty>/1.png  2.png  … 12.png

    For 4 videos × 3 picks each you need at least 12 images per folder.

    Raises
    ------
    FileNotFoundError  – if the questions/<difficulty>/ folder is missing.
    RuntimeError       – if the pool is exhausted (all images already used).
    """

    def __init__(self, questions_dir: Path) -> None:
        if not questions_dir.exists():
            raise FileNotFoundError(
                f"The questions sub-folder does not exist.\n"
                f"Expected it at: {questions_dir.resolve()}\n"
                f"Create the folder and drop your question images inside it "
                f"(e.g. 1.png … 12.png for 4 videos)."
            )

        self._available: list[Path] = sorted(
            [
                p for p in questions_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg"} and p.stem.isdigit()
            ],
            key=lambda p: int(p.stem),
        )

        if not self._available:
            raise FileNotFoundError(
                f"No numbered question images found in {questions_dir.resolve()}.\n"
                f"Add files like 1.png … 12.png (need 4 videos × 3 = 12 for no repeats)."
            )

        self._used: list[Path] = []
        print_step("🗂 ", f"Question pool   : {[p.name for p in self._available]} "
                          f"({len(self._available)} image(s)) "
                          f"[from {questions_dir.parent.name}/{questions_dir.name}/]")

    def pick(self) -> Path:
        """
        Randomly pick one image from the remaining pool, mark it as used,
        and return its path.  Raises RuntimeError when the pool is empty.
        """
        if not self._available:
            used_names = [p.name for p in self._used]
            raise RuntimeError(
                f"Question image pool is exhausted — all images have already "
                f"been used.\n"
                f"Used : {used_names}\n"
                f"You need at least 12 images per difficulty folder for 4 videos "
                f"with no repeats (4 videos × 3 questions each).\n"
                f"Add more numbered images to the questions/<difficulty>/ folder."
            )

        chosen = random.choice(self._available)
        self._available.remove(chosen)
        self._used.append(chosen)

        remaining = [p.name for p in self._available]
        print_step("🎲", f"Picked question : {chosen.name}  "
                         f"(remaining pool: {remaining or ['(empty)']}, "
                         f"used so far: {[p.name for p in self._used]})")
        return chosen


class TransitionPool:
    """
    Manages a pool of transition audio files (1.mp3 … N.mp3) so that each
    file is picked at most once across its lifetime.

    Instantiated in main.py ONCE per difficulty and shared across all videos
    for that difficulty, so the same transition audio never appears in more
    than one review scene across the entire batch.

    Raises
    ------
    FileNotFoundError  – if the transitions/ folder is missing or empty.
    RuntimeError       – if the pool is exhausted (all files already used).
    """

    _AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac"}

    def __init__(self, transitions_dir: Path) -> None:
        if not transitions_dir.exists():
            raise FileNotFoundError(
                f"The transitions/ folder does not exist.\n"
                f"Expected it at: {transitions_dir.resolve()}\n"
                f"Create the folder and drop your numbered mp3 files inside "
                f"(1.mp3, 2.mp3, … 55.mp3)."
            )

        self._available: list[Path] = [
            p for p in transitions_dir.iterdir()
            if p.suffix.lower() in self._AUDIO_EXTENSIONS
        ]

        if not self._available:
            raise FileNotFoundError(
                f"No audio files found in {transitions_dir.resolve()}.\n"
                f"Add numbered mp3 files (1.mp3 … 55.mp3) to that folder."
            )

        self._used: list[Path] = []
        print_step("🗂 ", f"Transition pool : {len(self._available)} audio file(s) available")

    def pick(self) -> Path:
        """
        Randomly pick one transition audio from the remaining pool, mark it
        as used, and return its path.  Raises RuntimeError when the pool is
        empty.
        """
        if not self._available:
            used_names = [p.name for p in self._used]
            raise RuntimeError(
                f"Transition pool is exhausted — all audio files have already "
                f"been used.\n"
                f"Used : {used_names}\n"
                f"You need at least 8 transition files per difficulty batch "
                f"(4 videos × 2 review scenes each).\n"
                f"Add more audio files to the transitions/ folder."
            )

        chosen = random.choice(self._available)
        self._available.remove(chosen)
        self._used.append(chosen)

        remaining_count = len(self._available)
        print_step("🎲", f"Picked transition: {chosen.name}  "
                         f"(used: {[p.name for p in self._used]}, "
                         f"{remaining_count} remaining in pool)")
        return chosen


# ── Main pipeline entry point ─────────────────────────────────────────────────


def run(
    difficulty: str,
    video_index: int = 1,
    question_pool: QuestionImagePool | None = None,
    transition_pool: TransitionPool | None = None,
) -> None:
    """
    Execute the full pipeline for the given difficulty level and video index.

    Args:
        difficulty:       "easy", "medium", or "hard".
        video_index:      1-based counter used in the output filename
                          (reel_easy_1.mp4, reel_easy_2.mp4, …).
        question_pool:    Pre-built pool shared across all videos for this
                          difficulty.  If None, a fresh pool is created (useful
                          for one-off testing of a single video).
        transition_pool:  Pre-built pool shared across all videos for this
                          difficulty.  If None, a fresh pool is created.
    """

    # Fall back to fresh pools when called standalone (e.g. during testing).
    if question_pool is None:
        print_step("⚠️ ", "No shared question pool supplied — creating a fresh one. "
                          "Images may repeat across videos if called multiple times.")
        question_pool = QuestionImagePool(QUESTIONS_DIR / difficulty)

    if transition_pool is None:
        print_step("⚠️ ", "No shared transition pool supplied — creating a fresh one. "
                          "Transitions may repeat across videos if called multiple times.")
        transition_pool = TransitionPool(TRANSITIONS_DIR)

    # ── 0. Thumbnail ──────────────────────────────────────────────────────────
    print_step("🖼 ", "=== THUMBNAIL SCENE ===")
    thumb_silent, thumb_audio = _build_thumbnail_scene(difficulty)

    # ── 1. Intro ──────────────────────────────────────────────────────────────
    print_step("🎬", "=== INTRO SCENE ===")
    intro_silent, intro_audio = build_intro_scene()

    # ── 2. Difficulty ─────────────────────────────────────────────────────────
    print_step("🏁", "=== DIFFICULTY SCENE ===")
    diff_silent, diff_audio = build_difficulty_scene(difficulty)

    # ── 3. Question 1 (with comment audio + subtitles) ────────────────────────
    print_step("❓", "=== QUESTION 1 SCENE ===")
    q1_image = question_pool.pick()
    print_step("🖼 ", f"Question 1 image → {q1_image}")
    q1_silent, q1_audio = build_question_scene(
        image_path=q1_image,
        bg_clip=diff_silent,
        fps=OUTPUT_FPS,
        comments_dir=COMMENTS_DIR,
    )

    # ── 4. Review 1 ───────────────────────────────────────────────────────────
    print_step("🔍", "=== REVIEW 1 SCENE ===")
    review1_audio_path = transition_pool.pick()
    review1_silent, review1_audio = build_review_scene(
        review_video_path=REVIEW_VIDEO_PATH,
        audio_path=review1_audio_path,
        fps=OUTPUT_FPS,
    )

    # ── 5. Question 2 (no comment audio) ──────────────────────────────────────
    print_step("❓", "=== QUESTION 2 SCENE ===")
    q2_image = question_pool.pick()
    print_step("🖼 ", f"Question 2 image → {q2_image}")
    q2_silent, q2_audio = build_question_scene(
        image_path=q2_image,
        bg_clip=review1_silent,
        fps=OUTPUT_FPS,
        comments_dir=None,
    )

    # ── 6. Review 2 (different transition audio than review 1) ────────────────
    print_step("🔍", "=== REVIEW 2 SCENE ===")
    review2_audio_path = transition_pool.pick()
    review2_silent, review2_audio = build_review_scene(
        review_video_path=REVIEW_VIDEO_PATH,
        audio_path=review2_audio_path,
        fps=OUTPUT_FPS,
    )

    # ── 7. Question 3 (no comment audio, no review follows) ───────────────────
    print_step("❓", "=== QUESTION 3 SCENE ===")
    q3_image = question_pool.pick()
    print_step("🖼 ", f"Question 3 image → {q3_image}")
    q3_silent, q3_audio = build_question_scene(
        image_path=q3_image,
        bg_clip=review2_silent,
        fps=OUTPUT_FPS,
        comments_dir=None,
    )

    # ── 8. Outro ──────────────────────────────────────────────────────────────
    print_step("🎬", "=== OUTRO SCENE ===")
    outro_silent, outro_audio = build_outro_scene()

    # ── 9. Stitch ─────────────────────────────────────────────────────────────
    print_step("🔗", "Stitching scenes  "
                     "[ thumbnail → intro → difficulty → q1 → review1 → "
                     "q2 → review2 → q3 → outro ] ...")
    final = _stitch(
        thumb_silent,    thumb_audio,
        intro_silent,    intro_audio,
        diff_silent,     diff_audio,
        q1_silent,       q1_audio,
        review1_silent,  review1_audio,
        q2_silent,       q2_audio,
        review2_silent,  review2_audio,
        q3_silent,       q3_audio,
        outro_silent,    outro_audio,
    )
    print(f"   Total duration : {final.duration:.2f} s")

    # ── 10. Export ────────────────────────────────────────────────────────────
    _export(final, difficulty, video_index)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_thumbnail_scene(difficulty: str):
    """
    Build a single-frame scene from thumbnails/<difficulty>.png.

    The frame lasts exactly 1/OUTPUT_FPS seconds (~0.033 s at 30 fps) — long
    enough for social platforms to detect it as the video's preview image but
    completely imperceptible during normal playback.

    Returns
    -------
    (silent_composite_clip, silent_audio_clip)
    """
    thumb_path = THUMBNAILS_DIR / f"{difficulty}.png"
    if not thumb_path.exists():
        raise FileNotFoundError(
            f"Thumbnail image not found.\n"
            f"Expected : {thumb_path.resolve()}\n"
            f"Create thumbnails/{difficulty}.png to fix this."
        )

    duration = 1.0 / OUTPUT_FPS   # one frame

    print_step("🖼 ", f"Loading thumbnail → {thumb_path.name}  "
                      f"(duration: 1 frame = {duration:.4f} s)")

    thumb_clip = (
        ImageClip(str(thumb_path), duration=duration)
        .resized((OUT_W, OUT_H))
    )
    thumb_silent = CompositeVideoClip(
        [thumb_clip],
        size=(OUT_W, OUT_H),
    ).with_duration(duration).with_fps(OUTPUT_FPS)

    AUDIO_FPS   = 44_100
    n_samples   = max(1, int(duration * AUDIO_FPS))
    silence_arr = np.zeros((n_samples, 2), dtype=np.float32)
    thumb_audio = AudioArrayClip(silence_arr, fps=AUDIO_FPS).with_duration(duration)

    return thumb_silent, thumb_audio


def _stitch(
    thumb_silent,    thumb_audio,
    intro_silent,    intro_audio,
    diff_silent,     diff_audio,
    q1_silent,       q1_audio,
    review1_silent,  review1_audio,
    q2_silent,       q2_audio,
    review2_silent,  review2_audio,
    q3_silent,       q3_audio,
    outro_silent,    outro_audio,
):
    """
    Concatenate video and audio tracks independently before recombining.

    MoviePy 2.x does not reliably carry audio through CompositeVideoClip /
    ImageClip chains, so tracks are joined separately then merged.

    Scene order: thumbnail → intro → difficulty → q1 → review1 →
                 q2 → review2 → q3 → outro
    """
    final_video = concatenate_videoclips(
        [
            thumb_silent,
            intro_silent,
            diff_silent,
            q1_silent,
            review1_silent,
            q2_silent,
            review2_silent,
            q3_silent,
            outro_silent,
        ],
        method="compose",
    )
    final_audio = concatenate_audioclips(
        [
            thumb_audio,
            intro_audio,
            diff_audio,
            q1_audio,
            review1_audio,
            q2_audio,
            review2_audio,
            q3_audio,
            outro_audio,
        ]
    )
    return final_video.with_audio(final_audio)


def _export(final, difficulty: str, video_index: int) -> None:
    """
    Write the final clip to output/<difficulty>/reel_<difficulty>_<index>.mp4.

    The output sub-folder is created if it does not already exist.
    """
    out_dir  = OUTPUT_DIR / difficulty
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"reel_{difficulty}_{video_index}.mp4"
    print_step("💾", f"Exporting → {out_path.relative_to(OUTPUT_DIR.parent)}")

    final.write_videofile(
        str(out_path),
        fps=OUTPUT_FPS,
        codec=OUTPUT_CODEC,
        audio_codec="aac",
        preset=OUTPUT_PRESET,
        threads=os.cpu_count() or 4,
        logger="bar",
    )

    final.close()

    print(f"\n✅  Done!  Output saved to: {out_path}")