"""
pipeline.py
───────────
Orchestrates the full reel-generation pipeline.

    run(difficulty)
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
        8.  Stitch all seven scenes   intro → difficulty → q1 → review1 →
                                       q2 → review2 → q3
        9.  Export to output/reel_<difficulty>.mp4
"""

import os
import random
from pathlib import Path

from moviepy import concatenate_audioclips, concatenate_videoclips

from config import OUTPUT_DIR, OUTPUT_FPS, OUTPUT_CODEC, OUTPUT_PRESET
from core.scenes import (
    build_intro_scene,
    build_difficulty_scene,
    build_question_scene,
    build_review_scene,
)
from utils import print_step

# Folder that holds the question images (1.png, 2.png, 3.png).
QUESTIONS_DIR = Path(__file__).parent / "questions"

# Folder that holds comment audio files (1.mp3, 2.mp3, …).
COMMENTS_DIR = Path(__file__).parent / "comments"

# The fixed review video shown after the question timer expires.
REVIEW_VIDEO_PATH = Path(__file__).parent / "question_review" / "question_review.mp4"

# Folder that holds the transition audio files used in the review scene
# (1.mp3, 2.mp3, … 55.mp3).
TRANSITIONS_DIR = Path(__file__).parent / "transition"


class QuestionImagePool:
    """
    Manages a pool of numbered question images (1.png, 2.png, …) so that
    each image is picked at most once per pipeline run.

    Usage
    -----
        pool = QuestionImagePool(QUESTIONS_DIR)
        img1 = pool.pick()   # e.g. 3.png  — now excluded
        img2 = pool.pick()   # e.g. 1.png  — now excluded
        img3 = pool.pick()   # e.g. 2.png  — last one

    Raises
    ------
    FileNotFoundError  – if the questions/ folder is missing.
    RuntimeError       – if the pool is exhausted (all images already used).
    """

    def __init__(self, questions_dir: Path) -> None:
        if not questions_dir.exists():
            raise FileNotFoundError(
                f"The questions/ folder does not exist.\n"
                f"Expected it at: {questions_dir.resolve()}\n"
                f"Create the folder and drop your question images inside it "
                f"(e.g. 1.png, 2.png, 3.png)."
            )

        # Collect all numbered images and sort them.
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
                f"Add files like 1.png, 2.png, 3.png."
            )

        self._used: list[Path] = []
        print_step("🗂 ", f"Question pool   : {[p.name for p in self._available]} "
                          f"({len(self._available)} image(s))")

    def pick(self) -> Path:
        """
        Randomly pick one image from the remaining pool, mark it as used,
        and return its path.  Raises RuntimeError when the pool is empty.
        """
        if not self._available:
            used_names = [p.name for p in self._used]
            raise RuntimeError(
                f"Question image pool is exhausted — all images have already "
                f"been used in this run.\n"
                f"Used : {used_names}\n"
                f"Add more numbered images to the questions/ folder if you "
                f"need additional picks."
            )

        chosen = random.choice(self._available)
        self._available.remove(chosen)
        self._used.append(chosen)

        remaining = [p.name for p in self._available]
        print_step("🎲", f"Picked question : {chosen.name}  "
                         f"(remaining pool: {remaining or ['(empty)']}, "
                         f"used: {[p.name for p in self._used]})")
        return chosen


class TransitionPool:
    """
    Manages a pool of transition audio files (1.mp3 … N.mp3) so that each
    file is picked at most once per pipeline run.  This prevents the same
    transition audio from repeating across the two review scenes within the
    same video.

    Usage
    -----
        pool = TransitionPool(TRANSITIONS_DIR)
        path1 = pool.pick()   # e.g. 17.mp3  — now excluded
        path2 = pool.pick()   # e.g.  3.mp3  — guaranteed different

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
                f"been used in this run.\n"
                f"Used : {used_names}\n"
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


def run(difficulty: str) -> None:
    """Execute the full pipeline for the given difficulty level."""

    # Initialise no-repeat pools once for the entire run.
    question_pool   = QuestionImagePool(QUESTIONS_DIR)
    transition_pool = TransitionPool(TRANSITIONS_DIR)

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
        comments_dir=COMMENTS_DIR,     # comment audio ON for Q1
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
    #   Slides in on top of the last frame of review 1, full 9:16.
    print_step("❓", "=== QUESTION 2 SCENE ===")
    q2_image = question_pool.pick()
    print_step("🖼 ", f"Question 2 image → {q2_image}")
    q2_silent, q2_audio = build_question_scene(
        image_path=q2_image,
        bg_clip=review1_silent,        # freeze last frame of review 1
        fps=OUTPUT_FPS,
        comments_dir=None,             # comment audio OFF for Q2
    )

    # ── 6. Review 2 (different transition audio than review 1) ────────────────
    print_step("🔍", "=== REVIEW 2 SCENE ===")
    review2_audio_path = transition_pool.pick()   # pool excludes already-used file
    review2_silent, review2_audio = build_review_scene(
        review_video_path=REVIEW_VIDEO_PATH,
        audio_path=review2_audio_path,
        fps=OUTPUT_FPS,
    )

    # ── 7. Question 3 (no comment audio, no review follows) ───────────────────
    #   Slides in on top of the last frame of review 2, full 9:16.
    print_step("❓", "=== QUESTION 3 SCENE ===")
    q3_image = question_pool.pick()
    print_step("🖼 ", f"Question 3 image → {q3_image}")
    q3_silent, q3_audio = build_question_scene(
        image_path=q3_image,
        bg_clip=review2_silent,        # freeze last frame of review 2
        fps=OUTPUT_FPS,
        comments_dir=None,             # comment audio OFF for Q3
    )

    # ── 8. Stitch ─────────────────────────────────────────────────────────────
    print_step("🔗", "Stitching scenes  "
                     "[ intro → difficulty → q1 → review1 → q2 → review2 → q3 ] ...")
    final = _stitch(
        intro_silent,    intro_audio,
        diff_silent,     diff_audio,
        q1_silent,       q1_audio,
        review1_silent,  review1_audio,
        q2_silent,       q2_audio,
        review2_silent,  review2_audio,
        q3_silent,       q3_audio,
    )
    print(f"   Total duration : {final.duration:.2f} s")

    # ── 9. Export ─────────────────────────────────────────────────────────────
    _export(final, difficulty)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stitch(
    intro_silent,    intro_audio,
    diff_silent,     diff_audio,
    q1_silent,       q1_audio,
    review1_silent,  review1_audio,
    q2_silent,       q2_audio,
    review2_silent,  review2_audio,
    q3_silent,       q3_audio,
):
    """
    Concatenate video and audio tracks independently before recombining.

    MoviePy 2.x does not reliably carry audio through CompositeVideoClip /
    ImageClip chains, so tracks are joined separately then merged.

    Scene order: intro → difficulty → q1 → review1 → q2 → review2 → q3
    """
    final_video = concatenate_videoclips(
        [
            intro_silent,
            diff_silent,
            q1_silent,
            review1_silent,
            q2_silent,
            review2_silent,
            q3_silent,
        ],
        method="compose",
    )
    final_audio = concatenate_audioclips(
        [
            intro_audio,
            diff_audio,
            q1_audio,
            review1_audio,
            q2_audio,
            review2_audio,
            q3_audio,
        ]
    )
    return final_video.with_audio(final_audio)


def _export(final, difficulty: str) -> None:
    """Write the final clip to output/reel_<difficulty>.mp4."""
    out_path = OUTPUT_DIR / f"reel_{difficulty}.mp4"
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

    # Explicitly close the clip so MoviePy shuts down the FFMPEG subprocess
    # cleanly. Without this, Python's garbage collector tries to close an
    # already-dead process handle on Windows, producing the WinError 6 noise.
    final.close()

    print(f"\n✅  Done!  Output saved to: {out_path}")