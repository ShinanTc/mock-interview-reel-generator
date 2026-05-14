"""
pipeline.py
───────────
Orchestrates the full reel-generation pipeline.

    run(difficulty)
        1. Build intro scene        (video + audio + subtitles)
        2. Build difficulty scene   (PNG + riser SFX)
        3. Build question scene     (PNG slide-in on top of difficulty +
                                     random comment audio + comment subtitles)
        4. Stitch all three scenes  intro → difficulty → question
        5. Export to output/reel_<difficulty>.mp4
"""

import os
from pathlib import Path

from moviepy import concatenate_audioclips, concatenate_videoclips

from config import OUTPUT_DIR, OUTPUT_FPS, OUTPUT_CODEC, OUTPUT_PRESET
from core.scenes import build_intro_scene, build_difficulty_scene, build_question_scene
from utils import print_step

# Folder that holds the question images (one per difficulty, or a single image).
QUESTIONS_DIR = Path(__file__).parent / "questions"

# Folder that holds comment audio files (1.mp3, 2.mp3, …).
COMMENTS_DIR = Path(__file__).parent / "comments"


def run(difficulty: str) -> None:
    """Execute the full pipeline for the given difficulty level."""

    # ── 1. Intro ──────────────────────────────────────────────────────────────
    print_step("🎬", "=== INTRO SCENE ===")
    intro_silent, intro_audio = build_intro_scene()

    # ── 2. Difficulty ─────────────────────────────────────────────────────────
    print_step("🏁", "=== DIFFICULTY SCENE ===")
    diff_silent, diff_audio = build_difficulty_scene(difficulty)

    # ── 3. Question ───────────────────────────────────────────────────────────
    #   Pass diff_silent so the question slides in ON TOP of the difficulty
    #   image rather than over a plain black background.
    print_step("❓", "=== QUESTION SCENE ===")
    question_image = _resolve_question_image(difficulty)
    print_step("🖼 ", f"Question image → {question_image}")
    q_silent, q_audio = build_question_scene(
        image_path=question_image,
        bg_clip=diff_silent,
        fps=OUTPUT_FPS,
        comments_dir=COMMENTS_DIR,   # ← new: random comment audio + subtitles
    )

    # ── 4. Stitch ─────────────────────────────────────────────────────────────
    print_step("🔗", "Stitching scenes  [ intro → difficulty → question ] ...")
    final = _stitch(
        intro_silent, intro_audio,
        diff_silent,  diff_audio,
        q_silent,     q_audio,
    )
    print(f"   Total duration : {final.duration:.2f} s")

    # ── 5. Export ─────────────────────────────────────────────────────────────
    _export(final, difficulty)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_question_image(difficulty: str) -> Path:
    """
    Look for a question image inside the `questions/` folder.

    Resolution order (first match wins):
      1. questions/<difficulty>.png   — difficulty-specific image
      2. questions/<difficulty>.jpg
      3. questions/question.png       — generic fallback
      4. questions/question.jpg

    Raises FileNotFoundError with a helpful message if nothing is found.
    """
    if not QUESTIONS_DIR.exists():
        raise FileNotFoundError(
            f"The questions/ folder does not exist.\n"
            f"Expected it at: {QUESTIONS_DIR.resolve()}\n"
            f"Create the folder and drop your question image inside it."
        )

    candidates = [
        QUESTIONS_DIR / f"{difficulty}.png",
        QUESTIONS_DIR / f"{difficulty}.jpg",
        QUESTIONS_DIR / "question.png",
        QUESTIONS_DIR / "question.jpg",
    ]
    for path in candidates:
        if path.exists():
            return path

    found = [p.name for p in QUESTIONS_DIR.iterdir()]
    raise FileNotFoundError(
        f"No question image found for difficulty '{difficulty}'.\n"
        f"Looked in : {QUESTIONS_DIR.resolve()}\n"
        f"Tried     : {[p.name for p in candidates]}\n"
        f"Found     : {found or ['(empty folder)']}"
    )


def _stitch(
    intro_silent, intro_audio,
    diff_silent,  diff_audio,
    q_silent,     q_audio,
):
    """
    Concatenate video and audio tracks independently before recombining.

    MoviePy 2.x does not reliably carry audio through CompositeVideoClip /
    ImageClip chains, so tracks are joined separately then merged.

    Scene order:  intro  →  difficulty  →  question
    """
    final_video = concatenate_videoclips(
        [intro_silent, diff_silent, q_silent],
        method="compose",
    )
    final_audio = concatenate_audioclips([intro_audio, diff_audio, q_audio])
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