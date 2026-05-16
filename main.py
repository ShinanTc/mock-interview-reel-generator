"""
main.py
───────
Instagram Reel Generator — batch entry point.

Usage
-----
    python main.py

Generates 4 reels for every difficulty level (easy | medium | hard) and
writes them to:

    output/
        easy/
            reel_easy_1.mp4  …  reel_easy_4.mp4
        medium/
            reel_medium_1.mp4  …  reel_medium_4.mp4
        hard/
            reel_hard_1.mp4  …  reel_hard_4.mp4

No-repeat guarantee
-------------------
A single QuestionImagePool and a single TransitionPool are created ONCE
per difficulty level and shared across all 4 videos for that difficulty.
This means:

  • The same question image will never appear in more than one video for a
    given difficulty.
  • The same transition audio will never appear in more than one review
    scene across all 4 videos for a given difficulty.

Required image counts per difficulty folder
-------------------------------------------
Each video uses 3 question images.  4 videos × 3 images = 12 image slots.
To guarantee zero repeats you therefore need at least 12 images per
difficulty folder (e.g. questions/easy/1.png … questions/easy/12.png).
"""

import io
import sys

import pipeline
from pipeline import QuestionImagePool, TransitionPool, QUESTIONS_DIR, TRANSITIONS_DIR
from config import VALID_DIFFICULTIES, OUTPUT_DIR

# ── Encoding fix for Windows terminals ────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

VIDEOS_PER_DIFFICULTY = 4


def main() -> None:
    total = len(VALID_DIFFICULTIES) * VIDEOS_PER_DIFFICULTY

    print("=" * 62)
    print(f"  Instagram Reel Generator  |  batch mode")
    print(f"  {len(VALID_DIFFICULTIES)} difficulties × {VIDEOS_PER_DIFFICULTY} videos = {total} reels")
    print("=" * 62)

    # Create output sub-folders up front so any early error is obvious.
    for difficulty in VALID_DIFFICULTIES:
        (OUTPUT_DIR / difficulty).mkdir(parents=True, exist_ok=True)

    completed = 0
    for difficulty in VALID_DIFFICULTIES:

        print()
        print("=" * 62)
        print(f"  DIFFICULTY: {difficulty.upper()}")
        print("=" * 62)

        # ── Create shared pools ONCE per difficulty ────────────────────────────
        # Both pools live for the entire batch of 4 videos so images and
        # transition audio are never reused across videos of the same difficulty.
        question_pool   = QuestionImagePool(QUESTIONS_DIR / difficulty)
        transition_pool = TransitionPool(TRANSITIONS_DIR)

        for idx in range(1, VIDEOS_PER_DIFFICULTY + 1):
            print()
            print("─" * 62)
            print(f"  [{completed + 1}/{total}]  {difficulty.upper()}  —  video {idx} of {VIDEOS_PER_DIFFICULTY}")
            print("─" * 62)

            pipeline.run(
                difficulty=difficulty,
                video_index=idx,
                question_pool=question_pool,
                transition_pool=transition_pool,
            )
            completed += 1

    print()
    print("=" * 62)
    print(f"  ✅  All {total} reels generated.  Check the output/ folder.")
    print("=" * 62)


if __name__ == "__main__":
    main()