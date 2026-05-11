"""
main.py
───────
Instagram Reel Generator — entry point.

Usage
-----
    python main.py <difficulty>
    difficulty: easy | medium | hard

Pipeline
--------
    1. intro_scene  — random video + random audio + word-synced subtitles
    2. difficulty   — difficulty/<difficulty>.png held for ≤ 3 s over sfx/riser.mp3

Directory layout
----------------
    intro_scene/
        audios/        1.mp3, 2.mp3, …
        videos/        1.mp4, 2.mp4, …
    difficulty/
        easy.png  /  medium.png  /  hard.png
    sfx/
        riser.mp3
    fonts/             (Inter-Black auto-downloaded here on first run)
    output/            (created automatically)
"""

import io
import os
import sys

from moviepy import concatenate_audioclips, concatenate_videoclips

from config import OUTPUT_DIR, OUTPUT_FPS, OUTPUT_CODEC, OUTPUT_PRESET, VALID_DIFFICULTIES
from core.scenes import build_intro_scene, build_difficulty_scene
from utils import print_step

# ── Encoding fix for Windows terminals ────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def main() -> None:
    difficulty = _parse_args()

    print("=" * 62)
    print(f"  Instagram Reel Generator  |  difficulty: {difficulty.upper()}")
    print("=" * 62)

    print_step("🎬", "=== INTRO SCENE ===")
    intro_silent, intro_audio = build_intro_scene()

    print_step("🏁", "=== DIFFICULTY SCENE ===")
    diff_silent, diff_audio = build_difficulty_scene(difficulty)

    print_step("🔗", "Stitching scenes...")
    final = _stitch(intro_silent, intro_audio, diff_silent, diff_audio)
    print(f"   Total duration : {final.duration:.2f} s")

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

    print(f"\n✅  Done!  Output saved to: {out_path}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_args() -> str:
    """Validate and return the difficulty argument from sys.argv."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <difficulty>")
        print(f"  difficulty: {' | '.join(VALID_DIFFICULTIES)}")
        sys.exit(1)

    difficulty = sys.argv[1].strip().lower()
    if difficulty not in VALID_DIFFICULTIES:
        print(f"Error: '{difficulty}' is not a valid difficulty.")
        print(f"  Choose from: {', '.join(VALID_DIFFICULTIES)}")
        sys.exit(1)

    return difficulty


def _stitch(intro_silent, intro_audio, diff_silent, diff_audio):
    """
    Concatenate the two scene pairs into a single final clip.

    Video and audio tracks are concatenated independently before being
    recombined.  This is required because MoviePy 2.x does not reliably
    carry audio through CompositeVideoClip / ImageClip chains.
    """
    final_video = concatenate_videoclips([intro_silent, diff_silent], method="compose")
    final_audio = concatenate_audioclips([intro_audio, diff_audio])
    return final_video.with_audio(final_audio)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()