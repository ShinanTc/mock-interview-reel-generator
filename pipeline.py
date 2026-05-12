"""
pipeline.py
───────────
Orchestrates the full reel-generation pipeline.

    run(difficulty)
        1. Build intro scene  (video + audio + subtitles)
        2. Build difficulty scene (PNG + riser SFX)
        3. Stitch both scenes into one clip
        4. Export to output/reel_<difficulty>.mp4
"""

import os

from moviepy import concatenate_audioclips, concatenate_videoclips

from config import OUTPUT_DIR, OUTPUT_FPS, OUTPUT_CODEC, OUTPUT_PRESET
from core.scenes import build_intro_scene, build_difficulty_scene
from utils import print_step


def run(difficulty: str) -> None:
    """Execute the full pipeline for the given difficulty level."""
    print_step("🎬", "=== INTRO SCENE ===")
    intro_silent, intro_audio = build_intro_scene()

    print_step("🏁", "=== DIFFICULTY SCENE ===")
    diff_silent, diff_audio = build_difficulty_scene(difficulty)

    print_step("🔗", "Stitching scenes...")
    final = _stitch(intro_silent, intro_audio, diff_silent, diff_audio)
    print(f"   Total duration : {final.duration:.2f} s")

    _export(final, difficulty)


def _stitch(intro_silent, intro_audio, diff_silent, diff_audio):
    """
    Concatenate video and audio tracks independently before recombining.

    MoviePy 2.x does not reliably carry audio through CompositeVideoClip /
    ImageClip chains, so tracks are joined separately then merged.
    """
    final_video = concatenate_videoclips([intro_silent, diff_silent], method="compose")
    final_audio = concatenate_audioclips([intro_audio, diff_audio])
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

    print(f"\n✅  Done!  Output saved to: {out_path}")