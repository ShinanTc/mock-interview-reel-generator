"""
utils.py
────────
Tiny shared helpers that don't belong to any specific domain.
"""

import random
from pathlib import Path


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    """Convert a hex colour string like '#385E4F' to an (R, G, B) tuple."""
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def print_step(emoji: str, msg: str) -> None:
    """Print a clearly visible pipeline step to stdout."""
    print(f"\n{emoji}  {msg}")


def pick_random_file(directory: Path, extensions: list[str]) -> Path:
    """
    Return a randomly chosen file from *directory* that has one of the
    given *extensions* (e.g. ['.mp4', '.mov']).

    Raises FileNotFoundError if no matching files exist.
    """
    files = [f for ext in extensions for f in directory.glob(f"*{ext}")]
    if not files:
        raise FileNotFoundError(
            f"No files with extensions {extensions} found in: {directory}"
        )
    chosen = random.choice(sorted(files))
    print(f"   -> Selected: {chosen.name}")
    return chosen