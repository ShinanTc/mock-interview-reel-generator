"""
main.py
───────
Instagram Reel Generator — CLI entry point.

Usage
-----
    python main.py <difficulty>
    difficulty: easy | medium | hard
"""

import io
import sys

import pipeline
from config import VALID_DIFFICULTIES

# ── Encoding fix for Windows terminals ────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def main() -> None:
    difficulty = _parse_args()

    print("=" * 62)
    print(f"  Instagram Reel Generator  |  difficulty: {difficulty.upper()}")
    print("=" * 62)

    pipeline.run(difficulty)


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


if __name__ == "__main__":
    main()