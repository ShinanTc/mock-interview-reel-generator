"""
core/font.py
────────────
Font discovery, downloading, and loading.

Lookup order for Inter-Black:
  1. Local fonts/ directory
  2. Common system font paths (Linux / macOS / Windows)
  3. Download from the official Inter GitHub release (one-time, cached)
  4. Pillow built-in fallback (last resort)
"""

import io
import os
import zipfile
import urllib.request
from pathlib import Path

from PIL import ImageFont

from config import FONTS_DIR


# ── Candidate paths ────────────────────────────────────────────────────────────

_SYSTEM_CANDIDATES = [
    FONTS_DIR / "Inter-Black.ttf",
    FONTS_DIR / "Inter-Black.otf",
    Path("/usr/share/fonts/truetype/inter/Inter-Black.ttf"),
    Path("/usr/local/share/fonts/Inter-Black.ttf"),
    Path(os.path.expanduser("~/Library/Fonts/Inter-Black.ttf")),
    Path("C:/Windows/Fonts/Inter-Black.ttf"),
]

_INTER_ZIP_URL = (
    "https://github.com/rsms/inter/releases/download/v4.0/Inter-4.0.zip"
)


# ── Public API ─────────────────────────────────────────────────────────────────

def find_or_download_font() -> Path | None:
    """
    Return a Path to Inter-Black.ttf, downloading it if necessary.
    Returns None if every strategy fails (caller should fall back to the
    Pillow built-in font via load_font()).
    """
    found = _find_locally()
    if found:
        print(f"   Font found: {found}")
        return found

    print("   Inter-Black not found locally — downloading from GitHub...")
    return _download_font()


def load_font(font_path: Path | None, size: int) -> ImageFont.FreeTypeFont:
    """
    Load a TrueType font at *size* px.  Falls back to the Pillow built-in
    font if *font_path* is None or the file cannot be loaded.
    """
    if font_path:
        try:
            return ImageFont.truetype(str(font_path), size)
        except Exception as exc:
            print(f"   Could not load font ({exc}); using Pillow default.")
    return ImageFont.load_default(size=max(size, 10))


# ── Private helpers ────────────────────────────────────────────────────────────

def _find_locally() -> Path | None:
    """Return the first candidate path that exists, or None."""
    return next((p for p in _SYSTEM_CANDIDATES if p.exists()), None)


def _download_font() -> Path | None:
    """
    Download the Inter release zip, extract Inter-Black.ttf into FONTS_DIR,
    and return its path.  Returns None on any failure.
    """
    try:
        with urllib.request.urlopen(_INTER_ZIP_URL, timeout=30) as resp:
            data = resp.read()

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            ttf_entries = [
                n for n in zf.namelist()
                if "Inter-Black" in n
                and n.endswith((".ttf", ".otf"))
                and "Variable" not in n
            ]
            if not ttf_entries:
                raise FileNotFoundError("Inter-Black not found inside the zip.")

            # Prefer .ttf over .otf
            entry = next(
                (n for n in ttf_entries if n.endswith(".ttf")),
                ttf_entries[0],
            )
            dest = FONTS_DIR / Path(entry).name
            dest.write_bytes(zf.read(entry))

        print(f"   Downloaded → {dest}")
        return dest

    except Exception as exc:
        print(f"   Font download failed: {exc}")
        print("   Falling back to Pillow built-in font.")
        print("   TIP: Manually place Inter-Black.ttf in the fonts/ folder.")
        return None