"""
core/clock.py
─────────────
Generates an animated countdown clock as a transparent MoviePy VideoClip.

The clock displays a sweeping arc that depletes clockwise (like a pie draining),
a numeric digit in the centre, and a thin sweep hand.  Colour shifts
green → amber → red as time runs out; the outer ring pulses in the final 3 s.

Usage
-----
    from core.clock import build_countdown_clip

    clock = build_countdown_clip(duration=10.0, size=210, fps=30)
    clock = clock.with_position((x, y))  # position before compositing

The clip has an alpha mask, so it composites cleanly over any background
inside a CompositeVideoClip without any rectangular border.
"""

import math
from pathlib import Path

import numpy as np
from moviepy import VideoClip
from PIL import Image, ImageDraw, ImageFont

# ── Font resolution ────────────────────────────────────────────────────────────
# Tries a handful of common system paths; falls back to PIL's built-in bitmap
# font if none are found.  Add your own preferred path at the top of the list.

_FONT_SEARCH_PATHS = [
    # Linux (DejaVu, installed on most distros)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    # Windows
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
]


def _get_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_SEARCH_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# ── Core drawing routine ───────────────────────────────────────────────────────

def _draw_clock_frame(t: float, duration: float, size: int) -> Image.Image:
    """
    Render one RGBA frame of the countdown clock at scene time `t`.

    Parameters
    ----------
    t        : Current time in seconds (0 → duration).
    duration : Total countdown duration in seconds.
    size     : Width/height of the square canvas in pixels.

    Returns
    -------
    PIL.Image in RGBA mode.
    """
    remaining  = max(0.0, duration - t)
    frac       = remaining / duration          # 1.0 → 0.0 (arc fill fraction)
    secs_shown = math.ceil(remaining) if remaining > 0.0 else 0

    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = max(6, size // 30)
    cx  = cy = size // 2

    # ── Pulse: outer ring and background throb in the last 3 s ────────────────
    if remaining <= 3.0 and remaining > 0.0:
        # ~2 Hz sinusoidal brightness modulation
        pulse = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 2)   # 0 → 1
        bg_alpha   = int(160 + 60 * pulse)                    # 160 – 220
        ring_extra = int(80  * pulse)                         # bonus brightness
    else:
        pulse      = 1.0
        bg_alpha   = 185
        ring_extra = 0

    # ── Background disc ───────────────────────────────────────────────────────
    draw.ellipse(
        [pad, pad, size - pad, size - pad],
        fill=(12, 12, 12, bg_alpha),
    )

    # ── Arc colour: green → amber → red ───────────────────────────────────────
    if frac > 0.55:
        r, g, b = 60, 210, 100       # green
    elif frac > 0.25:
        r, g, b = 255, 185, 20       # amber
    else:
        r, g, b = 235, 45, 45        # red

    arc_color  = (r, g, b, 255)
    ring_color = (
        min(255, r + ring_extra),
        min(255, g + ring_extra // 3),
        min(255, b + ring_extra // 3),
        min(255, 200 + ring_extra),
    )

    # ── Arc geometry ──────────────────────────────────────────────────────────
    arc_w   = max(9, size // 20)
    arc_pad = pad + arc_w // 2 + 2
    box     = [arc_pad, arc_pad, size - arc_pad, size - arc_pad]

    # "remaining" arc: -90° (top) clockwise for frac * 360°
    arc_end = -90.0 + frac * 360.0

    # Faint track showing the depleted portion
    if frac < 0.999:
        draw.arc(box, start=arc_end, end=270.0,
                 fill=(70, 70, 70, 110), width=arc_w)

    # Active arc
    if frac > 0.001:
        draw.arc(box, start=-90.0, end=arc_end,
                 fill=arc_color, width=arc_w)

    # ── Sweep hand ────────────────────────────────────────────────────────────
    hand_rad = math.radians(arc_end)
    # Keep the hand tip just inside the arc midline
    hand_len = cx - arc_pad - arc_w // 2 - 2
    hx = cx + int(hand_len * math.cos(hand_rad))
    hy = cy + int(hand_len * math.sin(hand_rad))
    draw.line([(cx, cy), (hx, hy)],
              fill=(255, 255, 255, 210), width=max(2, size // 70))

    # Centre pivot dot
    dot_r = max(4, size // 40)
    draw.ellipse(
        [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
        fill=(255, 255, 255, 230),
    )

    # ── Outer decorative ring ─────────────────────────────────────────────────
    draw.ellipse(
        [pad, pad, size - pad, size - pad],
        outline=ring_color,
        width=max(2, size // 60),
    )

    # ── Digit ─────────────────────────────────────────────────────────────────
    font_size = max(12, int(size * 0.30))
    font      = _get_bold_font(font_size)
    text      = str(secs_shown)

    # Compute tight bounding box so the digit is truly centred
    bbox = draw.textbbox((0, 0), text, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]
    tx   = cx - tw // 2 - bbox[0]
    ty   = cy - th // 2 - bbox[1]

    # Soft shadow for legibility over the arc
    draw.text((tx + 2, ty + 2), text, font=font, fill=(0, 0, 0, 120))
    draw.text((tx, ty),         text, font=font, fill=(255, 255, 255, 240))

    return img


# ── Public API ─────────────────────────────────────────────────────────────────

def build_countdown_clip(
    duration: float = 10.0,
    size: int = 210,
    fps: int = 30,
) -> VideoClip:
    """
    Return a transparent-background MoviePy VideoClip of an animated countdown.

    The clip contains:
    - A depleting arc (green → amber → red)
    - A sweeping clock hand pointing to the arc tip
    - A large digit showing the ceiling of the remaining seconds
    - A pulsing outer ring in the final 3 seconds
    - An alpha mask for clean compositing over any background

    Parameters
    ----------
    duration : Total countdown time in seconds (default 10).
    size     : Diameter of the clock in pixels (default 210).
    fps      : Must match the parent scene's fps (default 30).

    Returns
    -------
    MoviePy VideoClip with an attached mask clip.

    Example
    -------
    >>> from config import OUT_W, OUT_H, OUTPUT_FPS
    >>> clock = build_countdown_clip(duration=10.0, size=210, fps=OUTPUT_FPS)
    >>> clock = clock.with_position((Out_W - 210 - 40, OUT_H - 210 - 40))
    >>> # Then include `clock` in your CompositeVideoClip layers list.
    """
    n_frames   = int(math.ceil(duration * fps)) + 1
    timestamps = [i / fps for i in range(n_frames)]

    # Pre-render every frame once so we don't call PIL twice per frame
    # (once for RGB, once for the mask).  300 frames @ 210×210 ≈ 50 MB RAM.
    print(f"   Clock pre-render: {n_frames} frames  "
          f"({size}×{size} px, {fps} fps)…")

    frames_rgba = [
        np.array(_draw_clock_frame(t, duration, size), dtype=np.uint8)
        for t in timestamps
    ]

    rgb_frames   = [f[:, :, :3] for f in frames_rgba]
    alpha_frames = [f[:, :, 3].astype(np.float32) / 255.0 for f in frames_rgba]

    def _make_rgb(t: float) -> np.ndarray:
        idx = min(int(t * fps), n_frames - 1)
        return rgb_frames[idx]

    def _make_alpha(t: float) -> np.ndarray:
        idx = min(int(t * fps), n_frames - 1)
        return alpha_frames[idx]

    rgb_clip  = VideoClip(_make_rgb,   duration=duration).with_fps(fps)
    mask_clip = VideoClip(_make_alpha, duration=duration).with_fps(fps)

    return rgb_clip.with_mask(mask_clip)