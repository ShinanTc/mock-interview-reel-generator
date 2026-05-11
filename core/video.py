"""
core/video.py
─────────────
Low-level video and image geometry helpers.

All functions implement CSS `object-fit: cover` logic:
  - wider than target ratio → scale to height, crop left/right
  - narrower than target ratio → scale to width, crop top/bottom
"""

from PIL import Image
from moviepy import VideoFileClip, concatenate_videoclips

from config import OUT_W, OUT_H


def force_9_16(
    clip: VideoFileClip,
    target_w: int = OUT_W,
    target_h: int = OUT_H,
) -> VideoFileClip:
    """Centre-crop and scale a video clip to exactly target_w × target_h."""
    src_w, src_h     = clip.w, clip.h
    target_ratio     = target_w / target_h
    src_ratio        = src_w / src_h

    scale = target_h / src_h if src_ratio > target_ratio else target_w / src_w
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    resized = clip.resized((new_w, new_h))
    x1 = (new_w - target_w) // 2
    y1 = (new_h - target_h) // 2
    return resized.cropped(x1=x1, y1=y1, x2=x1 + target_w, y2=y1 + target_h)


def image_cover_crop(
    img: Image.Image,
    target_w: int = OUT_W,
    target_h: int = OUT_H,
) -> Image.Image:
    """Scale a PIL image to fully cover target_w × target_h, then centre-crop."""
    src_w, src_h = img.size
    target_ratio = target_w / target_h
    src_ratio    = src_w / src_h

    scale = target_h / src_h if src_ratio > target_ratio else target_w / src_w
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    img  = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top  = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def loop_clip_to(clip: VideoFileClip, duration: float) -> VideoFileClip:
    """Loop *clip* as many times as needed to fill exactly *duration* seconds."""
    if clip.duration >= duration:
        return clip.subclipped(0, duration)
    loops = int(duration / clip.duration) + 1
    return concatenate_videoclips([clip] * loops).subclipped(0, duration)