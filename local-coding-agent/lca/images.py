"""Images for vision models: files and pasted images become data URLs (data:image/png;base64,...)."""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}
MAX_IMAGE_BYTES = 5_000_000
MAX_IMAGES_PER_MESSAGE = 6
_DATA_URL = re.compile(r"^data:image/(png|jpeg|gif|webp|bmp);base64,[A-Za-z0-9+/=\s]+$")


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_TYPES


MAX_SIDE = 1568  # vision models work at about this size; larger images only cost context
SHRINK_ABOVE_BYTES = 1_500_000


def _shrink(path: Path) -> tuple[str, bytes] | None:
    """(media type, bytes) of a smaller copy, or None when Pillow is not installed or the image is small enough."""
    try:
        from PIL import Image  # optional: pip install "local-coding-agent[images]"
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            if max(im.size) <= MAX_SIDE and path.stat().st_size <= SHRINK_ABOVE_BYTES:
                return None
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            out = io.BytesIO()
            if im.mode in ("RGBA", "LA", "P"):  # keep transparency and sharp screenshot text
                im.save(out, "PNG", optimize=True)
                if out.tell() <= SHRINK_ABOVE_BYTES:
                    return "image/png", out.getvalue()
                out = io.BytesIO()
            im.convert("RGB").save(out, "JPEG", quality=85)
            return "image/jpeg", out.getvalue()
    except OSError:
        return None


def image_data_url(path: Path) -> str:
    """Raises ValueError when the file is not a supported image or is too large. Large images are shrunk to at most
    MAX_SIDE pixels when Pillow is installed."""
    kind = IMAGE_TYPES.get(path.suffix.lower())
    if kind is None:
        raise ValueError(f"{path.name} is not an image the model can read ({', '.join(sorted(IMAGE_TYPES))}).")
    shrunk = _shrink(path)
    if shrunk is not None:
        kind, data = shrunk
    else:
        data = path.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"{path.name} is too large ({len(data) // 1000} KB; the limit is {MAX_IMAGE_BYTES // 1_000_000} MB). "
            'Install Pillow (pip install "local-coding-agent[images]") to have large images shrunk automatically.'
        )
    return f"data:{kind};base64,{base64.b64encode(data).decode('ascii')}"


def valid_data_urls(urls: object) -> list[str]:
    """Pasted images from a client: keep well-formed image data URLs within the size limit."""
    if not isinstance(urls, list):
        return []
    ok = [u for u in urls if isinstance(u, str) and len(u) <= MAX_IMAGE_BYTES * 4 // 3 + 100 and _DATA_URL.match(u[:100] + "A=")]
    return ok[:MAX_IMAGES_PER_MESSAGE]
