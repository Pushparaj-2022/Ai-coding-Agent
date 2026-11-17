"""@-mentions: `@src/app.py` in a message attaches that file's contents (or a folder's listing, or an image) to it."""

from __future__ import annotations

import re

from lca.images import MAX_IMAGES_PER_MESSAGE, image_data_url, is_image
from lca.tools.workspace import Workspace, WorkspaceError

MENTION = re.compile(r"(?<![\w@/])@([\w.\-/\\]+[\w/])")
MAX_FILE_CHARS = 30_000
MAX_TOTAL_CHARS = 80_000
MAX_LISTING = 60


def expand_mentions(workspace: Workspace, text: str, images: bool = True) -> tuple[str, list[str], list[str]]:
    """Returns (text with the mentioned files attached, the mentioned paths, image data URLs). Unknown @words are left
    alone. Mentioned images are attached as images when `images` (the model can read them)."""
    blocks: list[str] = []
    found: list[str] = []
    pictures: list[str] = []
    total = 0
    for m in MENTION.finditer(text):
        wanted = m.group(1).rstrip("/\\") or "."
        try:
            path = workspace.resolve(wanted)
        except WorkspaceError:
            continue
        rel = workspace.rel(path)
        if rel in found or not path.exists() or workspace.is_secret(path):
            continue
        if images and path.is_file() and is_image(path) and len(pictures) < MAX_IMAGES_PER_MESSAGE:
            try:
                pictures.append(image_data_url(path))
                found.append(rel)
                continue
            except ValueError as e:
                blocks.append(f"[@{rel} not attached: {e}]")
                continue
        if path.is_dir():
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))[:MAX_LISTING]
            body = "\n".join(f"{e.name}{'/' if e.is_dir() else ''}" for e in entries)
            block = f"[Folder @{rel}:]\n{body}"
        else:
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if b"\0" in data[:8192]:
                block = f"[@{rel} is a binary file ({len(data)} bytes); not attached.]"
            else:
                content = data.decode("utf-8", errors="replace")
                if len(content) > MAX_FILE_CHARS:
                    content = content[:MAX_FILE_CHARS] + f"\n[... {len(content) - MAX_FILE_CHARS} more characters: read_file the rest ...]"
                block = f"[File @{rel}:]\n```\n{content}\n```"
        if total + len(block) > MAX_TOTAL_CHARS:
            blocks.append(f"[@{rel} not attached: the attachments are already large. Use read_file.]")
            continue
        total += len(block)
        blocks.append(block)
        found.append(rel)
    if not blocks:
        return text, found, pictures
    return text + "\n\n" + "\n\n".join(blocks), found, pictures
