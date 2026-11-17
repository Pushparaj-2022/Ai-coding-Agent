"""Project instructions: Markdown files that tell the agent how to work in a repository.

Loaded in order (all that exist, concatenated):
  ~/.lca/AGENTS.md          personal instructions for every project
  <root>/AGENTS.md          shared project instructions (also read by other agents)
  <root>/LCA.md             instructions specific to this agent
  <root>/.lca/instructions.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lca.config import lca_home

MAX_CHARS_PER_FILE = 12_000
MAX_TOTAL_CHARS = 24_000


@dataclass
class InstructionFile:
    source: str
    text: str


def load_instructions(root: Path) -> list[InstructionFile]:
    candidates = [
        ("~/.lca/AGENTS.md", lca_home() / "AGENTS.md"),
        ("AGENTS.md", root / "AGENTS.md"),
        ("LCA.md", root / "LCA.md"),
        (".lca/instructions.md", root / ".lca" / "instructions.md"),
    ]
    found: list[InstructionFile] = []
    total = 0
    for label, path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        if len(text) > MAX_CHARS_PER_FILE:
            text = text[:MAX_CHARS_PER_FILE] + "\n[...truncated]"
        if total + len(text) > MAX_TOTAL_CHARS:
            break
        total += len(text)
        found.append(InstructionFile(source=label, text=text))
    return found
