"""Trust for parts of a project's .lca/config.toml that run commands on this machine ([hooks], [mcp]).

A repository must not be able to run programs just because you opened it, so these sections are ignored until you
trust them (`lca hooks --trust`, `lca mcp --trust`). Trust is stored outside the repository, in
~/.lca/trusted/<project>.json, as a digest of the section: when the section changes it needs trusting again.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from lca.config import lca_home


def _trust_file(root: Path) -> Path:
    key = hashlib.sha256(str(root.resolve()).lower().encode()).hexdigest()[:16]
    return lca_home() / "trusted" / f"{key}.json"


def digest(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def read_project_section(root: Path, section: str) -> Any:
    path = root / ".lca" / "config.toml"
    if not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            return tomllib.load(f).get(section)
    except (OSError, tomllib.TOMLDecodeError):
        return None


def _load(root: Path) -> dict[str, Any]:
    try:
        data = json.loads(_trust_file(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def trust_section(root: Path, section: str) -> str | None:
    """Trust the project's current `section`. Returns its digest, or None when the project has no such section."""
    data = read_project_section(root, section)
    if not data:
        return None
    saved = _load(root)
    sections = dict(saved.get("sections") or {})
    sections[section] = digest(data)
    path = _trust_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"workspace": str(root), "sections": sections}, indent=2), encoding="utf-8")
    return sections[section]


def is_trusted(root: Path, section: str, data: Any) -> bool:
    return (_load(root).get("sections") or {}).get(section) == digest(data)
