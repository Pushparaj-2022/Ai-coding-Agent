"""Custom slash commands and skills, from Markdown files.

Commands: built-in ones (/init, /commit, /review, /pr), ~/.lca/commands/<name>.md (yours, every project) and
<project>/.lca/commands/<name>.md (shared with the repository). Later ones override earlier ones with the same name. `/name some text` sends the file's text as the message, with
$ARGUMENTS replaced by `some text` ($1, $2, ... by single words); without $ARGUMENTS the text is appended. Commands in
sub-folders are named folder:name. Optional front matter:

    ---
    description: Review the current diff for bugs
    argument-hint: [focus area]
    ---

Skills: ~/.lca/skills/<name>/SKILL.md and <project>/.lca/skills/<name>/SKILL.md, with front matter `name` and
`description`. The agent sees only each skill's name and description; it loads the full instructions (and any other
files in the skill's folder) with the load_skill tool when a task matches. `/name text` also runs a skill directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from lca.config import lca_home

MAX_SKILL_FILE = 100_000
BUILTIN_DIR = Path(__file__).with_name("builtin_commands")  # /init, /commit, /review, /pr


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip().strip("\"'")
    return meta, text[m.end():]


@dataclass
class CustomCommand:
    name: str
    description: str
    argument_hint: str
    body: str
    source: str  # user | project

    def expand(self, args: str) -> str:
        words = args.split()
        text = self.body.replace("$ARGUMENTS", args)
        text = re.sub(r"\$(\d)", lambda m: words[int(m.group(1)) - 1] if 0 < int(m.group(1)) <= len(words) else "", text)
        if "$ARGUMENTS" not in self.body and not re.search(r"\$\d", self.body) and args:
            text = f"{text.rstrip()}\n\n{args}"
        return text.strip()

    def as_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "argument_hint": self.argument_hint, "source": self.source, "kind": "command"}


@dataclass
class Skill:
    name: str
    description: str
    folder: Path
    source: str

    def as_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "argument_hint": "", "source": self.source, "kind": "skill"}


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def load_commands(root: Path) -> dict[str, CustomCommand]:
    found: dict[str, CustomCommand] = {}
    bases = (("builtin", BUILTIN_DIR), ("user", lca_home() / "commands"), ("project", root / ".lca" / "commands"))
    for source, base in bases:  # later sources override earlier ones
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.md")):
            text = _read(path)
            if text is None:
                continue
            rel = path.relative_to(base).with_suffix("")
            name = ":".join(rel.parts).lower()
            meta, body = parse_front_matter(text)
            first = next((ln.strip("# ").strip() for ln in body.splitlines() if ln.strip()), "")
            found[name] = CustomCommand(name, meta.get("description") or first[:100], meta.get("argument-hint", ""), body, source)
    return found


def load_skills(root: Path) -> dict[str, Skill]:
    found: dict[str, Skill] = {}
    for source, base in (("user", lca_home() / "skills"), ("project", root / ".lca" / "skills")):
        if not base.is_dir():
            continue
        for skill_md in sorted(base.glob("*/SKILL.md")):
            text = _read(skill_md)
            if text is None:
                continue
            meta, body = parse_front_matter(text)
            name = (meta.get("name") or skill_md.parent.name).strip().lower()
            description = meta.get("description") or next((ln.strip("# ").strip() for ln in body.splitlines() if ln.strip()), "")
            found[name] = Skill(name, description[:300], skill_md.parent, source)
    return found


def skill_text(skill: Skill, file: str = "") -> str:
    """The skill's instructions (without front matter) and its other files, or one of those files."""
    if file:
        target = (skill.folder / file).resolve()
        if skill.folder.resolve() not in target.parents or not target.is_file():
            raise ValueError(f"{file} is not a file of the {skill.name} skill.")
        if target.stat().st_size > MAX_SKILL_FILE:
            raise ValueError(f"{file} is too large to load ({target.stat().st_size} bytes).")
        return _read(target) or ""
    _, body = parse_front_matter(_read(skill.folder / "SKILL.md") or "")
    others = sorted(p.relative_to(skill.folder).as_posix() for p in skill.folder.rglob("*") if p.is_file() and p.name != "SKILL.md")
    extra = ("\n\nOther files of this skill (load them with load_skill(name, file=...)):\n" + "\n".join(f"- {o}" for o in others[:50])) if others else ""
    return f"# Skill: {skill.name}\n\n{body.strip()}{extra}"


def slash_items(root: Path) -> list[dict]:
    """Custom commands and skills for the editor's / menu."""
    skills = load_skills(root)
    commands = load_commands(root)
    items = [c.as_dict() for c in commands.values()]
    items += [s.as_dict() for name, s in skills.items() if name not in commands]
    return sorted(items, key=lambda i: i["name"])


def expand_input(root: Path, text: str) -> tuple[str, str | None]:
    """`/name args` → (the command's or skill's prompt, name). Other text is returned unchanged with name None.
    Raises KeyError for an unknown /name."""
    m = re.match(r"^/([\w:.-]+)(?:\s+(.*))?$", text.strip(), re.S)
    if not m:
        return text, None
    name, args = m.group(1).lower(), (m.group(2) or "").strip()
    commands = load_commands(root)
    if name in commands:
        return commands[name].expand(args), name
    skills = load_skills(root)
    if name in skills:
        prompt = f"Use the {name} skill: load it with load_skill and follow its instructions."
        return (f"{prompt}\n\n{args}" if args else prompt), name
    raise KeyError(name)
