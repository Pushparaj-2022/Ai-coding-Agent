"""Secret detection: files the agent must never read, and redaction of secret-looking values in output."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable

# Directory names whose contents are never exposed to the model.
SECRET_DIRS = {".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".git"}

# File-name patterns (matched case-insensitively against the base name).
SECRET_FILE_PATTERNS = [
    ".env",
    ".env.*",
    "*.env",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "*.kdbx",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "_netrc",
    ".git-credentials",
    "credentials",
    "credentials.json",
    "*credentials*.json",
    "service-account*.json",
    "secrets.*",
    "*.secret",
    "*.secrets",
]

# Templates that document variables without real values are fine to read.
SAFE_FILE_PATTERNS = [".env.example", ".env.sample", ".env.template", ".env.dist", "*.example", "*.sample"]

_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED PRIVATE KEY]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED AWS KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), "[REDACTED GITHUB TOKEN]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"), "[REDACTED GITHUB TOKEN]"),
    (re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}\b"), "[REDACTED API KEY]"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), "[REDACTED SLACK TOKEN]"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "[REDACTED GOOGLE KEY]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), "[REDACTED JWT]"),
    # key = "value" style assignments of credentials
    (
        re.compile(
            r"(?i)\b((?:api[_-]?key|secret[_-]?key|client[_-]?secret|access[_-]?token|auth[_-]?token|password|passwd)\s*[:=]\s*)"
            r"(['\"]?)([^\s'\"]{8,})\2"
        ),
        r"\1\2[REDACTED]\2",
    ),
]


def is_secret_name(name: str) -> bool:
    lower = name.lower()
    if any(fnmatch.fnmatchcase(lower, p) for p in SAFE_FILE_PATTERNS):
        return False
    return any(fnmatch.fnmatchcase(lower, p) for p in SECRET_FILE_PATTERNS)


def is_secret_path(parts: Iterable[str]) -> bool:
    """True if a workspace-relative path (given as its parts) points at secret material."""
    parts = list(parts)
    if any(p.lower() in SECRET_DIRS for p in parts[:-1]):
        return True
    if parts and parts[-1].lower() in SECRET_DIRS:
        return True
    return bool(parts) and is_secret_name(parts[-1])


def redact_secrets(text: str) -> str:
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text
