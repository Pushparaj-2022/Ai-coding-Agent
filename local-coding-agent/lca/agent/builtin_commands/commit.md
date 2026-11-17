---
description: Commit the current changes with a message in this repository's style
argument-hint: [what to commit or a message hint]
---
Commit the current changes. $ARGUMENTS

1. Run `git status` and `git diff` (and `git diff --staged`) to see exactly what changed, and `git log -5 --oneline`
   to see this repository's commit message style.
2. Stage only the files that belong to this change (never secrets such as .env files or credentials). If unrelated
   changes are mixed in, ask with ask_user which to include.
3. Commit with a concise message in the repository's style: a short summary line, and a body explaining why if it
   is not obvious.
4. Do not push, amend earlier commits, or use --no-verify. If a pre-commit hook fails, fix the problem and commit again.
5. Finish by showing `git log -1 --stat`.
