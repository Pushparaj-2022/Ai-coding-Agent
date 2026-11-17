---
description: Review the uncommitted changes (or a branch, file or folder) for bugs
argument-hint: [branch, file or folder]
---
Review code for real problems. Target: $ARGUMENTS
(If no target is given, review the uncommitted changes: `git diff` and `git diff --staged`. For a branch, review
`git diff main...<branch>` or the repository's default branch.)

Read every changed part with enough surrounding code to understand it. Look for: wrong logic, unhandled errors and
edge cases, broken callers of changed functions, missing tests for new behaviour, security problems (injection,
secrets, unsafe input), and resource leaks. Do not change any files.

Report each finding as `path:line - problem - how to fix`, most severe first. Leave out style nitpicks. If you find
nothing significant, say so plainly.
