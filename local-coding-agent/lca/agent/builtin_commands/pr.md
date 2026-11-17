---
description: Open a pull request for the current work (branch, commit, push, gh pr create)
argument-hint: [title or notes]
---
Open a pull request for the current work. $ARGUMENTS

1. Check `git status`, the current branch and `git log` against the default branch (main or master).
2. If on the default branch, create a new branch with a short descriptive name first.
3. Commit any uncommitted changes that belong to this work (in the repository's message style; never secrets).
4. Push the branch with `git push -u origin <branch>` (the user approves the push).
5. Create the pull request with `gh pr create`, with a title and a body that summarizes what changed and why, and how
   it was tested. If `gh` is not installed or not logged in, stop and give the user the exact commands to run.
6. Finish with the pull request URL.
