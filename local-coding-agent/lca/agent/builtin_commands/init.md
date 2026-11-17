---
description: Write AGENTS.md: how this project is built, tested and organized, for future sessions
---
Analyze this project and create (or improve, if it exists) the file AGENTS.md at the workspace root. Future agent
sessions read it first, so it must contain what an agent needs to work here, concisely:

1. Commands: how to install dependencies, build, run the tests (all tests and a single test), lint and type-check.
   Find them in the manifests (package.json scripts, pyproject.toml, Makefile, ...) and the README, and run the test
   command once to confirm it works.
2. Architecture: the main folders and modules and how they fit together. Only what is not obvious from file names.
3. Conventions: code style, naming, error handling and testing patterns this codebase actually uses.
4. Gotchas: anything surprising you noticed (generated files, required environment variables, platform quirks).

Do not invent anything; state only what you found. Keep it under about 80 lines. If AGENTS.md exists, keep what is
still correct and fix or add the rest. $ARGUMENTS
