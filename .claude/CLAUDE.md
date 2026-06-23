# CLAUDE.md

The project guide for agents lives in **[AGENTS.md](../AGENTS.md)** (the SSOT,
read by every agent host). Read it first.

@AGENTS.md

## Claude Code specifics

- Dev skills are in `skills/` and surfaced here via per-skill relative symlinks
  (`.claude/skills/<name> -> ../../skills/<name>`). Invoke them as
  `/ek-dev-architecture`, `/ek-dev-licensing`, `/ek-dev-add-metric`,
  `/ek-dev-add-signal`, `/ek-dev-ocr`. Start with `ek-dev-architecture`.
- Per-session handoff notes go in the gitignored `.claude/handoffs/`.
