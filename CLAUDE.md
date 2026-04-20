# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 2.5. Defensibility First

**Write code we could defend in a paper. No ad hoc tricks.**

- This codebase is very recent, and almost nothing should be assumed frozen.
- We are in an experimentation phase, so don't be timid about questioning or replacing existing implementations.
- Do not add hardcoded lexical substitutions, phrase-specific rewrites, dataset-specific fixes, or content-aware string repair rules unless the user explicitly asks for an experimental heuristic.
- Do not patch isolated failures with hidden special cases.
- Prefer generic, principled mechanisms over benchmark-tuned behavior.
- If a change would feel embarrassing to justify in a methods section, do not implement it.
- We want clean and defensible systems, not "screugneugneu" adjustment.
- Do not overuse tests. Add tests when they protect an important invariant, a real bug fix, or a reusable mechanism, but avoid bloating the repo with low-signal tests during experimentation.
- If existing code is weak, awkward, or poorly motivated, it is acceptable and often preferable to remove it aggressively rather than preserve it out of habit.