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

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
