# Context Injection Plan

Active mission for a dedicated 24h agent.

This file replaces the broader repo roadmap for now. The current objective is
to earn one strong, paper-defensible extra-context mechanism for the
IWSLT 2026 simultaneous Speech-to-Text with Extra Context sub-track, using this
repo's cascade runtime rather than a separate baseline script stack.

Historical broad planning belongs in `docs/archive/`. Session-level decisions
should still be appended to `DECISIONS.md`.

## External framing

Verified against the IWSLT 2026 simultaneous track page on 2026-04-16:

- There is a main Speech-to-Text track and a Speech-to-Text with Extra Context
  sub-track.
- For the extra-context sub-track, participants may preprocess the ACL paper
  PDFs before running the streaming system.
- Main English directions are `en->de`, `en->zh`, and `en->it`; `cs->en` is a
  separate direction without ACL-paper PDF context.
- Ranking is by quality under two non-computation-aware LongYAAL regimes:
  `0-2 s` and `2-4 s`.
- Docker submissions are expected to run on a single `H100 80GB`.

Relevant links:

- https://iwslt.org/2026/simultaneous
- https://github.com/owaski/iwslt-2026-baselines

## Mission

Build and validate a clean extra-context path that helps long-form ACL-talk
translation by giving the MT model access to compact, relevant information from
the corresponding ACL paper PDF.

The result must be something we could defend honestly in a paper:

- no hand-written lexical substitution tables
- no talk-specific prompt hacks
- no benchmark-artifact patching
- no giant raw PDF dump stuffed into the prompt
- no hidden oracle information

If the 24h investigation ends in a negative result, that is acceptable, but the
agent must leave behind a clean artifact, honest measurements, and a clear
recommendation about whether to continue.

## Default architectural bet

Use **Gemma on the MT side** as the main context-injection substrate.

Concretely:

- Keep `alignment_backend_name="qwen_forced"` as the main ASR path.
- Keep Gemma as the translation model.
- Inject paper context into the **Gemma MT prompt contract**, not into the
  Qwen ASR prompt by default.

Why this is the main bet:

- Qwen ASR is currently the strongest and most stable source frontend in this
  repo; we should avoid destabilizing it unless MT-side context clearly fails.
- The MT prompt contract is already structured and under our control in
  `cascade_translation_variants.py`.
- Extra paper context is semantically much closer to translation disambiguation,
  terminology consistency, and long-form discourse consistency than to raw ASR.
- AlignAtt remains well-defined if paper context is kept as a separate
  non-source prompt region outside the mapped current-source span.

## Existing assets to reuse, not worship

Two scripts already exist in `context_injection/`:

- `context_injection/extract_abstract.py`
- `context_injection/ner_llm.py`

These are useful bootstrap assets copied from the public baseline pattern, but
they should be treated as **starting points**, not final architecture.

What they are good for:

- fast PDF-to-text bootstrapping
- extracting title / authors / abstract
- producing a first terminology list for analysis or ablation

What they are *not* yet:

- a clean runtime integration
- a principled retrieval pipeline
- a justified final mechanism for this repo

## Primary hypothesis

The best extra-context mechanism in this repo is likely:

1. offline preprocessing of each ACL paper PDF into a compact structured
   artifact
2. runtime retrieval of a small number of relevant context snippets based on
   the current English ASR prefix and recent confirmed history
3. injection of those snippets into the Gemma MT prompt as an explicit
   `[Paper context]` block
4. strict token-budgeting so the extra context helps without destroying latency
   or crowding out the live source prefix

This is the main branch.

## Secondary hypotheses

These are worth measuring, but they are not the main contribution:

- Static `title + abstract` context may already help and should be the first
  baseline.
- Entity-only context may help terminology consistency, but on its own it is
  probably too weak and too heuristic-looking to be our main story.
- ASR-side terminology priming may help on names, but it should be treated as a
  fallback branch, not the default direction.

## Non-goals for this 24h run

- Do not revive Gemma ASR as the main path.
- Do not turn this into a generic RAG framework.
- Do not build a multi-document retrieval system; there is one paper per talk.
- Do not optimize for broad benchmark coverage before one single PDF-backed talk
  behaves convincingly.
- Do not sink the whole budget into PDF parsing churn or embedding-model churn.
- Do not make uncontrolled prompt changes across both ASR and MT at once.

## Success condition

By the end of the 24h run, the repo should contain:

- one clean extra-context mechanism integrated into the main runtime
- one or two compact offline artifacts derived from the paper PDF
- one single-audio validation story on a PDF-backed ACL talk
- at least one honest ablation against `no context`
- a short written note saying whether the mechanism is worth scaling out

The best-case outcome is a measurable quality/consistency win with acceptable
latency drift. A good second-best outcome is a negative result that clearly
explains why the mechanism is not worth pursuing.

## Hard constraints from this repo

- Use `.venv-inference`.
- Avoid unnecessary model reloads; hot-start reuse matters.
- Treat full SimulStream runs as expensive.
- Validate on one audio first.
- Keep backend runs sequential and isolated.
- Prefer `run_simulstream_compare.py` for canonical single-audio iteration when
  that is sufficient.
- SimulStream is the canonical inference path.
- OmniSTEval is the canonical evaluation path.

Also keep the current stable runtime assumptions intact unless there is a real
scientific reason to change them:

- `qwen_forced` for ASR
- Gemma MT as the translation backend
- lazy model loading through `LoadedModelBundle.load()`
- no broad "let's reload everything and see" experiments

## Design rules

The mechanism must obey all of these:

- Extra context must be explicitly represented in code, not smuggled in through
  ad hoc prompt-string concatenation spread across files.
- The live current-source span must remain clearly delimited for
  `PromptSourceMap`; paper context is non-source prompt content.
- The accepted-prefix contract must remain intact.
- The default behavior must stay `off` unless context is explicitly configured.
- The agent should prefer a small number of generic knobs over many fragile
  prompt variants.
- If retrieval is used, retrieval must depend only on information available at
  the current time step.

## Recommended implementation shape

Prefer a design close to this:

1. A small offline module, likely a new file, that turns a PDF into a structured
   artifact such as:
   - title
   - authors
   - abstract
   - section headers
   - chunked text passages
   - optional entity list

2. A runtime-side context selector that chooses a tiny context package such as:
   - static title + abstract
   - or top-k retrieved chunks
   - or title + top-k chunks

3. A structured prompt integration on the MT side, probably by extending
   `RenderedTranslationPrompt` and `TranslationVariant.render_messages()` so the
   paper context is rendered intentionally rather than injected as a random
   string.

4. Minimal targeted tests around:
   - config defaults
   - prompt rendering
   - source-span preservation
   - retrieval budget determinism

## Files the agent should inspect first

- `docs/CONTEXT_INJECTION.md` — current mechanism + reproducible commands
- `context_injection/paper_artifact.py`, `context_injection/context_selector.py`
- `context_injection/extract_abstract.py`, `context_injection/ner_llm.py`
  (legacy bootstrap scripts; the new `paper_artifact` module supersedes them)
- `cascade_translation_variants.py`
- `cascade_runtime.py`
- `cascade_mt_backend.py`
- `run_simulstream_compare.py`
- `README.md`
- `docs/RUNTIME_ARCHITECTURE.md`

## Preferred execution order

### Step 0 - Re-anchor the problem without GPU churn

Before editing code:

- Read the IWSLT task description and the public baseline for the extra-context
  sub-track.
- Inspect where this repo currently builds the MT prompt.
- Write down one explicit main mechanism and at most one fallback.

Acceptance gate:

- The agent can explain, in `DECISIONS.md`, exactly where context should enter
  the runtime and why Gemma MT is the primary substrate.

### Step 1 - Define the offline paper artifact

Create a compact, reusable representation for one talk's paper.

Minimum viable artifact:

- `paper_id` or source path
- title
- authors
- abstract
- chunk list with stable chunk ids

Preferred richer artifact:

- section headers when extraction is reliable
- a normalized text form for retrieval
- optional entity list as metadata, not as the only signal

Important:

- Reuse the current `context_injection` scripts if that is faster, but feel free
  to refactor or replace them if the result becomes cleaner.
- Avoid designing around only the abstract if reliable paragraph chunking is
  available.

Acceptance gate:

- One PDF can be deterministically converted into a compact JSON artifact with a
  shape the runtime can consume directly.

### Step 2 - Land a static-context baseline first

Do the smallest credible runtime integration before building retrieval.

Recommended first baseline:

- inject `title + abstract` on the MT side only
- keep the context block explicit, e.g. `[Paper context]`
- keep it outside the current-source span used by AlignAtt

Why this first:

- it is the cheapest proof that the runtime integration is sound
- it gives a proper baseline against retrieval
- it de-risks prompt and token-budget plumbing before smarter selection

Acceptance gate:

- The system runs with context on one PDF-backed example without breaking source
  mapping, accepted-prefix handling, or the streaming loop.

### Step 3 - Add principled retrieval over paper chunks

After the static baseline works, build the main mechanism.

Preferred retrieval order:

1. start simple with a transparent lexical scorer or very small retriever
2. only add a heavier embedding/reranking model if the simple version is clearly
   inadequate

The query should be built from:

- current English ASR prefix
- optionally a short window of earlier confirmed English source context

The query should *not* use:

- future text
- references
- manually curated term lists

The selected context package should be small and stable:

- top-k chunks only
- fixed token or character budget
- deterministic ordering

Acceptance gate:

- Retrieval produces a compact context block that is obviously relevant on at
  least one real example and does not explode prompt length.

### Step 4 - Measure three minimal conditions only

Do not start with a large matrix.

Measure on one PDF-backed ACL talk:

- `no context`
- `static title+abstract`
- `retrieved paper chunks`

If time allows, add exactly one extra diagnostic ablation:

- `entities only`

Track at minimum:

- translation output examples
- one quality metric bundle if references are available
- latency drift
- prompt length or context budget

Acceptance gate:

- There is a clear written comparison showing whether retrieved context is
  better than no context and better than naive static context.

### Step 5 - Only then decide whether ASR-side context is worth touching

Fallback branch only.

Touch ASR-side context injection **only if** the evidence strongly suggests that
the dominant remaining failure mode is name recognition rather than translation
choice.

If this branch is opened:

- keep it narrow
- reuse the offline paper artifact
- prefer term priming over broad prompt redesign
- keep Qwen ASR as the backend

But do not let this fallback eat the whole night.

Acceptance gate:

- Either a narrow ASR-side experiment lands, or the agent explicitly records why
  the branch was not worth opening.

### Step 6 - Leave the repo in a reusable state

Before stopping:

- document the chosen mechanism
- record what worked and what failed
- keep defaults conservative
- do not leave half-integrated prompt hacks behind

Ideal deliverables:

- a small doc such as `docs/CONTEXT_INJECTION.md`
- a clean config surface
- one reproducible command line for the winning experiment
- one reproducible command line for the baseline

## What to optimize for scientifically

The intended paper story is not "we pasted a glossary into the prompt."

The intended story is closer to:

*A simultaneous cascade can exploit document-level extra context by retrieving a
compact, time-local, paper-grounded support set from the associated ACL paper
and exposing it to the MT model through an explicit structured prompt contract,
improving long-form technical translation without violating streaming
constraints.*

That story becomes stronger if:

- the context selector is simple and measurable
- the runtime integration is explicit and typed
- the added prompt region is clearly separated from the live source span
- the win is visible on terminology or discourse consistency, not just on one
  cherry-picked string edit

## What to avoid

- no talk-specific exceptions
- no giant entity dumps
- no "if paper mentions X force translation Y" logic
- no hidden lexical replacement post-processing
- no proliferating prompt variants that differ only in wording
- no broad benchmark sweeps before the single-example mechanism is convincing

## Best first guess

If the agent needs a default path and should not spend an hour debating
alternatives, choose this:

1. keep `qwen_forced` ASR
2. keep Gemma MT
3. preprocess one paper into `title + abstract + chunked body`
4. integrate a `[Paper context]` block into the MT prompt
5. first run `title + abstract`
6. then replace static context with top-k retrieved chunks
7. compare against `no context`

This is the shortest path to an actual result.
