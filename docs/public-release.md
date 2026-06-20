# Public Release Trail

This branch is the public-facing code surface for AlignAtt4LLM.

## Scope

- Keep active runtime, presets, evaluation scripts, reporting utilities, and
  reusable AlignAtt head metadata.
- Keep compact result anchors in `docs/`.
- Link to the paper through arXiv: https://arxiv.org/abs/2606.03967.
- Do not vendor paper source/PDF, Docker submission packaging, local logs,
  model weights, or dataset audio.

## Decisions

- vLLM remains part of the supported inference runtime because it is the path
  used by the Gemma and MiLMMT AlignAtt backends.
- Docker is not an official public entrypoint on this branch. If a container is
  added later, it should be a tested optional recipe, not the canonical surface.
- Audio files are local inputs. The repo tracks metadata and references but
  ignores `.wav` payloads.
- Report-generation scripts write to `outputs/reports` by default. Historical
  `--paper-generated-dir` flags remain as deprecated aliases where already
  exposed.

## Release Checks

- Publish from a sanitized branch whose reachable history does not include
  removed private packaging or paper-source material.
- Run the maintained lightweight test suite before pushing.
- Run an A100 inference smoke before using the public branch for new result
  claims.
