"""Lightweight console-script entrypoints.

The inference and evaluation runners intentionally import heavy optional
dependencies. Public `--help` should still work from a dev-only install, so the
console scripts serve compact static help before importing the full runner.
"""

from __future__ import annotations

from importlib import import_module
import sys
from textwrap import dedent


_HELP: dict[str, str] = {
    "alignatt-batch": """
        usage: alignatt-batch --inputs WAV [WAV ...] --output-dir DIR [options]
               alignatt-batch --input-dir DIR --output-dir DIR [options]

        Run the streaming AlignAtt cascade over one or more media files.

        common options:
          --inputs PATH [PATH ...]          input audio/video files
          --input-dir DIR                  directory of input media
          --output-dir DIR                 artifact output directory
          --source {en}                    source language code
          --target {de,it,zh}              target language code
          --chunk-ms MS                    streaming chunk size
          --preset NAME                    runtime preset, e.g. gemma_low_latency
          --alignment-backend-name NAME    qwen_forced or gemma_vllm_qk_fast
          --mt-backend-name NAME           gemma_vllm_alignatt or milmmt_vllm_alignatt
    """,
    "alignatt-compare": """
        usage: alignatt-compare [--wav WAV] [--output-dir DIR] [options]

        Run the canonical single-audio validation loop on one local WAV.

        common options:
          --wav PATH                       input WAV
          --reference PATH                 optional ASR reference text
          --output-dir DIR                 comparison output directory
          --source {en}                    source language code
          --target {de,it,zh}              target language code
          --chunk-ms MS                    streaming chunk size
          --alignment-backend-name NAME    run one backend instead of all stable backends
          --mt-backend-name NAME           Gemma or MiLMMT MT backend
    """,
    "alignatt-eval": """
        usage: alignatt-eval --output-dir DIR [options]

        Score AlignAtt inference artifacts with OmniSTEval.

        common options:
          --output-dir DIR                 directory containing hypothesis.jsonl
          --speech-segmentation PATH       long-form segmentation file
          --target-reference PATH          target-language reference text
          --target-lang-code CODE          target code, e.g. de, it, zh
          --source-reference PATH          source reference text
          --comet-model MODEL              COMET or XCOMET model id
          --skip-comet                     compute local metrics without COMET
    """,
    "alignatt-preset": """
        usage: alignatt-preset PRESET --output-dir DIR [options]

        Run a named runtime preset through the batch runner.

        common presets:
          gemma_low_latency
          gemma_high_latency
    """,
    "alignatt-gemma-asr": """
        usage: alignatt-gemma-asr --wav WAV [options]

        Run the Gemma AlignAtt ASR research harness on one local audio file.
    """,
    "alignatt-mt-parity": """
        usage: alignatt-mt-parity [options]

        Compare MT backend behavior for parity diagnostics.
    """,
}


def _wants_help(argv: list[str]) -> bool:
    return any(arg in {"-h", "--help"} for arg in argv[1:])


def _print_help(command: str) -> int:
    print(dedent(_HELP[command]).strip())
    return 0


def _dispatch(command: str, module_name: str) -> int | None:
    if _wants_help(sys.argv):
        return _print_help(command)
    module = import_module(module_name)
    return module.main()


def batch_main() -> int | None:
    return _dispatch("alignatt-batch", "alignatt4llm.cli.batch")


def compare_main() -> int | None:
    return _dispatch("alignatt-compare", "alignatt4llm.cli.compare")


def evaluate_main() -> int | None:
    return _dispatch("alignatt-eval", "alignatt4llm.cli.evaluate")


def preset_main() -> int | None:
    return _dispatch("alignatt-preset", "alignatt4llm.cli.preset")


def gemma_asr_main() -> int | None:
    return _dispatch("alignatt-gemma-asr", "alignatt4llm.cli.gemma_asr")


def mt_parity_main() -> int | None:
    return _dispatch("alignatt-mt-parity", "alignatt4llm.cli.mt_parity")
