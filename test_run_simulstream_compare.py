from __future__ import annotations

from types import SimpleNamespace
import sys

import run_simulstream_compare as compare


def _parse_compare_args(monkeypatch, extra_args: list[str]):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_simulstream_compare.py", *extra_args],
    )
    return compare.parse_args()


def test_compare_config_can_select_milmmt_clean_source_bearing(monkeypatch):
    args = _parse_compare_args(
        monkeypatch,
        [
            "--target",
            "zh",
            "--mt-backend-name",
            "milmmt_vllm_alignatt",
            "--translation-alignatt-acceptance-variant",
            "unit_mass_source_bearing",
            "--translation-alignatt-online-normalization",
            "raw",
            "--translation-alignatt-min-source-mass",
            "0.0",
            "--translation-alignatt-frontier-min-inaccessible-mass",
            "0.005",
            "--translation-alignatt-source-frontier-action",
            "trim_unrecovered",
            "--translation-alignatt-max-inaccessible-source-mass",
            "1.0",
            "--translation-alignatt-min-accessible-inaccessible-margin",
            "-1.0",
            "--translation-alignatt-source-bearing-min-source-mass",
            "0.005",
            "--translation-alignatt-source-bearing-hard-inaccessible-cap",
            "0.60",
            "--asr-gpu-memory-utilization",
            "0.40",
            "--mt-vllm-gpu-memory-utilization",
            "0.60",
            "--mt-vllm-enable-prefix-caching",
            "--milmmt-temperature",
            "0.0",
            "--milmmt-top-p",
            "1.0",
            "--milmmt-top-k",
            "1",
            "--milmmt-repetition-penalty",
            "1.0",
            "--paper-context-mode",
            "off",
        ],
    )

    config = compare.build_processor_config(args, backend_name="qwen_forced")

    assert config.mt_backend_name == "milmmt_vllm_alignatt"
    assert config.target_lang_code == "zh"
    assert config.translation_alignatt_acceptance_variant == "unit_mass_source_bearing"
    assert config.translation_alignatt_online_normalization == "raw"
    assert config.translation_alignatt_min_source_mass == 0.0
    assert config.translation_alignatt_frontier_min_inaccessible_mass == 0.005
    assert config.translation_alignatt_source_frontier_action == "trim_unrecovered"
    assert config.translation_alignatt_max_inaccessible_source_mass == 1.0
    assert config.translation_alignatt_min_accessible_inaccessible_margin == -1.0
    assert config.translation_alignatt_source_bearing_min_source_mass == 0.005
    assert config.translation_alignatt_source_bearing_hard_inaccessible_cap == 0.60
    assert config.translation_alignatt_max_non_source_prompt_mass == 1.0
    assert config.translation_alignatt_max_source_regression == -1
    assert config.translation_alignatt_token_argmax_frontier_gate is False
    assert config.translation_acceptance_policy == "alignatt"
    assert config.asr_gpu_memory_utilization == 0.40
    assert config.mt_vllm_gpu_memory_utilization == 0.60
    assert config.mt_vllm_enable_prefix_caching is True
    assert config.milmmt_prompt_mode == "direct"
    assert config.milmmt_temperature == 0.0
    assert config.milmmt_top_p == 1.0
    assert config.milmmt_top_k == 1
    assert config.milmmt_repetition_penalty == 1.0
    assert config.paper_context_mode == "off"


def test_compare_can_limit_smoke_to_one_alignment_backend(monkeypatch):
    args = _parse_compare_args(
        monkeypatch,
        [
            "--alignment-backend-name",
            "qwen_forced",
        ],
    )

    assert compare.selected_backend_ids(args) == ("qwen_forced",)


def test_compare_runs_both_alignment_backends_by_default(monkeypatch):
    args = _parse_compare_args(monkeypatch, [])

    assert compare.selected_backend_ids(args) == compare.BACKEND_IDS
    assert args.translation_alignatt_min_source_mass == 0.0
    assert args.translation_alignatt_max_inaccessible_source_mass == 1.0
    assert args.translation_alignatt_source_bearing_min_source_mass == 0.005
    assert args.translation_alignatt_source_bearing_hard_inaccessible_cap == 1.0


def test_compare_default_config_disables_inaccessible_source_cap(monkeypatch):
    args = _parse_compare_args(monkeypatch, ["--alignment-backend-name", "qwen_forced"])

    config = compare.build_processor_config(args, backend_name="qwen_forced")

    assert config.translation_alignatt_min_source_mass == 0.0
    assert config.translation_alignatt_max_inaccessible_source_mass == 1.0
    assert config.translation_alignatt_source_bearing_min_source_mass == 0.005
    assert config.translation_alignatt_source_bearing_hard_inaccessible_cap == 1.0


def test_compare_subprocess_forwards_clean_alignatt_flags(monkeypatch, tmp_path):
    args = _parse_compare_args(
        monkeypatch,
        [
            "--target",
            "zh",
            "--mt-backend-name",
            "milmmt_vllm_alignatt",
            "--translation-alignatt-acceptance-variant",
            "unit_mass_source_bearing",
            "--translation-alignatt-max-inaccessible-source-mass",
            "1.0",
            "--translation-alignatt-source-frontier-action",
            "trim_unrecovered",
            "--translation-alignatt-max-non-source-prompt-mass",
            "0.80",
            "--translation-alignatt-source-bearing-min-source-mass",
            "0.005",
            "--translation-alignatt-source-bearing-hard-inaccessible-cap",
            "0.60",
            "--translation-alignatt-min-accessible-source-units",
            "6",
            "--translation-alignatt-min-accessible-source-units-mode",
            "target_unit_cap",
            "--translation-alignatt-token-argmax-frontier-gate",
            "--asr-gpu-memory-utilization",
            "0.40",
            "--mt-vllm-gpu-memory-utilization",
            "0.60",
            "--mt-vllm-enable-prefix-caching",
            "--paper-context-mode",
            "off",
        ],
    )
    captured: dict[str, list[str]] = {}

    def fake_run_subprocess(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["cwd"] = [str(kwargs.get("cwd"))]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(compare, "run_subprocess", fake_run_subprocess)

    compare.run_backend_subprocess(
        python_executable=sys.executable,
        backend_name="qwen_forced",
        args=args,
        output_dir=tmp_path / "qwen_forced",
    )

    cmd = captured["cmd"]
    assert "--mt-backend-name" in cmd
    assert cmd[cmd.index("--mt-backend-name") + 1] == "milmmt_vllm_alignatt"
    assert "--translation-alignatt-acceptance-variant" in cmd
    assert (
        cmd[cmd.index("--translation-alignatt-acceptance-variant") + 1]
        == "unit_mass_source_bearing"
    )
    assert "--translation-alignatt-max-inaccessible-source-mass" in cmd
    assert cmd[cmd.index("--translation-alignatt-max-inaccessible-source-mass") + 1] == "1.0"
    assert "--translation-alignatt-source-frontier-action" in cmd
    assert (
        cmd[cmd.index("--translation-alignatt-source-frontier-action") + 1]
        == "trim_unrecovered"
    )
    assert "--translation-alignatt-max-non-source-prompt-mass" in cmd
    assert (
        cmd[cmd.index("--translation-alignatt-max-non-source-prompt-mass") + 1]
        == "0.8"
    )
    assert "--translation-alignatt-source-bearing-min-source-mass" in cmd
    assert (
        cmd[cmd.index("--translation-alignatt-source-bearing-min-source-mass") + 1]
        == "0.005"
    )
    assert "--translation-alignatt-min-accessible-source-units" in cmd
    assert cmd[cmd.index("--translation-alignatt-min-accessible-source-units") + 1] == "6"
    assert "--translation-alignatt-min-accessible-source-units-mode" in cmd
    assert (
        cmd[cmd.index("--translation-alignatt-min-accessible-source-units-mode") + 1]
        == "target_unit_cap"
    )
    assert "--translation-alignatt-token-argmax-frontier-gate" in cmd
    assert "--asr-gpu-memory-utilization" in cmd
    assert cmd[cmd.index("--asr-gpu-memory-utilization") + 1] == "0.4"
    assert "--mt-vllm-gpu-memory-utilization" in cmd
    assert cmd[cmd.index("--mt-vllm-gpu-memory-utilization") + 1] == "0.6"
    assert "--mt-vllm-enable-prefix-caching" in cmd
    assert "--paper-context-mode" in cmd
    assert cmd[cmd.index("--paper-context-mode") + 1] == "off"
