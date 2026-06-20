# Status

This branch is the public companion-code surface for the AlignAtt4LLM paper.

- Paper: https://arxiv.org/abs/2606.03967
- Default ASR route: `qwen_forced`
- Stable MT route: `gemma_vllm_alignatt`
- Active EN->ZH research MT route: `milmmt_vllm_alignatt`
- Maintained presets: `gemma_low_latency`, `gemma_high_latency`

The public branch intentionally omits paper source/PDF files, model weights,
dataset audio, historical Docker packaging, and local experiment logs.

For new claims, keep manifests, score files, and exact commands. Run one local
clip before broader sweeps, and run an A100 smoke before reporting new
inference results.
