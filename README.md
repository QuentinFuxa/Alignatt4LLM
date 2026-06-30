<h1 align="center">
  <img src="src/assets/alignatt_logo.svg" alt="AlignAtt4LLM icon" width="64" />
  AlignAtt4LLM
</h1>

> [AlignAtt4LLM: Fast AlignAtt for Decoder-Only LLMs at IWSLT 2026
> Simultaneous Speech Translation Task](https://arxiv.org/abs/2606.03967)

**AlignAtt4LLM** adapts [AlignAtt](https://arxiv.org/abs/2305.11408) to decoder-only LLMs for simultaneous speech
translation. The MT model drafts a translation from the current source prefix,
the runtime reconstructs target-to-source attention from selected decoder
attention heads, and only the target prefix that is supported by accessible
source evidence is emitted.


![Chunk-synchronous AlignAtt4LLM cascade](src/assets/cascade.png)

## Scope & what it brings ?

The IWSLT implementation is end-to-end: it includes ASR, chunk-synchronous runtime code (synchronicity comes from the requirement to use [SimulStream](https://arxiv.org/abs/2512.17648)), and MT. This makes the full
ASR + MT cascade runnable from audio input to simultaneous translation output. But the core of the innovation here is what happens in the MT part:

**1.** The idea of reconstructing the attention, to know *where to cut* :

<img src="src/assets/where_to_cut.png" width="500"/>


**2.** The way of recomputing attention from a fused kernel to keep inference *fast*

<img src="src/assets/run_fast.png" width="500"/>


**3.** The way of capturing keys and queries at runtime in [VLLM](https://github.com/vllm-project/vllm) to keep inference *really fast*

<img src="src/assets/run_really_fast.png" width="500"/>


**Thus, this package contains:**

- a reproducible end-to-end cascade
- A focus on the implementation of AlignAtt to decoders only LLMs.

## See where Gemma listens

The runtime already reconstructs, for every drafted token, **where in the source
it attends**: an audio frame for ASR, a source token for MT. `--trace-attention`
prints that live on stderr as each token is committed or held. It is a pure read
of the signal the policy already uses, so it does not change what the model emits.

Standalone Gemma AlignAtt ASR. Watch where each transcript token lands on the
audio timeline (`src@frame (seconds)`):

```bash
alignatt-gemma-asr \
  --wavs audio.wav \
  --output-dir outputs/gemma_asr_trace \
  --trace-attention
```

```
[chunk   1] commit "Hi"         → src@2 (0.12s)
[chunk   2] commit " Si"        → src@52 (2.12s)
[chunk   2] commit " Yuan"      → src@52 (2.12s)
[chunk   3] commit " F"         → src@75 (3.04s)
[chunk   3] commit "udan"       → src@89 (3.60s)
[chunk   3] commit " Universit" → src@92 (3.72s)
```

The full cascade, end to end. The MT trace adds the accessible / inaccessible
attention-mass split that drives the *where to cut* decision:

```bash
alignatt-batch \
  --inputs audio.wav --target zh \
  --mt-backend-name gemma_vllm_alignatt \
  --trace-attention \
  --output-dir outputs/gemma_zh_smoke
```

```
[chunk   1] commit "大家好"   → src@0   mass acc 0.34 inacc 0.01
[chunk   2] commit "来自"     → src@9   mass acc 0.47 inacc 0.10
[chunk   2] commit "复"       → src@9   mass acc 0.63 inacc 0.06
[chunk   9] HOLD   "经常"     → src@26  mass acc 0.03 inacc 0.68 > frontier → cut
```

The last line is the policy at work: that draft token's attention is 0.68 on
source that has not arrived yet, so it is held rather than emitted.

## Bring your own LLM

The portable part of AlignAtt4LLM is the MT-side policy, not the model. A new
decoder-only LLM plugs into the same runtime by supplying a `VLLMAttentionSpec`
(which vLLM attention class to patch and how its `forward` recomputes Q/K) plus a
thin backend subclass, and reuses the shared capture/reconstruction/acceptance
machinery in [`src/alignatt4llm/vllm_qk/`](src/alignatt4llm/vllm_qk/).

The shipped worked example is [Qwen3](src/alignatt4llm/mt/qwen_vllm_backend.py)
(`qwen_vllm_alignatt`):

```bash
alignatt-batch \
  --inputs audio.wav --target de \
  --mt-backend-name qwen_vllm_alignatt \
  --output-dir outputs/qwen_de_smoke
```

The full recipe (find your attention class → write a spec → subclass the backend
and worker → register → calibrate heads) is in
[Adding a New LLM](docs/adding_a_model.md).

## Public CLI

- `alignatt-batch`: run the streaming cascade over one or more media files.
- `alignatt-compare`: single-WAV A/B of two backends with WER/CER/latency.
- `alignatt-eval`: score emitted hypotheses with OmniSTEval-compatible files.
- `alignatt-preset`: run named operating points (`gemma_low_latency`, `gemma_high_latency`) in batch or server mode.
- `alignatt-gemma-asr`: standalone Gemma AlignAtt ASR probe.
- `alignatt-mt-parity`: MT backend parity/diagnostic harness.

## Documentation

- [Architecture](docs/architecture.md)
- [Generalizing AlignAtt4LLM to other LLMs](docs/generalizing.md)
- [Adding a New LLM (bring your own model)](docs/adding_a_model.md)
- [Data](docs/data.md)
- [Reproducibility](docs/reproducibility.md)
- [Results](docs/results.md)
- [Development](docs/development.md)

## Citation

```bibtex
@article{fuxa2026alignatt4llm,
  title = {AlignAtt4LLM: Fast AlignAtt for Decoder-Only LLMs at IWSLT 2026 Simultaneous Speech Translation Task},
  author = {Fuxa, Quentin and Macháček, Dominik},
  year = {2026},
  doi = {10.48550/arXiv.2606.03967},
  url = {https://arxiv.org/abs/2606.03967}
}
```
