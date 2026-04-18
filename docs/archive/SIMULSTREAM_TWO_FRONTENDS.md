# SimulStream Two-Frontend Runtime

Current supported cascade frontends:

- `qwen_forced`: `Qwen3-ASR-1.7B` + `Qwen3-ForcedAligner-0.6B`
- `gemma_onepass_qk_fast`: Gemma 4 ASR + audio AlignAtt `qk_fast` in one pass

The canonical inference path is `CascadeAlignAttProcessor` in SimulStream.
The canonical single-audio comparison entrypoint is `run_simulstream_compare.py`.

Operational defaults:

- validate ideas on exactly one clip first: `tmp/alignatt_smoke18.wav`
- compare the two frontends sequentially and in isolated processes
- keep Qwen loading local to the Qwen backend; importing the runtime must not
  patch or resolve Qwen unless `qwen_forced` is actually selected
- keep `eager` only for explicit calibration / debug workflows in
  `run_alignment_single_audio.py`

Runtime architecture:

- `cascade/runtime.py` owns the neutral runtime surface
- `CascadeRuntimeConfig` holds immutable-ish experiment config
- `LoadedModelBundle` loads the selected ASR frontend and the Gemma MT backend
  lazily
- `CascadeSession` owns mutable per-stream state and MT prompt cache
- `qwen3asr_gemma_cascade_core.py` is a temporary compatibility shim only

Validation loop for this iteration:

1. `py_compile`
2. pure-Python tests
3. one real SimulStream comparison on `tmp/alignatt_smoke18.wav`

Historical design and audit notes live under `docs/archive/`.
