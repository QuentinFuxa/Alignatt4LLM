# Existing Mission Context

## Mission

La mission Ralph comporte deux objectifs ordonnes.

Objectif 1 consiste a verrouiller un setup reproductible de cascade En->DE sur
`test-set/audio/ccpXHNfaoy.wav`:

- inference dans `.venv-inference` avec Simulstream et la cascade full vLLM
  `Qwen3-ASR-1.7B` + `gemma-4-E4B-it`
- artefacts ecrits dans `outputs/cascade_v1/`
- evaluation dans `.venv-evaluation` avec OmniSTEval pour produire dans
  `outputs/cascade_v1/` les scores `BLEU`, `CHRF`, `XCOMETXL`, `LongYAAL CU`,
  `LongYAAL CA`
- workflow notebook/kernel assez propre pour iterer vite sur le prompt Gemma
  sans recharger inutilement les modeles
- un commit local propre quand cet objectif est atteint

Objectif 2 ne s'ouvre qu'apres un Objectif 1 propre et un repo clean. Il
s'agit d'iterer sur l'implementation de la cascade pour ameliorer `BLEU`,
`CHRF`, `XCOMETXL` tout en gardant `LongYAAL CU < 2s` sur le meme audio, avec
un commit a chaque experimentation. Les seules familles de changements
autorisees sont le prompt, la reinjection de la phrase precedente, et des
coupes prudentes sur la fin de prediction; pas de custom model, pas de
fine-tuning, pas de nouvelle stack exotique.

Le succes est donc d'abord une baseline reproductible avec sorties et
evaluations persistantes, puis seulement ensuite des iterations bornees sur la
qualite/latence. Sont hors scope les refactors larges non necessaires, les
changements de famille de modeles, et toute iteration qui redecouvre des bugs
de stack deja contournes dans le repo.

## Repo Surface

- entrypoints:
  - `qwen3asr_gemma_cascade_core.py`: logique actuelle de la cascade full vLLM,
    chargement des modeles, boucle streaming, et emission des artefacts
    d'inference
  - `qwen3asr_gemma_cascade_notebook.py`: facade notebook pour
    `load_models()` puis `run_baseline()` ou `run_stream(...)`, sans execution
    automatique a l'import
  - `run_cascade_baseline.py`: entree repo-locale `.venv-inference` pour
    produire `outputs/cascade_v1/`
  - `evaluate_cascade_outputs.py`: entree repo-locale `.venv-evaluation`
    pour resegmenter et scorer `hypothesis.jsonl`
  - `setup_inference_qwen_asr_vllm.sh`: bootstrap de `.venv-inference`
- evaluation stack:
  - `pyproject.toml` et `uv.lock`: separation des deps inference/evaluation
  - `.venv-evaluation` avec `OmniSTEval`
- assets or checkpoints:
  - snapshots HF locaux references dans `qwen3asr_gemma_cascade_core.py`
  - `test-set/audio/ccpXHNfaoy.wav`
  - `test-set/ref/en.txt`, `test-set/ref/de.txt`,
    `test-set/audio-segments.yaml`
- logs or artifacts already available:
  - `cascade_artifacts.py`: schema versionne pour `manifest.json`,
    `hypothesis.jsonl`, `stream_updates.jsonl`, `scores.tsv`,
    `evaluation.json`
  - scripts utilitaires de kernel: `check_jupyter_kernels.sh`,
    `restart_jupyter_kernel.py`

## Constraints

- ne pas recharger ASR/Gemma sauf necessite prouvee; reutiliser le kernel
  `.venv-inference` persistant quand les modeles sont deja en memoire
- garder `Qwen3-ASR` et `Gemma 4` tous les deux sur `vLLM`
- respecter les reglages GPU actuellement stables tant qu'aucune preuve ne
  force un changement:
  - ASR `gpu_memory_utilization=0.2`
  - Gemma `gpu_memory_utilization=0.44`, `max_model_len=1024`,
    `enforce_eager=True`
- conserver les snapshots HF locaux, les monkey-patches runtime `qwen_asr`, et
  le chargement des modeles a l'interieur de `load_models()`
- l'Objectif 1 doit ecrire les sorties d'inference et d'evaluation sous
  `outputs/cascade_v1/`
- l'Objectif 1 doit faire l'inference dans `.venv-inference` et l'evaluation
  dans `.venv-evaluation`
- l'Objectif 2 ne commence qu'apres un Objectif 1 propre et un repo clean
- pour l'Objectif 2, seules sont autorisees les variantes de prompt, de
  contexte precedent, et de post-traitement conservateur sur la fin de sortie
- si un kernel propre ne peut pas etre conserve, il faut le dire explicitement
  et expliquer pourquoi

## Known Blockers

- blocker: `Unbabel/XCOMET-XL` n'est toujours pas present dans le cache HF
  local offline; `XCOMETXL` reste donc un blocage externe explicite meme si
  `BLEU`, `CHRF`, `LongYAAL CU`, et `LongYAAL CA` sont maintenant persistes
- blocker: la baseline runtime corrigee existe maintenant, mais `LongYAAL CU`
  reste a `2638.8114 ms`, donc l'Objectif 2 doit encore reduire la latence
  sous `2s` sans casser `BLEU=40.3126` et `CHRF=68.5453`
- blocker: aucun kernel Jupyter `.venv-inference` persistant n'etait vivant au
  moment de la relance; toute nouvelle rerun devra encore justifier un
  rechargement unique des modeles
- blocker: la reutilisation du kernel persistant est une contrainte forte; tout
  workflow qui redemarre silencieusement le kernel compte comme regression

## First Bounded Slice Candidates

- candidate: utiliser le bundle reel corrige `outputs/cascade_v1/` comme
  baseline verrouillee et ouvrir enfin l'Objectif 2 sur une seule variante
  prompt-only ou contexte-1-phrase
- candidate: mesurer si une coupe prudente sur la fin de prediction ou une
  reinjection minimale du contexte precedent peut faire passer `LongYAAL CU`
  sous `2s` sans casser materiallement `BLEU`/`CHRF`
- candidate: garder `XCOMETXL` hors focus tant que le modele n'est pas cache
  localement, mais conserver son blocage structure dans les artefacts
