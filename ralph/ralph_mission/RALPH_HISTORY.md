## bootstrap

### Decision

- initialize a blank Ralph mission scaffold with one active bootstrap
  hypothesis

### Impact runtime

- none yet; this is bootstrap only

### Portee de preuve

- no runtime proof yet
- no paper claim yet
- bootstrap only

### Trigger de sortie

- the scaffold is ready once the mission files validate and the first bounded
  hypothesis is explicit

### Hypothesis IDs touched

- `h_bootstrap_first_bounded_slice`

### Status transitions

- `h_bootstrap_first_bounded_slice: new -> active`

### Ce qui a ete fait

- added a blank `ralph_mission/` directory
- added a generic worker prompt, registry bootstrap, and first plan

### Validations lancees

- `bash scripts/ralph_registry.sh render-state ralph_mission/RALPH_HYPOTHESES.json > ralph_mission/RALPH_STATE.md`: expected

### Resultats utiles

- the mission scaffold is versioned and ready to edit

### Blocages restants

- the placeholders still need to be replaced with the real mission

### Hypothese de la prochaine iteration

- `h_bootstrap_first_bounded_slice`:
  replace the placeholders with the actual first bounded slice for the mission

## mission_objectives_bootstrap

### Decision

- remplacer le bootstrap vide par une vraie mission a deux objectifs
- activer uniquement l'Objectif 1 comme baseline reproductible
- geler l'Objectif 2 tant qu'il n'existe pas de baseline propre et de repo
  clean

### Impact runtime

- aucun changement runtime direct dans cette iteration
- le pilotage Ralph est maintenant aligne sur le vrai contrat de travail

### Portee de preuve

- preuve locale par inspection du repo et consigne humaine
- aucune nouvelle preuve runtime
- aucun claim papier

### Trigger de sortie

- cette iteration sort une fois que la mission Ralph cible explicitement
  `outputs/cascade_v1/`, les deux environnements, et la contrainte de kernel
  persistant

### Hypothesis IDs touched

- `h_bootstrap_first_bounded_slice`
- `h_obj1_reproducible_single_audio_eval_loop`
- `h_obj2_prompt_only_quality_latency_tuning`

### Status transitions

- `h_bootstrap_first_bounded_slice: active -> superseded`
- `h_obj1_reproducible_single_audio_eval_loop: new -> active`
- `h_obj2_prompt_only_quality_latency_tuning: new -> frozen_annex`

### Ce qui a ete fait

- remplace les placeholders de `EXISTING.md` par la mission reelle
- reecrit `PLAN.md` autour d'un seul focus actif: l'Objectif 1
- remanie `RALPH_HYPOTHESES.json` pour encoder la baseline active et la
  branche prompt-only gelee
- prepare la regeneration de `RALPH_STATE.md` a partir du JSON

### Validations lancees

- `bash ralph/scripts/ralph_registry.sh render-state ralph/ralph_mission/RALPH_HYPOTHESES.json > ralph/ralph_mission/RALPH_STATE.md`
- `bash ralph/scripts/ralph_registry.sh validate-json ralph/ralph_mission/RALPH_HYPOTHESES.json`
- `bash ralph/scripts/ralph_registry.sh validate-state ralph/ralph_mission/RALPH_HYPOTHESES.json ralph/ralph_mission/RALPH_STATE.md`
- `bash ralph/scripts/ralph_registry.sh validate-plan ralph/ralph_mission/RALPH_HYPOTHESES.json ralph/ralph_mission/PLAN.md`
- `bash ralph/scripts/ralph_registry.sh validate-history-tail ralph/ralph_mission/RALPH_HISTORY.md`

### Resultats utiles

- Ralph sait maintenant que la prochaine iteration utile est de fermer
  l'Objectif 1, pas d'improviser des experiences prompt-only
- la contrainte de kernel persistant et de worktree propre est explicite

### Blocages restants

- aucun contrat d'artefacts `outputs/cascade_v1/` n'est encore implemente
- aucun driver OmniSTEval repo-local n'est encore branche
- le repo est dirty, donc l'Objectif 2 reste volontairement hors focus

### Hypothese de la prochaine iteration

- `h_obj1_reproducible_single_audio_eval_loop`:
  verrouiller la baseline single-audio avec sorties et evaluation persistantes

## objective1_artifact_contract

### Decision

- figer un vrai contrat d'artefacts `outputs/cascade_v1/` avant toute relance
  GPU
- separer proprement l'entree inference `.venv-inference` et l'entree
  evaluation `.venv-evaluation`
- garder l'Objectif 1 actif tant qu'aucun bundle reel `ccpXHNfaoy.wav` n'a ete
  produit

### Impact runtime

- le runtime sait maintenant ecrire `manifest.json`, `hypothesis.jsonl`,
  `stream_updates.jsonl`, `transcript.en.txt`, `translation.de.txt`
- le notebook n'execute plus la cascade a l'import
- rien n'a encore ete charge en GPU dans cette iteration

### Portee de preuve

- preuve locale uniquement
- offline seulement pour les validations lancees
- smoke test single-audio synthetique, pas une vraie inference
- aucun claim papier

### Trigger de sortie

- cette iteration sort une fois que le repo contient un chemin unique pour
  produire puis evaluer `outputs/cascade_v1/` sans re-decouvrir la stack

### Hypothesis IDs touched

- `h_obj1_reproducible_single_audio_eval_loop`

### Status transitions

- none

### Ce qui a ete fait

- ajoute `cascade_artifacts.py` comme schema partage inference/evaluation
- etend `qwen3asr_gemma_cascade_core.py` pour persister la baseline et suivre
  des timestamps mot-a-mot
- retire l'execution automatique de
  `qwen3asr_gemma_cascade_notebook.py`
- ajoute `run_cascade_baseline.py` et `evaluate_cascade_outputs.py`
- filtre l'evaluation OmniSTEval sur les `source` reels du `hypothesis.jsonl`
  pour le cas single-audio

### Validations lancees

- `python -m py_compile cascade_artifacts.py run_cascade_baseline.py evaluate_cascade_outputs.py qwen3asr_gemma_cascade_core.py qwen3asr_gemma_cascade_notebook.py`
- smoke test local `write_inference_artifacts(...)`: PASS
- `.venv-evaluation/bin/python evaluate_cascade_outputs.py --skip-comet` sur un
  bundle synthetique `ccpXHNfaoy.wav`: PASS

### Resultats utiles

- l'evaluation repo-locale ecrit bien `BLEU`, `CHRF`, `XCOMETXL`,
  `LongYAAL CU`, `LongYAAL CA` dans `scores.tsv`
- le faux blocage "1 hypothese contre tout le corpus" a ete elimine en
  filtrant `audio-segments.yaml` sur les `source` reels
- un smoke test synthetique a donne `BLEU=100`, `CHRF=100`,
  `XCOMETXL=NA`, donc le contrat de sortie est verifie sans preuve runtime

### Blocages restants

- aucun kernel `.venv-inference` vivant n'etait disponible au moment de
  l'iteration
- aucun bundle reel `outputs/cascade_v1/` n'a encore ete produit
- `Unbabel/XCOMET-XL` n'est pas present dans le cache HF local, donc le score
  `XCOMETXL` reste a confirmer en conditions reelles

### Hypothese de la prochaine iteration

- `h_obj1_reproducible_single_audio_eval_loop`:
  produire le premier bundle reel `ccpXHNfaoy.wav`, puis lancer
  l'evaluation reelle ou enregistrer un blocage `XCOMETXL` explicite
