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

## objective1_real_bundle_blockers

### Decision

- produire enfin le premier bundle runtime reel `outputs/cascade_v1/` pour
  `ccpXHNfaoy.wav`
- durcir l'entree d'evaluation pour ecrire un blocage `XCOMETXL` explicite au
  lieu d'echouer a vide
- garder l'Objectif 1 actif car la traduction finale persistee est encore un
  prefixe court malgre un transcript ASR complet

### Impact runtime

- `outputs/cascade_v1/` contient maintenant un vrai bundle inference +
  evaluation pour `ccpXHNfaoy.wav`
- `.venv-evaluation` declare maintenant `setuptools<81` pour que COMET reste
  importable avec `pkg_resources` sous Python 3.13
- l'evaluation ecrit des `metric_blockers` structures quand `XCOMETXL` n'est
  pas disponible localement

### Portee de preuve

- preuve locale runtime
- un seul talk `objective1`
- un seul delai `single_audio_baseline`
- evaluation offline seulement pour `XCOMETXL`
- aucun claim papier

### Trigger de sortie

- cette iteration sort une fois que le premier bundle reel et son evaluation
  offline sont persistes et que le blocage `XCOMETXL` est explicite dans les
  artefacts

### Hypothesis IDs touched

- `h_obj1_reproducible_single_audio_eval_loop`

### Status transitions

- none

### Ce qui a ete fait

- ajoute un chemin d'evaluation qui separe les metriques locales de `XCOMETXL`
  et persiste les blocages structures
- epingle `setuptools<81` dans le groupe `evaluation` puis resynchronise
  `.venv-evaluation`
- justifie un chargement unique `.venv-inference` car aucun kernel ni
  `VLLM::EngineCore` n'etait vivant
- lance la premiere baseline reelle puis l'evaluation offline sur
  `outputs/cascade_v1/`

### Validations lancees

- `python -m py_compile cascade_artifacts.py evaluate_cascade_outputs.py`: PASS
- smoke test synthetique offline du blocage `XCOMETXL`: PASS
- `UV_PROJECT_ENVIRONMENT=/home/cascade_simultaneous/.venv-evaluation uv sync --group evaluation`: PASS
- `.venv-inference/bin/python run_cascade_baseline.py`: PASS
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-evaluation/bin/python evaluate_cascade_outputs.py --output-dir outputs/cascade_v1`: PASS

### Resultats utiles

- premier bundle runtime reel present sous `outputs/cascade_v1/`
- scores reels: `BLEU=0.0013`, `CHRF=6.5501`, `LongYAAL CU=144127.5131`,
  `LongYAAL CA=-988.2935`, `XCOMETXL=NA`
- `metric_blockers` enregistre maintenant
  `comet_model_not_cached_locally` pour `Unbabel/XCOMET-XL`
- `translation.de.txt` ne contient que 62 mots alors que `transcript.en.txt`
  est complet, donc le bug de completude baseline est maintenant prouve

### Blocages restants

- `Unbabel/XCOMET-XL` n'est toujours pas dans le cache HF local offline
- la traduction finale persistee est un prefixe court, ce qui rend la baseline
  qualitativement inutilisable telle quelle
- la run process a vide la GPU a la fin; toute rerun devra donc recharger une
  fois les modeles et doit rester strictement justifiee

### Hypothese de la prochaine iteration

- `h_obj1_reproducible_single_audio_eval_loop`:
  expliquer puis corriger la traduction finale prefix-only, rerun une seule
  fois la baseline reelle, puis reevaluer offline avec le meme contrat

## objective1_incremental_translation_unlock_objective2

### Decision

- remplacer la retraduction Gemma du transcript complet par une traduction
  incrementale par utterance ponctuee
- rerun une seule baseline reelle car aucun kernel `.venv-inference` ni
  `VLLM::EngineCore` reutilisable n'etait vivant
- geler l'Objectif 1 en `blocked_external` sur `XCOMETXL` et ouvrir l'Objectif
  2 car la baseline corrigee existe enfin

### Impact runtime

- `outputs/cascade_v1/translation.de.txt` n'est plus un prefixe court: la
  traduction finale persistee fait maintenant `709` mots
- `outputs/cascade_v1/manifest.json` declare maintenant un mode
  `translation_mode=utterance_incremental` avec budget tokens par segment
- `outputs/cascade_v1/evaluation.json` contient des scores reels rafraichis:
  `BLEU=40.3126`, `CHRF=68.5453`, `LongYAAL CU=2638.8114`,
  `LongYAAL CA=-12600.8309`, `XCOMETXL=NA`

### Portee de preuve

- preuve locale runtime
- un seul talk `objective1`
- un seul delai `single_audio_baseline`
- evaluation offline seulement pour `XCOMETXL`
- aucun claim papier

### Trigger de sortie

- cette iteration sort une fois que le bundle reel corrige remplace le prefixe
  62 mots, que l'evaluation offline est rerafraichie, et que le prochain focus
  Ralph peut passer a une variante prompt-only bornee

### Hypothesis IDs touched

- `h_obj1_reproducible_single_audio_eval_loop`
- `h_obj2_prompt_only_quality_latency_tuning`

### Status transitions

- `h_obj1_reproducible_single_audio_eval_loop: active -> blocked_external`
- `h_obj2_prompt_only_quality_latency_tuning: frozen_annex -> active`

### Ce qui a ete fait

- ajoute un chemin de traduction Gemma incremental par utterance stabilisee
  avec budget `max_new_tokens` calcule par segment au lieu de retranscrire tout
  le transcript courant
- conserve `Qwen3-ASR` + `Gemma` sur `vLLM` et les reglages GPU stables
  (`0.2`, `0.44`, `1024`, `enforce_eager=True`)
- justifie un seul rechargement modele car la GPU etait propre et aucun kernel
  `.venv-inference` persistant n'etait vivant
- rerun la baseline reelle puis relance l'evaluation offline sur le meme
  contrat `outputs/cascade_v1/`

### Validations lancees

- `python -m py_compile qwen3asr_gemma_cascade_core.py run_cascade_baseline.py cascade_artifacts.py evaluate_cascade_outputs.py`: PASS
- `.venv-inference/bin/python run_cascade_baseline.py`: PASS
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-evaluation/bin/python evaluate_cascade_outputs.py --output-dir outputs/cascade_v1`: PASS

### Resultats utiles

- la traduction finale est maintenant materiellement complete (`709` mots) et
  les compteurs `delays`/`elapsed` couvrent toute la prediction persistee
- la baseline corrigee remonte fortement `BLEU` et `CHRF` par rapport au bundle
  prefix-only precedent
- `LongYAAL CU` reste au-dessus de la cible Objectif 2 (`2638.8114 ms > 2s`)
- `XCOMETXL` reste explicitement bloque offline par
  `comet_model_not_cached_locally`

### Blocages restants

- `Unbabel/XCOMET-XL` manque toujours du cache HF local offline
- la latence reste trop haute pour l'Objectif 2 (`LongYAAL CU=2638.8114 ms`)
- la baseline script se termine en dechargeant la GPU; toute rerun suivante
  doit encore rester unique et justifiee

### Hypothese de la prochaine iteration

- `h_obj2_prompt_only_quality_latency_tuning`:
  tester une seule variante prompt-only ou contexte minimal pour faire passer
  `LongYAAL CU` sous `2s` sans casser materiallement la baseline corrigee

## objective2_emission_freeze14_annex

### Decision

- separer proprement la timeline de calcul brut et la timeline d'emission
- tester offline une seule branche de coupe conservative:
  `freeze_major_tail_rewrites` avec fenetre `14`
- garder cette branche en annexe et restaurer `outputs/cascade_v1/` comme
  baseline canonique

### Impact runtime

- aucun rechargement `Qwen3-ASR`/`Gemma`: la preuve vient d'un replay offline
  du stream brut deja capture
- `outputs/cascade_v1/` reste la baseline raw-passthrough
  (`BLEU=40.3126`, `CHRF=68.5453`, `LongYAAL CU=2638.8114`)
- `outputs/cascade_v1_emit_freeze14/` capture l'annexe `freeze14`
  (`BLEU=40.3126`, `CHRF=68.5453`, `LongYAAL CU=1940.9677`)
- la politique gelante n'est pas active par defaut dans le runtime live

### Portee de preuve

- preuve locale runtime
- un seul talk `objective2`
- un seul delai `prompt_only_latency_quality`
- replay offline seulement sur le bundle reel verrouille
- aucun claim papier

### Trigger de sortie

- cette iteration sort une fois que la baseline canonique est restauree,
  qu'une annexe `freeze14` distincte existe, et que la branche reste
  explicitement hors focus pour la suite

### Hypothesis IDs touched

- `h_obj2_prompt_only_quality_latency_tuning`
- `h_obj2_freeze14_emission_annex`

### Status transitions

- `h_obj2_freeze14_emission_annex: new -> frozen_annex`

### Ce qui a ete fait

- ajoute `cascade_emission.py` pour separer traduction brute, politique
  d'emission, `delays`, et `elapsed`
- ajoute `reemit_cascade_outputs.py` pour rejouer offline un bundle capture
  sans relancer `vLLM`
- etend `stream_updates.jsonl` avec `raw_translation_text` et
  `emission_policy_action`
- produit une annexe `outputs/cascade_v1_emit_freeze14/`, puis restaure
  `outputs/cascade_v1/` en raw-passthrough

### Validations lancees

- `python -m py_compile cascade_artifacts.py cascade_emission.py reemit_cascade_outputs.py qwen3asr_gemma_cascade_core.py qwen3asr_gemma_cascade_notebook.py run_cascade_baseline.py evaluate_cascade_outputs.py`: PASS
- `python reemit_cascade_outputs.py --input-dir outputs/cascade_v1 --output-dir outputs/cascade_v1_emit_freeze14 --emit-policy freeze_major_tail_rewrites --max-tail-rewrite-words 14`: PASS
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-evaluation/bin/python evaluate_cascade_outputs.py --output-dir outputs/cascade_v1_emit_freeze14`: PASS
- `python reemit_cascade_outputs.py --input-dir outputs/cascade_v1 --output-dir outputs/cascade_v1 --emit-policy raw_passthrough --max-tail-rewrite-words 14`: PASS
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-evaluation/bin/python evaluate_cascade_outputs.py --output-dir outputs/cascade_v1`: PASS

### Resultats utiles

- `freeze14` passe `LongYAAL CU` sous `2s` sur le bundle reel
  (`1940.9677 ms`)
- `BLEU`, `CHRF`, `XCOMETXL=NA`, et `LongYAAL CA` restent inchanges dans
  l'annexe
- `319/357` updates sont gelees par la politique, ce qui confirme une coupe
  trop agressive pour en faire la voie par defaut
- les autres metriques CU (`LongAL`, `LongLAAL`, `LongDAL`) restent
  pathologiques dans l'annexe, donc le gain n'est pas promu comme vraie
  amelioration runtime

### Blocages restants

- aucun kernel `.venv-inference` persistant n'est vivant pour une vraie
  variante prompt/context
- `Unbabel/XCOMET-XL` manque toujours du cache HF local offline
- aucune variante prompt/context n'a encore ameliore la qualite au-dessus de
  la baseline corrigee

### Hypothese de la prochaine iteration

- `h_obj2_prompt_only_quality_latency_tuning`:
  tester une seule vraie variante prompt/context sur la baseline restauree, en
  gardant `h_obj2_freeze14_emission_annex` hors focus

## objective2_context1_terminology_guard_annex

### Decision

- introduire une vraie registry de variantes de traduction au lieu de modifier
  `config` a la main
- lancer une seule variante live `context1_terminology_guard`
- garder cette branche en annexe: meilleure qualite, meilleure latence CU,
  mais gate `2s` toujours rate

### Impact runtime

- un seul rechargement `Qwen3-ASR`/`Gemma` a ete justifie car aucun kernel
  `.venv-inference` n'etait vivant
- le runtime sait maintenant selectionner une variante nommee depuis le core,
  le notebook et le CLI, sans ecraser `outputs/cascade_v1/`
- `outputs/cascade_v1_context1_terminology_guard/` capture le run live
  (`BLEU=40.6694`, `CHRF=69.8671`, `LongYAAL CU=2339.1869`)
- la baseline canonique `outputs/cascade_v1/` n'est pas remplacee pour
  l'instant

### Portee de preuve

- preuve locale runtime
- un seul talk `objective2`
- un seul delai `prompt_only_latency_quality`
- un seul rerun live puis evaluation offline
- aucun claim papier

### Trigger de sortie

- cette iteration sort une fois que la variante live est reproduisible via une
  option nommee, qu'un bundle reel distinct est evalue, et que la branche est
  explicitement gelee si elle ne passe pas sous `2s`

### Hypothesis IDs touched

- `h_obj2_prompt_only_quality_latency_tuning`
- `h_obj2_context1_terminology_guard_annex`

### Status transitions

- `h_obj2_context1_terminology_guard_annex: new -> frozen_annex`

### Ce qui a ete fait

- ajoute `cascade_translation_variants.py` pour definir des variantes
  prompt/context nommees
- branche la selection de variante dans
  `qwen3asr_gemma_cascade_core.py`,
  `qwen3asr_gemma_cascade_notebook.py`, et `run_cascade_baseline.py`
- justifie l'absence de kernel persistant puis lance un seul run live
  `context1_terminology_guard`
- evalue offline `outputs/cascade_v1_context1_terminology_guard/`

### Validations lancees

- `python -m py_compile cascade_translation_variants.py qwen3asr_gemma_cascade_core.py qwen3asr_gemma_cascade_notebook.py run_cascade_baseline.py`: PASS
- `python restart_jupyter_kernel.py --list`: PASS (`No active notebook kernels found.`)
- `.venv-inference/bin/python run_cascade_baseline.py --output-dir outputs/cascade_v1_context1_terminology_guard --translation-variant context1_terminology_guard`: PASS
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-evaluation/bin/python evaluate_cascade_outputs.py --output-dir outputs/cascade_v1_context1_terminology_guard`: PASS

### Resultats utiles

- la variante live ameliore la qualite par rapport a la baseline:
  `BLEU +0.3568` (`40.6694` vs `40.3126`) et
  `CHRF +1.3218` (`69.8671` vs `68.5453`)
- `LongYAAL CU` baisse de `299.6245 ms`
  (`2339.1869` vs `2638.8114`) mais reste au-dessus du gate `2s`
- `LongAL`, `LongLAAL`, et `LongDAL` cote CU s'ameliorent aussi, mais
  `LongYAAL CA` se degrade a `-15065.4140`
- le gain semble venir d'un meilleur cadrage prompt/terminologie, sans preuve
  que la reinjection d'une phrase precedente vaut encore son cout latence

### Blocages restants

- aucun kernel `.venv-inference` persistant n'est vivant apres le script
- `Unbabel/XCOMET-XL` manque toujours du cache HF local offline
- la meilleure variante live reste au-dessus du gate
  (`LongYAAL CU=2339.1869 ms > 2s`)

### Hypothese de la prochaine iteration

- `h_obj2_prompt_only_quality_latency_tuning`:
  tester une variante prompt-only issue de la nouvelle registry pour isoler le
  gain de wording sans le cout de `max_history_utterances=1`

## objective2_prompt_only_terminology_guard_annex

### Decision

- ajouter une seule variante live `prompt_only_terminology_guard`
- geler cette branche en annexe: meilleure latence que la baseline, mais gate
  `2s` encore rate et `BLEU` en baisse

### Impact runtime

- un seul rechargement `Qwen3-ASR`/`Gemma` a encore ete justifie car
  `restart_jupyter_kernel.py --list` et `ps` n'ont trouve ni kernel
  `.venv-inference` ni `VLLM::EngineCore` reutilisable
- le runtime sait maintenant selectionner une variante prompt-only gardee dans
  la meme registry que les variantes contexte
- `outputs/cascade_v1_prompt_only_terminology_guard/` capture le run live
  distinct sans ecraser `outputs/cascade_v1/`

### Portee de preuve

- preuve locale runtime
- un seul talk `objective2`
- un seul delai `prompt_only_latency_quality`
- un seul rerun live puis evaluation offline
- aucun claim papier

### Trigger de sortie

- cette iteration sort une fois que la variante prompt-only est reproductible
  via la registry, qu'un bundle reel distinct est evalue, et que la branche
  est explicitement gelee si elle ne passe pas sous `2s`

### Hypothesis IDs touched

- `h_obj2_prompt_only_quality_latency_tuning`
- `h_obj2_prompt_only_terminology_guard_annex`

### Status transitions

- `h_obj2_prompt_only_terminology_guard_annex: new -> frozen_annex`

### Ce qui a ete fait

- factorise les regles de prompt partagees dans
  `cascade_translation_variants.py`
- ajoute la variante `prompt_only_terminology_guard` avec
  `max_history_utterances=0`
- justifie l'absence de kernel persistant puis lance un seul run live
  `prompt_only_terminology_guard`
- evalue offline `outputs/cascade_v1_prompt_only_terminology_guard/`

### Validations lancees

- `python -m py_compile cascade_translation_variants.py qwen3asr_gemma_cascade_core.py qwen3asr_gemma_cascade_notebook.py run_cascade_baseline.py`: PASS
- `python restart_jupyter_kernel.py --list`: PASS (`No active notebook kernels found.`)
- `ps -eo pid=,command= | rg 'VLLM::EngineCore|qwen3asr_gemma_cascade.py|run_cascade_baseline.py'`: PASS (`aucun runtime reutilisable`)
- `.venv-inference/bin/python run_cascade_baseline.py --output-dir outputs/cascade_v1_prompt_only_terminology_guard --translation-variant prompt_only_terminology_guard`: PASS
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-evaluation/bin/python evaluate_cascade_outputs.py --output-dir outputs/cascade_v1_prompt_only_terminology_guard`: PASS

### Resultats utiles

- par rapport a la baseline canonique, la variante prompt-only baisse
  `LongYAAL CU` de `420.2378 ms` (`2218.5736` vs `2638.8114`) et ameliore
  `LongYAAL CA` de `1833.0147 ms` (`-10767.8162` vs `-12600.8309`)
- `CHRF` monte legerement (`68.5887` vs `68.5453`) mais `BLEU` baisse
  (`39.8939` vs `40.3126`), donc le wording seul ne conserve pas le gain
  qualite de `context1_terminology_guard`
- la branche reste au-dessus du gate de `218.5736 ms`
  (`2218.5736 ms > 2s`) mais elle rapproche suffisamment la latence pour
  justifier un replay d'emission offline sur ce nouveau bundle

### Blocages restants

- aucun kernel `.venv-inference` persistant n'est vivant apres le script
- `Unbabel/XCOMET-XL` manque toujours du cache HF local offline
- la meilleure branche live qualite reste `context1_terminology_guard`, tandis
  que la meilleure branche prompt-only live reste encore au-dessus du gate

### Hypothese de la prochaine iteration

- `h_obj2_prompt_only_quality_latency_tuning`:
  rejouer exactement une politique d'emission conservative sur
  `outputs/cascade_v1_prompt_only_terminology_guard/` pour tester si le bundle
  prompt-only peut enfin passer sous `2s` sans nouveau rerun live
