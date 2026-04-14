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
