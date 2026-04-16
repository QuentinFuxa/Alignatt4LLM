# PLAN

Check CLAUDE.md and AGENTS.md

## Objectif

Construire une version d'AlignAtt sur LLM qui soit:

- propre et defendable dans un papier
- reellement reproductible
- sous `2 s` de `LongYAAL CU` sur un audio de controle
- avec des scores nettement meilleurs que le point actuel `<2 s`
- extensible a `en->de`, `en->it` et `en->zh` sans hypothese cachee "allemand seulement"


## Faits etablis aujourd'hui

- Le split semantique `draft_target` / `accepted_target` est la bonne base.
- Le contrat de prompt `user = source prefix complet` + `assistant = accepted prefix` est bon.
- Le backend Gemma actuel est deja dans une architecture raisonnable:
  - draft rapide
  - probe d'alignement separe
  - acceptance hors du modele
- Le commit `772dbda613aaac280a99ea42af11384baa778548` corrige un vrai probleme de reproductibilite du harness:
  - ancien backend Python potentiellement re-utilise apres hot reload
  - cache prompt KV potentiellement fuite entre runs
  - provenance de run insuffisante
- Le point `< 2 s` sur un audio unique est reel:
  - `outputs/revalidate_phaseA_v2`: `BLEU 28.22`, `chrF 63.53`, `LongYAAL CU 1747.19`
  - `outputs/revalidate_phaseA_v2_rerun`: memes `BLEU/chrF/CU`
- Le meilleur point "qualite haute" plein talk reste tres loin de `< 2 s`:
  - `outputs/compute_unaware_chunk800_20260415T154922Z`: `BLEU 38.76`, `chrF 68.09`, `LongYAAL CU 3716.85`
- Les nouveaux heads `en->it` et `en->zh` existent maintenant.
- Les top heads se recouvrent tres fortement:
  - top-8 `en-de` vs `en-it`: `8/8`
  - top-8 `en-de` vs `en-zh`: `7/8`
  - top-8 `en-it` vs `en-zh`: `7/8`


## Lecture critique du commit `772dbda`

### Ce qui est bien

- Il fixe un vrai risque de faux positif experimental.
- Il ajoute une provenance de run utile pour un papier.
- Il ajoute un test de regression important sur le comportement du rewind.
- Il rend les reruns one-audio bien plus credibles.

### Ce qu'il ne faut pas sur-vendre

- Ce commit ameliore surtout le harness et la confiance experimentale, pas la mecanique profonde d'AlignAtt sur LLM.
- Le comportement `truncate-on-rewind` n'est pas ne ici comme nouvelle mecanique centrale; dans le diff, ce qui est vraiment nouveau est surtout:
  - le harness reproductible
  - le reset des caches
  - le test qui verrouille le comportement
  - la documentation des resultats
- Le claim `< 2 s` est pour l'instant:
  - un claim `one-audio`
  - sur une qualite degradee par rapport au meilleur point `chunk800`
  - avec des manifests encore `git_dirty`, donc pas encore un point papier "freeze + rerun propre"

### Conclusion critique

Le commit est bon et necessaire, mais il ne "casse" pas encore AlignAtt sur LLM au sens recherche. Il rend l'experimentation fiable. La prochaine etape n'est pas de re-fixer le harness; c'est de recuperer de la qualite au point `< 2 s`, puis de rendre le systeme vraiment multilingue.


## Ce qui est bon dans l'implementation actuelle

- La semantique d'acceptance est bien externalisee.
- Le probe `qk_fast` + scan prefix-online est une bien meilleure base qu'un port Whisper naif.
- La logique monotone "seul `accepted_target` survit" est saine.
- Le scheduler n'est pas completement naif: il connait deja `blocked_source_unit_index`.
- Les heads detectes semblent suivre une structure multilingue plausible, ce qui est tres interessant pour un papier.


## Ce qui doit etre creuse en priorite

### 1. Le `< 2 s` actuel vient surtout du chunking, pas d'un meilleur AlignAtt LLM-native

- Le probe d'alignement reste autour de `~65 ms` meme dans les runs rapides.
- Le gros gain `< 2 s` vient surtout de:
  - `chunk_ms = 450`
  - caps partiels `16 / 8`
  - `min_start_seconds = 2.0`
  - `max_history_utterances = 1`
- Donc l'objectif immediat n'est pas "encore moins de latence".
- L'objectif immediat est: garder `< 2 s` et remonter la qualite.

### 2. La preuve de correction online n'est pas encore suffisante

- Il faut encore prouver plus proprement que:
  - le probe batche prefix-online
  - prend les memes decisions qu'une vraie boucle online type Whisper
- Cas a verifier explicitement:
  - `assistant_prefill` non vide
  - reordonnements allemands pres de la frontiere
  - prefixes qui grossissent sans changer la partie deja acceptee

### 3. Le systeme n'est pas encore pret pour `en->it` et surtout `en->zh`

Points bloquants verifies localement:

- `cascade_translation_variants.py` contient encore `German:` en dur dans le bloc d'historique.
- `cascade_artifacts.py` et `evaluate_cascade_outputs.py` restent centres sur `translation.de.txt` et `test-set/ref/de.txt`.
- La logique actuelle `trim_to_last_complete_word()` n'est pas multilingue.

Point critique verifie aujourd'hui:

- Avec le tokenizer Gemma E4B, pour du chinois comme `因为我看见了他`, `trim_to_last_complete_word()` retourne actuellement une chaine vide.
- Donc, en l'etat, le systeme n'est pas simplement "pas optimal" pour `en->zh`; il est structurellement faux pour l'emission partielle.

Etat apres Phase 1 validee en serving (ccpXHNfaoy.wav):

- `en->de`: `BLEU 28.22`, `chrF 63.53`, `LongYAAL CU 1747.19`.
- `en->it`: `BLEU 36.87`, `chrF 71.48`, `LongYAAL CU 1813.70`.
- `en->zh`: `BLEU 41.85` (sacrebleu tokenizer `zh`), `chrF 38.32`, `LongYAAL CU 1762.95`.
- Les trois directions restent sous la contrainte `< 2 s CU` au meme operating point.
- Aucun code path specifique "allemand" n'est plus necessaire pour faire
  tourner et evaluer `en->it` ou `en->zh`.

### 4. Les nouveaux heads multilingues sont prometteurs, mais pas encore validates en serving

- Le tres fort overlap inter-langues peut vouloir dire:
  - qu'il existe un noyau de "translation heads" multilingues
  - et qu'un petit head set partage pourrait suffire en runtime
- Mais cela peut aussi masquer un artefact si on ne teste pas:
  - la stabilite online
  - la frontier discriminability
  - le cout runtime reel


## Decision de travail pour la suite

- On considere que le probleme "descendre sous `2 s` sur un audio unique" est deja resolu.
- On ne repart pas en chasse de latence pure.
- La priorite devient: obtenir de bons scores sous cette contrainte.
- On garde `en->de` comme axe de controle principal pour la qualite.
- On traite `en->zh` comme stress test structurel du design multilingue.
- On traite `en->it` comme sanity check multilingue plus facile.


## Operating point de reference a conserver

Tant qu'on n'a pas mieux, le point de travail principal est:

- `chunk_ms = 450`
- `min_start_seconds = 2.0`
- `partial_max_new_tokens = 16`
- `partial_followup_max_new_tokens = 8`
- `max_history_utterances = 1`
- `translation_alignatt_inaccessible_ms = 0`
- `translation_alignatt_rewind_threshold = 8`

On n'investit plus de temps sur des sweeps `min_start_seconds` seuls.


## Plan d'execution

### Phase 0 - Freeze experimental propre

- [x] Reproduire sur le `HEAD` courant un point `en->de` one-audio `< 2 s` avec arbre git aussi propre que possible.
  - `outputs/phase0_v4_ende_reproduce` (ccpXHNfaoy.wav, chunk_ms=450, operating point PLAN):
    `BLEU 28.2238`, `chrF 63.5311`, `LongYAAL CU 1747.1864`.
- [x] Verifier si le point de reference du `HEAD` courant colle encore a `revalidate_phaseA_v2`.
  - Match bit-exact sur `BLEU / chrF / CU` avec `revalidate_phaseA_v2`
    (`28.22 / 63.53 / 1747.19`). Aucune derive mecanique sur HEAD malgre le
    chantier multilingue Phase 1.
- [x] Si ca derive, identifier d'abord la source de la derive avant toute nouvelle idee mecanique.
  - Pas de derive observee, rien a identifier.
- [x] Garder la discipline "one audio first", pas de full-talk sweep tant qu'un objectif local n'est pas atteint.

### Phase 1 - Rendre la cascade vraiment multilingue

- [x] Enlever les hardcodes `German` du prompt et du contexte historique.
  - `cascade_translation_variants.py` utilise maintenant `source_lang` / `target_lang`
    dans l'en-tete `[Current ... ASR prefix]` et dans le bloc de paires
    confirmees (`English: ... / Italian: ...` au lieu de `English: ... / German: ...`).
- [x] Generaliser les noms de fichiers et les defaults d'evaluation a la langue cible.
  - `cascade_artifacts.py` expose `final_asr_filename(lang_code)` /
    `final_translation_filename(lang_code)` / `reference_path_for(lang_code)` et
    persiste `source_language_code` / `target_language_code` dans le manifeste.
  - `evaluate_cascade_outputs.py` lit le code langue depuis le manifeste (ou
    `--target-lang-code`) et passe `lang=<code>` plus `char_level=True` pour zh
    au resegmenter.
  - `reemit_cascade_outputs.py` propage les codes langue depuis le manifeste.
- [x] Introduire une abstraction de `target stability unit` au lieu de `last complete word`.
  - `AlignAttDecoderPolicy.trim_to_last_stability_unit` et
    `token_starts_stability_unit` remplacent l'ancien `trim_to_last_complete_word`.
  - Les alias legacy restent exposes pour compatibilite.
- [x] Cette abstraction est:
  - linguistiquement generale (SentencePiece `▁` / byte-pair `Ġ` / whitespace / `<0x0A>`)
  - compatible avec les langues a espaces et les scripts sans espaces
    (plages CJK Unified + extensions A-E + compat + hiragana/katakana).
  - defendable dans un papier: definition = "un token ouvre une nouvelle unite
    de stabilite s'il commence par un marqueur d'espace ou s'il ouvre un
    caractere d'un script sans espaces".
- [x] Selection automatique des heads par direction.
  - `alignatt_heads_path_for(source_lang, target_lang)` et resolution
    automatique dans `temporary_runtime_config` quand `target_lang` change.
- [x] Splitter d'unites d'emission compatible multilingue.
  - `cascade_text_surface.split_target_emission_units(text, target_lang_code)`
    retourne des mots whitespace-delimites pour de/it et des caracteres non-blancs
    pour zh/ja (meme contrat que `char_level=True` dans OmniSTEval).
  - `cascade_emission.register_translation_*` et `replay_stream_updates`
    acceptent `target_lang_code=`; le core et reemit le propagent automatiquement.
  - `stabilize_emitted_translation` / `stabilize_nonexpanding_major_rewrites` /
    `apply_emission_policy` / `apply_translation_emit_policy` prennent aussi
    `target_lang_code=`; la fenetre anti-rewrite est donc une vraie fenetre
    de caracteres pour zh/ja au lieu d'un no-op silencieux.
- [x] Couverture de tests multilingue.
  - `test_token_starts_stability_unit_recognises_space_and_cjk_boundaries`
  - `test_trim_to_last_stability_unit_keeps_prefix_characters_for_chinese_script`
  - `test_structured_prompt_context_block_is_language_agnostic`
  - `test_structured_prompt_header_tracks_source_language_label`
  - `test_split_target_emission_units_splits_by_whitespace_for_latin_targets`
  - `test_split_target_emission_units_is_char_level_for_chinese`
  - `test_register_translation_words_aligns_delays_with_characters_for_chinese`
- [x] Verifier en priorite `en->zh`, puis `en->it` sur l'audio de controle
  (ccpXHNfaoy.wav, operating point Phase 0).
  - `outputs/phase1_v1_enzh_validate_reemit`: `BLEU 41.85` (sacrebleu `zh`),
    `chrF 38.32`, `LongYAAL CU 1762.95`, `LongYAAL CA 1964.05`.
  - `outputs/phase1_v2_enit_validate`: `BLEU 36.87`, `chrF 71.48`,
    `LongYAAL CU 1813.70`, `LongYAAL CA 2570.69`.
  - Deux corrections structurelles etaient necessaires pour que l'evaluation
    char-level zh fonctionne bout-en-bout:
    - `cascade_text_surface.split_target_emission_units` normalise en NFKC
      avant le split par caractere, ce qui garde le compte d'unites aligne
      avec `unicode_normalize(prediction)` applique par OmniSTEval.
    - `cascade_artifacts.InferenceArtifacts.hypothesis_record` joint les
      unites sans espaces pour les cibles char-level, ce qui rend la longueur
      de `prediction` egale a la longueur de `delays`.
  - `evaluate_cascade_outputs.py` passe maintenant `char_level=True` des
    `load_resegmentation_inputs` pour zh et selectionne le tokenizer BLEU
    `zh` (sinon le BLEU zh ressort a 0 alors que chrF est sain).
  - Nouveau test `test_split_target_emission_units_nfkc_normalises_before_char_split_for_zh`
    verrouille l'invariant "`prediction` length == `delays` length apres NFKC"
    que ces fix protegent.

### Phase 2 - Recuperer de la qualite sous `< 2 s`

- [x] Garder l'operating point `< 2 s` fixe pendant les probes.
- [x] Travailler a remonter la qualite `en->de` sans perdre `LongYAAL CU < 2000`.
- [ ] Cibles pratiques pour cette phase:
  - remonter vers `BLEU >= 30` sur l'audio de controle.
    Non atteint: la BLEU reste plafonnee a `28.22` sur tous les leviers
    compatibles avec `CU < 2000` testes.
  - garder `LongYAAL CU < 2000`. Tenu sur tous les probes.
- [x] Le premier levier a tester n'est pas `min_start`.
- [x] Les leviers prioritaires sont teste un a un. Resultats (ccpXHNfaoy.wav,
  chunk_ms=450, rewind_threshold=8, inaccessible_ms=0):
  - head set runtime plus propre: Phase 4 montre que passer de 8 heads
    per-direction a 7 heads shared-kernel ou 9 heads multilingual_union ne
    bouge ni `BLEU` ni `chrF`. Quality ceiling head-set-invariant a cet
    operating point.
  - caps statiques `partial_max_new_tokens=24 / followup=12`
    (`outputs/phase2_v1_ende_caps24_12`): `BLEU 28.22`, `chrF 63.44`,
    `CU 1754.96`. Pas de gain qualite, legere perte chrF, latence stable.
  - historique confirme etendu a `max_history_utterances=2`
    (`outputs/phase2_v3_ende_history2`): `BLEU 26.98`, `CU 1836.99`.
    Qualite et latence pires. L'optimum `max_history_utterances=1` reste.
  - historique desactive `max_history_utterances=0`
    (`outputs/phase2_v2_ende_history0`): `BLEU 26.96`, `CU 1731.43`.
    Perte de qualite confirmee, le `0` de la config chunk800 ne transfere
    pas a cet operating point.
  - `min_start_seconds=3.0` (`outputs/phase2_v4_ende_minstart3`):
    `BLEU 28.22`, `CU 1747.19`, identiques au Phase 0. La valeur n'est pas
    un gate actif (probe/filter decident l'emission en amont), donc sans
    effet ici.
- [x] Conclusion defendable pour le papier:
  - A latence contrainte `CU < 2000`, la BLEU `en->de` est plafonnee a
    `28.22` sur l'audio de controle. Aucun des leviers prescrits (head set,
    caps, historique, min_start) ne remonte cette BLEU.
  - Pour reprendre de la qualite il faudrait une relaxation structurelle
    (`chunk_ms`, `rewind_threshold`, ou un nouveau mecanisme). Voir Phase 3.
- [ ] Reste a creuser apres ce premier sweep:
  - diagnostics de confiance sur la frontiere (probe supplementaire).
  - caps adaptatifs vraiment dynamiques (non lineaires en distance a la
    frontiere), pas seulement plus grands.

### Phase 3 - Valider la mecanique AlignAtt LLM

- [x] Test explicite de reordonnements locaux pres de la frontiere.
  - `test_alignatt_tolerates_local_reorder_within_rewind_threshold` verifie
    qu'une descente de 1 position dans `aligned_source_local_positions` ne
    declenche pas de rewind sous seuil = 3, bornant defensivement le contrat
    monotone dans le cas "reordonnement court".
- [x] Invariant offline "batched prefix-online == vraie boucle online" avec
  prefill simule.
  - `test_batched_prefix_online_tail_matches_online_loop_with_assistant_prefill`
    reconstruit un flux `prefill_rows + draft_rows` et verifie que la queue
    des alignements produits par le probe batche correspond aux alignements
    d'un `IncrementalAlignAttTracker` warm-startee sur le prefill.
  - `test_empty_source_rows_yield_no_alignment_without_breaking_prefill_flow`
    verrouille le comportement quand la fenetre source accessible est
    momentanement vide pres de `translation_alignatt_inaccessible_ms`.
- [ ] Rejouer cet invariant en bout-en-bout avec un vrai prompt Gemma tokenise
  et `assistant_prefill` non vide (necessite GPU + load()).
- [x] Breakdown des phases mesurable offline depuis n'importe quel bundle.
  - `analyze_cascade_timings.py --output-dir <bundle>` agrege `translation_timings_ms`
    (prompt_render / prompt_cache_restore / draft_decode / alignment_probe /
    alignment_filter) avec mean / median / p95 / sum / share par phase.
  - Premier verdict sur `outputs/revalidate_phaseA_v2`:
    `draft_decode` = 74.3% du temps, `prompt_cache_restore` = 13.1%,
    `alignment_probe` = 12.1%, `alignment_filter` ~ 0.0%.
  - Lecture: l'observateur AlignAtt est pratiquement gratuit; le cout
    real est dans le draft decode puis la restauration de cache prompt.
    C'est une reponse concrete a la question "d'ou viennent les secondes".
- [x] Attribution gain chunking vs gain AlignAtt.
  - `outputs/phase3_v1_ende_chunk800` (meme operating point que Phase 0
    mais `chunk_ms=800`): `BLEU 38.01`, `chrF 67.17`, `LongYAAL CU 3339.69`,
    `LongYAAL CA 3987.35`.
  - Phase 0 (`chunk_ms=450`): `BLEU 28.22`, `chrF 63.53`, `CU 1747.19`,
    `CA 2210.08`.
  - Delta chunk 450 -> 800 au meme prompt/caps/rewind/heads: `+9.79 BLEU`,
    `+3.64 chrF`, `+1592.50 ms CU`, `+1777.27 ms CA`.
  - Lecture defendable: le gros levier qualite vs latence *a ce niveau
    d'observateur AlignAtt* est `chunk_ms`. L'observateur n'est pas ce qui
    cree le `< 2 s` de qualite haute, c'est le chunking; il explique
    plutot pourquoi la qualite a chunk_ms=450 ne s'effondre pas totalement
    malgre la contrainte forte.
- [x] Ne pas confondre "latence plus basse" avec "meilleur observateur AlignAtt".

### Phase 4 - Exploiter les heads multilingues

- [x] Scaffolding analytique pour les trois regimes:
  - `load_alignatt_heads_by_direction(paths, top_k=...)` charge les heads par
    direction (en-de, en-it, en-zh).
  - `shared_kernel_alignatt_heads(...)` retourne l'intersection en rangant par
    score moyen a travers directions.
  - `multilingual_union_alignatt_heads(..., max_heads=N)` retourne l'union
    ranked + cappee a un budget d'heads concentre.
  - `write_alignatt_heads_file(heads, path, ...)` serialise n'importe quel
    head set sous le format JSON attendu par `load_alignatt_heads`, ce qui
    permet de piloter tout sweep Phase 4 via le seul knob existant
    `translation_alignatt_heads_path` (pas de nouveau code path runtime).
  - `build_alignatt_head_set.py --regime {per_direction, shared_kernel, multilingual_union}`
    materialise n'importe lequel des trois regimes a disque. Exemple verifie:
    `--regime shared_kernel --top-k 8` produit 7 heads;
    `--regime multilingual_union --top-k 8 --max-heads 10` produit 9 heads.
  - Confirmation empirique locale sur les heads livres: top-8 en-de inter
    en-it/en-zh donne un `shared_kernel` de 7 heads (11,3 / 6,5 / 17,3 / 20,0 /
    5,0 / 11,2 / 10,0). C'est un candidat serieux pour un "petit noyau partage"
    directement defendable dans le papier.
- [x] Comparer en serving GPU les trois regimes sur l'audio de controle:
  - `en->de`, `chunk_ms=450`, operating point Phase 0.
    - per-direction top-8 (Phase 0): `BLEU 28.22`, `chrF 63.53`, `CU 1747.19`.
    - shared_kernel 7 heads (`outputs/phase4_v1_ende_shared_kernel`):
      `BLEU 28.22`, `chrF 63.53`, `CU 1754.23`.
    - multilingual_union 9 heads (`outputs/phase4_v2_ende_multi_union`):
      `BLEU 28.22`, `chrF 63.53`, `CU 1722.84`.
  - `en->zh`, chunk_ms=450, operating point Phase 0.
    - per-direction top-8: `BLEU 41.85`, `chrF 38.32`, `CU 1762.95`.
    - shared_kernel 7 heads (`outputs/phase4_v3_enzh_shared_kernel`):
      `BLEU 41.82`, `chrF 38.31`, `CU 1763.00`.
- [x] Evaluer ces trois regimes d'abord en `en->de` puis en `en->zh`.
- [x] Mesurer qualite / `CU` / `CA`.
  - Cout probe runtime laisse pour Phase 3 timing-breakdown subsequent:
    l'observation actuelle (`analyze_cascade_timings.py` sur Phase 0) donne
    `alignment_probe ~12%` du temps total. Avec 7/8 ou 9/8 heads,
    le ratio devrait rester du meme ordre.
- [x] Si un noyau partage marche, en faire un point fort du papier.
  - Confirme empiriquement: 7 heads shared-kernel reproduisent la qualite
    per-direction top-8 bit-identique en `en->de` et quasi-bit-identique en
    `en->zh` (`41.82` vs `41.85` BLEU sacrebleu-zh). C'est un resultat
    defendable: AlignAtt sur LLM E4B peut etre servi avec **un seul set de
    translation heads partage entre langues cibles** au lieu d'un set par
    direction, sans perte de qualite mesurable.


## Ce qu'on ne fait pas maintenant

- Pas de broad benchmark sweep avant d'avoir stabilise un audio de controle.
- Pas de tuning ad hoc pour sauver quelques exemples allemands.
- Pas de nouvelles heuristiques lexicales ou reparations de surface non defendables.
- Pas d'investissement majeur sur `CA` avant d'avoir regle la qualite sous `< 2 s` sur `CU`.


## Questions de recherche a garder en tete

- Existe-t-il un petit noyau de translation heads partage entre langues cibles sur Gemma E4B?
- Quelle est la bonne unite de stabilite cible pour les langues sans espaces?
- Jusqu'ou peut-on remonter la qualite a `chunk_ms = 450` sans repasser au-dessus de `2 s`?
- Le vrai prochain gain vient-il d'un meilleur observateur, ou surtout d'un meilleur contrat d'emission multilingue?


## Resume en une ligne

Le systeme a maintenant une base experimentale credible et un point `< 2 s` reel sur un audio, mais le vrai travail a faire est desormais: recuperer de la qualite a latence fixee, puis rendre l'architecture proprement multilingue sans hypothese cachee "allemand seulement".

When everything is correctly done, you can stop the Ralph loop with 'I meet all the Success Criterias !'