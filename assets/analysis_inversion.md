# Inversion / Reordering Analysis for `ccpXHNfaoy`

## Scope

Requested scope was the first 50 chunks of `mcif-long-trans/reference_clean/ccpXHNfaoy/segments.json`.

This file only contains 47 segments, so this note covers the full file, segments `0..46`.

The goal here is not to score translation quality. The goal is to identify where the reference translations show real target-side reordering that a future flagger should detect, especially language-specific word order changes.

## Short takeaways

- `en->de` is the clearest case for hard syntactic inversion. The dominant signal is not random movement, but regular German clause structure: verb-second in matrix clauses, verb-final in subordinate clauses, and infinitive-at-end in `um ... zu`-like structures.
- `en->it` is much closer to English surface order. The useful cases are softer: adjunct movement, purpose clauses, packaging changes, and occasional fronting of an instrument or condition.
- `en->zh` has many strong rewrites, but they are often better described as restructuring than as a single inversion. The main signals are topic/comment packaging, fronted cause/purpose phrases, prenominal modifier packing, and serial-verb style decomposition.
- For a detector, German gives the cleanest first benchmark. Chinese gives the richest reordering pressure. Italian gives fewer hard positives and many medium-strength cases.

## Best positive examples

- `seg 1`: EN `I'm here to introduce ...` -> DE `Ich bin hier, um ... vorzustellen.` Strong German purpose-clause reordering, lexical verb at the end.
- `seg 2`: EN `... plan their actions by following ...` -> DE `... planen ..., indem sie ... folgen.` Strong English gerund -> German subordinate clause, verb final.
- `seg 7`: EN `... planning which imposes different constraints ...` -> DE `..., bei dem ... unterliegen.` Relative clause with verb final.
- `seg 11`: EN `Since no dataset ... exists ..., we have to ... first.` -> DE `Da es ... gibt, mussen ... erst ...` and ZH `由于...因此...`. Very useful cause-clause reordering example in both DE and ZH.
- `seg 16`: EN `conduct detailed analysis to investigate why ... fail` -> DE `fuhren ... durch, um ... zu untersuchen` and ZH `进行了..., 探究...`. Good purpose/reason restructuring.
- `seg 17`: EN `show that ... is acceptable but ... cannot be guaranteed` -> DE `zeigen, dass ... , dass aber ...`. Heavy clause packaging.
- `seg 27`: EN `We only keep the script if ...` -> ZH `只有当...时, ...才...`. Very good Chinese conditional inversion pattern.
- `seg 30`: EN `Since ... are costly to deploy, it's essential to enable ...` -> DE `Da ..., ist es wichtig, ... zu ermoglichen` and ZH `由于..., 因此至关重要的是要...`. Strong causal reordering.
- `seg 34`: EN `We apply our method ... named as CoScript.` -> DE `Wir wenden ... an, der ... genannt wird.` Strong separable-verb + relative clause behavior.
- `seg 35`: EN `In total, we generate ... with scripts.` -> IT `con gli script generiamo ...` and ZH `我们用脚本生成了...`. Instrument phrase moves before the verb.
- `seg 36`: EN `To ensure ..., we ask ... to find and revise ...` -> DE `Um ... sicherzustellen, bitten wir ...` and ZH `为了确保..., 我们要求...`. Very useful purpose-clause fronting.
- `seg 40`: EN `..., indicating that ... when properly trained ...` -> DE `..., was darauf hindeutet, dass ... wenn ...` and ZH `这表明在...时, ...`. Strong final-clause packing.
- `seg 44`: EN `We hope ... can be a valuable resource to advance ...` -> DE `Wir hoffen, dass ... sein wird, um ... voranzutreiben.` Good embedded-clause + infinitive-final case.
- `seg 46`: EN `Please find more details ... in our paper.` -> DE `Weitere Details ... finden Sie ...`. Imperative with fronted object.

## German (`en->de`)

German is where a future flagger can be the most precise. The important thing is to flag structural movement, not just lexical differences.

- `seg 1`: `I'm here to introduce ...` -> `Ich bin hier, um ... vorzustellen.`
  The English infinitive sits close to the matrix verb. In German the introducing verb is postponed to the end of the purpose clause.

- `seg 2`: `... plan their actions by following ...`
  -> `... planen Menschen ihre Handlungen oft, indem sie ... folgen.`
  English uses a gerundial manner phrase. German turns it into a subordinate clause with the finite verb at the end.

- `seg 3`: `... exploited language models to plan for abstract goals ...`
  -> `... haben Sprachmodelle verwendet, um ... zu planen.`
  Another clean purpose-clause pattern with the lexical verb delayed.

- `seg 7`: `... constrained language planning which imposes ...`
  -> `... Sprachplanung, bei dem ... unterliegen.`
  Relative clause moves the main predicate to the end. Good example of clause-internal inversion.

- `seg 11`: `Since no dataset ... exists ...`
  -> `Da es keinen Datensatz ... gibt, mussen wir ... erst beschaffen.`
  Three signals at once: cause clause first, existential `es gibt`, and delayed main predicate in the subordinate clause.

- `seg 12`: `As shown in the table, ...`
  -> `Wie in der Tabelle zu sehen ist, ...`
  The English participial frame becomes a finite copular clause placed before the main clause.

- `seg 15`: `We find that all language models ...`
  -> `Wir stellen fest, dass ... liefern.`
  Good basic `that`-clause example: the proposition is pushed into a subordinate clause with final verb.

- `seg 16`: `conduct detailed analysis to investigate why ... fail`
  -> `fuhren ... durch, um die Grunde ... zu untersuchen.`
  Useful because the English simple verb becomes the German split predicate `fuhren ... durch`, plus a final infinitive.

- `seg 17`: `show that ... is acceptable but ... cannot be guaranteed`
  -> `zeigen, dass ... akzeptabel ist, dass aber ... nicht garantiert werden kann.`
  Strong example of German clause packaging with repeated embedded clauses and final verbal material.

- `seg 20`: `... output quality ... falls in high variance, leading to ...`
  -> `... mit hoher Varianz abnimmt, was zu ... fuhrt.`
  The English participial result clause becomes a relative-like continuation `was ... fuhrt`, with German final verbal position again.

- `seg 21`: `over-generate-then-filter`
  -> `Ubergenerierung und anschliessenden Filterung`
  Not a pure inversion, but important: German often resolves process descriptions into nominal compounds rather than preserving English surface order.

- `seg 25`: `... calculate the cosine similarity ... to measure semantic similarity`
  -> `... berechnen ... , um ... zu messen.`
  Straightforward purpose-clause reordering.

- `seg 27`: `We only keep the script if ...`
  -> `Wir behalten das Skript nur, wenn ... hat.`
  Conditional subordinate clause with final predicate.

- `seg 30`: `Since ... are costly to deploy, it's essential to enable ...`
  -> `Da ... teuer ... sind, ist es wichtig, ... zu ermoglichen.`
  This is a strong benchmark case because both causal fronting and infinitival extraposition happen.

- `seg 33`: `we follow the idea ... to distil ...`
  -> `... verfolgen wir die Idee ..., um ... zu destillieren.`
  Again, purpose clause at the end.

- `seg 34`: `We apply our method ...`
  -> `Wir wenden unsere Methode ... an, der CoScript genannt wird.`
  Excellent separable-verb example. A detector should absolutely learn to treat `wenden ... an` as one predicate spanning the clause.

- `seg 36`: `To ensure ..., we ask ...`
  -> `Um ... sicherzustellen, bitten wir ...`
  Canonical purpose-clause fronting.

- `seg 38`: `We find CoScript shows ...`
  -> `Es ist zu erkennen, dass CoScript ... aufweist.`
  This is less a clean inversion and more a stylistic recast through an expletive frame. It is useful as a warning case: not every order change is a simple word-order phenomenon.

- `seg 40`: `..., indicating that smaller models can surpass larger models when ...`
  -> `..., was darauf hindeutet, dass ... ubertreffen konnen, wenn ... trainiert werden.`
  Very high reordering pressure: reporting clause, embedded complement, and final `wenn` clause.

- `seg 44`: `We hope ... can be ... to advance ...`
  -> `Wir hoffen, dass ... sein wird, um ... voranzutreiben.`
  Another strong complement + infinitive-final example.

- `seg 46`: `Please find more details ...`
  -> `Weitere Details ... finden Sie ...`
  Clear imperative inversion: object comes first, finite verb before subject pronoun.

## Italian (`en->it`)

Italian stays much closer to English order, so a detector should be more conservative. Many differences are medium-strength reorderings, not hard inversions.

- `seg 1`: `I'm here to introduce ...` -> `Sono qui per presentare ...`
  Mild purpose-clause shift, but still mostly monotonic.

- `seg 2`: `... by following ...`
  -> `... pianificano ... seguendo ...`
  The gerund is preserved, so this is weaker than German. Still useful as a low-pressure positive.

- `seg 7`: `... planning which imposes ...`
  -> `... pianificazione ... che impone ...`
  Relative clause stays fairly aligned with English.

- `seg 11`: `Since no dataset ... exists ...`
  -> `Dal momento che non esiste ..., dobbiamo ...`
  Good causal-fronting case, but much less extreme than German.

- `seg 16`: `conduct detailed analysis to investigate why ... fail`
  -> `conduciamo ... per esaminare la ragione per cui ... falliscono`
  Purpose phrase and embedded explanation both remain close, but there is enough clause repackaging to mark as moderate.

- `seg 17`: `... is acceptable but ... cannot be guaranteed`
  -> `... e accettabile, ma ... non puo essere garantita`
  Mostly monotonic; useful as a near-negative.

- `seg 20`: `..., leading to bad performance`
  -> `..., portando a prestazioni scadenti`
  English result participle is preserved by an Italian gerund-like continuation. Soft positive.

- `seg 21`: `over-generate-then-filter`
  -> `sovra-generare e poi filtrare`
  Italian keeps the event sequence explicitly, which is closer to English than German.

- `seg 27`: `We only keep the script if ...`
  -> `Manteniamo lo script solo se ...`
  Good conditional case, but still close to English surface order.

- `seg 30`: `Since ... are costly to deploy ...`
  -> `Poiche ... sono costosi da implementare, e essenziale abilitare ...`
  Clear cause-fronting plus extraposed infinitival target.

- `seg 33`: `... to distil ... from large language models`
  -> `... per distillare ... da grandi modelli linguistici`
  Mild but real purpose structure.

- `seg 35`: `In total, we generate ... with scripts.`
  -> `In totale, con gli script generiamo ...`
  This is one of the best Italian examples. The instrumental phrase moves before the verb.

- `seg 36`: `To ensure ..., we ask ... to find and revise ...`
  -> `Per garantire ..., chiediamo ... di trovare e rivedere ...`
  Useful purpose-fronting example.

- `seg 40`: `..., indicating that ... if properly trained ...`
  -> `..., indicando che ... se opportunamente addestrati ...`
  Good medium-strength case: reporting participle plus conditional packing.

- `seg 45`: `Thanks for your time.`
  -> `Grazie per il tempo che mi avete dedicato.`
  Not an inversion in the narrow sense, but a strong structural expansion. Useful as a warning case for false positives.

- `seg 46`: `Please find more details ...`
  -> `Troverete maggiori dettagli ...`
  Light imperative recast; little true inversion.

## Chinese (`en->zh`)

Chinese gives many strong cases, but many are not just `verb moved left/right`. They are broader structural transformations. A detector should therefore look for construction types, not only token crossing.

- `seg 1`: `I'm here to introduce ...`
  -> `我将介绍...`
  Chinese removes `I am here to` and recasts it as future/intention. This is more compression than inversion.

- `seg 2`: `... plan their actions by following ...`
  -> `... 会遵循..., 按照...来规划行为`
  Very useful example: English one-clause gerund becomes a serial construction with ordered sub-actions.

- `seg 6`: `Planning for the goals with specific constraints ... remains under-studied`
  -> `对于有特定限制的目标..., 规划工作仍然研究不足`
  Strong topic-fronting. This is exactly the type of Chinese word order change that a naive inversion detector may miss.

- `seg 7`: `... planning which imposes different constraints on the goals ...`
  -> `... 该问题对规划目标施加了不同的约束`
  Relative structure becomes a direct clause with explicit topic reference.

- `seg 8`: `An abstract goal can be inherited by different real-life specific goals ...`
  -> `一个抽象目标可以被赋予不同的多方面约束, 形成特定的实际目标`
  Strong restructuring with a resultative continuation.

- `seg 11`: `Since no dataset ... exists ..., we have to acquire ... first`
  -> `由于没有...的数据集..., 因此我们必须首先获取...`
  Very clean Chinese cause-first pattern.

- `seg 12`: `As shown in the table, we extend ... using InstructGPT ...`
  -> `如表所示, 我们使用 InstructGPT, 通过...来扩展..., 以实现...`
  Excellent example of Chinese serial packaging. The `using InstructGPT` phrase is pulled earlier than in English.

- `seg 16`: `conduct detailed analysis to investigate why ... fail`
  -> `进行了详细分析, 探究...`
  Straight serial-verb restructuring.

- `seg 18`: `constraints defined in wikiHow`
  -> `wikiHow 中定义的更细化的约束条件主题类别`
  Important head-final nominal behavior: the full modifier stack appears before the noun phrase head.

- `seg 21`: `adopt the idea of over-generate-then-filter to improve generation quality`
  -> `采用“先过度生成再筛选”的方法来提高...`
  The process name is packed into a quoted modifier before `方法`.

- `seg 24`: `a filter model is developed to select the faithful scripts`
  -> `我们开发了一个过滤模型来选择...`
  Passive-like English becomes active Chinese. This is a structural recast, not a simple inversion.

- `seg 27`: `We only keep the script if ...`
  -> `只有当...时, 我们才会保留脚本`
  One of the best Chinese positives. The condition is fronted and wrapped by the correlative `只有...才...`.

- `seg 29`: `improves the planning ability both in ... and ...`
  -> `在...方面都大大提高了规划能力`
  Domain phrases move before the verb.

- `seg 30`: `Since ... are costly to deploy, it's essential to enable ...`
  -> `由于..., 因此至关重要的是要实现...`
  Very strong causal and predicative recast.

- `seg 33`: `... to distil constrained language planning datasets from large language models`
  -> `从大语言模型中提炼出受限语言规划数据集`
  Source phrase `from ...` is fronted before the verb.

- `seg 34`: `We apply our method ...`
  -> `我们将我们的方法应用于构建...的数据集`
  The `将` construction explicitly front-loads the object before the verb phrase.

- `seg 35`: `In total, we generate ... with scripts`
  -> `我们用脚本生成了...`
  Instrument phrase before the verb. Clean positive.

- `seg 36`: `To ensure ..., we ask ...`
  -> `为了确保..., 我们要求...查找并修改...`
  Purpose fronting plus serial action chain.

- `seg 40`: `..., indicating that smaller models can surpass larger models when ...`
  -> `这表明在合适的数据集上进行适当的训练时, 较小模型的表现可以超越较大的模型`
  The time/condition phrase moves before the main predicate.

- `seg 44`: `... resource to advance research on language planning`
  -> `推进语言规划研究的宝贵资源`
  Verbal phrase becomes a prenominal modifier with `的`.

- `seg 46`: `Please find more details ... in our paper`
  -> `请在我们的论文中找到有关...的更多详细信息`
  Prepositional location phrase is fronted before the verb.

## Good negative or low-pressure examples

These are useful because a future detector should not over-flag.

- `seg 14`: `This table reports the overall accuracy of the results.` All three target languages stay close to the English information order.
- `seg 19`: `The heat map in the figure shows that ...` There is some clause packaging, especially in German, but the information flow remains mostly aligned.
- `seg 28`: `With our method, InstructGPT can generate scripts of higher quality.` This is mostly monotonic across languages.
- `seg 37`: `This figure shows the constraint distribution of CoScript.` Very low reordering pressure.
- `seg 41`: `In summary, we establish the constrained language planning problem.` Minimal structural movement.
- `seg 43`: `We use large language models to generate ...` Mostly monotonic in all targets.

## Places where reordering and paraphrase are mixed

These should be treated carefully in an automatic pipeline because the target change is not only about order.

- `seg 4`: EN is a fragment (`And show that ...`), but all targets normalize it into a full sentence. A flagger should not read this as pure inversion.
- `seg 8`: ZH strongly restructures the logic, so token crossing alone would be misleading.
- `seg 22`: DE turns the English active coordination into a more passive-like formulation. This is not a neat word-order case.
- `seg 24`: ZH makes the clause explicitly active (`we develop ...`) although the English is passive-like.
- `seg 38`: DE uses `Es ist zu erkennen, dass ...`, which is more a discourse recast than a simple inversion.
- `seg 45`: IT expands `Thanks for your time` into `Grazie per il tempo che mi avete dedicato`. Good reminder that politeness and discourse style can change syntax.

## What this suggests for a future flagger

- Start with `en->de` and explicitly flag:
  `subordinate_verb_final`, `purpose_clause_infinitive_final`, `separable_verb_span`, `fronted_condition_or_cause`, `object_fronted_imperative`.

- For `en->it`, use softer labels:
  `fronted_adjunct`, `purpose_clause_repackaging`, `instrument_fronting`, `conditional_clause_shift`.

- For `en->zh`, do not reduce everything to token inversion. Better labels are:
  `topic_fronting`, `cause_first`, `purpose_first`, `serial_verb_repackaging`, `prenominal_modifier_packing`, `correlative_condition_only_if_then`, `object_fronting_with_jiang_like_pattern`.

- If we want a compact benchmark slice from this talk, the highest-yield segments are:
  `1, 2, 7, 11, 12, 16, 17, 27, 30, 34, 35, 36, 40, 44, 46`.

## Bottom line

This talk is a good mini-benchmark for inversion analysis because it contains all three useful regimes:

- clean German hard reordering
- moderate Italian repackaging
- strong Chinese structural rewrites

If we want the first operational version of a flagger, this file already gives enough examples to define a hand-checked taxonomy without using any illegal static vocabulary mapping.
