Stp lis  @CLAUDE.md, @AGENTS.md, ALIGNATT_LLM.md et E4B_ALIGNATT_CASCADE_DESIGN.md 
Regarde très consciensement. ce qu'est alignatt et comment il est implémenté dans whisper, dans  assets/alignatt_doc. 


1. Regarde dans la cascade. Là on se concentre sur En->De
2. Cette cascade ne correspond pas exactement à l'implmélentation de alignatt dans whisper, puisque alignatt  dans whisper est sur cross attention, et là on travaille sur un LLM
3. Reflechis profondeement à ce que ca implique que ce soit un LLM et non un transformers, comment gérer au mieux alignatt x performance x kv cache x decisions inteliigentes. On veut "casser" la meilleure maniere de faire du alignatt sur un LLM.
4. Sois critique de l'implémentation actuelle. Elle a principalement été générée par IA, et peut etre over complexe à certains endroits, foireuse à d'autre, et tout peut être challengé, et modifié.
5. Ne te lance pas dans des longs runs, les audios durent en moyenne qq minutes chacun, et lancer un run sur audio est largement suffisant pour débugger.
6. Le problème principal qu'on a aujourd'hui, c'est la latence : On souhaite passer sous 2s de latence LongYaal compute unware. Quelles sont les pistes que tu suggeres pour avoir une implémentation plus "rapides" ? Mets les dans assets/alignatt_doc/ALIGNATT_LLM.md si elles n'y sont pas déjà. On est OK de perdre en qualité pour arriver à ces métriques. On a même très probablement pas le choix.

Travaille donc à arriver sous les 2 secondes.

Tu peux terminer la boucle avec le message 'Okay, my Longyaal CU one audio is below 2s, and I consider the quality okay'.

