# Rapport technique

Chaque membre rédige la partie de sa brique.

## 1. Data & Retrieval
## 2. Agent & MCP
## 3. Évaluation & Fine-tuning

Détail des métriques et des scripts : [`docs/evaluation.md`](evaluation.md).

### 3.1 Protocole

- **Jeu de test** : FinanceBench (150 questions en accès libre). Seules les questions dont le PDF est indexé sont évaluées : **75 questions sur 37 documents (15 sociétés)**.
- **Retrieval** : Hit@k, Recall@k et MRR. Les rangs sont comptés en pages distinctes du bon document, et les filtres société / année sont activés (comme dans l'agent).
- **Génération** : l'agent LangGraph répond à chaque question. Un LLM juge compare sa réponse à la réponse de référence (**exactitude**) et aux passages que l'agent a reçus (**fidélité**), en un seul appel qui renvoie un JSON. Les citations sont comparées aux pages de preuve de FinanceBench.
  - Agent : Groq `openai/gpt-oss-120b`. Juge : Groq `openai/gpt-oss-20b` (`JUDGE_MODEL`).
  - `llama-3.3-70b-versatile`, prévu au départ, a été retiré par Groq le 16/08/2026. `gpt-oss-120b` est le remplaçant recommandé par Groq. Le juge utilise un autre modèle pour avoir son propre quota (offre gratuite : 8 000 tokens/min et 200 000 tokens/jour par modèle).
- **Fine-tuning** : `BAAI/bge-small-en-v1.5`, perte MultipleNegativesRankingLoss, **124 paires** (question, texte de la page de preuve). Le split est fait **par société** (graine 42, 30 % en test) : 22 sociétés en train, 10 en test, dont 5 indexées (AMCOR, AMD, BLOCK, BOEING, COCACOLA). Entraînement : 3 époques, batch 16, lr 2e-5, environ 7 min sur CPU.

### 3.2 Retrieval

75 questions, index complet (37 documents) :

| Méthode | Questions | MRR | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 75 | 0.173 | 0.080 | 0.200 | 0.227 | 0.373 | 0.080 | 0.193 | 0.220 | 0.360 |
| vector | 75 | 0.283 | 0.147 | 0.320 | 0.480 | 0.533 | 0.133 | 0.287 | 0.427 | 0.507 |
| hybrid | 75 | 0.270 | 0.133 | 0.320 | 0.387 | 0.627 | 0.120 | 0.307 | 0.367 | 0.587 |

- **BM25 seul est le plus faible** (Hit@5 = 23 %). Les questions reformulent souvent les libellés des rapports (« capital expenditure » au lieu de « Purchases of property, plant and equipment »).
- **La recherche vectorielle est la meilleure en haut de liste** (Hit@5 = 48 %, MRR = 0,283).
- **L'hybride (RRF) est la meilleure sur 10 résultats** (Hit@10 = 63 %) : la fusion ramène des pages que chaque méthode seule rate, mais elle les classe un peu plus bas.
- Même à 10 résultats, la bonne page manque pour plus d'un tiers des questions. C'est la première cause d'erreur de l'agent (voir 3.3).
- Le reranker cross-encoder (`--methods hybrid rerank`) n'a pas été évalué : il est facultatif et prend 15 à 30 minutes sur CPU.

### 3.3 Génération

75 questions, agent complet (routage simple / calcul / comparaison) :

| Métrique | Valeur |
|---|---|
| questions | 75 |
| exactitude | 40.0% |
| fidélité | 80.0% |
| cite une page de preuve | 28.0% |
| précision des citations | 26.6% |
| secondes / question | 2.9 |
| exactitude (domain-relevant, 31) | 41.9% |
| exactitude (metrics-generated, 27) | 40.7% |
| exactitude (novel-generated, 17) | 35.3% |

Par route de l'agent :

| Route | Questions | Exactitude | Fidélité |
|---|---|---|---|
| simple | 31 | 15 (48 %) | 29 (94 %) |
| calcul | 23 | 8 (35 %) | 20 (87 %) |
| comparaison | 21 | 7 (33 %) | 11 (52 %) |

**La recherche décide de la réponse.** Quand l'agent cite une page de preuve, sa réponse est juste dans **15 cas sur 21 (71 %)**. Sinon, elle n'est juste que dans **15 cas sur 54 (28 %)**. Dans environ 19 réponses, l'agent dit qu'il ne trouve pas l'information dans les passages reçus : il refuse de répondre plutôt que d'inventer.

Exemples :

- **Bonne réponse** (AMCOR, EBITDA ajusté FY2023) : « $2,018 million », avec la citation [AMCOR, 2023, p. 1]. C'est exactement la réponse attendue.
- **Erreur de recherche** (Coca-Cola, ROA FY2017) : l'agent répond qu'aucun passage ne donne le résultat net ni le total des actifs. La réponse est fidèle, mais pas exacte, car les bonnes pages n'ont pas été trouvées.
- **Erreur de raisonnement** (Boeing, principaux clients FY2022) : la réponse décrit les clients par zone géographique et attribue au gouvernement américain une part fausse (89 % d'un segment au lieu de 40 % du chiffre d'affaires total).

**Vérification manuelle du juge** (10 verdicts tirés au hasard, 5 OK et 5 NON) : accord sur **8 sur 10**.
- Les 2 désaccords vont dans le même sens : **le juge est sévère**. Par exemple, il refuse 0,389 Md$ de dividendes (American Water Works) alors que la référence est 0,40 Md$, ce qui n'est qu'un arrondi. Il refuse aussi une réponse qui donne le bon segment (3M, Consumer) mais avec le chiffre en dollars au lieu du chiffre organique.
- L'exactitude réelle est donc probablement un peu au-dessus de 40 %.

### 3.4 Fine-tuning : avant / après

31 questions des 5 sociétés de test indexées (jamais vues à l'entraînement). « baseline » = index `data/index` (modèle d'origine), « finetune » = index `data/index-ft` (modèle fine-tuné, 14 documents) :

| Évaluation | Méthode | Questions | MRR | Hit@1 | Hit@5 | Hit@10 | Recall@5 |
|---|---|---|---|---|---|---|---|
| baseline-test | vector | 31 | 0.315 | 0.129 | 0.581 | 0.613 | 0.548 |
| baseline-test | hybrid | 31 | 0.285 | 0.129 | 0.452 | 0.677 | 0.452 |
| finetune-test | vector | 31 | 0.298 | 0.129 | 0.484 | 0.710 | 0.452 |
| finetune-test | hybrid | 31 | 0.314 | 0.129 | 0.516 | 0.677 | 0.468 |

- La loss d'entraînement passe de 1,55 à 0,72 : le modèle apprend bien les paires d'entraînement.
- Sur la ligne `vector`, qui isole l'effet du modèle, **le résultat est mitigé** :
  - le Hit@10 monte de 61 % à 71 % (+3 questions) ;
  - le Hit@5 baisse de 58 % à 48 % (−3 questions) ;
  - le MRR est stable (0,315 puis 0,298).
- L'hybride gagne un peu (Hit@5 de 45 % à 52 %, soit +2 questions ; MRR de 0,285 à 0,314).
- Sur 31 questions, une question vaut environ 3 points. Ces écarts ne sont donc **pas significatifs**.
- **Conclusion : le fine-tuning n'améliore pas clairement la recherche, et le modèle d'origine reste celui de l'équipe.** Avec 124 paires seulement, le modèle a peu appris de règles générales. Il a pu en partie sur-apprendre les formulations des sociétés d'entraînement.

### 3.5 Analyse et limites

- **Recherche** : c'est le maillon faible. La bonne page n'est dans les 5 premiers résultats que pour 39 à 48 % des questions, et l'agent ne peut pas répondre sans elle. Les pistes sont le reranker cross-encoder, des chunks par tableau pour les états financiers, et l'expansion de requête (synonymes des libellés comptables).
- **Calculs et comparaisons** : ce sont les routes les moins exactes (33 à 35 %). Une seule valeur mal extraite (mauvaise année, unité, périmètre organique ou publié) fausse tout le calcul.
- **Juge LLM** : il est plutôt sévère sur les arrondis et les réponses partielles. La **fidélité est sous-estimée sur la route comparaison** : l'outil `compare_years` ne renvoie pas les passages, donc le juge ne voit pas les preuves (10 des 15 réponses « non fidèles » viennent de cette route). Il faudrait transmettre au juge les passages utilisés par `compare_years`.
- **Taille du jeu** : 75 questions pour la génération et 31 pour la comparaison du fine-tuning. Un écart de quelques points correspond à 1 ou 2 questions. Pour conclure plus sûrement, il faudrait indexer davantage de sociétés de FinanceBench.
- **Coût et quotas** : avec l'offre gratuite de Groq, les 75 questions consomment plus de 200 000 tokens. L'évaluation a dû être reprise en plusieurs fois, ce que le CSV permet. Les modèles sont restés les mêmes pour toutes les questions.

## 4. API, Web & DevOps
## Résultats
## Conclusion
