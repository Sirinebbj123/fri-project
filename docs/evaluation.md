# Évaluation et fine-tuning

Mesurer la qualité du système sur FinanceBench, puis essayer d'améliorer la recherche en fine-tunant le modèle d'embeddings.

## Données

`data/financebench_open_source.jsonl` : 150 questions annotées (réponse attendue, document, pages de preuve, texte des pages). Seules les questions dont le document est **dans l'index** sont évaluées (75 questions pour les 37 documents de `--companies 15`).

- Pages : FinanceBench est 0-indexé, nos citations 1-indexées. `EvalItem.evidence_pages` est déjà converti (`evidence_page_num + 1`).
- Types de questions : `metrics-generated` (un chiffre d'un état financier), `domain-relevant` (questions d'analyste), `novel-generated` (questions variées).
- Split train / test **par société**, graine fixe (`seed=42`, 30 % des sociétés en test). Il est calculé sur les 150 questions, donc il ne change pas selon l'index. Le modèle fine-tuné ne voit jamais les sociétés du test.

```bash
python -m evaluation.dataset --split test --index-dir data/index
```

## 1. Retrieval

```bash
python -m evaluation.retrieval_eval                                   # bm25, vector, hybrid
python -m evaluation.retrieval_eval --methods hybrid rerank --tag rerank
python -m evaluation.retrieval_eval --no-filters --tag sans-filtres
```

| Métrique | Définition |
|---|---|
| Hit@k (page hit rate) | part des questions dont une page de preuve est dans les k premières pages retrouvées |
| Recall@k | part des pages de preuve retrouvées dans les k premières pages (une question peut en avoir plusieurs) |
| MRR | moyenne de 1 / rang de la première page de preuve (0 si absente des 10 premières) |

Les rangs sont comptés en **pages distinctes** (deux chunks de la même page = une page), et une page compte seulement si c'est la bonne page **du bon document**. Par défaut, la recherche est filtrée par la société et l'année du document, comme quand l'utilisateur les choisit dans l'interface ; `--no-filters` mesure la recherche sans aide.

Sorties : `evaluation/results/retrieval_<tag>.csv` (une ligne par question et par méthode, avec les pages retrouvées) et `retrieval_<tag>_summary.md`.

## 2. Génération (LLM-as-judge)

```bash
python -m evaluation.generation_eval --limit 5        # essai
python -m evaluation.generation_eval --pause 2        # toutes les questions indexées
```

Pour chaque question, l'agent répond (même graphe LangGraph que l'API), puis un LLM juge lit la question, la réponse attendue, les passages donnés à l'agent et sa réponse :

| Métrique | Définition |
|---|---|
| exactitude | le juge dit que la réponse donne la même information que la réponse attendue |
| fidélité | le juge dit que tout ce qu'affirme l'agent est dans les passages (pas d'invention) |
| cite une page de preuve | au moins une citation `[Société, Année, p. X]` pointe une page de preuve |
| précision des citations | part des citations qui pointent une page de preuve |

L'exactitude est aussi donnée par type de question. Il faut une clé API dans `.env` (le juge utilise le même LLM ; `JUDGE_PROVIDER` / `JUDGE_MODEL` pour en prendre un autre, idéalement plus fort). Le CSV est écrit question par question : si le script s'arrête (quota), relancer la même commande reprend où il s'était arrêté. Un nouveau `--tag` repart de zéro.

Limites à écrire dans le rapport : le juge peut se tromper (vérifier à la main une dizaine de lignes du CSV, colonne `reason`) ; la précision des citations est sévère, car une page voisine qui contient aussi le chiffre compte comme fausse.

## 3. Fine-tuning des embeddings

```bash
python -m finetuning.train_embeddings
```

Paires (question avec le préfixe BGE, texte de la page de preuve) des sociétés du split `train` (~124 paires), perte `MultipleNegativesRankingLoss`, 3 époques, batch 16, lr 2e-5. Résultat : `finetuning/models/bge-small-en-financebench/` (+ `training_info.json`).

Comparer sur les sociétés du test seulement :

```bash
python -m evaluation.retrieval_eval --split test --methods vector hybrid --tag baseline-test
python -m fri.ingestion.indexer --index-dir data/index-ft --embedding-model finetuning/models/bge-small-en-financebench --docs <documents du test>
python -m evaluation.retrieval_eval --split test --methods vector hybrid --index-dir data/index-ft --tag finetune-test
python -m evaluation.compare baseline-test finetune-test
```

`data/index-ft` ne contient que les documents du test (plus rapide à construire). Avec les filtres société / année, la recherche vectorielle ne voit de toute façon que le document de la question ; seul le BM25 de l'hybride change un peu (statistiques des mots calculées sur moins de documents). Comparer d'abord la ligne `vector`.

Si le modèle fine-tuné est meilleur : mettre `EMBEDDING_MODEL=finetuning/models/bge-small-en-financebench` dans `.env` et reconstruire l'index complet. Avec ~124 paires, un gain n'est pas garanti : un résultat neutre ou négatif est aussi un résultat à expliquer dans le rapport (trop peu de données, sur-apprentissage).

## Tests

`tests/test_evaluation.py` : métriques, chargement et split, évaluation du retrieval de bout en bout sur un petit index, évaluation de la génération avec un faux agent et un faux juge (et reprise après arrêt), paires du fine-tuning. Aucune clé API ni téléchargement.
