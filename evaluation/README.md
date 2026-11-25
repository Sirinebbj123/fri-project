# evaluation/ (Rôle 3)

- `dataset.py` : charge les questions FinanceBench (question, réponse, pages de preuve 1-indexées) et sépare train / test par société.
- `retrieval_eval.py` : Recall@k, MRR, page hit rate pour BM25, vecteurs, hybride et reranker.
- `generation_eval.py` : exactitude, fidélité et citations des réponses de l'agent (LLM-as-judge).
- `compare.py` : met plusieurs évaluations côte à côte (ex. avant / après fine-tuning).
- `results/` : CSV et tableaux Markdown (ignorés par Git sauf ceux qu'on ajoute au rapport).

Mode d'emploi complet : [`docs/evaluation.md`](../docs/evaluation.md).
