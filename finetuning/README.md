# finetuning/ (Rôle 3)

Fine-tuning de `BAAI/bge-small-en-v1.5` sur les paires (question, texte de la page de preuve)
de FinanceBench, avec `MultipleNegativesRankingLoss` (sentence-transformers).

```bash
python -m finetuning.train_embeddings
```

- Entraînement seulement sur les sociétés du split `train` ; les sociétés du split `test` servent
  à mesurer le gain (`python -m evaluation.retrieval_eval --split test`).
- Le modèle est enregistré dans `finetuning/models/bge-small-en-financebench/` (ignoré par Git),
  avec `training_info.json`.
- Il remplace celui de l'index (`EMBEDDING_MODEL` dans `.env`) seulement si l'évaluation montre un gain.

Mode d'emploi complet : [`docs/evaluation.md`](../docs/evaluation.md).
