# Retrieval : baseline

Index : `C:\fri-project\data\index` ; split : all ; filtres société/année : oui.

| Méthode | Questions | MRR | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 75 | 0.173 | 0.080 | 0.200 | 0.227 | 0.373 | 0.080 | 0.193 | 0.220 | 0.360 |
| vector | 75 | 0.283 | 0.147 | 0.320 | 0.480 | 0.533 | 0.133 | 0.287 | 0.427 | 0.507 |
| hybrid | 75 | 0.270 | 0.133 | 0.320 | 0.387 | 0.627 | 0.120 | 0.307 | 0.367 | 0.587 |
