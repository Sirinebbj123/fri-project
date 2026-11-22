# Corrections de l'agent : années FY, année des données, unités

Trois corrections dans l'agent et les outils MCP, signalées pendant les essais sur les questions
FinanceBench.

## 1. Le routeur lit « FY2021 » et « FY22 »

La regex `\b(?:19|20)\d{2}\b` ne trouvait aucune année dans « Compare FY2021 and FY2022 revenue for
AMD » : `\b` ne coupe pas entre « FY » et « 2021 ». La question partait en route « calcul » et la
comparaison n'était jamais lancée, sans message d'erreur.

`find_years` reconnaît maintenant :

| Écrit | Année |
|---|---|
| `2021` | 2021 |
| `FY2021`, `FY 2021` | 2021 |
| `FY22`, `FY'22` | 2022 |
| `fiscal 2020`, `fiscal year 2021` | 2020, 2021 |

## 2. compare_years travaille sur l'année des données

Un 10-K de 2018 contient aussi les chiffres de 2017. Avant, `compare_years("3M", …, 2017, 2018)`
cherchait 2017 seulement dans le rapport 2017 : s'il manquait, tout échouait.

Maintenant `get_financial_metric(company, year, …)` cherche d'abord dans le rapport de `year`, puis,
s'il n'y a pas de passage ou pas de valeur, dans le rapport de `year + 1`. Le prompt demande au LLM
de prendre la colonne de l'exercice demandé. La citation pointe le rapport vraiment utilisé
(ex. 2017 lu dans `[3M, 2018, p. 42]`).

Une métrique annuelle se lit dans le 10-K : si le 10-K de l'année a des passages, les passages des
10-Q de la même année sont écartés (on garde les 10-Q seulement s'il n'y a pas de 10-K).

## 3. Les unités dépendent de la métrique

Le prompt imposait `"USD millions"` pour tout. Un BPA (EPS) de 8,89 $ ou une marge de 21,4 % sortait
faux. Le LLM choisit maintenant l'unité, puis `normalize_unit` la ramène à une des unités connues :

| Métrique | Unité |
|---|---|
| Montant (CA, résultat, capex…) | `USD millions` (milliards et milliers convertis) |
| Par action (EPS, dividende par action) | `USD per share` |
| Pourcentage (marge, taux) | `%` |
| Nombre d'actions | `millions of shares` |
| Ratio | `ratio` |

L'écart entre deux pourcentages est donné en points (« +2,5 points ») et les montants gardent deux
décimales (« 8,89 USD per share »). `compare_years` refuse toujours deux unités différentes.

## Noms de société

Le garde-fou de `search_filings` accepte aussi un nom plus long que celui de l'index
(« JPMorgan Chase & Co. » pour `JPMORGAN`, « Apple Inc. » pour `APPLE`). La partie de Maryem
(`company_key`, `resolve_company`) complète ce point dans le retrieval.
