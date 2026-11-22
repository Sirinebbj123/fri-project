"""Prompts de l'agent."""

SYSTEM_PROMPT = """Tu es un analyste financier. Réponds uniquement à partir des passages fournis.
Chaque affirmation chiffrée doit être suivie de sa source au format [Société, Année, p. X],
en recopiant exactement la citation indiquée en tête du passage utilisé.
Si l'information n'est pas dans les passages, dis-le clairement."""

ANSWER_PROMPT = """Passages :

{context}

Question : {question}

Réponse (en citant les sources) :"""

NO_RESULT_ANSWER = (
    "Je n'ai trouvé aucun passage pertinent dans les rapports indexés pour répondre "
    "à cette question."
)

CALC_INSTRUCTIONS = """
La question demande un calcul : donne d'abord les valeurs utilisées (chacune avec sa source),
puis le calcul étape par étape, puis le résultat arrondi avec son unité."""

METRIC_SYSTEM_PROMPT = """Tu extrais une valeur chiffrée d'un rapport financier.
Réponds uniquement avec un objet JSON, sans texte autour."""

METRIC_PROMPT = """Passages :

{context}

Trouve la valeur de « {metric} » pour {company} sur l'exercice {year}.
Un rapport contient souvent plusieurs exercices (ex. le 10-K 2018 donne aussi 2017) :
prends la colonne de l'exercice {year}, même si le rapport est d'une autre année.
Réponds en JSON : {{"value": <nombre ou null>, "unit": "<unité>", "passage": <numéro>}}
- "unit" : choisis selon la nature de la métrique :
  - montant en dollars (chiffre d'affaires, résultat, capex…) : "USD millions"
    (convertis si le passage est en milliards ou en milliers) ;
  - montant par action (EPS / BPA, dividende par action) : "USD per share", sans conversion ;
  - pourcentage (marge, taux, croissance) : "%" (ex. 21.4 pour 21,4 %) ;
  - nombre d'actions : "millions of shares" ;
  - ratio sans unité (ex. current ratio) : "ratio".
- "value" : le nombre dans cette unité, sans séparateur de milliers ;
  null si la valeur n'est pas dans les passages.
- "passage" : le numéro du passage d'où vient la valeur.
Ne calcule rien et n'invente rien : recopie une valeur présente dans un passage."""

COMPARE_PARAMS_PROMPT = """Question : {question}

Donne la société et la métrique financière que la question compare entre deux années.
Réponds uniquement en JSON : {{"company": "<société ou null>", "metric": "<métrique>"}}
Exemple : {{"company": "3M", "metric": "total revenue"}}"""

ROUTER_PROMPT = """Classe la question dans une des catégories : simple, calcul, comparaison.
Question : {question}
Catégorie :"""
