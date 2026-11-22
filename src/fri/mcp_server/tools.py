"""Logique des 3 outils MCP, indépendante du transport (testable sans serveur).

Le retriever est remplaçable : les tests et la démo utilisent un retriever en
mémoire (`set_retriever`). Sinon, `get_retriever()` charge le vrai `HybridRetriever`
(BM25 + ChromaDB) construit par `python -m fri.ingestion.indexer`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from fri.ingestion.metadata import company_key
from fri.schemas import DocType, MetricValue, SearchResult, YearComparison


class Retriever(Protocol):
    """Ce que les outils attendent d'un retriever (c'est l'interface de HybridRetriever)."""

    def search(
        self, query: str, company: str | None = None, year: int | None = None, k: int = 8
    ) -> list[SearchResult]: ...


# extracteur(company, year, metric, results) -> MetricExtraction | None (voir fri.agent.metrics)
MetricExtractor = Callable[[str, int, str, list[SearchResult]], Any]


class MetricNotFoundError(LookupError):
    """La valeur demandée n'a pas été trouvée dans les passages."""


_retriever: Retriever | None = None
_extractor: MetricExtractor | None = None


def set_metric_extractor(extractor: MetricExtractor | None) -> None:
    """Choisit l'extracteur de valeurs (None = le LLM du .env)."""
    global _extractor
    _extractor = extractor


def get_metric_extractor() -> MetricExtractor:
    global _extractor
    if _extractor is None:
        from fri.agent.metrics import LLMMetricExtractor

        _extractor = LLMMetricExtractor()
    return _extractor


def set_retriever(retriever: Retriever | None) -> None:
    """Choisit le retriever utilisé par les outils (None = revenir au vrai index)."""
    global _retriever
    _retriever = retriever


def get_retriever() -> Retriever:
    """Renvoie le retriever courant ; charge le vrai index au premier appel."""
    global _retriever
    if _retriever is None:
        _retriever = _load_index_retriever()
    return _retriever


def _load_index_retriever() -> Retriever:
    """Charge le HybridRetriever construit par `python -m fri.ingestion.indexer`."""
    from fri.config import settings
    from fri.ingestion.indexer import MANIFEST
    from fri.retrieval.hybrid import HybridRetriever

    if not (settings.index_dir / MANIFEST).is_file():
        raise RuntimeError(
            f"Index introuvable dans {settings.index_dir}. Lance d'abord "
            "`python -m fri.ingestion.indexer`, ou utilise la démo : "
            "`python -m fri.mcp_server.server --demo`."
        )
    return HybridRetriever.load(settings.index_dir)


_MIN_PREFIX = 4  # « AMD » ne doit pas valoir pour « AMDOCS »


def _same_company(indexed: str, asked: str) -> bool:
    """La société d'un passage (nom du fichier) est-elle celle demandée ?

    'AMERICANEXPRESS' = 'American Express' ; 'APPLE' = 'Apple Inc.' ;
    'JPMORGAN' = 'JPMorgan Chase & Co.' (le nom de l'index est le début du nom demandé).
    """
    a, b = company_key(indexed), company_key(asked)
    return a == b or (len(a) >= _MIN_PREFIX and b.startswith(a))


def search_filings(
    query: str, company: str | None = None, year: int | None = None, k: int = 5
) -> list[SearchResult]:
    """Renvoie les passages pertinents avec leur page, filtrés par société et année."""
    query = query.strip()
    if not query:
        raise ValueError("La question (query) est vide.")
    if k < 1:
        raise ValueError(f"k doit être >= 1, reçu {k}")
    company = company.strip() if company and company.strip() else None

    results = get_retriever().search(query, company=company, year=year, k=k)

    # Garde-fou : on ne renvoie jamais un passage d'une autre société ou d'une autre
    # année que celles demandées, même si le retriever filtre mal.
    kept = [
        r
        for r in results
        if (company is None or _same_company(r.chunk.metadata.company, company))
        and (year is None or r.chunk.metadata.year == year)
    ]
    return kept[:k]


# Les rapports n'emploient pas toujours le mot de la question : 3M écrit « Net sales »,
# pas « total revenue ». On ajoute les synonymes et le nom de l'état financier à la
# recherche pour que la ligne du tableau (compte de résultat, flux de trésorerie) remonte.
METRIC_HINTS: dict[str, str] = {
    "revenue": "total revenue net sales revenues consolidated statement of income",
    "sales": "net sales total revenue consolidated statement of income",
    "net income": "net income attributable consolidated statement of income",
    "operating income": "operating income consolidated statement of income",
    "gross profit": "gross profit gross margin cost of sales consolidated statement of income",
    "capital expenditure": "purchases of property plant and equipment capital expenditures "
    "statement of cash flows",
    "capex": "purchases of property plant and equipment capital expenditures "
    "statement of cash flows",
    "operating cash flow": "net cash provided by operating activities statement of cash flows",
    "cash from operations": "net cash provided by operating activities statement of cash flows",
    "dividend": "dividends paid statement of cash flows",
    "total assets": "total assets consolidated balance sheet",
    "inventory": "inventories consolidated balance sheet",
}


def metric_queries(metric: str) -> list[str]:
    """Requêtes pour trouver une métrique : avec synonymes d'abord, puis le mot seul."""
    lowered = metric.lower()
    hints = [h for key, h in METRIC_HINTS.items() if key in lowered]
    expanded = " ".join([metric, *dict.fromkeys(hints)]) if hints else ""
    return [q for q in (expanded, metric) if q]


def _search_metric(metric: str, company: str, year: int, k: int) -> list[SearchResult]:
    """Fusionne les résultats des requêtes de ``metric_queries``, sans doublon.

    Une métrique annuelle se lit dans le 10-K : si le rapport annuel de l'année a des
    passages, on écarte ceux des 10-Q de la même année (un trimestre n'est pas l'exercice).
    """
    merged: dict[str, SearchResult] = {}
    for query in metric_queries(metric):
        for r in search_filings(query, company=company, year=year, k=k * 2):
            merged.setdefault(r.chunk.id, r)
    results = list(merged.values())
    annual = [r for r in results if r.chunk.metadata.doc_type == DocType.TEN_K]
    return (annual or results)[:k]


# Le 10-K de l'année N donne aussi les chiffres de N-1 (et N-2 pour le compte de résultat) :
# si le rapport de l'année manque ou n'a pas la valeur, on cherche dans les suivants.
NEXT_REPORTS = 1


def get_financial_metric(
    company: str,
    year: int,
    metric: str,
    k: int = 8,
    extractor: MetricExtractor | None = None,
) -> MetricValue:
    """Renvoie une valeur chiffrée avec sa source.

    ``year`` est l'exercice de la donnée, pas forcément celui du rapport : on cherche
    d'abord dans le rapport de ``year``, puis (s'il manque ou n'a pas la valeur) dans
    le rapport suivant, qui reprend l'exercice précédent en comparaison. L'extracteur
    (le LLM par défaut) lit la valeur ; la citation pointe le passage d'où elle vient
    (donc le rapport réellement utilisé).
    Lève MetricNotFoundError si la valeur n'est pas trouvée.
    """
    from fri.agent.citations import citation_from_result

    metric = metric.strip()
    if not metric:
        raise ValueError("La métrique (metric) est vide.")
    if not company or not company.strip():
        raise ValueError("La société (company) est obligatoire.")

    extract = extractor or get_metric_extractor()
    searched = False
    for report_year in range(year, year + NEXT_REPORTS + 1):
        results = _search_metric(metric, company, report_year, k)
        if not results:
            continue
        searched = True
        found = extract(company, year, metric, results)
        if found is not None:
            break
    else:
        if not searched:
            raise MetricNotFoundError(f"Aucun passage pour {company} {year} sur « {metric} ».")
        raise MetricNotFoundError(f"« {metric} » introuvable pour {company} {year}.")

    source = results[found.passage - 1]
    return MetricValue(
        company=source.chunk.metadata.company,
        year=year,
        metric=metric,
        value=found.value,
        unit=found.unit,
        citation=citation_from_result(source),
    )


def compute_comparison(a: MetricValue, b: MetricValue) -> YearComparison:
    """Calcule l'écart et le pourcentage d'évolution entre deux valeurs."""
    delta = b.value - a.value
    pct = None if a.value == 0 else delta / abs(a.value) * 100
    return YearComparison(
        company=a.company, metric=a.metric, year_a=a, year_b=b, delta=delta, pct_change=pct
    )


def compare_years(
    company: str,
    metric: str,
    year_a: int,
    year_b: int,
    extractor: MetricExtractor | None = None,
) -> YearComparison:
    """Compare une métrique d'une société entre deux exercices (year_a = année de référence).

    Chaque valeur peut venir du rapport de son année ou du suivant (voir
    ``get_financial_metric``) : un seul 10-K suffit souvent pour deux exercices.
    """
    if year_a == year_b:
        raise ValueError("Les deux années à comparer doivent être différentes.")
    a = get_financial_metric(company, year_a, metric, extractor=extractor)
    b = get_financial_metric(company, year_b, metric, extractor=extractor)
    if a.unit.casefold() != b.unit.casefold():
        raise ValueError(f"Unités différentes : {a.unit} ({year_a}) et {b.unit} ({year_b}).")
    return compute_comparison(a, b)
