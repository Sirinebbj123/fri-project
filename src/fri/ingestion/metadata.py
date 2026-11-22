"""Déduction des métadonnées (société, année, trimestre, type) depuis le nom du document."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import NamedTuple

from fri.schemas import DocType

_DOC_RE = re.compile(
    r"^(?P<company>.+?)_(?P<year>\d{4})(?:Q(?P<quarter>[1-4]))?_(?P<type>[0-9A-Z]+)", re.I
)
_NOT_ALNUM_RE = re.compile(r"[^0-9a-z]+")

_TYPES = {
    "10K": DocType.TEN_K,
    "10Q": DocType.TEN_Q,
    "8K": DocType.EIGHT_K,
    "EARNINGS": DocType.EARNINGS,
}

# Formes juridiques et articles ignorés pour comparer les noms de sociétés :
# « 3M Company » = « 3M », « Apple Inc. » = « APPLE », « The AES Corporation » = « AES ».
_LEGAL_WORDS = frozenset(
    {
        "the",
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "company",
        "co",
        "ltd",
        "limited",
        "plc",
        "llc",
    }
)

# Un nom de l'index plus court que ça n'est pas accepté comme préfixe (« AMD » ne doit
# pas capter « AMDOCS ») : il faut alors le nom exact.
_MIN_PREFIX = 4


class DocInfo(NamedTuple):
    company: str
    year: int
    quarter: int | None  # 1 à 4 pour un 10-Q (``PEPSICO_2023Q1_10Q``), sinon None
    doc_type: DocType


def parse_doc_info(doc_name: str) -> DocInfo:
    """Ex. 'PEPSICO_2023Q1_10Q' -> DocInfo('PEPSICO', 2023, 1, DocType.TEN_Q)."""
    match = _DOC_RE.match(doc_name)
    if not match:
        raise ValueError(f"Nom de document inattendu : {doc_name}")
    doc_type = _TYPES.get(match["type"].upper(), DocType.OTHER)
    quarter = int(match["quarter"]) if match["quarter"] else None
    return DocInfo(match["company"], int(match["year"]), quarter, doc_type)


def parse_doc_name(doc_name: str) -> tuple[str, int, DocType]:
    """Ex. '3M_2018_10K' -> ('3M', 2018, DocType.TEN_K). Le trimestre : ``parse_doc_info``."""
    info = parse_doc_info(doc_name)
    return info.company, info.year, info.doc_type


def company_key(company: str) -> str:
    """Forme normalisée d'un nom de société, pour les filtres.

    Les noms de fichiers FinanceBench collent les mots (``AMERICANEXPRESS``,
    ``JOHNSON_JOHNSON``) alors que les questions les écrivent normalement
    (``American Express``, ``Johnson & Johnson``, ``Apple Inc.``) : on enlève les
    formes juridiques (Inc., Corp., Company, The…), puis on garde seulement les
    lettres et les chiffres, en minuscules.
    Ex. ``"Johnson & Johnson" -> "johnsonjohnson"`` ; ``"3M Company" -> "3m"``.
    """
    words = [w for w in _NOT_ALNUM_RE.split(company.casefold()) if w]
    kept = [w for w in words if w not in _LEGAL_WORDS]
    return "".join(kept or words)


def resolve_company(company: str, known: Iterable[str]) -> str:
    """Nom de société tel qu'il est dans l'index, à partir du nom écrit dans la question.

    1. même clé (``company_key``) : « Apple Inc. » -> « APPLE » ;
    2. sinon, un seul nom de l'index est le début du nom demandé :
       « JPMorgan Chase & Co. » -> « JPMORGAN », « Costco Wholesale » -> « COSTCO ».
    Si rien ne correspond (ou si c'est ambigu), on renvoie le nom tel quel : le filtre
    ne trouvera alors rien, comme avant.
    """
    asked = company_key(company)
    names = {company_key(name): name for name in known}
    if asked in names:
        return names[asked]
    prefixes = [k for k in names if len(k) >= _MIN_PREFIX and asked.startswith(k)]
    if prefixes:
        best = max(prefixes, key=len)
        if sum(len(k) == len(best) for k in prefixes) == 1:
            return names[best]
    return company
