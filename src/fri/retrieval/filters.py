"""Filtres de métadonnées communs à BM25 et à la recherche vectorielle."""

from __future__ import annotations

import re
from typing import Any

from fri.ingestion.metadata import company_key
from fri.schemas import ChunkMetadata, DocType

_KNOWN = {"company", "year", "doc_type"}
_DOC_TYPE_KEY_RE = re.compile(r"[^0-9a-z]+")
_DOC_TYPES = {_DOC_TYPE_KEY_RE.sub("", t.value.casefold()): t for t in DocType}


def parse_doc_type(value: str | DocType) -> DocType:
    """'10-K', '10K', '10k' ou DocType.TEN_K -> DocType.TEN_K."""
    if isinstance(value, DocType):
        return value
    found = _DOC_TYPES.get(_DOC_TYPE_KEY_RE.sub("", str(value).casefold()))
    if found is None:
        choices = ", ".join(t.value for t in DocType)
        raise ValueError(f"Type de document inconnu : {value!r} ({choices})")
    return found


def normalize_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    """Garde seulement ``company``, ``year`` et ``doc_type`` renseignés ; les normalise.

    ``{"company": "American Express", "year": 2022, "doc_type": "10K"}``
    -> ``{"company_key": "americanexpress", "year": 2022, "doc_type": "10-K"}``.
    """
    filters = filters or {}
    unknown = set(filters) - _KNOWN
    if unknown:
        raise ValueError(f"Filtres inconnus : {sorted(unknown)} ({', '.join(sorted(_KNOWN))})")
    out: dict[str, Any] = {}
    company = filters.get("company")
    if company and company.strip():
        out["company_key"] = company_key(company)
    if filters.get("year") is not None:
        out["year"] = int(filters["year"])
    if filters.get("doc_type"):
        out["doc_type"] = parse_doc_type(filters["doc_type"]).value
    return out


def matches(metadata: ChunkMetadata, normalized: dict[str, Any]) -> bool:
    """Vrai si le chunk respecte les filtres déjà normalisés."""
    if "company_key" in normalized and company_key(metadata.company) != normalized["company_key"]:
        return False
    if "doc_type" in normalized and metadata.doc_type.value != normalized["doc_type"]:
        return False
    return "year" not in normalized or metadata.year == normalized["year"]
