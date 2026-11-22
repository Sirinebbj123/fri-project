"""Nom de société souple, trimestre des 10-Q et filtre par type de document."""

import pytest

from fri.ingestion.chunking import chunk_page, search_text
from fri.ingestion.metadata import company_key, parse_doc_info, parse_doc_name, resolve_company
from fri.ingestion.pdf_parser import Page
from fri.retrieval.bm25 import BM25Retriever
from fri.retrieval.filters import matches, normalize_filters, parse_doc_type
from fri.retrieval.hybrid import HybridRetriever
from fri.retrieval.vector import chunk_to_record, record_to_chunk
from fri.schemas import DocType

# --------------------------------------------------------------------------- #
# Nom de société : « 3M Company » = « 3M »
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "asked, indexed",
    [
        ("3M Company", "3M"),
        ("Apple Inc.", "APPLE"),
        ("AES Corporation", "AES"),
        ("The AES Corporation", "AES"),
        ("The Coca-Cola Company", "COCACOLA"),
        ("Pfizer Inc", "PFIZER"),
        ("Johnson & Johnson", "JOHNSON_JOHNSON"),
    ],
)
def test_company_key_ignores_legal_words(asked, indexed):
    assert company_key(asked) == company_key(indexed)


def test_company_key_keeps_a_name_made_only_of_legal_words():
    assert company_key("The Company") == "thecompany"


def test_company_key_unchanged_for_file_names():
    # Les clés déjà stockées dans Chroma (noms de fichiers) ne changent pas : pas de réindexation.
    for name in ("3M", "AMERICANEXPRESS", "JOHNSON_JOHNSON", "PEPSICO", "AES"):
        assert company_key(name) == "".join(c for c in name.casefold() if c.isalnum())


KNOWN = ["3M", "APPLE", "AES", "AMD", "JPMORGAN", "COSTCO", "AMERICANEXPRESS"]


@pytest.mark.parametrize(
    "asked, expected",
    [
        ("Apple Inc.", "APPLE"),
        ("3M Company", "3M"),
        ("JPMorgan Chase & Co.", "JPMORGAN"),
        ("Costco Wholesale Corporation", "COSTCO"),
        ("American Express Company", "AMERICANEXPRESS"),
        ("AMD", "AMD"),
    ],
)
def test_resolve_company_against_indexed_names(asked, expected):
    assert resolve_company(asked, KNOWN) == expected


def test_resolve_company_short_names_are_not_prefixes():
    # « AMD » (3 lettres) ne doit pas capter « AMDOCS » ; le nom inconnu est renvoyé tel quel.
    assert resolve_company("Amdocs Limited", KNOWN) == "Amdocs Limited"
    assert resolve_company("Netflix", KNOWN) == "Netflix"


# --------------------------------------------------------------------------- #
# Trimestre et type de document
# --------------------------------------------------------------------------- #


def test_parse_doc_info_keeps_quarter():
    assert parse_doc_info("PEPSICO_2023Q1_10Q") == ("PEPSICO", 2023, 1, DocType.TEN_Q)
    assert parse_doc_info("PEPSICO_2022_10K") == ("PEPSICO", 2022, None, DocType.TEN_K)
    # parse_doc_name garde sa signature (compatibilité)
    assert parse_doc_name("PEPSICO_2023Q1_10Q") == ("PEPSICO", 2023, DocType.TEN_Q)


def test_chunk_metadata_has_quarter_and_type():
    chunk = chunk_page(Page(doc_name="PEPSICO_2023Q1_10Q", page=3, text="Net revenue"))[0]
    assert (chunk.metadata.quarter, chunk.metadata.doc_type) == (1, DocType.TEN_Q)
    assert search_text(chunk) == "PEPSICO 2023 Q1 10-Q\nNet revenue"


def test_chroma_record_round_trip_with_and_without_quarter():
    for name in ("PEPSICO_2023Q1_10Q", "PEPSICO_2022_10K"):
        chunk = chunk_page(Page(doc_name=name, page=1, text="x"))[0]
        record = chunk_to_record(chunk)
        assert None not in record.values()  # Chroma refuse None
        assert record_to_chunk(chunk.id, "x", record).metadata == chunk.metadata


def test_old_chroma_record_without_quarter_still_loads():
    record = {"company": "3M", "year": 2018, "doc_type": "10-K", "doc_name": "3M_2018_10K",
              "page": 2}  # fmt: skip
    assert record_to_chunk("id", "x", record).metadata.quarter is None


@pytest.mark.parametrize("value", ["10-K", "10K", "10k", DocType.TEN_K])
def test_parse_doc_type(value):
    assert parse_doc_type(value) is DocType.TEN_K


def test_parse_doc_type_rejects_unknown():
    with pytest.raises(ValueError):
        parse_doc_type("20-F")


def test_normalize_filters_with_doc_type():
    assert normalize_filters({"company": "3M Company", "year": 2023, "doc_type": "10q"}) == {
        "company_key": "3m",
        "year": 2023,
        "doc_type": "10-Q",
    }


PEPSI = [
    chunk_page(Page(doc_name="PEPSICO_2023Q1_10Q", page=5, text="Net revenue quarter"))[0],
    chunk_page(Page(doc_name="PEPSICO_2023_10K", page=40, text="Net revenue full year"))[0],
]


def test_matches_filters_by_doc_type():
    annual = normalize_filters({"year": 2023, "doc_type": "10-K"})
    assert [matches(c.metadata, annual) for c in PEPSI] == [False, True]


def test_bm25_annual_question_skips_10q():
    other = chunk_page(Page(doc_name="3M_2023_10K", page=1, text="Capital expenditures"))
    bm25 = BM25Retriever(PEPSI + other * 3)  # BM25 a besoin d'autres documents (IDF)
    assert len(bm25.search("net revenue", filters={"company": "PepsiCo, Inc.", "year": 2023})) == 2
    results = bm25.search("net revenue", filters={"year": 2023, "doc_type": "10-K"})
    assert [r.chunk.metadata.doc_name for r in results] == ["PEPSICO_2023_10K"]


class RecordingRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def search(self, query, k=8, filters=None):
        self.calls.append(filters)
        return []


def test_hybrid_resolves_company_and_passes_doc_type():
    bm25 = RecordingRetriever(PEPSI + [chunk_page(Page(doc_name="JPMORGAN_2022_10K", page=1,
                                                       text="x"))[0]])  # fmt: skip
    vec = RecordingRetriever([])
    hybrid = HybridRetriever(bm25, vec)
    hybrid.search("revenue", company="JPMorgan Chase & Co.", year=2022, doc_type="10-K")
    assert bm25.calls[-1] == {"company": "JPMORGAN", "year": 2022, "doc_type": "10-K"}
    assert vec.calls[-1] == bm25.calls[-1]
