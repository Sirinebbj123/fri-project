"""compare_years sur l'année des données, 10-K avant 10-Q, unités (EPS, %), noms de société."""

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from fri.agent.metrics import LLMMetricExtractor, normalize_unit  # noqa: E402
from fri.agent.prompts import METRIC_PROMPT  # noqa: E402
from fri.mcp_server import tools  # noqa: E402
from fri.mcp_server.demo import InMemoryRetriever  # noqa: E402
from fri.schemas import Chunk, ChunkMetadata, Citation, DocType, MetricValue  # noqa: E402


def chunk(doc_name: str, page: int, text: str, doc_type=DocType.TEN_K) -> Chunk:
    company, year = doc_name.split("_")[:2]
    meta = ChunkMetadata(
        company=company, year=int(year[:4]), doc_type=doc_type, doc_name=doc_name, page=page
    )
    return Chunk(id=f"{doc_name}-p{page}", text=text, metadata=meta)


def llm_extractor(*replies: str) -> LLMMetricExtractor:
    return LLMMetricExtractor(FakeListChatModel(responses=list(replies)))


@pytest.fixture(autouse=True)
def reset_tools():
    yield
    tools.set_retriever(None)
    tools.set_metric_extractor(None)


# --- l'année des données, pas celle du rapport ---------------------------------------

ONLY_2018_REPORT = [
    chunk(
        "3M_2018_10K", 42, "Net sales total revenue: 2018 $32,765 million; 2017 $31,657 million."
    ),
]


def test_compare_years_with_only_the_later_report():
    tools.set_retriever(InMemoryRetriever(ONLY_2018_REPORT))
    ext = llm_extractor(
        '{"value": 31657, "unit": "USD millions", "passage": 1}',  # 2017 lu dans le 10-K 2018
        '{"value": 32765, "unit": "USD millions", "passage": 1}',
    )
    comp = tools.compare_years("3M", "total revenue", 2017, 2018, extractor=ext)
    assert (comp.year_a.year, comp.year_a.value) == (2017, 31657.0)
    assert comp.delta == 1108.0
    # la citation dit d'où vient vraiment la valeur : le rapport 2018
    assert (comp.year_a.citation.year, comp.year_a.citation.doc_name) == (2018, "3M_2018_10K")


def test_next_report_is_used_when_own_report_lacks_the_value():
    tools.set_retriever(
        InMemoryRetriever([chunk("3M_2017_10K", 5, "Total revenue overview."), *ONLY_2018_REPORT])
    )
    ext = llm_extractor(
        '{"value": null, "unit": "USD millions", "passage": 1}',  # pas dans le 10-K 2017
        '{"value": 31657, "unit": "USD millions", "passage": 1}',  # trouvé dans le 10-K 2018
    )
    value = tools.get_financial_metric("3M", 2017, "total revenue", extractor=ext)
    assert (value.year, value.value, value.citation.year) == (2017, 31657.0, 2018)


def test_own_report_comes_first():
    tools.set_retriever(
        InMemoryRetriever(
            [chunk("3M_2017_10K", 40, "Total revenue 2017 $31,657 million."), *ONLY_2018_REPORT]
        )
    )
    ext = llm_extractor('{"value": 31657, "unit": "USD millions", "passage": 1}')
    value = tools.get_financial_metric("3M", 2017, "total revenue", extractor=ext)
    assert value.citation.doc_name == "3M_2017_10K"


def test_still_raises_when_no_report_has_the_value():
    tools.set_retriever(InMemoryRetriever(ONLY_2018_REPORT))
    ext = llm_extractor('{"value": null, "unit": "USD millions", "passage": 1}')
    with pytest.raises(tools.MetricNotFoundError):
        tools.get_financial_metric("3M", 2016, "total revenue", extractor=ext)


# --- 10-K avant 10-Q ------------------------------------------------------------------


def test_annual_metric_ignores_10q_of_same_year():
    tools.set_retriever(
        InMemoryRetriever(
            [
                chunk(
                    "PEPSICO_2023Q1_10Q", 3, "Net revenue quarter $17,846 million.", DocType.TEN_Q
                ),
                chunk("PEPSICO_2023_10K", 40, "Net revenue full year $91,471 million."),
            ]
        )
    )
    seen = []

    def extractor(company, year, metric, results):
        seen.extend(r.chunk.metadata.doc_name for r in results)
        return None

    with pytest.raises(tools.MetricNotFoundError):
        tools.get_financial_metric("PEPSICO", 2023, "net revenue", extractor=extractor)
    assert "PEPSICO_2023Q1_10Q" not in seen and "PEPSICO_2023_10K" in seen


def test_10q_is_used_when_no_10k():
    tools.set_retriever(
        InMemoryRetriever(
            [chunk("PEPSICO_2023Q1_10Q", 3, "Net revenue $17,846 million.", DocType.TEN_Q)]
        )
    )
    ext = llm_extractor('{"value": 17846, "unit": "USD millions", "passage": 1}')
    assert tools.get_financial_metric("PEPSICO", 2023, "net revenue", extractor=ext).value == 17846


# --- unités ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, unit, metric, expected",
    [
        (5.61, "USD per share", "diluted EPS", (5.61, "USD per share")),
        (5.61, "$/share", "EPS", (5.61, "USD per share")),
        (5.61, None, "diluted EPS", (5.61, "USD per share")),
        (21.4, "%", "operating margin", (21.4, "%")),
        (21.4, None, "gross margin", (21.4, "%")),
        (550.0, "millions of shares", "shares outstanding", (550.0, "millions of shares")),
        (1.4, "ratio", "current ratio", (1.4, "ratio")),
        (3.2, "USD billions", "revenue", (3200.0, "USD millions")),
        (32765.0, "USD millions", "revenue", (32765.0, "USD millions")),
        (32765.0, None, "net deposits", (32765.0, "USD millions")),
    ],
)
def test_normalize_unit(value, unit, metric, expected):
    assert normalize_unit(value, unit, metric) == expected


def test_metric_prompt_no_longer_forces_millions():
    assert "USD per share" in METRIC_PROMPT and '"%"' in METRIC_PROMPT
    assert '"unit": "USD millions"' not in METRIC_PROMPT


def test_eps_is_not_converted_to_millions():
    tools.set_retriever(InMemoryRetriever([chunk("3M_2018_10K", 50, "Diluted EPS was $8.89.")]))
    ext = llm_extractor('{"value": 8.89, "unit": "USD per share", "passage": 1}')
    value = tools.get_financial_metric("3M", 2018, "diluted EPS", extractor=ext)
    assert (value.value, value.unit) == (8.89, "USD per share")


def _mv(year, value, unit):
    return MetricValue(
        company="3M",
        year=year,
        metric="m",
        value=value,
        unit=unit,
        citation=Citation(company="3M", year=year, page=1),
    )


def test_format_comparison_eps_keeps_cents():
    pytest.importorskip("langgraph")
    from fri.agent.graph import format_comparison

    text = format_comparison(
        tools.compute_comparison(_mv(2017, 7.93, "USD per share"), _mv(2018, 8.89, "USD per share"))
    )
    assert "7,93 USD per share" in text and "8,89 USD per share" in text
    assert "+0,96 USD per share" in text


def test_format_comparison_percent_in_points():
    pytest.importorskip("langgraph")
    from fri.agent.graph import format_comparison

    text = format_comparison(tools.compute_comparison(_mv(2017, 20.0, "%"), _mv(2018, 22.5, "%")))
    assert "Écart : +2,5 points." in text


# --- nom de société -------------------------------------------------------------------


@pytest.mark.parametrize(
    "indexed, asked, same",
    [
        ("AMERICANEXPRESS", "American Express", True),
        ("APPLE", "Apple Inc.", True),
        ("JPMORGAN", "JPMorgan Chase & Co.", True),
        ("AMD", "AMDOCS", False),  # trop court pour servir de préfixe
        ("3M", "AMD", False),
    ],
)
def test_same_company(indexed, asked, same):
    assert tools._same_company(indexed, asked) is same


# --- affichage du client MCP ----------------------------------------------------------


def test_client_shows_eps_with_cents_and_its_unit():
    from fri.mcp_server.client import Call, summarize

    cite = {"company": "3M", "year": 2018, "page": 16}
    eps = {"value": 8.89, "unit": "USD per share", "citation": cite}
    lines = summarize(Call("get_financial_metric", {}, {"metric": "diluted EPS", **eps}, None, 1.0))
    assert lines == ["diluted EPS = 8,89 $ par action  [3M, 2018, p. 16]"]

    comp = {
        "year_a": {"year": 2017, "value": 7.93, "unit": "USD per share", "citation": cite},
        "year_b": {"year": 2018, "value": 8.89, "unit": "USD per share", "citation": cite},
        "delta": 0.96,
        "pct_change": 12.1,
    }
    lines = summarize(Call("compare_years", {}, comp, None, 1.0))
    assert lines[0].startswith("2017 : 7,93 $ par action")
    assert lines[2] == "écart : +0,96 $ par action  (+12.1 %)"


def test_client_still_shows_amounts_in_millions():
    from fri.mcp_server.client import _amount

    assert _amount(32765.0, "USD millions") == "32 765 M$"
    assert _amount(1108.0, None, sign=True) == "+1 108 M$"
    assert _amount(2.5, "%", sign=True) == "+2,5 points"
