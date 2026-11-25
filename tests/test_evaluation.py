import csv
import json

import pytest

from evaluation import compare, generation_eval, retrieval_eval
from evaluation.dataset import load_financebench, select_split, split_companies
from evaluation.generation_eval import citation_precision, parse_verdict
from evaluation.retrieval_eval import (
    evaluate,
    evaluate_item,
    mean_reciprocal_rank,
    page_hit,
    recall_at_k,
    summarize,
    unique_pages,
)
from finetuning.train_embeddings import build_pairs, clean_passage
from fri.schemas import Chunk, ChunkMetadata, Citation, DocType, SearchResult
from tests.conftest import FakeEmbedder

ROWS = [
    {
        "financebench_id": "q1",
        "company": "3M",
        "doc_name": "3M_2018_10K",
        "question_type": "metrics-generated",
        "question": "What is the FY2018 total revenue for 3M?",
        "answer": "$32,765 million",
        "evidence": [{"evidence_text": "Net sales total revenue 32,765", "evidence_page_num": 1}],
    },
    {
        "financebench_id": "q2",
        "company": "American Express",
        "doc_name": "AMERICANEXPRESS_2022_10K",
        "question_type": "domain-relevant",
        "question": "What were total revenues net of interest expense in 2022?",
        "answer": "$52,862 million",
        "evidence": [
            {
                "evidence_text": "Total revenues net of interest expense 52,862",
                "evidence_page_num": 1,
            },
            {"evidence_text": "Card member loans", "evidence_page_num": 0},
        ],
    },
    {
        "financebench_id": "q3",
        "company": "AMD",
        "doc_name": "AMD_2022_10K",
        "question_type": "novel-generated",
        "question": "What drove AMD revenue growth?",
        "answer": "Xilinx",
        "evidence": [{"evidence_text": "x" * 80, "evidence_page_num": 9}],
    },
]


@pytest.fixture
def questions(tmp_path):
    path = tmp_path / "fb.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n", encoding="utf-8")
    return path


def result(doc: str, page: int, cid: str | None = None) -> SearchResult:
    company, year, _ = doc.split("_")[0], int(doc.split("_")[-2][:4]), None
    meta = ChunkMetadata(
        company=company, year=year, doc_type=DocType.TEN_K, doc_name=doc, page=page
    )
    return SearchResult(chunk=Chunk(id=cid or f"{doc}-{page}", text="t", metadata=meta), score=1)


# --------------------------------------------------------------------------- #
# Métriques de base
# --------------------------------------------------------------------------- #


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, k=2) == 0.5


def test_mrr():
    assert mean_reciprocal_rank([(["a", "b"], {"b"}), (["c"], {"c"})]) == 0.75


def test_page_hit():
    assert page_hit([3, 7, 9], [9], k=3)
    assert not page_hit([3, 7, 9], [9], k=2)


# --------------------------------------------------------------------------- #
# Jeu de données
# --------------------------------------------------------------------------- #


def test_load_financebench_converts_pages_to_1_indexed(questions):
    items = load_financebench(questions)
    assert [it.id for it in items] == ["q1", "q2", "q3"]
    assert items[0].evidence_pages == [2]  # evidence_page_num 1 -> page 2
    assert items[1].evidence_pages == [1, 2]
    assert (items[1].year, items[1].doc_company) == (2022, "AMERICANEXPRESS")
    assert items[0].evidence_texts == ["Net sales total revenue 32,765"]


def test_load_financebench_filters_documents(questions):
    assert [it.id for it in load_financebench(questions, ["AMD_2022_10K"])] == ["q3"]


def test_missing_file_explains_the_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_financebench"):
        load_financebench(tmp_path / "absent.jsonl")


def test_split_is_by_company_and_reproducible(questions):
    items = load_financebench(questions)
    train, test = split_companies(items, test_ratio=0.34)
    assert train and test and not train & test
    assert train | test == {"3m", "americanexpress", "amd"}
    assert split_companies(items, test_ratio=0.34) == (train, test)
    train_items = select_split(items, "train", test_ratio=0.34)
    test_items = select_split(items, "test", test_ratio=0.34)
    assert len(train_items) + len(test_items) == 3
    assert not {i.doc_company for i in train_items} & {i.doc_company for i in test_items}
    with pytest.raises(ValueError):
        select_split(items, "valid")


# --------------------------------------------------------------------------- #
# Évaluation du retrieval
# --------------------------------------------------------------------------- #


def test_unique_pages_keeps_rank_order():
    results = [
        result("3M_2018_10K", 5, "a"),
        result("3M_2018_10K", 5, "b"),
        result("3M_2018_10K", 2),
    ]
    assert unique_pages(results) == ["3M_2018_10K#p5", "3M_2018_10K#p2"]


def test_evaluate_item_counts_ranks_in_pages(questions):
    item = load_financebench(questions)[0]  # page de preuve : 2
    results = [
        result("3M_2018_10K", 7, "a"),
        result("3M_2018_10K", 7, "b"),
        result("3M_2018_10K", 2),
    ]
    row = evaluate_item(item, results, ks=(1, 3))
    assert row["hit@1"] == 0 and row["hit@3"] == 1
    assert row["rr"] == 0.5
    assert row["retrieved_pages"] == "7 2"


def test_same_page_of_another_document_does_not_count(questions):
    item = load_financebench(questions)[0]
    row = evaluate_item(item, [result("3M_2017_10K", 2)], ks=(1,))
    assert row["hit@1"] == 0


def test_evaluate_passes_filters_and_summarizes(questions):
    items = load_financebench(questions)[:2]
    calls = []

    def search(q, company, year, k):
        calls.append((company, year, k))
        return [result("3M_2018_10K", 2)] if company == "3M" else []

    rows = evaluate(items, search, ks=(1, 5))
    assert calls == [("3M", 2018, 15), ("AMERICANEXPRESS", 2022, 15)]
    summary = summarize(rows, ks=(1, 5))
    assert summary["questions"] == 2 and summary["Hit@1"] == 0.5 and summary["MRR"] == 0.5
    evaluate(items, search, ks=(1,), use_filters=False)
    assert calls[-1] == (None, None, 3)


@pytest.fixture
def index(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    fitz = pytest.importorskip("fitz")
    from fri.ingestion.indexer import build_index
    from fri.retrieval import vector

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    docs = {
        "3M_2018_10K": ["Risk factors.", "Net sales total revenue was $32,765 million."],
        "AMERICANEXPRESS_2022_10K": ["Card member loans.", "Total revenues net of interest 52,862"],
    }
    for name, pages in docs.items():
        doc = fitz.open()
        for text in pages:
            doc.new_page().insert_text((72, 72), text, fontsize=11)
        doc.save(pdf_dir / f"{name}.pdf")
        doc.close()
    index_dir = tmp_path / "index"
    build_index(pdf_dir, index_dir, embedding_model="fake", embedder=FakeEmbedder())
    monkeypatch.setattr(vector, "load_embedder", lambda name, prefix=None: FakeEmbedder())
    return index_dir


def test_retrieval_run_end_to_end(index, questions, tmp_path):
    out = tmp_path / "results"
    summaries = retrieval_eval.run(
        methods=["bm25", "vector", "hybrid"],
        index_dir=index,
        questions=questions,
        out_dir=out,
        tag="t",
    )
    assert set(summaries) == {"bm25", "vector", "hybrid"}
    assert summaries["hybrid"]["questions"] == 2  # q3 (AMD) n'est pas indexée
    assert summaries["bm25"]["Hit@1"] == 1.0
    with (out / "retrieval_t.csv").open(encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 6
    assert "| hybrid | 2 |" in (out / "retrieval_t_summary.md").read_text(encoding="utf-8")

    table = compare.compare(["t"], methods=["bm25"], results_dir=out)
    assert "| t | bm25 | 2 | 1.000 |" in table


# --------------------------------------------------------------------------- #
# Évaluation de la génération
# --------------------------------------------------------------------------- #


def test_citation_precision():
    cites = [Citation(company="3M", year=2018, page=p) for p in (2, 5)]
    assert citation_precision(cites, [2]) == 0.5
    assert citation_precision([], [2]) == 0.0


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"correct": true, "faithful": false, "reason": "ok"}', (True, False)),
        ('```json\n{"correct": "false", "faithful": "true"}\n```', (False, True)),
        ("pas de JSON", None),
    ],
)
def test_parse_verdict(text, expected):
    verdict = parse_verdict(text)
    assert (verdict.correct, verdict.faithful) == expected if expected else verdict is None


class FakeJudge:
    """Juge factice : « correct » si la réponse contient le chiffre attendu."""

    def __init__(self):
        self.prompts = []

    def invoke(self, messages):
        prompt = messages[0].content
        self.prompts.append(prompt)
        expected = prompt.split("Réponse attendue (référence FinanceBench) : ")[1].split("\n")[0]
        predicted = prompt.split("Réponse de l'assistant :\n")[1].split("\n")[0]
        number = expected.strip("$").split()[0]
        ok = str(number in predicted).lower()
        return f'{{"correct": {ok}, "faithful": true, "reason": "test"}}'


def fake_agent(question, company, year):
    if company == "3M":
        cites = [Citation(company="3M", year=2018, page=2, doc_name="3M_2018_10K")]
        return (
            "Revenue was $32,765 million [3M, 2018, p. 2]",
            cites,
            "simple",
            [result("3M_2018_10K", 2)],
        )
    cites = [Citation(company="AMERICANEXPRESS", year=2022, page=9)]
    return "About 50 billion [AMERICANEXPRESS, 2022, p. 9]", cites, "simple", []


def test_generation_run_writes_rows_and_resumes(index, questions, tmp_path):
    out = tmp_path / "results"
    judge = FakeJudge()
    summary = generation_eval.run(
        questions=questions,
        index_dir=index,
        out_dir=out,
        tag="t",
        agent=fake_agent,
        judge_model=judge,
    )
    assert summary["questions"] == 2
    assert summary["exactitude"] == 0.5
    assert summary["cite une page de preuve"] == 0.5
    assert len(judge.prompts) == 2
    assert "[3M_2018_10K, p. 2]" in judge.prompts[0] or "p. 2" in judge.prompts[0]

    with (out / "generation_t.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["id"] for r in rows] == ["q1", "q2"]
    assert rows[0]["correct"] == "1" and rows[1]["correct"] == "0"
    assert rows[1]["citation_precision"] == "0.0"

    # Relancer ne refait pas les questions déjà jugées.
    again = generation_eval.run(
        questions=questions,
        index_dir=index,
        out_dir=out,
        tag="t",
        agent=fake_agent,
        judge_model=judge,
    )
    assert len(judge.prompts) == 2
    assert again["exactitude"] == 0.5
    assert "exactitude" in (out / "generation_t_summary.md").read_text(encoding="utf-8")


def test_unreadable_verdict_counts_as_wrong():
    class Mute:
        calls = 0

        def invoke(self, messages):
            Mute.calls += 1
            return "je ne sais pas"

    verdict = generation_eval.judge("q", "1", "1", "", Mute(), retries=1)
    assert not verdict.correct and Mute.calls == 2


# --------------------------------------------------------------------------- #
# Fine-tuning
# --------------------------------------------------------------------------- #


def test_build_pairs_uses_every_evidence_page(questions):
    items = load_financebench(questions)
    pairs = build_pairs(items, query_prefix="Q: ", min_chars=10)
    assert ("Q: " + ROWS[0]["question"], "Net sales total revenue 32,765") in pairs
    assert sum(q.endswith("in 2022?") for q, _ in pairs) == 2
    assert all(len(p) >= 10 for _, p in pairs)


def test_clean_passage():
    assert clean_passage("  a \n\n b  ", max_chars=3) == "a b"
