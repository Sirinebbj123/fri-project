"""Évaluation du retrieval (Hit@k, Recall@k, MRR) sur les questions FinanceBench.

Méthodes : ``bm25``, ``vector``, ``hybrid`` (celle de l'agent) et ``rerank``.
Usage : ``python -m evaluation.retrieval_eval [--methods ...] [--split test] [--tag ...]``
"""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from evaluation.dataset import DEFAULT_PATH, EvalItem, load_financebench, select_split
from fri.schemas import SearchResult

RESULTS_DIR = Path(__file__).parent / "results"
METHODS = ("bm25", "vector", "hybrid", "rerank")

# search(question, company, year, k) -> résultats
SearchFn = Callable[[str, str | None, int | None, int], list[SearchResult]]


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(runs: list[tuple[list[str], set[str]]]) -> float:
    if not runs:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in runs) / len(runs)


def page_hit(retrieved_pages: list[int], evidence_pages: list[int], k: int) -> bool:
    """Vrai si une des k premières pages retrouvées est une page de preuve."""
    return bool(set(retrieved_pages[:k]) & set(evidence_pages))


def page_key(doc_name: str, page: int) -> str:
    """Identifiant d'une page : ``3M_2018_10K#p60``."""
    return f"{doc_name}#p{page}"


def unique_pages(results: Sequence[SearchResult]) -> list[str]:
    """Pages des résultats, dans l'ordre, sans doublon (plusieurs chunks par page)."""
    pages: list[str] = []
    for r in results:
        key = page_key(r.chunk.metadata.doc_name, r.chunk.metadata.page)
        if key not in pages:
            pages.append(key)
    return pages


def evaluate_item(item: EvalItem, results: Sequence[SearchResult], ks: Sequence[int]) -> dict:
    """Métriques d'une question. Les rangs sont comptés en pages distinctes."""
    retrieved = unique_pages(results)
    relevant = {page_key(item.doc_name, p) for p in item.evidence_pages}
    row: dict = {
        "id": item.id,
        "doc_name": item.doc_name,
        "question_type": item.question_type,
        "evidence_pages": " ".join(map(str, item.evidence_pages)),
        "retrieved_pages": " ".join(
            r.split("#p")[1] if r.startswith(item.doc_name) else r for r in retrieved[: max(ks)]
        ),
        "rr": round(reciprocal_rank(retrieved, relevant), 4),
    }
    for k in ks:
        row[f"hit@{k}"] = int(bool(set(retrieved[:k]) & relevant))
        row[f"recall@{k}"] = round(recall_at_k(retrieved, relevant, k), 4)
    return row


def summarize(rows: Sequence[dict], ks: Sequence[int]) -> dict:
    n = len(rows)
    summary: dict = {"questions": n, "MRR": sum(r["rr"] for r in rows) / n if n else 0.0}
    for k in ks:
        summary[f"Hit@{k}"] = sum(r[f"hit@{k}"] for r in rows) / n if n else 0.0
        summary[f"Recall@{k}"] = sum(r[f"recall@{k}"] for r in rows) / n if n else 0.0
    return summary


def evaluate(
    items: Sequence[EvalItem],
    search: SearchFn,
    ks: Sequence[int] = (1, 3, 5, 10),
    use_filters: bool = True,
) -> list[dict]:
    """Rejoue chaque question. ``use_filters`` : filtre société + année du document,
    comme quand l'utilisateur les choisit dans l'interface."""
    depth = max(ks)
    rows = []
    for item in items:
        company = item.doc_company if use_filters else None
        year = item.year if use_filters else None
        start = time.perf_counter()
        results = search(item.question, company, year, depth * 3)  # plusieurs chunks par page
        row = evaluate_item(item, results, ks)
        row["seconds"] = round(time.perf_counter() - start, 3)
        rows.append(row)
    return rows


def build_searchers(index_dir: Path, methods: Sequence[str]) -> dict[str, SearchFn]:
    """Charge l'index une seule fois et prépare une fonction de recherche par méthode."""
    from fri.retrieval.hybrid import HybridRetriever
    from fri.retrieval.reranker import Reranker

    base = HybridRetriever.load(index_dir, use_reranker=False)

    def single(retriever) -> SearchFn:
        def search(q, company, year, k):
            filters = {"company": company, "year": year}
            return retriever.search(q, k=k, filters={f: v for f, v in filters.items() if v})

        return search

    searchers: dict[str, SearchFn] = {}
    for method in methods:
        if method == "bm25":
            searchers[method] = single(base.bm25)
        elif method == "vector":
            searchers[method] = single(base.vector)
        elif method == "hybrid":
            searchers[method] = lambda q, c, y, k: base.search(q, company=c, year=y, k=k)
        elif method == "rerank":
            from fri.config import settings

            reranked = HybridRetriever(
                base.bm25, base.vector, reranker=Reranker(settings.reranker_model)
            )
            reranked.candidates = 30
            searchers[method] = lambda q, c, y, k, r=reranked: r.search(q, company=c, year=y, k=k)
        else:
            raise ValueError(f"Méthode inconnue : {method} ({', '.join(METHODS)})")
    return searchers


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(summaries: dict[str, dict], ks: Sequence[int]) -> str:
    cols = ["MRR"] + [f"Hit@{k}" for k in ks] + [f"Recall@{k}" for k in ks]
    lines = [
        "| Méthode | Questions | " + " | ".join(cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    for method, s in summaries.items():
        values = " | ".join(f"{s[c]:.3f}" for c in cols)
        lines.append(f"| {method} | {s['questions']} | {values} |")
    return "\n".join(lines)


def run(
    k: int = 5,
    *,
    methods: Sequence[str] = ("bm25", "vector", "hybrid"),
    index_dir: str | Path | None = None,
    questions: str | Path = DEFAULT_PATH,
    split: str = "all",
    limit: int | None = None,
    use_filters: bool = True,
    tag: str = "baseline",
    out_dir: Path = RESULTS_DIR,
) -> dict[str, dict]:
    """Évalue chaque méthode et écrit les résultats ; renvoie le résumé par méthode."""
    from evaluation.dataset import indexed_documents
    from fri.config import settings

    index_dir = Path(index_dir or settings.index_dir)
    ks = sorted({1, 3, k, 10})
    docs = indexed_documents(index_dir)
    # Séparation calculée sur les 150 questions : la même quel que soit l'index.
    items = [it for it in select_split(load_financebench(questions), split) if it.doc_name in docs]
    items = items[:limit] if limit else items
    if not items:
        raise RuntimeError("Aucune question FinanceBench ne porte sur les documents indexés.")
    print(f"{len(items)} questions sur {len(docs)} documents indexés (split={split})")

    searchers = build_searchers(index_dir, methods)
    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for method, search in searchers.items():
        print(f"- {method}...", flush=True)
        rows = evaluate(items, search, ks, use_filters=use_filters)
        for row in rows:
            row["method"] = method
        all_rows.extend(rows)
        summaries[method] = summarize(rows, ks)

    write_csv(out_dir / f"retrieval_{tag}.csv", all_rows)
    table = markdown_table(summaries, ks)
    header = (
        f"# Retrieval : {tag}\n\nIndex : `{index_dir}` ; split : {split} ; "
        f"filtres société/année : {'oui' if use_filters else 'non'}.\n\n"
    )
    (out_dir / f"retrieval_{tag}_summary.md").write_text(header + table + "\n", encoding="utf-8")
    print()
    print(table)
    print(f"\nRésultats : {out_dir / f'retrieval_{tag}.csv'}")
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Évalue le retrieval sur FinanceBench.")
    parser.add_argument("-k", type=int, default=5, help="k principal (en plus de 1, 3 et 10)")
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=["bm25", "vector", "hybrid"]
    )
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--questions", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    parser.add_argument("--limit", type=int, help="seulement les N premières questions (essai)")
    parser.add_argument("--no-filters", action="store_true", help="sans filtre société / année")
    parser.add_argument("--tag", default="baseline", help="nom des fichiers de résultats")
    args = parser.parse_args(argv)
    run(
        args.k,
        methods=args.methods,
        index_dir=args.index_dir,
        questions=args.questions,
        split=args.split,
        limit=args.limit,
        use_filters=not args.no_filters,
        tag=args.tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
