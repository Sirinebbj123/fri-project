"""Évaluation des réponses de l'agent par un LLM juge : exactitude, fidélité, citations.

Usage : ``python -m evaluation.generation_eval [--limit N] [--pause S]``.
Le CSV est écrit au fil de l'eau : relancer la commande reprend où elle s'était arrêtée.
Le juge peut utiliser un autre modèle (``JUDGE_PROVIDER`` / ``JUDGE_MODEL``).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.dataset import DEFAULT_PATH, EvalItem, load_financebench, select_split
from fri.schemas import Citation, SearchResult

RESULTS_DIR = Path(__file__).parent / "results"

JUDGE_PROMPT = """Tu évalues la réponse d'un assistant financier.

Question : {question}

Réponse attendue (référence FinanceBench) : {expected}

Passages donnés à l'assistant :
{context}

Réponse de l'assistant :
{predicted}

Réponds uniquement en JSON :
{{"correct": true/false, "faithful": true/false, "reason": "<une phrase>"}}
- "correct" : true si la réponse de l'assistant donne la même information que la réponse
  attendue (un arrondi ou une autre unité équivalente est accepté ; un chiffre différent,
  une réponse vide ou « je ne sais pas » = false).
- "faithful" : true si tout ce qu'affirme l'assistant est appuyé par les passages
  ci-dessus (même si la réponse est fausse) ; false s'il invente un chiffre ou un fait."""


def citation_precision(cited: list[Citation], evidence_pages: list[int]) -> float:
    """Part des citations qui pointent vers une page de preuve."""
    if not cited:
        return 0.0
    return sum(c.page in evidence_pages for c in cited) / len(cited)


@dataclass
class Verdict:
    correct: bool
    faithful: bool
    reason: str = ""


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "oui", "1"}
    return bool(value)


def parse_verdict(text: str) -> Verdict | None:
    from fri.agent.metrics import parse_json_reply

    data = parse_json_reply(text)
    if not data or "correct" not in data:
        return None
    return Verdict(
        correct=_as_bool(data.get("correct")),
        faithful=_as_bool(data.get("faithful", False)),
        reason=str(data.get("reason", ""))[:300],
    )


def judge(
    question: str,
    expected: str,
    predicted: str,
    context: str,
    llm: Any,
    retries: int = 2,
) -> Verdict:
    """Demande au LLM juge son verdict. Réponse illisible = faux (et on le signale)."""
    from langchain_core.messages import HumanMessage

    from fri.agent.metrics import message_text

    prompt = JUDGE_PROMPT.format(
        question=question, expected=expected, predicted=predicted, context=context or "(aucun)"
    )
    for _ in range(retries + 1):
        verdict = parse_verdict(message_text(llm.invoke([HumanMessage(prompt)])))
        if verdict is not None:
            return verdict
    return Verdict(False, False, "verdict du juge illisible")


def judge_correctness(question: str, expected: str, predicted: str, llm: Any = None) -> bool:
    """Exactitude seule (sans passages), avec le LLM juge."""
    return judge(question, expected, predicted, "", llm or judge_llm()).correct


def judge_llm() -> Any:
    from fri.agent.llm import get_chat_model

    return get_chat_model(os.getenv("JUDGE_PROVIDER") or None, os.getenv("JUDGE_MODEL") or None)


# agent(question, company, year) -> (answer, citations, route, passages)
AgentFn = Callable[
    [str, str | None, int | None], tuple[str, list[Citation], str, list[SearchResult]]
]


def default_agent(llm: Any = None) -> AgentFn:
    """L'agent LangGraph ; renvoie aussi les passages reçus (pour juger la fidélité)."""
    from fri.agent.graph import build_graph

    graph = build_graph(llm=llm)

    def agent(question, company, year):
        state = graph.invoke({"question": question, "company": company, "year": year})
        answer = state["answer"]
        results = state.get("results") or []  # vide si la réponse vient de compare_years
        return answer.answer, list(answer.citations), answer.route or "", results

    return agent


FIELDS = [
    "id",
    "doc_name",
    "question_type",
    "route",
    "correct",
    "faithful",
    "cites_evidence",
    "citation_precision",
    "n_citations",
    "citations",
    "evidence_pages",
    "seconds",
    "question",
    "expected",
    "predicted",
    "reason",
]


def evaluate_item(item: EvalItem, agent: AgentFn, judge_model: Any, use_filters: bool) -> dict:
    from fri.agent.citations import format_context

    start = time.perf_counter()
    company = item.doc_company if use_filters else None
    year = item.year if use_filters else None
    text, citations, route, results = agent(item.question, company, year)
    seconds = time.perf_counter() - start
    verdict = judge(item.question, item.answer, text, format_context(results), judge_model)
    on_doc = [c for c in citations if c.doc_name in (None, item.doc_name)]
    return {
        "id": item.id,
        "doc_name": item.doc_name,
        "question_type": item.question_type,
        "route": route,
        "correct": int(verdict.correct),
        "faithful": int(verdict.faithful),
        "cites_evidence": int(any(c.page in item.evidence_pages for c in on_doc)),
        "citation_precision": round(citation_precision(on_doc, item.evidence_pages), 4)
        if citations
        else "",
        "n_citations": len(citations),
        "citations": " ".join(str(c) for c in citations),
        "evidence_pages": " ".join(map(str, item.evidence_pages)),
        "seconds": round(seconds, 2),
        "question": item.question,
        "expected": item.answer,
        "predicted": text,
        "reason": verdict.reason,
    }


def read_done(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows: Sequence[dict]) -> dict:
    n = len(rows)

    def mean(key: str, subset: Sequence[dict] | None = None) -> float:
        values = [float(r[key]) for r in (subset or rows) if str(r[key]) != ""]
        return sum(values) / len(values) if values else 0.0

    summary = {
        "questions": n,
        "exactitude": mean("correct"),
        "fidélité": mean("faithful"),
        "cite une page de preuve": mean("cites_evidence"),
        "précision des citations": mean("citation_precision"),
        "secondes / question": mean("seconds"),
    }
    by_type: dict[str, float] = {}
    for qtype in sorted({r["question_type"] for r in rows}):
        subset = [r for r in rows if r["question_type"] == qtype]
        by_type[f"exactitude ({qtype}, {len(subset)})"] = mean("correct", subset)
    return {**summary, **by_type}


def markdown_summary(summary: dict) -> str:
    lines = ["| Métrique | Valeur |", "|---|---|"]
    for key, value in summary.items():
        shown = (
            str(value)
            if key == "questions"
            else (f"{value:.1f}" if key.startswith("secondes") else f"{value:.1%}")
        )
        lines.append(f"| {key} | {shown} |")
    return "\n".join(lines)


def run(
    *,
    questions: str | Path = DEFAULT_PATH,
    index_dir: str | Path | None = None,
    split: str = "all",
    limit: int | None = None,
    use_filters: bool = True,
    tag: str = "baseline",
    out_dir: Path = RESULTS_DIR,
    agent: AgentFn | None = None,
    judge_model: Any = None,
    pause: float = 0.0,
) -> dict:
    """Rejoue les questions sur l'agent, fait juger chaque réponse, écrit le CSV."""
    from evaluation.dataset import indexed_documents
    from fri.config import settings

    index_dir = Path(index_dir or settings.index_dir)
    docs = indexed_documents(index_dir)
    items = [it for it in select_split(load_financebench(questions), split) if it.doc_name in docs]
    items = items[:limit] if limit else items
    path = out_dir / f"generation_{tag}.csv"
    done = read_done(path)
    done_ids = {r["id"] for r in done}
    todo = [it for it in items if it.id not in done_ids]
    print(
        f"{len(items)} questions, {len(done_ids & {it.id for it in items})} déjà faites, "
        f"{len(todo)} à faire -> {path}"
    )

    if agent is None:
        from fri.mcp_server import tools
        from fri.retrieval.hybrid import HybridRetriever

        tools.set_retriever(HybridRetriever.load(index_dir))  # l'index choisi, pas celui du .env
        agent = default_agent()
    judge_model = judge_model or judge_llm()
    out_dir.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, item in enumerate(todo, start=1):
            row = evaluate_item(item, agent, judge_model, use_filters)
            writer.writerow(row)
            f.flush()  # écrit tout de suite : rien n'est perdu si le script s'arrête
            done.append({k: str(v) for k, v in row.items()})
            status = "OK " if row["correct"] else "NON"
            print(f"[{i}/{len(todo)}] {status} {item.id} ({row['seconds']} s)", flush=True)
            if pause:
                time.sleep(pause)

    wanted = {it.id for it in items}
    rows = [r for r in done if r["id"] in wanted]
    summary = summarize(rows)
    table = markdown_summary(summary)
    (out_dir / f"generation_{tag}_summary.md").write_text(
        f"# Génération : {tag}\n\nSplit : {split}.\n\n{table}\n", encoding="utf-8"
    )
    print()
    print(table)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Évalue les réponses de l'agent (LLM-as-judge).")
    parser.add_argument("--questions", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    parser.add_argument("--limit", type=int, help="seulement les N premières questions")
    parser.add_argument("--no-filters", action="store_true", help="sans filtre société / année")
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--pause", type=float, default=0.0, help="secondes entre deux questions")
    args = parser.parse_args(argv)
    run(
        questions=args.questions,
        index_dir=args.index_dir,
        split=args.split,
        limit=args.limit,
        use_filters=not args.no_filters,
        tag=args.tag,
        pause=args.pause,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
