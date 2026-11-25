"""Chargement de FinanceBench et séparation train / test par société.

FinanceBench numérote les pages à partir de 0 ; ``EvalItem.evidence_pages`` est converti
(+1) pour être comparé directement à nos citations.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from fri.ingestion.metadata import company_key, parse_doc_name

DEFAULT_PATH = Path("data/financebench_open_source.jsonl")


@dataclass
class EvalItem:
    question: str
    answer: str
    company: str
    doc_name: str
    evidence_pages: list[int]  # 1-indexées, comme nos citations
    id: str = ""
    year: int = 0
    question_type: str = ""  # metrics-generated | domain-relevant | novel-generated
    evidence_texts: list[str] = field(default_factory=list)
    justification: str = ""

    @property
    def doc_company(self) -> str:
        """Société telle qu'écrite dans le nom du PDF (ex. AMERICANEXPRESS)."""
        return parse_doc_name(self.doc_name)[0]


def _to_item(row: dict) -> EvalItem:
    evidence = row.get("evidence") or []
    _, year, _ = parse_doc_name(row["doc_name"])
    pages = sorted({int(e["evidence_page_num"]) + 1 for e in evidence if "evidence_page_num" in e})
    return EvalItem(
        id=row.get("financebench_id", ""),
        question=row["question"],
        answer=str(row.get("answer", "")),
        company=row["company"],
        doc_name=row["doc_name"],
        year=year,
        question_type=row.get("question_type", ""),
        evidence_pages=pages,
        evidence_texts=[e.get("evidence_text", "") for e in evidence],
        justification=row.get("justification") or "",
    )


def load_financebench(
    path: str | Path = DEFAULT_PATH, doc_names: Iterable[str] | None = None
) -> list[EvalItem]:
    """Lit le JSONL ; ``doc_names`` garde seulement les questions de ces documents."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} introuvable. Lance `python data/download_financebench.py --companies 15`."
        )
    with path.open(encoding="utf-8") as f:
        items = [_to_item(json.loads(line)) for line in f if line.strip()]
    if doc_names is not None:
        wanted = set(doc_names)
        items = [it for it in items if it.doc_name in wanted]
    return items


def indexed_documents(index_dir: str | Path) -> list[str]:
    """Les documents présents dans l'index (lus dans ``manifest.json``)."""
    from fri.ingestion.indexer import read_manifest

    return list(read_manifest(index_dir)["documents"])


def split_companies(
    items: Sequence[EvalItem], test_ratio: float = 0.3, seed: int = 42
) -> tuple[set[str], set[str]]:
    """Sépare les sociétés en train / test (graine fixe), pour ne pas tester sur des
    rapports déjà vus à l'entraînement."""
    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio doit être entre 0 et 1, reçu {test_ratio}")
    companies = sorted({company_key(it.doc_company) for it in items})
    random.Random(seed).shuffle(companies)
    n_test = max(1, round(len(companies) * test_ratio)) if len(companies) > 1 else 0
    return set(companies[n_test:]), set(companies[:n_test])


def select_split(
    items: Sequence[EvalItem], split: str, test_ratio: float = 0.3, seed: int = 42
) -> list[EvalItem]:
    """``split`` : ``all`` | ``train`` | ``test``."""
    if split == "all":
        return list(items)
    if split not in {"train", "test"}:
        raise ValueError(f"split inconnu : {split} (all, train ou test)")
    train, test = split_companies(items, test_ratio, seed)
    keep = train if split == "train" else test
    return [it for it in items if company_key(it.doc_company) in keep]


def main(argv: Sequence[str] | None = None) -> int:
    """Affiche les documents d'un split (pour ``indexer --docs``) ou un résumé."""
    import argparse

    parser = argparse.ArgumentParser(description="Questions FinanceBench et split train / test.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    parser.add_argument("--index-dir", type=Path, help="garder les documents de cet index")
    parser.add_argument("--docs", action="store_true", help="afficher seulement les documents")
    args = parser.parse_args(argv)

    items = select_split(load_financebench(args.questions), args.split)
    if args.index_dir:
        indexed = set(indexed_documents(args.index_dir))
        items = [it for it in items if it.doc_name in indexed]
    docs = sorted({it.doc_name for it in items})
    if args.docs:
        print(" ".join(docs))
        return 0
    companies = sorted({it.doc_company for it in items})
    print(
        f"split={args.split} : {len(items)} questions, {len(docs)} documents, "
        f"{len(companies)} sociétés"
    )
    print("Sociétés :", ", ".join(companies))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
