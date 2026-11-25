"""Compare plusieurs évaluations du retrieval (ex. avant / après fine-tuning).

Usage : ``python -m evaluation.compare baseline-test finetune-test``
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
COLUMNS = ("MRR", "Hit@1", "Hit@5", "Hit@10", "Recall@5")


def load_rows(tag: str, results_dir: Path = RESULTS_DIR) -> list[dict]:
    path = results_dir / f"retrieval_{tag}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} introuvable : lance d'abord retrieval_eval --tag {tag}")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows: Sequence[dict]) -> dict[str, float]:
    n = len(rows)

    def mean(key: str) -> float:
        return sum(float(r[key]) for r in rows) / n if n else 0.0

    return {
        "questions": n,
        "MRR": mean("rr"),
        "Hit@1": mean("hit@1"),
        "Hit@5": mean("hit@5"),
        "Hit@10": mean("hit@10"),
        "Recall@5": mean("recall@5"),
    }


def compare(
    tags: Sequence[str], methods: Sequence[str] | None = None, results_dir: Path = RESULTS_DIR
) -> str:
    lines = [
        "| Évaluation | Méthode | Questions | " + " | ".join(COLUMNS) + " |",
        "|---|---|---|" + "---|" * len(COLUMNS),
    ]
    for tag in tags:
        rows = load_rows(tag, results_dir)
        for method in dict.fromkeys(r["method"] for r in rows):
            if methods and method not in methods:
                continue
            s = summarize([r for r in rows if r["method"] == method])
            values = " | ".join(f"{s[c]:.3f}" for c in COLUMNS)
            lines.append(f"| {tag} | {method} | {s['questions']} | {values} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare des évaluations du retrieval.")
    parser.add_argument("tags", nargs="+", help="ex. baseline-test finetune-test")
    parser.add_argument("--methods", nargs="+", help="ex. vector hybrid")
    args = parser.parse_args(argv)
    table = compare(args.tags, args.methods)
    (RESULTS_DIR / "comparison.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nTableau écrit dans {RESULTS_DIR / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
