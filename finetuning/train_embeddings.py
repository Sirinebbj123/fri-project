"""Fine-tuning de bge-small sur les paires (question, page de preuve) de FinanceBench.

Perte MultipleNegativesRankingLoss ; entraînement sur les sociétés du split ``train`` seulement.
Usage : ``python -m finetuning.train_embeddings [--epochs N] [--limit N]``
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from evaluation.dataset import DEFAULT_PATH, EvalItem, load_financebench, split_companies
from fri.retrieval.vector import default_query_prefix

BASE_MODEL = "BAAI/bge-small-en-v1.5"
OUTPUT_DIR = Path("finetuning/models/bge-small-en-financebench")
MAX_PASSAGE_CHARS = 2000  # ~500 tokens, la limite du modèle


def clean_passage(text: str, max_chars: int = MAX_PASSAGE_CHARS) -> str:
    """Espaces normalisés, coupé à ``max_chars`` (le texte FinanceBench a beaucoup de blancs)."""
    return " ".join(text.split())[:max_chars]


def build_pairs(
    items: Sequence[EvalItem], query_prefix: str = "", min_chars: int = 50
) -> list[tuple[str, str]]:
    """Paires (question, passage de preuve), avec le préfixe BGE comme à la recherche."""
    pairs: list[tuple[str, str]] = []
    for item in items:
        for text in item.evidence_texts:
            passage = clean_passage(text)
            if len(passage) >= min_chars:
                pairs.append((query_prefix + item.question, passage))
    return pairs


def train(
    base_model: str = BASE_MODEL,
    output_dir: str | Path = OUTPUT_DIR,
    *,
    questions: str | Path = DEFAULT_PATH,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    test_ratio: float = 0.3,
    seed: int = 42,
    limit: int | None = None,
) -> dict:
    """Entraîne et enregistre le modèle ; renvoie les informations d'entraînement."""
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )

    try:  # sentence-transformers >= 6
        from sentence_transformers.sentence_transformer.losses import (
            MultipleNegativesRankingLoss,
        )
        from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    except ImportError:  # versions 3 à 5
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        from sentence_transformers.training_args import BatchSamplers

    output_dir = Path(output_dir)
    items = load_financebench(questions)
    train_companies, test_companies = split_companies(items, test_ratio, seed)
    from fri.ingestion.metadata import company_key

    train_items = [it for it in items if company_key(it.doc_company) in train_companies]
    prefix = default_query_prefix(base_model)
    pairs = build_pairs(train_items, prefix)
    random.Random(seed).shuffle(pairs)
    pairs = pairs[:limit] if limit else pairs
    if len(pairs) < 2:
        raise RuntimeError("Pas assez de paires pour entraîner.")
    print(
        f"{len(pairs)} paires (question, passage) de {len(train_companies)} sociétés ; "
        f"{len(test_companies)} sociétés gardées pour le test."
    )

    model = SentenceTransformer(base_model)
    dataset = Dataset.from_dict(
        {"anchor": [q for q, _ in pairs], "positive": [p for _, p in pairs]}
    )
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=max(1, math.ceil(len(pairs) / batch_size) * epochs // 10),
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # pas deux fois le même passage par paquet
        save_strategy="no",
        logging_steps=5,
        report_to="none",
        seed=seed,
        use_cpu=not _has_gpu(),
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=MultipleNegativesRankingLoss(model),
    )
    start = time.perf_counter()
    trainer.train()
    model.save(str(output_dir))

    info = {
        "base_model": base_model,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pairs": len(pairs),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "test_ratio": test_ratio,
        "train_companies": sorted(train_companies),
        "test_companies": sorted(test_companies),
        "query_prefix": prefix,
        "seconds": round(time.perf_counter() - start, 1),
    }
    (output_dir / "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Modèle enregistré dans {output_dir} ({info['seconds']} s)")
    return info


def _has_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune bge-small sur FinanceBench.")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--questions", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--limit", type=int, help="seulement N paires (essai)")
    args = parser.parse_args(argv)
    train(
        args.base_model,
        args.output_dir,
        questions=args.questions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
