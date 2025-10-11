"""Utilities for loading and grouping SQuAD data for branch-style decoding."""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download


_SQUAD_REPO = "rajpurkar/squad"


def _extract_answers(example: Mapping[str, Any]) -> List[str]:
    answers = example.get("answers", {})
    texts: Iterable[str]
    if isinstance(answers, Mapping):
        texts = answers.get("text", []) or []
    elif isinstance(answers, Sequence):
        texts = [ans.get("text", "") for ans in answers if isinstance(ans, Mapping)]
    else:
        texts = []
    cleaned = [str(t).strip() for t in texts if str(t).strip()]
    return cleaned or ["<no_answer>"]


def _group_squad_split(
    split: Dataset,
    max_questions: int,
    min_questions: int,
) -> Dataset:
    buckets: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for example in split:
        context = example.get("context", "")
        entry = buckets.setdefault(context, {"context": context, "qas": []})
        entry["qas"].append(
            {
                "question": example.get("question", ""),
                "answers": _extract_answers(example),
            }
        )

    grouped_examples: List[Dict[str, Any]] = []
    for entry in buckets.values():
        qas = entry["qas"]
        if len(qas) < min_questions:
            continue
        grouped_examples.append(
            {
                "context": entry["context"],
                "qas": qas[: max_questions],
            }
        )

    return Dataset.from_list(grouped_examples)


def _load_squad_raw(local_files_only: bool = False) -> DatasetDict:
    try:
        return load_dataset(_SQUAD_REPO, local_files_only=local_files_only)
    except ValueError as exc:
        if "Feature type 'List'" not in str(exc):
            raise
        if local_files_only:
            raise RuntimeError(
                "SQuAD dataset features are not supported by the current datasets version "
                "and local_files_only=True prevents downloading JSON fallback."
            ) from exc

        def _load_split(filename: str) -> Dataset:
            path = hf_hub_download(_SQUAD_REPO, filename)
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            records: List[Dict[str, Any]] = []
            for article in payload.get("data", []):
                for paragraph in article.get("paragraphs", []):
                    context = paragraph.get("context", "")
                    for qa in paragraph.get("qas", []):
                        answers = qa.get("answers", [])
                        texts = [ans.get("text", "") for ans in answers] or ["<no_answer>"]
                        records.append(
                            {
                                "context": context,
                                "question": qa.get("question", ""),
                                "answers": {"text": texts},
                            }
                        )
            return Dataset.from_list(records)

        return DatasetDict(
            {
                "train": _load_split("train-v1.1.json"),
                "validation": _load_split("dev-v1.1.json"),
            }
        )


def load_grouped_squad(
    splits: Sequence[str] = ("train", "validation"),
    max_questions_per_context: int = 4,
    min_questions_per_context: int = 2,
    max_samples_per_split: Optional[Mapping[str, Optional[int]] | int] = None,
    local_files_only: bool = False,
) -> DatasetDict:
    raw = _load_squad_raw(local_files_only=local_files_only)

    def _max_for(split: str) -> Optional[int]:
        if isinstance(max_samples_per_split, Mapping):
            return max_samples_per_split.get(split)
        if isinstance(max_samples_per_split, int):
            return max_samples_per_split
        return None

    grouped: Dict[str, Dataset] = {}
    for split in splits:
        if split not in raw:
            continue
        ds = _group_squad_split(raw[split], max_questions_per_context, min_questions_per_context)
        max_samples = _max_for(split)
        if max_samples is not None:
            max_samples = min(max_samples, len(ds))
            ds = ds.select(range(max_samples))
        grouped[split] = ds

    return DatasetDict(grouped)


__all__ = [
    "load_grouped_squad",
]

