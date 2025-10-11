import argparse
import json
from collections import OrderedDict
from typing import Any, Dict, List

import torch
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download
from transformers import Trainer, TrainingArguments

from model import (
    ParallelDecoder,
    build_flat_linear_layout,
    set_rope_pos2d,
)


class ParallelDecodingDataCollator:
    def __init__(self, tokenizer, max_length: int = 1024, max_questions: int = 4) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_questions = max(1, max_questions)

    def _convert_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        context = example.get("context", "")
        qas = example.get("qas", [])[: self.max_questions]
        if not qas:
            raise ValueError("样本缺少问答对，无法构建分支")

        def _clean(text: str) -> str:
            return text.strip().replace("\n", " ")

        main_question = _clean(qas[0].get("question", ""))
        main_answer = _clean(qas[0].get("answer", "<no_answer>"))
        main = f"背景: {context}\n问题1: {main_question}\n答案: {main_answer}"

        branches: List[str] = []
        for idx, qa in enumerate(qas[1:], start=2):
            question = _clean(qa.get("question", ""))
            answer = _clean(qa.get("answer", "<no_answer>"))
            branches.append(f"问题{idx}: {question}\n答案: {answer}")

        return {"main": main, "branches": branches}

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        samples = [self._convert_example(feat) for feat in features]
        layout = build_flat_linear_layout(
            self.tokenizer,
            samples,
            device=torch.device("cpu"),
            pad_to=self.max_length,
        )

        labels = layout.input_ids.clone()
        labels[layout.attention_mask == 0] = -100

        return {
            "input_ids": layout.input_ids,
            "attention_mask": layout.attention_mask,
            "position_ids": layout.pos1d,
            "pos2d": layout.pos2d,
            "labels": labels,
        }


class ParallelDecodingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pos2d_tensor = inputs.pop("pos2d")

        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            module = model.module
        else:
            module = model

        input_ids = inputs.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            target_device = input_ids.device
        else:
            try:
                target_device = next(module.parameters()).device
            except StopIteration:
                target_device = torch.device("cpu")

        set_rope_pos2d(module, pos2d_tensor.to(target_device))
        outputs = model(
            use_cache=False,
            **inputs,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune 2D RoPE parallel decoder on SQuAD")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--pair-indices", type=int, nargs="*", default=[8, 16, 24])
    parser.add_argument("--output-dir", type=str, default="./parallel-decoder-squad")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-branches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    return parser.parse_args()


def _default_answer_text(example: Dict[str, Any]) -> str:
    answers = example.get("answers", {})
    if isinstance(answers, dict):
        texts = answers.get("text")
        if texts:
            return texts[0]
    return "<no_answer>"


def _group_squad_split(split: Dataset, max_questions: int, min_questions: int = 1) -> Dataset:
    buckets: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for example in split:
        context = example.get("context", "")
        entry = buckets.setdefault(context, {"context": context, "qas": []})
        entry["qas"].append(
            {
                "question": example.get("question", ""),
                "answer": _default_answer_text(example),
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


def load_squad_dataset(
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    max_questions_per_context: int = 4,
    min_questions_per_context: int = 2,
) -> DatasetDict:
    try:
        raw_dataset = load_dataset("rajpurkar/squad")
    except ValueError as exc:
        if "Feature type 'List'" not in str(exc):
            raise

        def _load_split(filename: str) -> Dataset:
            path = hf_hub_download("rajpurkar/squad", filename)
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            records = []
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

        raw_dataset = DatasetDict(
            {
                "train": _load_split("train-v1.1.json"),
                "validation": _load_split("dev-v1.1.json"),
            }
        )

    grouped = DatasetDict(
        {
            split: _group_squad_split(raw_dataset[split], max_questions_per_context, min_questions_per_context)
            for split in raw_dataset
        }
    )

    if max_train_samples:
        grouped["train"] = grouped["train"].select(range(max_train_samples))
    if "validation" in grouped and max_eval_samples:
        grouped["validation"] = grouped["validation"].select(range(max_eval_samples))

    return grouped


def main():
    args = parse_args()

    decoder = ParallelDecoder(
        model_name=args.model_name,
        pair_indices_1based=tuple(args.pair_indices),
    )
    model = decoder.model
    tokenizer = decoder.tokenizer

    max_questions = max(1, args.max_branches + 1)
    dataset = load_squad_dataset(
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_questions_per_context=max_questions,
        min_questions_per_context=2,
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("validation")

    data_collator = ParallelDecodingDataCollator(
        tokenizer,
        max_length=args.max_length,
        max_questions=max_questions,
    )

    eval_strategy = "steps" if eval_dataset is not None else "no"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_eval=eval_dataset is not None,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        eval_strategy=eval_strategy,
        save_strategy="steps",
        logging_steps=50,
        save_steps=500,
        eval_steps=500 if eval_dataset is not None else None,
        seed=args.seed,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = ParallelDecodingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
