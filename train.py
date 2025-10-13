import argparse
from typing import Any, Dict, List

import torch
from transformers import Trainer, TrainingArguments

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore

from model import (
    ParallelDecoder,
    build_flat_linear_layout,
    set_rope_pos2d,
    build_columnar_causal_mask,
)
from data_utils import load_grouped_squad


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

        def _first_answer(answers: Any) -> str:
            if isinstance(answers, list):
                for candidate in answers:
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate
            return "<no_answer>"

        main = f"背景: {context}"

        branches: List[str] = []
        for idx, qa in enumerate(qas, start=1):
            question = _clean(qa.get("question", ""))
            answer = _clean(_first_answer(qa.get("answers", [])))
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
            "time_ids": layout.time_ids,
            "labels": labels,
        }


class ParallelDecodingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pos2d_tensor = inputs.pop("pos2d")
        time_ids = inputs.pop("time_ids")
        pad_mask = inputs["attention_mask"]

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

        try:
            param_dtype = next(module.parameters()).dtype
        except StopIteration:
            param_dtype = torch.float32

        set_rope_pos2d(module, pos2d_tensor.to(target_device))
        attn_mask = build_columnar_causal_mask(time_ids.to(target_device), pad_mask.to(target_device))
        inputs["attention_mask"] = attn_mask.to(dtype=param_dtype)
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
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--min-questions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA adapters for parameter-efficient fine-tuning")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA scaling factor")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout probability")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="*",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Module names to inject LoRA adapters into",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    decoder = ParallelDecoder(
        model_name=args.model_name,
        pair_indices_1based=tuple(args.pair_indices),
    )
    model = decoder.model
    tokenizer = decoder.tokenizer

    if args.use_lora:
        if LoraConfig is None or get_peft_model is None:
            raise RuntimeError("peft 未安装，无法启用 LoRA。请先 pip install peft")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        decoder.model = model
        try:
            model.print_trainable_parameters()  # type: ignore[attr-defined]
        except AttributeError:
            pass

    max_questions = max(1, args.max_branches + 1)
    dataset = load_grouped_squad(
        splits=("train", "validation"),
        max_questions_per_context=max_questions,
        min_questions_per_context=max(1, args.min_questions),
        max_samples_per_split={
            "train": args.max_train_samples,
            "validation": args.max_eval_samples,
        },
        local_files_only=args.local_files_only,
    )
    train_dataset = dataset.get("train")
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
