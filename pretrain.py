import argparse
from typing import Dict, List, Optional

import torch
from torch.utils.data import IterableDataset
from datasets import DownloadConfig, load_dataset
from transformers import TrainingArguments

from model import (
    ParallelDecoder,
    build_flat_linear_layout,
    set_rope_pos2d,
    build_columnar_causal_mask,
)
from train import ParallelDecodingTrainer

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore


class FineWebColumnarDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        branch_count: int,
        seq_length: int,
        split: str,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        config_name: Optional[str] = None,
        max_samples: Optional[int] = None,
        local_files_only: bool = False,
        streaming: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.branch_count = branch_count
        self.seq_length = seq_length
        self.split = split
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.max_samples = max_samples
        self.local_files_only = local_files_only
        self.streaming = streaming

    def __iter__(self):
        download_config = DownloadConfig(local_files_only=True) if self.local_files_only else None
        if self.config_name is not None:
            dataset = load_dataset(
                self.dataset_name,
                self.config_name,
                split=self.split,
                streaming=self.streaming,
                download_config=download_config,
            )
        else:
            dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                streaming=self.streaming,
                download_config=download_config,
            )
        buffer: List[str] = []
        produced = 0
        for row in dataset:
            text = row.get("text", "")
            if not text:
                continue
            tokenized = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False)
            tokens = tokenized["input_ids"]
            if not tokens:
                continue
            start = 0
            while start < len(tokens):
                chunk = tokens[start : start + self.seq_length]
                if not chunk:
                    break
                chunk_text = self.tokenizer.decode(
                    chunk,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                buffer.append(chunk_text)
                start += self.seq_length
                if len(buffer) == self.branch_count:
                    yield {"main": "", "branches": buffer}
                    buffer = []
                    produced += 1
                    if self.max_samples is not None and produced >= self.max_samples:
                        return
        # drop remaining buffer


class ColumnarPretrainCollator:
    def __init__(self, tokenizer, seq_length: int, branch_count: int) -> None:
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.branch_count = branch_count
        self.total_length = seq_length * max(1, branch_count)

    def __call__(self, features: List[Dict[str, List[str]]]) -> Dict[str, torch.Tensor]:
        samples = []
        for feat in features:
            branches = feat.get("branches", [])
            if len(branches) < self.branch_count:
                branches = branches + [""] * (self.branch_count - len(branches))
            samples.append({"main": feat.get("main", ""), "branches": branches})

        layout = build_flat_linear_layout(
            self.tokenizer,
            samples,
            device=torch.device("cpu"),
            pad_to=self.total_length,
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


class PretrainTrainer(ParallelDecodingTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pos2d = inputs.pop("pos2d")
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

        param_iter = iter(module.parameters())
        first_param = next(param_iter, None)
        param_dtype = first_param.dtype if first_param is not None else torch.float32

        set_rope_pos2d(module, pos2d.to(target_device))
        causal_mask = build_columnar_causal_mask(time_ids.to(target_device), pad_mask.to(target_device))
        inputs["attention_mask"] = causal_mask.to(dtype=param_dtype)
        outputs = model(use_cache=False, **inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser(description="Columnar pretraining on FineWeb")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--output-dir", type=str, default="./pretrained-columnar")
    parser.add_argument("--dataset-name", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", type=str, default="sample-10BT")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--branch-count", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-samples", type=int, help="Limit number of training samples from the iterable dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--streaming", action="store_true", help="Use HuggingFace streaming dataset interface")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA adapters for pretraining")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="*",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    known_splits = {"train", "validation", "test"}
    if args.dataset_split not in known_splits:
        if args.dataset_config in (None, args.dataset_split):
            args.dataset_config = args.dataset_split
        args.dataset_split = "train"

    if args.dataset_config is not None and args.dataset_config.strip() == "":
        args.dataset_config = None

    decoder = ParallelDecoder(model_name=args.model_name)
    model = decoder.model
    tokenizer = decoder.tokenizer

    if args.use_lora:
        if LoraConfig is None or get_peft_model is None:
            raise RuntimeError("peft 未安装，无法启用 LoRA。请先 pip install peft")
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        decoder.model = model

    dataset = FineWebColumnarDataset(
        tokenizer=tokenizer,
        branch_count=args.branch_count,
        seq_length=args.seq_length,
        split=args.dataset_split,
        dataset_name=args.dataset_name,
        config_name=args.dataset_config,
        max_samples=args.max_samples,
        local_files_only=args.local_files_only,
        streaming=args.streaming,
    )

    collator = ColumnarPretrainCollator(
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        branch_count=args.branch_count,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_train_epochs=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        save_steps=args.max_steps,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    trainer = PretrainTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
