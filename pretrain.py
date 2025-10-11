import argparse
from typing import Dict, List, Optional

import torch
from torch.utils.data import IterableDataset
from datasets import DownloadConfig, load_dataset
from transformers import TrainingArguments

from model import ParallelDecoder, build_flat_linear_layout, set_rope_pos2d, build_columnar_causal_mask
from train import ParallelDecodingTrainer


class FineWebColumnarDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        branch_count: int,
        seq_length: int,
        split: str,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        max_samples: Optional[int] = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.branch_count = branch_count
        self.seq_length = seq_length
        self.split = split
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.local_files_only = local_files_only

    def __iter__(self):
        download_config = DownloadConfig(local_files_only=True) if self.local_files_only else None
        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True,
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
                chunk_text = self.tokenizer.decode(chunk)
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
            pad_to=self.seq_length * max(1, self.branch_count),
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
    parser.add_argument("--dataset-split", type=str, default="sample-10BT")
    parser.add_argument("--branch-count", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-samples", type=int, help="Limit number of training samples from the iterable dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    decoder = ParallelDecoder(model_name=args.model_name)
    model = decoder.model
    tokenizer = decoder.tokenizer

    dataset = FineWebColumnarDataset(
        tokenizer=tokenizer,
        branch_count=args.branch_count,
        seq_length=args.seq_length,
        split=args.dataset_split,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        local_files_only=args.local_files_only,
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
