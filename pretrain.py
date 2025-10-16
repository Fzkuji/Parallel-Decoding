import argparse
import math
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
        main_segments: int = 1,
        max_total_tokens: Optional[int] = 4096,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.total_segments = max(1, branch_count)
        self.seq_length = seq_length
        self.split = split
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.max_samples = max_samples
        self.local_files_only = local_files_only
        self.streaming = streaming
        self.main_segments = max(1, min(main_segments, self.total_segments))
        self.expected_branches = max(0, self.total_segments - self.main_segments)
        self.max_total_tokens = max_total_tokens if (max_total_tokens is None or max_total_tokens > 0) else None

    def _append_eos(self, text: str) -> str:
        eos = self.tokenizer.eos_token
        if eos and not text.endswith(eos):
            return text + eos
        return text

    @staticmethod
    def _ensure_double_newline(text: str) -> str:
        if not text:
            return text
        stripped = text.rstrip()
        if stripped.endswith("\n\n"):
            return stripped
        if stripped.endswith("\n"):
            return stripped + "\n"
        return stripped + "\n\n"

    def _prepare_segment(self, token_ids: List[int]) -> List[int]:
        max_len = self.seq_length
        if max_len <= 0:
            return []
        eos_id = self.tokenizer.eos_token_id
        segment = token_ids[:max_len]
        if eos_id is None:
            return segment

        if len(segment) == max_len:
            if segment[-1] != eos_id:
                segment = segment[:-1] + [eos_id]
        else:
            segment = segment + [eos_id]
            if len(segment) > max_len:
                segment = segment[:max_len]
        return segment

    def _decode_tokens(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _build_sample_from_text(self, text: str) -> Optional[Dict[str, List[str]]]:
        if not text:
            return None
        normalized = text.replace("\r\n", "\n")
        paragraphs = [paragraph.strip() for paragraph in normalized.split("\n") if paragraph.strip()]
        if len(paragraphs) < 2:
            return None

        token_budget = self.max_total_tokens
        if token_budget is None:
            token_budget = self.seq_length * self.total_segments
        else:
            token_budget = min(token_budget, self.seq_length * self.total_segments)
        if token_budget <= 0:
            token_budget = self.seq_length * self.total_segments

        selected_paragraphs: List[str] = []
        used_tokens = 0
        for paragraph in paragraphs:
            encoded = self.tokenizer(paragraph, add_special_tokens=False, return_attention_mask=False)
            ids = encoded["input_ids"]
            if not ids:
                continue
            if used_tokens + len(ids) > token_budget and selected_paragraphs:
                break
            if used_tokens + len(ids) > token_budget and not selected_paragraphs:
                # paragraph itself exceeds budget; truncate tokens and decode back
                truncated = ids[: token_budget]
                paragraph = self._decode_tokens(self._prepare_segment(truncated))
                used_tokens = token_budget
                selected_paragraphs.append(paragraph)
                break
            selected_paragraphs.append(paragraph)
            used_tokens += len(ids)
            if used_tokens >= token_budget:
                break

        if len(selected_paragraphs) < 2:
            return None

        num_paragraphs = len(selected_paragraphs)
        main_count = max(self.main_segments, min(num_paragraphs // 2, num_paragraphs - 1))
        if main_count >= num_paragraphs:
            main_count = num_paragraphs - 1
        if main_count <= 0:
            main_count = min(self.main_segments, num_paragraphs - 1)

        main_paragraphs = selected_paragraphs[:main_count]
        branch_candidates = selected_paragraphs[main_count:]
        if not branch_candidates:
            return None

        main_text = "\n\n".join(main_paragraphs)

        if self.expected_branches <= 0:
            main_text = self._ensure_double_newline(main_text)
            return {"main": self._append_eos(main_text), "branches": []}

        chunk_size = max(1, math.ceil(len(branch_candidates) / self.expected_branches))
        branch_texts: List[str] = []
        for idx in range(self.expected_branches):
            start = idx * chunk_size
            if start >= len(branch_candidates):
                break
            end = min(len(branch_candidates), (idx + 1) * chunk_size)
            branch_chunk = "\n\n".join(branch_candidates[start:end]).strip()
            branch_chunk = self._ensure_double_newline(branch_chunk)
            branch_texts.append(branch_chunk)

        branch_texts = [b for b in branch_texts if b]
        if not branch_texts:
            return None

        main_text = self._append_eos(self._ensure_double_newline(main_text))
        branch_texts = [self._append_eos(branch) for branch in branch_texts]

        return {"main": main_text, "branches": branch_texts}

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

        produced = 0
        for row in dataset:
            text = row.get("text", "")
            sample = self._build_sample_from_text(text)
            if sample is None:
                continue
            yield sample
            produced += 1
            if self.max_samples is not None and produced >= self.max_samples:
                return


class ColumnarPretrainCollator:
    def __init__(self, tokenizer, seq_length: int, branch_count: int, main_segments: int) -> None:
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.total_segments = max(1, branch_count)
        self.main_segments = max(1, min(main_segments, self.total_segments))
        self.expected_branches = max(0, self.total_segments - self.main_segments)
        self.total_length = seq_length * self.total_segments

    def __call__(self, features: List[Dict[str, List[str]]]) -> Dict[str, torch.Tensor]:
        samples = []
        for feat in features:
            branches = feat.get("branches", [])
            if len(branches) < self.expected_branches:
                branches = branches + [""] * (self.expected_branches - len(branches))
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
    parser.add_argument("--branch-count", type=int, default=12)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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
    parser.add_argument("--main-segments", type=int, default=2, help="Number of segments assigned to the main branch")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="启用梯度检查点降低显存占用（会禁用 use_cache）",
    )
    parser.add_argument("--fp16", action="store_true", help="在 fp16 模式下训练")
    parser.add_argument("--bf16", action="store_true", help="在 bf16 模式下训练")
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=4096,
        help="单条原始文本在分段前允许使用的最大 token 数，将按段落切分并截断",
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
        main_segments=args.main_segments,
        max_total_tokens=args.max_total_tokens,
    )

    preview_sample = None
    try:
        preview_sample = next(iter(dataset))
    except StopIteration:
        preview_sample = None

    if preview_sample is not None:
        def _preview_text(text: str, limit: int = 200) -> str:
            text = text.strip()
            return text if len(text) <= limit else text[:limit] + "..."

        print("=== Preview sample ===")
        print("Main:\n", _preview_text(preview_sample.get("main", "")))
        for idx, branch_text in enumerate(preview_sample.get("branches", []), start=1):
            print(f"Branch {idx}:\n{_preview_text(branch_text)}")
        print("=====================")

    collator = ColumnarPretrainCollator(
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        branch_count=args.branch_count,
        main_segments=args.main_segments,
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
        fp16=args.fp16,
        bf16=args.bf16,
    )

    trainer = PretrainTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    if args.gradient_checkpointing:
        trainer.model.gradient_checkpointing_enable()
        if hasattr(trainer.model.config, "use_cache"):
            trainer.model.config.use_cache = False

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
