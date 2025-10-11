import argparse
import math
import re
import string
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import load_grouped_squad
from model import ParallelDecoder

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is missing
    tqdm = None  # type: ignore


def _normalize_answer(text: str) -> str:
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # collapse whitespace
    return " ".join(text.split())


def _first_line(text: str) -> str:
    line = text.splitlines()[0]
    if ":" in line:
        # drop leading labels like "答案:" or "answer:"
        parts = line.split(":", 1)
        if parts[0].strip().lower() in {"answer", "答案"}:
            line = parts[1]
    return line.strip(" ：: ")


def _exact_match(prediction: str, references: Sequence[str]) -> bool:
    pred_norm = _normalize_answer(prediction)
    for ref in references:
        if _normalize_answer(ref) == pred_norm:
            return True
    return False


def _iter_dataset(dataset) -> Iterable[Dict[str, List[Dict[str, Sequence[str]]]]]:
    if dataset is None:
        return []
    if hasattr(dataset, "to_list"):
        return dataset.to_list()
    return dataset


def evaluate_baseline(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str],
    dataset,
    device: torch.device,
    max_new_tokens: int,
    max_questions: int,
    local_files_only: bool,
) -> Tuple[float, int]:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path or model_name_or_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()

    total = 0
    correct = 0

    iterable = list(_iter_dataset(dataset))
    iterator = tqdm(iterable, desc="Baseline", unit="context") if tqdm else iterable

    for entry in iterator:
        context = entry["context"]
        for qa in entry["qas"][:max_questions]:
            question = qa.get("question", "")
            prompt = f"背景: {context}\n问题: {question}\n答案:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            generated_tokens = output[0, inputs["input_ids"].shape[1] :]
            prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            prediction = _first_line(prediction)
            if _exact_match(prediction, qa.get("answers", [""])):
                correct += 1
            total += 1

    return (correct / total if total else 0.0, total)


def _build_branch_sample(context: str, qas: Sequence[Dict[str, Sequence[str]]]) -> Dict[str, str]:
    def _clean(text: str) -> str:
        return text.strip().replace("\n", " ")

    def _first_answer(answers: Sequence[str]) -> str:
        for ans in answers:
            if ans.strip():
                return ans
        return "<no_answer>"

    main_prompt = f"背景: {context}"

    branches: List[str] = []
    for idx, qa in enumerate(qas, start=1):
        question = _clean(qa.get("question", ""))
        branches.append(f"问题{idx}: {question}\n答案:")

    return {"main": main_prompt, "branches": branches}


def evaluate_parallel(
    decoder: ParallelDecoder,
    dataset,
    max_new_tokens: int,
    max_questions: int,
    batch_size: int,
) -> Tuple[float, int]:
    entries = list(_iter_dataset(dataset))
    total = 0
    correct = 0

    iterator = range(0, len(entries), batch_size)
    iterator = tqdm(iterator, desc="Parallel", unit="batch") if tqdm else iterator

    for start in iterator:
        batch_entries = entries[start : start + batch_size]
        samples = []
        reference_answers: List[List[Sequence[str]]] = []

        for entry in batch_entries:
            qas = entry["qas"][:max_questions]
            if not qas:
                continue
            samples.append(_build_branch_sample(entry["context"], qas))
            reference_answers.append([qa.get("answers", [""]) for qa in qas])

        if not samples:
            continue

        result = decoder.generate(
            samples,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        for sample_result, refs in zip(result.samples, reference_answers):
            branch_map = {branch.branch_id: branch.text for branch in sample_result.branches}
            for idx, answers in enumerate(refs):
                prediction = branch_map.get(idx, "")
                prediction = _first_line(prediction)
                if _exact_match(prediction, answers):
                    correct += 1
                total += 1

    return (correct / total if total else 0.0, total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and parallel decoders on SQuAD")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B", help="Base Hugging Face model identifier")
    parser.add_argument("--base-tokenizer", type=str, help="Tokenizer to pair with the base model")
    parser.add_argument("--ft-model", type=str, help="Fine-tuned model path or identifier")
    parser.add_argument("--ft-tokenizer", type=str, help="Tokenizer to pair with the fine-tuned model")
    parser.add_argument("--max-branches", type=int, default=3, help="Number of additional questions beyond the main branch")
    parser.add_argument("--min-questions", type=int, default=2, help="Minimum questions per context to keep")
    parser.add_argument("--max-eval-samples", type=int, help="Cap on the number of grouped contexts for evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Generation length for each answer")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for parallel decoder evaluation")
    parser.add_argument("--device", type=str, help="Device override, e.g., cuda or cpu")
    parser.add_argument("--local-files-only", action="store_true", help="Force using local cached datasets and models")
    parser.add_argument("--pair-indices", type=int, nargs="*", default=[8, 16, 24], help="1-based frequency indices for 2D RoPE interleave")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    max_questions = max(1, args.max_branches + 1)

    dataset_dict = load_grouped_squad(
        splits=("validation",),
        max_questions_per_context=max_questions,
        min_questions_per_context=max(1, args.min_questions),
        max_samples_per_split={"validation": args.max_eval_samples},
        local_files_only=args.local_files_only,
    )
    validation_set = dataset_dict.get("validation")

    if validation_set is None or len(validation_set) == 0:
        raise RuntimeError("Validation split is empty; adjust min_questions or sample limits")

    print(f"Loaded {len(validation_set)} grouped contexts for evaluation")

    if args.base_model:
        base_accuracy, total_questions = evaluate_baseline(
            model_name_or_path=args.base_model,
            tokenizer_name_or_path=args.base_tokenizer,
            dataset=validation_set,
            device=device,
            max_new_tokens=args.max_new_tokens,
            max_questions=max_questions,
            local_files_only=args.local_files_only,
        )
        print(f"Baseline exact match: {base_accuracy:.4f} ({total_questions} questions)")

    if args.ft_model:
        tokenizer_obj = None
        tokenizer_kwargs = {"local_files_only": args.local_files_only}
        model_kwargs = {"local_files_only": args.local_files_only}
        if args.ft_tokenizer:
            tokenizer_obj = AutoTokenizer.from_pretrained(
                args.ft_tokenizer,
                trust_remote_code=True,
                local_files_only=args.local_files_only,
            )
            tokenizer_kwargs = None

        decoder = ParallelDecoder(
            model_name=args.ft_model,
            tokenizer=tokenizer_obj,
            pair_indices_1based=tuple(args.pair_indices),
            device=str(device),
            tokenizer_kwargs=tokenizer_kwargs,
            model_kwargs=model_kwargs,
        )
        parallel_accuracy, total_questions = evaluate_parallel(
            decoder,
            validation_set,
            max_new_tokens=args.max_new_tokens,
            max_questions=max_questions,
            batch_size=max(1, args.batch_size),
        )
        print(f"Parallel decoder exact match: {parallel_accuracy:.4f} ({total_questions} questions)")


if __name__ == "__main__":
    main()
