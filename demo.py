#!/usr/bin/env python3
"""Quick demo script for the columnar parallel decoder."""

import argparse
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer

from model import ParallelDecoder

BACKGROUND_PREFIXES = ("背景:", "Background:")
QUESTION_PREFIX_TEMPLATE_CH = "问题{idx}:"
QUESTION_PREFIX_TEMPLATE_EN = "Question {idx}:"
ANSWER_PREFIXES = ("答案:", "Answer:")


def _default_model_path() -> str:
    local_path = Path("./parallel-decoder-squad")
    if local_path.exists():
        return str(local_path)
    return "Qwen/Qwen3-4B"


def _build_sample(background: str, questions: List[str]) -> Dict[str, List[str]]:
    background = background.strip()
    if not background:
        raise ValueError("Background text cannot be empty.")
    if not questions:
        raise ValueError("At least one question is required to build branches.")

    if any(background.startswith(prefix) for prefix in BACKGROUND_PREFIXES):
        main = background
    else:
        main = f"Background: {background}"

    branches: List[str] = []
    for idx, question in enumerate(questions, start=1):
        q = question.strip()
        if not q:
            continue
        prefixes = [
            QUESTION_PREFIX_TEMPLATE_CH.format(idx=idx),
            QUESTION_PREFIX_TEMPLATE_EN.format(idx=idx),
        ]
        if any(q.startswith(prefix) for prefix in prefixes):
            header = q
        else:
            header = QUESTION_PREFIX_TEMPLATE_EN.format(idx=idx) + f" {q}"
        if header.startswith("问题"):
            answer_prefix = ANSWER_PREFIXES[0]
        else:
            answer_prefix = ANSWER_PREFIXES[1]
        branches.append(f"{header}\n{answer_prefix}")

    if not branches:
        raise ValueError("All questions are empty; unable to build branches.")

    return {"main": main, "branches": branches}


_DEFAULT_EXAMPLES: List[Dict[str, List[str]]] = [
    _build_sample(
        "Write a warm coming-of-age story about Lin Xue, who just graduated high school and wants to thank her friends.",
        [
            "Which emotions should the story highlight? Friendship, love, and family support.",
            "What should happen during the climax?",
            "What sense of hope should the ending deliver?",
        ],
    ),
    _build_sample(
        "Introduce our company's new smart watch, highlighting daily-life use cases.",
        [
            "What is the flagship feature of the watch?",
            "Which improvements does it have compared to the previous generation?",
            "Provide an example of a user's experience.",
        ],
    ),
]


def _map_branch_prompt(sample: Dict[str, List[str]], branch_id: int) -> str:
    has_background = bool(sample.get("main", "").strip())
    offset = branch_id - 1 if has_background else branch_id
    branches = sample.get("branches", [])
    if 0 <= offset < len(branches):
        header = branches[offset].splitlines()[0]
        return header
    return f"Branch {branch_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel decoding demo with custom branches")
    parser.add_argument("--model-name", type=str, default=_default_model_path(), help="Model checkpoint or HF Hub id")
    parser.add_argument("--base-model", type=str, help="Base model when --model-name points to a LoRA adapter")
    parser.add_argument("--tokenizer-name", type=str, help="Tokenizer path/identifier (defaults to model)")
    parser.add_argument("--device", type=str, help="Device override, e.g. cuda or cpu")
    parser.add_argument("--pair-indices", type=int, nargs="*", default=[8, 16, 24], help="1-based frequency indices for 2D RoPE patch")
    parser.add_argument("--background", type=str, help="Custom background text for a single-sample run")
    parser.add_argument("--questions", nargs="*", help="Custom questions forming branches; provide multiple entries")
    parser.add_argument("--max-branches", type=int, help="Optional cap on the number of branches to use")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum tokens to generate per branch")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling instead of greedy decoding")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Penalty factor (>1.0) to discourage repeated tokens during generation",
    )
    parser.add_argument("--local-files-only", action="store_true", help="Restrict model/tokenizer loading to local cache")
    parser.add_argument("--show-linear", action="store_true", help="Print linearized merged outputs in addition to branch answers")
    return parser.parse_args()


def build_samples(args: argparse.Namespace) -> List[Dict[str, List[str]]]:
    if args.background:
        questions = args.questions or []
        if not questions:
            raise ValueError("Custom background requires at least one question; provide them with --questions.")
        if args.max_branches is not None:
            questions = questions[: args.max_branches]
        sample = _build_sample(args.background, questions)
        return [sample]

    samples = _DEFAULT_EXAMPLES
    if args.max_branches is not None:
        trimmed: List[Dict[str, List[str]]] = []
        for sample in samples:
            branches = sample["branches"][: args.max_branches]
            trimmed.append({"main": sample["main"], "branches": branches})
        samples = trimmed
    return samples


def main() -> None:
    args = parse_args()

    samples = build_samples(args)

    tokenizer_obj = None
    tokenizer_kwargs = {"local_files_only": args.local_files_only}
    if args.tokenizer_name:
        tokenizer_obj = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        tokenizer_kwargs = None

    model_kwargs = {"local_files_only": args.local_files_only}

    decoder = ParallelDecoder(
        model_name=args.model_name,
        tokenizer=tokenizer_obj,
        pair_indices_1based=tuple(args.pair_indices),
        device=args.device,
        tokenizer_kwargs=tokenizer_kwargs,
        model_kwargs=model_kwargs,
        adapter_base_model=args.base_model,
    )

    result = decoder.generate(
        samples,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )

    for idx, sample_result in enumerate(result.samples):
        print(f"Sample {idx}:")
        prompt_sample = samples[idx]
        if args.show_linear:
            print("  Linearized sequence:")
            print(f"    {sample_result.linear_text}")
        for branch in sample_result.branches:
            prompt_header = _map_branch_prompt(prompt_sample, branch.branch_id)
            generated = branch.text.strip()
            if not generated:
                generated = "(no new tokens generated)"
            print(f"  {prompt_header}")
            print(f"    -> {generated}")
        print()


if __name__ == "__main__":
    main()
