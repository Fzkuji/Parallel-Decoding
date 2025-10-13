#!/usr/bin/env python3
"""Quick demo script for the columnar parallel decoder."""

import argparse
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer

from model import ParallelDecoder


def _default_model_path() -> str:
    local_path = Path("./parallel-decoder-squad")
    if local_path.exists():
        return str(local_path)
    return "Qwen/Qwen3-4B"


def _build_sample(background: str, questions: List[str]) -> Dict[str, List[str]]:
    background = background.strip()
    if not background:
        raise ValueError("背景不能为空")
    if not questions:
        raise ValueError("至少需要一个问题来构建分支")

    main = background if background.startswith("背景:") else f"背景: {background}"
    branches: List[str] = []
    for idx, question in enumerate(questions, start=1):
        q = question.strip()
        if not q:
            continue
        if q.startswith(f"问题{idx}:"):
            header = q
        else:
            header = f"问题{idx}: {q}"
        branches.append(f"{header}\n答案:")

    if not branches:
        raise ValueError("所有问题为空，无法构建分支")

    return {"main": main, "branches": branches}


_DEFAULT_EXAMPLES: List[Dict[str, List[str]]] = [
    _build_sample(
        "帮我写一个温暖的成长故事，主角是高中毕业的林雪，她希望向朋友表达感谢。",
        [
            "故事里希望强调哪些情感？友情、爱情和家人的陪伴",
            "高潮部分应该发生什么？",
            "结尾想传达怎样的希望？",
        ],
    ),
    _build_sample(
        "介绍我们公司推出的新款智能手表，突出其日常生活场景。",
        [
            "这款手表的主打功能是什么？",
            "和上一代产品相比的改进有哪些？",
            "给出一个用户的使用体验。",
        ],
    ),
]


def _map_branch_prompt(sample: Dict[str, List[str]], branch_id: int) -> str:
    has_background = bool(sample.get("main", "").strip())
    offset = branch_id - 1 if has_background else branch_id
    branches = sample.get("branches", [])
    if 0 <= offset < len(branches):
        return branches[offset].splitlines()[0]
    return f"分支 {branch_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel decoding demo with custom branches")
    parser.add_argument("--model-name", type=str, default=_default_model_path(), help="Model checkpoint or HF Hub id")
    parser.add_argument("--base-model", type=str, help="Base model when --model-name 指向 LoRA adapter")
    parser.add_argument("--tokenizer-name", type=str, help="Tokenizer path/identifier (defaults to model)")
    parser.add_argument("--device", type=str, help="Device override, e.g. cuda or cpu")
    parser.add_argument("--pair-indices", type=int, nargs="*", default=[8, 16, 24], help="1-based frequency indices for 2D RoPE patch")
    parser.add_argument("--background", type=str, help="Custom background text for a single-sample run")
    parser.add_argument("--questions", nargs="*", help="Custom questions forming branches; provide multiple entries")
    parser.add_argument("--max-branches", type=int, help="Optional cap on the number of branches to use")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum tokens to generate per branch")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling instead of greedy decoding")
    parser.add_argument("--local-files-only", action="store_true", help="Restrict model/tokenizer loading to local cache")
    parser.add_argument("--show-linear", action="store_true", help="Print linearized merged outputs in addition to branch answers")
    return parser.parse_args()


def build_samples(args: argparse.Namespace) -> List[Dict[str, List[str]]]:
    if args.background:
        questions = args.questions or []
        if not questions:
            raise ValueError("提供自定义背景时需要至少一个问题。使用 --questions 指定问题文本。")
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
    )

    for idx, sample_result in enumerate(result.samples):
        print(f"样本 {idx}:")
        prompt_sample = samples[idx]
        if args.show_linear:
            print("  线性化序列:")
            print(f"    {sample_result.linear_text}")
        for branch in sample_result.branches:
            prompt_header = _map_branch_prompt(prompt_sample, branch.branch_id)
            generated = branch.text.strip()
            if not generated:
                generated = "(无新增生成)"
            print(f"  {prompt_header}")
            print(f"    -> {generated}")
        print()


if __name__ == "__main__":
    main()
