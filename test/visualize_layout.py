import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from transformers import AutoTokenizer

from model import (
    build_columnar_causal_mask,
    build_flat_linear_layout,
)


def _load_samples(path: str | None) -> List[Dict[str, Sequence[str]]]:
    """Load samples from a JSON/JSONL file or return a default example."""

    if path is None:
        return [
            {
                "main": "背景: 人工智能的发展历程可以追溯到二十世纪。以下总结三个领域的里程碑。",
                "branches": [
                    "问题1: 机器学习领域的关键事件是什么？答案: 神经网络的复兴探索。",
                    "问题2: 计算机视觉有哪些突破？答案: 卷积网络在图像识别的广泛应用。",
                    "问题3: 自然语言处理的代表成果？答案: Transformer 架构的提出。",
                ],
            }
        ]

    samples: List[Dict[str, Sequence[str]]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("[") or line.startswith("{"):
                payload = json.loads(line)
                if isinstance(payload, list):
                    samples.extend(payload)
                elif isinstance(payload, dict):
                    samples.append(payload)
            else:
                raise ValueError("仅支持 JSON/JSONL 格式")
    if not samples:
        raise ValueError("未在输入文件中解析出任何样本")
    return samples


def _format_attention(attn: torch.Tensor, max_rows: int | None = None) -> str:
    """Format the causal mask as a string matrix (1=可见, 0=受限)."""

    attn_slice = attn[0, 0]
    if max_rows is not None:
        attn_slice = attn_slice[:max_rows, :max_rows]
    visible = attn_slice.eq(0).to(dtype=torch.int16)
    rows = []
    for row in visible:
        rows.append(" ".join(f"{int(val):1d}" for val in row.tolist()))
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化列式布局与注意力掩码")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-4B", help="用于分词的模型或本地分词器路径")
    parser.add_argument("--input", type=str, help="JSON/JSONL 文件，包含 main/branches 字段")
    parser.add_argument("--max-cols", type=int, default=64, help="打印注意力矩阵时保留的最大列数")
    parser.add_argument("--show-tokens", action="store_true", help="逐列打印 token / branch / time 信息")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    samples = _load_samples(args.input)

    layout = build_flat_linear_layout(
        tokenizer=tokenizer,
        samples=samples,
        device=torch.device("cpu"),
    )

    causal_mask = build_columnar_causal_mask(layout.time_ids, layout.attention_mask)

    print("=== Layout Summary ===")
    for idx, meta in enumerate(layout.metadata):
        print(f"- Sample {idx}:")
        print(f"  branch_ids       : {meta.branch_ids}")
        print(f"  branch_positions : {meta.branch_positions}")
        print(f"  branch_lengths   : {meta.branch_lengths}")
        print(f"  branch_start_y   : {meta.branch_start_y}")
        print(f"  branch_pos1d_end : {meta.branch_pos1d_end}")
        print(f"  background_branch: {meta.background_branch_id}")
        print(f"  sequence_length  : {layout.attention_mask[idx].sum().item()}")

    if args.show_tokens:
        print("\n=== Token Timeline (batch=0) ===")
        seq_len = int(layout.attention_mask[0].sum().item())
        token_ids = layout.input_ids[0, :seq_len].tolist()
        branches = layout.pos2d[0, :seq_len, 0].tolist()
        times = layout.pos2d[0, :seq_len, 1].tolist()
        decoded = tokenizer.convert_ids_to_tokens(token_ids)
        for i, (token, bid, tid) in enumerate(zip(decoded, branches, times)):
            print(f"[{i:02d}] time={tid:03d} branch={bid:02d} token={token}")

    print("\n=== Attention Mask (1=可见, 0=屏蔽) ===")
    formatted = _format_attention(causal_mask, max_rows=args.max_cols)
    print(formatted)


if __name__ == "__main__":
    main()
