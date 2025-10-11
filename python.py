from model import ParallelDecoder


SAMPLES = [
    {
        "main": "背景: 帮我写一个小故事",
        "branches": [
            "问题1: 故事的主题是什么？\n答案: 友情",
            "问题2: 需要加入怎样的情节？\n答案: 爱情",
            "问题3: 还想强调什么元素？\n答案: 成长",
        ],
    },
    {
        "main": "背景: 文章 + A 第二个样本",
        "branches": [
            "问题1: 补充第一段？\n答案: + B2",
            "问题2: 补充第二段？\n答案: + C2",
            "问题3: 补充第三段？\n答案: + D2",
        ],
    },
]


def run_demo():
    decoder = ParallelDecoder()
    result = decoder.generate(SAMPLES, max_new_tokens=4)

    print("线性化解码结果:")
    for idx, sample in enumerate(result.samples):
        print(f"样本 {idx}: {sample.linear_text}")

    print("\n分支新增 token:")
    for idx, sample in enumerate(result.samples):
        print(f"样本 {idx}:")
        for branch in sample.branches:
            text_display = branch.text if branch.text else "(无新增)"
            print(f"  x={branch.branch_id}: {text_display}")


if __name__ == "__main__":
    run_demo()
