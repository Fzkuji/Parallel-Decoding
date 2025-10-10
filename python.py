from model import ParallelDecoder


SAMPLES = [
    {
        "main": "帮我写一个小故事，要关于友情的",
        "branches": [
            "要关于爱情的",
            "要关于家庭的",
            "要关于成长的",
        ],
    },
    {
        "main": "文章 + A 第二个样本",
        "branches": ["+ B2", "+ C2", "+ D2"],
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
