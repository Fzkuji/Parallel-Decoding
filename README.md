# Parallel-Decoding

## 设计概述
我们正在探索一种“列式并行”的解码/训练方式：把同一个上下文下的多条分支（例如多个问题或多段无关文本）同时放入模型，借助二维位置编码实现共享背景与分支独立展开。

- **二维位置**：每个 token 拥有 `(branch_id, column_id)` 形式的坐标；`branch_id` 区分分支，`column_id` 代表列时间轴。背景可被视作 `branch_id=0`、列从 0 开始，其他分支会靠右对齐到同一列。
- **列同步注意力**：在同一列生成的新 token 视为“同时刻”出现，只能看到更早列的数据（以及自身），因此所有分支可以共享背景、互相偷窥历史，却不会在同一列相互泄漏。
- **生成策略**：推理时按列轮询每个分支，一次只为分支追加一个 token，并保持独立的 KV Cache；训练/推理共用同一布局与掩码，确保行为一致。

### 示例：背景 + 三条问题分支
假设背景 `B` 含 8 个 token，三个问题 `Q1/Q2/Q3` 分别为 4、3、2 个 token。我们会得到如下布局（同列对齐）：

| 列序号 (`column_id`) | 分支 (`branch_id`) | Token |
|---------------------|--------------------|-------|
| 0‥7                | 0 (背景)            | B₀‥B₇ |
| 8                  | 0 / 1 / 2 / 3       | Q1₀ / Q2₀ / Q3₀ |
| 9                  | 0 / 1 / 2           | Q1₁ / Q2₁ / Q3₁ |
| 10                 | 0 / 1 / 2           | Q1₂ / Q2₂ / *PAD* |
| 11                 | 0 / 1               | Q1₃ / *PAD* |

- 列 8 表示各分支的第一个 token，同列互不可见，但都能看到背景列 0‥7。
- 列 9 的 token 可以访问背景 + 列 8。
- 长度不足的分支在靠右列使用 `PAD` 填充，只参与布局不参与损失/注意力。

推理时，若要生成每个分支的新 token，会让列时间增加到 12，并依次为每个 `branch_id` 写入新的答案 token。从训练到推理，列时间的定义完全一致。

Parallel-Decoding 是一个实验性项目，用于探索如何在大语言模型中引入 2D 位置编码并实现“并行分支”式的推理与训练流程。本仓库针对 Qwen3 系列模型打了定制补丁，使其能够在同一段背景文本后并行生成多个分支答案。

## 项目流程总览
完整使用流程包含三大阶段：

1. **预训练**：使用 FineWeb 采样多段文本，在列式时间轴上预训练模型，使其适应 2D 位置编码与列同步注意力。
2. **任务微调**：把 SQuAD 中相同 `context` 的问答聚合成分支样本，在预训练权重基础上做任务定向微调。
3. **效果评估**：并行对照原始 Hugging Face 模型与分支模型，在 SQuAD 验证集上比较精确匹配率。

下文按这一流程展开说明。

## 环境要求
- Python 3.10+
- PyTorch 2.2+（推荐 GPU 环境，CUDA/MPS 均可）
- `transformers>=4.55`
- `datasets>=2.14`
- 其他依赖：`huggingface_hub`, `accelerate`, `torchvision`（使用 GPU 时需要）

建议提前缓存好 Qwen/Qwen3-4B 模型与分词器，或保证可访问 Hugging Face Hub。若网络受限，脚本会 fallback 到官方 JSON 构建数据集，但模型与分词器仍需本地存在。

## 阶段一：FineWeb 列式预训练
`pretrain.py` 会从 FineWeb 采样多段文本，将每段视为一个分支，构造列同步的注意力掩码后执行自回归训练。

```bash
python pretrain.py \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --dataset-split train \
  --branch-count 16 \
  --seq-length 256 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-steps 20480
```

- `--branch-count` 决定每个样本并行的分支数量；`--seq-length` 为每个分支的截断长度。
- 默认使用本地（非 streaming）加载。如需避免一次性下载，可附加 `--streaming` 切换到 HF streaming 接口。
- `--dataset-config` 选择 FineWeb 的子集（默认 `sample-10BT`），`--dataset-split` 通常保持 `train`。
- 结果会保存到 `--output-dir`（默认 `./pretrained-columnar`）。后续微调可把该目录作为 `train.py --model-name` 输入。
- 若只能使用本地缓存数据，可加 `--local-files-only`。
- `--learning-rate` 默认 `4e-4`，可根据 batch 大小或是否启用 LoRA 调整。
- 若显存有限，可加 `--use-lora` 与 `--lora-*` 参数，仅训练 LoRA 适配器，实现低开销预训练。

## 阶段二：SQuAD 任务微调
`train.py` 会把 SQuAD 中相同 `context` 的问答聚合成单条样本，主干保存背景，后续问题作为分支。推荐在预训练权重基础上继续训练：

```bash
python train.py \
  --model-name ./pretrained-columnar \
  --max-train-samples 128 \
  --max-eval-samples 64 \
  --max-branches 3 \
  --batch-size 1 \
  --epochs 1
```

常用参数：
- `--pair-indices`：2D RoPE 使用的 1-based 频率索引。
- `--max-length`：背景与分支拼接后的最大 token 数（默认 1024，可按 `分支数 × 单分支长度` 估算）。
- `--max-branches`：额外保留的问题数量，实际分支数 = `max_branches + 1`（包含主干）。样本问题不足时不会补空分支，超过上限则截断。
- `--min-questions`：过滤掉问题数不足的 context。
- `--gradient-accumulation-steps`、`--learning-rate`、`--warmup-ratio` 等与 `TrainingArguments` 一致。
- `--learning-rate` 默认 `4e-4`，可根据是否启用 LoRA 或 batch 大小自行调整。
- 若显存紧张，可附加 `--use-lora` 及相关参数（`--lora-r`, `--lora-alpha`, `--lora-dropout`, `--lora-target-modules`），只更新少量 LoRA 权重，大幅节省显存。LoRA 训练完成后输出目录包含适配器权重，需要在推理和评估时同时指定底模。

训练过程中会自动把 `pos2d` 设置到模型的 2D RoPE 上，并为每个 batch 构造列同步 causal mask。如需多卡训练，可直接使用：

```bash
torchrun --nproc_per_node 8 train.py --model-name ./pretrained-columnar
```

## 阶段三：评估与对比
`evaluate.py` 会在 SQuAD 验证集上比较基线模型与分支模型的精确匹配率：

```bash
python evaluate.py \
  --base-model Qwen/Qwen3-4B \
  --ft-model ./parallel-decoder-squad \
  --max-branches 3 \
  --max-eval-samples 64 \
  --max-new-tokens 32
```

- **Baseline**：按传统方式逐个问题生成答案。
- **Parallel**：将同一背景的多个问题一次性并行生成，并对每个分支计算 Exact Match。

如果 `--ft-model` 指向 LoRA 适配器目录，请同时传入 `--ft-base-model` 指向底模：

```bash
python evaluate.py \
  --base-model Qwen/Qwen3-4B \
  --ft-model ./parallel-decoder-squad-lora \
  --ft-base-model Qwen/Qwen3-4B \
  --max-branches 3
```

脚本会输出两组准确率，帮助评估预训练 + 微调后的收益。若需要离线评测，可同时传入 `--local-files-only`，确保仅使用本地缓存。

如果想快速用最原始的 demo 脚本体验分支推理，可直接运行：

```bash
python demo.py --model-name ./parallel-decoder-squad
```

脚本会加载指定模型（默认自动检测本地微调目录），构造示例分支样本并打印线性化文本及每个分支新增的 token，便于快速验证推理流程是否正常。可通过 `--background` 与 `--questions` 自定义输入。若使用 LoRA 适配器，需额外提供 `--base-model` 指向底模，例如：

```bash
python demo.py --model-name ./parallel-decoder-squad-lora --base-model Qwen/Qwen3-4B
```

## 其他工具
- `demo.py`：快速演示脚本，展示列式布局与分支生成的基本流程，可加载任意微调或 LoRA 权重（需配合 `--base-model`）。
- `model.py`：核心实现，涵盖 `Interleaved2DRoPE`、列同步布局构建、分支生成器等组件。
- `data_utils.py`：SQuAD 聚合与过滤逻辑。

## 注意事项
- 若无法联网，请提前下载模型权重并在初始化时传入本地路径；必要时设置 `HF_HUB_ENABLE_HF_TRANSFER=0` 禁止额外请求。
- 默认使用贪心解码。若想启用采样，可在 `ParallelDecoder.generate` 中设置 `do_sample=True` 与合适的 `temperature`。
- 项目仍在快速迭代，细节（如分支对齐策略、掩码构造）随实验推进可能调整，请保持 README 与代码同步。

## 设计思路速记
1. **建模**：通过 2D RoPE 把多个分支映射到不同的 x 轴，使分支共享背景上下文但各自保持时间顺序。
2. **训练**：预训练阶段学习列同步的注意力模式，任务微调将背景-问题组合映射为多分支序列。
3. **推理**：`ParallelDecoder.generate` 在列同步约束下轮询分支，保持缓存独立并行生成。
4. **评估**：`evaluate.py` 同时报告 baseline 与 parallel 的 EM 指标，方便量化收益。

## 后续计划：列式并行时间轴
我们正在进一步收敛“列式时间”设计：
1. 背景 token 按 `[0,0]…[0,N)` 排列，为所有分支提供统一起点。
2. 分支首 token 对齐在同一列，后续 token 靠右对齐，确保列同步。
3. 注意力掩码以列为时间步：只能看到更早列，禁止同列互相可见。
4. 推理阶段保持与训练一致的掩码与 2D RoPE 设置。

请在继续实验时同步更新 README，避免遗忘设计初衷与约束。

## 许可证
本项目基于 [MIT License](LICENSE) 发布。
