# Parallel-Decoding

Parallel-Decoding 是一个实验性项目，用于探索如何在大语言模型中引入 2D 位置编码并实现“并行分支”式的推理与训练流程。本仓库针对 Qwen3 系列模型打了一个定制补丁，使其支持将同一段背景文本后的多个问题视作独立分支，分别生成答案。

## 功能概览
- 自定义 `Interleaved2DRoPE`，在有限的频率通道中交错输入 1D/2D 位置，兼容 Qwen3 的 RoPE 实现。
- `ParallelDecoder` 封装模型加载、布局构建、推理与输出整理，可按分支拆解生成文本。
- 针对 SQuAD 的数据预处理管线：将相同 `context` 下的问题聚合为一个样本，主干存放问题 1，其余问题归入分支。
- 基于 Hugging Face `Trainer` 的微调脚本，支持单卡启动，也可直接切换到 `torchrun` 进行多卡/分布式训练。

## 运行环境
- Python 3.10+
- PyTorch 2.2+（支持 CUDA/MPS 更佳）
- `transformers>=4.55`
- `datasets>=2.14`
- 其余依赖：`huggingface_hub`, `accelerate`, `torchvision`（若使用 GPU）

建议提前手动下载 Qwen3-4B 模型与分词器，或确保当前机器能够访问 Hugging Face Hub。若网络受限，脚本会尝试拉取官方 JSON 并构建数据集，但模型与分词器仍需要本地缓存。

## 快速体验
```bash
python python.py
```

该脚本会：
1. 自动选择当前可用的设备与 dtype；
2. 加载 Qwen/Qwen3-4B，打上 2D RoPE 补丁；
3. 构造两个示例样本（主干 + 分支），执行一次前向与生成；
4. 输出线性化文本以及每个分支新增的 token。

## 数据预处理与训练
微调脚本会将 SQuAD 中相同 `context` 的问答聚合成一个分支样本：主干放问题 1，其余问题依次作为分支。示例命令：

```bash
python train.py \
  --max-train-samples 128 \
  --max-eval-samples 64 \
  --max-branches 3 \
  --batch-size 1 \
  --epochs 1
```

参数说明：
- `--max-branches`：主干之外最多保留的额外问题数，最终每个样本最多包含 `max_branches + 1` 条问答。
- `--max-length`：编码后的序列统一填充到该长度，默认 1024。
- `--max-train-samples / --max-eval-samples`：可选，截断样本数量以便快速调试。

训练时会自动将 `pos2d` 移动到模型所在设备，并在每个 batch 里调用补丁后的 RoPE。若要使用多卡并行，可直接用 `torchrun` 启动，例如：

```bash
torchrun --nproc_per_node 8 train.py --batch-size 1 --eval-batch-size 1
```

## 目录结构
- `model.py`：核心模块，包含 RoPE 补丁、布局构建、推理接口以及结果数据结构。
- `python.py`：演示脚本，可快速验证推理流程。
- `train.py`：微调脚本，含自定义数据整理与 `Trainer` 子类。
- `parallel-decoder-squad/`：默认输出目录（训练权重、分词器等会保存于此）。
- `LICENSE`：项目许可证。

## 注意事项
- 若无法访问 Hugging Face，请先行下载模型权重与分词器到本地，并在 `ParallelDecoder` 初始化时传入本地路径；也可以通过设置 `HF_HUB_ENABLE_HF_TRANSFER=0` 禁止额外的 HTTP 请求。
- 当前实现默认使用贪心解码。若要尝试采样，可在 `ParallelDecoder.generate` 调用中传入 `temperature` 与 `do_sample=True`。
- 项目仍处于实验阶段，数据预处理与分支布局策略都可以根据需求调整。

## 许可证

本项目基于 [MIT License](LICENSE) 发布。
