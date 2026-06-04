# BirdCLEF Plus 2026

BirdCLEF Plus 2026 是面向 Kaggle BirdCLEF+ 2026 的声学生物多标签识别项目。仓库提供可复现的本地训练流程、可在 Kaggle CPU 环境运行的推理 Notebook，以及覆盖核心数据和提交逻辑的回归测试。

项目将竞赛数据、模型权重、生成提交文件、公开 Kernel 下载内容和本地运行输出排除在 Git 之外。公开仓库仅保留适合展示和复现的工程主体。

## 项目结构

| 路径 | 用途 |
| --- | --- |
| `src/` | 音频读取、特征生成、元数据处理、训练、推理和 Notebook 生成模块。 |
| `tests/` | 音频裁剪、数据解析、指标计算和提交格式的回归测试。 |
| `notebooks/` | 生成的 Kaggle CPU 推理 Notebook。 |
| `data/raw/` | 本地 Kaggle 官方数据目录；原始文件不纳入 Git。 |
| `models/` | 本地模型权重和 Kaggle Dataset 上传内容；模型产物不纳入 Git。 |
| `submissions/` | 本地生成的 `submission.csv`；提交文件不纳入 Git。 |
| `logs/` | 本地训练、推理和自动化日志；日志文件不纳入 Git。 |
| `archive/` | 不属于主复现流程的历史赛务记录或脚本。 |

## 环境配置

Conda 环境：

```bash
conda env create -f environment.yml
conda activate kaggle-birdclef-2026
```

Pip 环境：

```bash
python -m pip install -r requirements.txt
```

环境依赖包含 Python 数据处理工具、PyTorch、torchaudio、timm、librosa、onnxruntime、Kaggle API 客户端和 pytest。

## 数据准备

Kaggle API 下载：

```bash
python -m src.download_data --competition birdclef-2026
```

手动准备数据时，官方竞赛文件应解压到 `data/raw/`。本地目录至少包含：

- `train.csv`
- `taxonomy.csv`
- `sample_submission.csv`
- `recording_location.txt`
- `train_audio/`

可选的 `test_soundscapes/` 目录用于本地推理 smoke test。Kaggle 正式提交使用竞赛运行环境提供的隐藏测试音景。

## 常用流程

生成数据概览：

```bash
python -m src.eda --data-dir data/raw
```

运行小规模调试训练：

```bash
python -m src.train --debug --epochs 1 --limit 256
```

运行基线训练流程：

```bash
python -m src.train --model efficientnet_b0 --folds 5 --epochs 10
```

生成本地提交文件：

```bash
python -m src.infer --checkpoint models/baseline.pt --output submissions/submission.csv
```

生成 Kaggle CPU 推理 Notebook：

```bash
python -m src.make_notebook --checkpoint models/baseline.pt
```

生成的 Notebook 从 `/kaggle/input/birdclef-2026` 读取官方数据，从单独的 Kaggle Dataset 加载模型权重，并写出 `/kaggle/working/submission.csv`。提交 Notebook 不依赖联网下载，也不要求 GPU 推理。

## 基线配置

| 配置项 | 默认值 |
| --- | --- |
| 采样率 | 32000 Hz |
| 音频片段长度 | 5 秒 |
| Mel bins | 128 |
| FFT size | 2048 |
| Hop length | 512 |
| 模型 | EfficientNet-B0 多标签分类头 |
| 损失函数 | `BCEWithLogitsLoss` |
| Fold 数量 | 5 |
| 推理输出 | 按 `sample_submission.csv` 列顺序排列的 sigmoid 概率 |

训练命令会写出 fold checkpoint、`models/baseline.pt` ensemble manifest，以及 `models/baseline_metrics.json` 指标摘要。`model-dataset-metadata.json` 提供默认 Kaggle Dataset 元数据模板，用于模型权重上传。

## 测试

运行回归测试：

```bash
python -m pytest
```

测试覆盖音频 crop/pad 行为、元数据解析、指标计算、提交文件形状和列顺序约束。

## Git 策略

仓库默认不纳入以下内容：

- Kaggle 官方竞赛数据
- 模型 checkpoint 和中间训练产物
- 生成的提交文件
- 日志和临时输出
- 下载或复现的公开 Kaggle Kernel
- 大型生成二进制文件

该策略保持公开仓库轻量，同时保留可复现的源码流程。

## English Documentation

English documentation is available in [`README.md`](README.md).

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE) for details.
