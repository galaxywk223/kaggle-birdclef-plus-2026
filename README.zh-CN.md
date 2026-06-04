# BirdCLEF Plus 2026

BirdCLEF Plus 2026 是面向 Kaggle BirdCLEF+ 2026 竞赛的声学生物识别项目。系统将长时野外录音转换为多标签物种预测，并覆盖从训练代码到 Kaggle CPU 提交 Notebook 的完整链路。

项目聚焦真实竞赛场景：固定时间窗、mel-spectrogram 特征、多标签分类、fold checkpoint、ensemble 推理，以及提交列顺序严格对齐。

## 项目亮点

- 面向 BirdCLEF+ 2026 音频分类任务的端到端 Python 管线。
- 具备确定性 crop 和 padding 行为的 mel-spectrogram 预处理。
- EfficientNet-B0 多标签基线模型，并提供轻量 CNN fallback。
- Fold checkpoint 导出和 ensemble manifest 支持。
- 生成可在 Kaggle CPU 环境离线运行的提交 Notebook。
- 回归测试覆盖音频窗口、元数据解析、macro AUC 和提交文件形状。

## 系统设计

| 阶段 | 实现 |
| --- | --- |
| 数据读取 | 竞赛元数据、taxonomy 标签、sample submission 列和 soundscape row ID。 |
| 音频预处理 | 32000 Hz 波形读取、五秒窗口截取和 mel-spectrogram 转换。 |
| 模型训练 | Fold 划分、多标签目标、PyTorch dataloader、EfficientNet 风格图像分类器和 BCE loss。 |
| 验证 | 基于标签列的 macro AUC 计算。 |
| Checkpoint | Fold 权重、ensemble manifest、音频配置和类别顺序。 |
| 推理 | Soundscape 分段、批量预测、sigmoid 概率和列顺序校验。 |
| 提交打包 | 生成读取竞赛输入并写出 `submission.csv` 的 Kaggle Notebook。 |

## 推理流程

推理链路面向 Kaggle CPU 运行环境设计：

1. 读取 `sample_submission.csv`，锁定 row ID 和类别列顺序。
2. 将每个 row ID 解析为 soundscape ID 和结束时间戳。
3. 每条 soundscape 只读取一次，并在多个时间窗之间复用。
4. 为每一行截取目标五秒音频片段。
5. 将音频片段转换为归一化 mel-spectrogram。
6. 使用 ensemble manifest 时，对多个成员模型的预测取平均。
7. 按竞赛提交格式写出概率结果。

生成的 Notebook 使用 `/kaggle/input/birdclef-2026` 读取官方数据，并通过单独的 Kaggle Dataset 加载模型权重。提交推理不依赖联网下载，也不要求 GPU。

## 仓库内容

| 路径 | 内容 |
| --- | --- |
| `src/audio.py` | 音频读取、波形分段和 mel-spectrogram 特征。 |
| `src/data.py` | 元数据读取、类别提取、row ID 解析和 fold 划分。 |
| `src/train.py` | 基线 fold 训练和 checkpoint 导出。 |
| `src/infer.py` | 基于单 checkpoint 或 ensemble manifest 的提交推理。 |
| `src/make_notebook.py` | Kaggle CPU Notebook 生成。 |
| `tests/` | 核心竞赛契约的回归测试。 |
| `notebooks/` | 生成的推理 Notebook 产物。 |

## 复现命令

创建 Conda 环境：

```bash
conda env create -f environment.yml
conda activate kaggle-birdclef-2026
```

使用 pip 安装依赖：

```bash
python -m pip install -r requirements.txt
```

通过 Kaggle API 下载竞赛数据：

```bash
python -m src.download_data --competition birdclef-2026
```

生成数据概览：

```bash
python -m src.eda --data-dir data/raw
```

运行调试训练：

```bash
python -m src.train --debug --epochs 1 --limit 256
```

训练基线模型：

```bash
python -m src.train --model efficientnet_b0 --folds 5 --epochs 10
```

生成提交文件：

```bash
python -m src.infer --checkpoint models/baseline.pt --output submissions/submission.csv
```

生成 Kaggle CPU Notebook：

```bash
python -m src.make_notebook --checkpoint models/baseline.pt
```

运行测试：

```bash
python -m pytest
```

## 基线配置

| 配置项 | 数值 |
| --- | --- |
| 采样率 | 32000 Hz |
| 窗口长度 | 5 秒 |
| Mel bins | 128 |
| FFT size | 2048 |
| Hop length | 512 |
| 模型 | EfficientNet-B0 多标签分类器 |
| 损失函数 | `BCEWithLogitsLoss` |
| Fold 数量 | 5 |
| 输出 | 按 `sample_submission.csv` 对齐的 sigmoid 概率 |

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE) for details.
