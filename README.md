# Day03 BirdCLEF Plus 2026

面向 BirdCLEF+ 2026 的声学生物识别项目。项目目标是建立可复现的本地训练工程，并生成可在 Kaggle CPU Notebook 中运行的首版有效提交文件。

## 项目内容

- `src/` 下的音频读取、特征生成、训练、推理和 Notebook 生成流程
- `tests/` 下针对数据接口、音频裁剪、指标和提交文件的回归测试
- `environment.yml` 与 `requirements.txt` 两套环境定义
- `notebooks/` 下由脚本生成的 Kaggle 提交 Notebook

## 目录说明

- `data/raw/`：本地 Kaggle 官方下载文件，包括 `train.csv`、`taxonomy.csv`、`sample_submission.csv`、`recording_location.txt`、`test_soundscapes/`、`train_audio/`
- `src/`：可复用 Python 模块与命令入口
- `tests/`：核心数据处理、指标和提交格式测试
- `models/`：训练得到的模型权重与元数据
- `submissions/`：本地生成的 `submission.csv`
- `logs/`：训练与推理日志

## 环境配置

### conda

```bash
conda env create -f environment.yml
conda activate kaggle-birdclef-2026
```

### pip

```bash
python -m pip install -r requirements.txt
```

## 数据准备

Kaggle API 下载：

```bash
python -m src.download_data --competition birdclef-2026
```

手动下载时，官方数据文件应解压到 `data/raw/`。目录内应至少包含 `train.csv`、`taxonomy.csv`、`sample_submission.csv`、`train_audio/`。

## 常用命令

生成数据概览：

```bash
python -m src.eda --data-dir data/raw
```

运行调试训练：

```bash
python -m src.train --debug --epochs 1 --limit 256
```

运行首版训练：

```bash
python -m src.train --model efficientnet_b0 --folds 5 --epochs 10
```

生成提交文件：

```bash
python -m src.infer --checkpoint models/baseline.pt --output submissions/submission.csv
```

生成 Kaggle CPU 提交 Notebook：

```bash
python -m src.make_notebook --checkpoint models/baseline.pt
```

生成权重 Dataset 元数据后，`models/` 目录可作为 Kaggle Dataset 上传。`model-dataset-metadata.json` 提供默认权重数据集描述。

运行测试：

```bash
python -m pytest
```

## 基线约束

- 输入音频采样率固定为 32000 Hz。
- 单个训练样本默认裁剪或补齐为 5 秒。
- mel-spectrogram 默认使用 128 个 mel bins。
- 模型首版使用 EfficientNet-B0 多标签分类头。
- 训练损失为 `BCEWithLogitsLoss`。
- 推理输出为 sigmoid 概率，提交列顺序以 `sample_submission.csv` 为准。

## Kaggle 提交说明

Kaggle Notebook 使用 `/kaggle/input/birdclef-2026` 读取官方测试数据，模型权重通过单独 Kaggle Dataset 挂载。Notebook 不依赖联网下载，不要求 GPU 推理，最终输出 `/kaggle/working/submission.csv`。

## Git 约定

以下内容不纳入 Git 版本控制：

- 官方竞赛数据
- 模型权重与中间训练产物
- 日志文件
- 本地提交文件
- 临时 Notebook 输出

## License

This repository is released under the MIT License. See `LICENSE` for details.
