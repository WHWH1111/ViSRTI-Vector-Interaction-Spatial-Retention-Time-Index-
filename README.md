# VisNet-RT: Graph Neural Network for Retention Time Prediction

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[RDKit](https://www.rdkit.org/)

VisNet-RT是一个基于图神经网络（GNN）的小分子液相色谱保留时间预测系统。该系统通过端到端学习获得分子的数据驱动表示，并利用这些表示预测小分子的保留时间。

## 目录
- [简介](#简介)
- [安装](#安装)
- [数据准备](#数据准备)
- [使用方法](#使用方法)
- [项目结构](#项目结构)
- [模型介绍](#模型介绍)
- [评估指标](#评估指标)
- [可视化分析](#可视化分析)
- [开发工具](#开发工具)

## 简介

VisNet-RT接收分子图作为输入，输出预测的保留时间。它利用图神经网络的强大能力，从分子结构中自动学习特征，并结合多种辅助特征（物理化学性质、毒性等），提供高精度的保留时间预测。

## 安装

### 环境要求

- Python 3.9
- PyTorch >= 2.7.1
- RDKit
- 其他依赖项请参考 [environment.yml](environment.yml)

### 安装步骤

windows 系统可以参考文档：[pytorch_pyg_gpu_installation](docs/setup/pytorch_pyg_gpu_installation.md)。

> 不过可能会会出现兼容性问题，需要具体情况具体处理了。

## 数据准备

> 具体情况具体处理。

## 使用方法

### 训练模型

要训练新的模型，可以参考 cli 脚本 [cli-4.md](docs/cli-4.md)。

训练脚本支持多种参数配置，以下是一些VisNet V2模型训练示例：

```bash
# 正离子模式训练示例
python train_visnet_v2.py --visnet-v2-feature-level graph_physchem_toxicity --standardize --loss-function l1 --batch-train 64 --standardize-features --visnet-v2-physchem-mask 1 1 1 0

# 负离子模式训练示例
python train_visnet_v2.py --visnet-v2-feature-level graph_physchem_toxicity --standardize --loss-function l1 --batch-train 64 --batch-test 64 --standardize-features --visnet-v2-physchem-mask 1 1 1 0 --visnet-v2-toxicity-mask 1 0 1 1 --standardize --standardize-features -it 100

# 使用不同物理化学特征掩码的训练示例
python train_visnet_v2.py --visnet-v2-feature-level graph_physchem_toxicity --standardize --loss-function l1 --batch-train 64 --batch-test 64 --standardize-features --visnet-v2-physchem-mask 1 0 1 1 --standardize --standardize-features -it 100

# 仅使用图特征的训练示例
python train_visnet_v2.py --visnet-v2-feature-level graph --standardize --loss-function l1 --batch-train 64 --batch-test 64 --standardize-features --standardize --standardize-features -it 100
```

有关更多详细的训练命令示例，请参考 [cli-4.md](docs/cli-4.md) 文件。

### 预测保留时间

使用训练好的模型进行预测：

```bash
# 使用基础预测脚本
python predict.py --model_path <path_to_model> --input_file <input_csv>

# 使用VisNet V2进行预测
python predict_visnet_v2.py --model_path <path_to_model> --params_path <path_to_params> --input_file <input_csv> --output_dir <output_directory> --batch_size <batch_size>
```

### 带有置信度的预测（仅参考）

对于需要预测置信度的场景，可以使用专门的置信度预测脚本：

```bash
python predict_visnet_v2_with_confidence.py --model_path <path_to_model> --params_path <path_to_params> --input_file <input_csv> --output_dir <output_directory> --batch_size <batch_size> --n_iterations <number_of_iterations> --dataset_type <train|test>
```

### 五折交叉验证预测

五折交叉验证预测示例：

```bash
python predict_visnet_v2.py --model_path <path_to_model> --params_path <path_to_params> --input_file <input_csv> --output_dir <output_directory> --batch_size <batch_size> --dataset_file <fold_file> --cache_name <cache_name>

# 批量五折预测
python scripts/batch_predict_visnet_v2.py --model_root <model_directory> --data_root <data_directory> --ion_type <pos|neg>
```

有关更多详细的预测命令示例，请参考 [cli-4.md](docs/cli-4.md) 文件。

## 项目结构

该项目按照模块化原则组织，便于维护和扩展。以下是项目的整体结构：

```
visnet-rt/
├── core/                     # 核心组件
│   ├── data_preprocessor.py  # 数据预处理模块
│   └── trainer_tester.py     # 训练和测试模块
├── models/                   # 模型定义
│   ├── base_model.py         # 基础模型
│   ├── visnet.py             # VisNet模型
│   ├── visnet_core.py        # VisNet核心组件
│   ├── visnet_v2.py          # VisNet V2模型（当前主要模型）
│   ├── dual_output_model.py  # 双输出模型
│   └── extended_model.py     # 扩展模型
├── data/                     # 数据相关文件和脚本
│   ├── MMF/                  # MMF数据集
│   ├── MMF-3/                # MMF-3数据集
│   └── convert_csv_to_excel.py  # 数据格式转换工具
├── utils/                    # 工具函数
│   ├── atom_types.py         # 原子类型处理
│   ├── evaluation_utils.py   # 评估工具
│   ├── feature_utils.py      # 特征处理工具
│   ├── log_utils.py          # 日志工具
│   ├── molecule.py           # 分子处理工具
│   ├── plot_utils.py         # 绘图工具
│   ├── prediction_plot_utils.py  # 预测结果绘图工具
│   └── shap/                 # SHAP分析工具
│       └── draw.py           # SHAP可视化工具
├── scripts/                  # 脚本目录
│   ├── batch_predict_visnet_v2.py  # VisNet V2批量预测脚本
│   └── utils/                # 脚本工具
│       ├── convert_csv.py    # CSV转换工具
│       └── csv2xlsx.py       # CSV转Excel工具
├── docs/                     # 文档目录
│   ├── dev-exp/              # 开发经验分享
│   ├── setup/                # 安装配置文档
│   ├── summary/              # 项目总结文档
│   ├── cli-4.md              # CLI命令文档
│   ├── shap-cli.md           # SHAP CLI命令文档
│   ├── shap.md               # SHAP分析文档
│   ├── uncertainty_pred.md   # 不确定性预测文档
│   └── visnet_v2_model_architecture.md  # VisNet V2模型架构文档
├── predictions/              # 预测结果目录
└── 主要执行文件              # 核心功能入口
    ├── analyze_shap.py       # SHAP分析主程序
    ├── cross_validation.py   # 交叉验证主程序
    ├── error_threshold_clustering.py  # 误差阈值聚类分析
    ├── predict.py            # 预测主程序
    ├── predict_visnet_v2.py  # VisNet V2专用预测程序
    ├── predict_visnet_v2_with_confidence.py  # VisNet V2带置信度预测程序
    ├── preprocess.py         # 预处理程序
    ├── train.py              # 训练主程序
    ├── train_visnet_v2.py    # VisNet V2专用训练程序
    ├── train_visnet_v2_5.py  # VisNet V2五折交叉验证训练程序
    └── visnet_v2_for_gendata.py  # VisNet V2数据生成程序
```

项目遵循清晰的功能分离原则：
- [core/](core/) 目录包含核心训练和数据处理逻辑
- [models/](models/) 目录包含各种模型实现
- [utils/](utils/) 目录包含通用工具函数
- [data/](data/) 目录存储数据集和数据处理脚本
- [scripts/](scripts/) 目录包含各种分析、处理和可视化脚本
- [tests/](tests/) 目录包含单元测试和集成测试
- [dev-tools/](dev-tools/) 目录包含开发者工具
- [docs/](docs/) 目录包含项目文档
- [predictions/](predictions/) 目录存储预测结果
- [support/](support/) 目录包含支持文件
- [tmp/](tmp/) 目录存储临时文件

## 模型介绍

项目实现了多种模型架构：

### 1. 基础GNN模型 (base_model.py)

> from gnn-rt project;

传统基于分子指纹的图神经网络模型。

### 2. VisNet V2模型 (visnet_v2.py)
最新版本的多模态保留时间预测器，支持：
- 图结构特征提取
- 物理化学特征处理
- 毒性特征处理
- 色谱质谱特征处理
- 可选的注意力机制和门控融合机制
- 模块化特征级别控制

VisNet V2支持四种特征级别：
- `graph`: 仅使用图结构特征
- `graph_physchem`: 使用图结构+物理化学特征
- `graph_physchem_toxicity`: 使用图结构+物理化学+毒性特征
- `all`: 使用所有特征

## 评估指标

系统使用以下评估指标衡量模型性能：

1. **R² Score**: 决定系数，衡量预测值与真实值的相关性
2. **RMSE**: 均方根误差，衡量预测误差的大小
3. **MAE**: 平均绝对误差
4. **MedAE**: 中位数绝对误差
5. **Accuracy Metrics**: 准确度相关指标（如在特定误差范围内的预测比例）

## 可视化分析

项目提供丰富的可视化工具，主要位于 `scripts/plt/` 和 `utils/` 目录下：

### 训练过程可视化
```bash
# 绘制训练指标曲线
python scripts/plt/plot_training_metrics.py
```

### 预测结果分析
```bash
# 绘制预测值与实际值的散点图和残差分析
python scripts/plt/plot_prediction_residuals.py

# 绘制测试指标的柱状图和折线图
python scripts/plt/plot_test_metrics_bar_line.py
```

### SHAP分析
```bash
# 进行SHAP解释性分析
python analyze_shap.py

# 绘制SHAP特征重要性指标
python scripts/plt/plot_pos_shap_metrics.py
```

### 特征分析
```bash
# 绘制特征类别热力图
python scripts/plt/plot_feature_category_heatmap.py
```

### 小提琴图分析
```bash
# 生成小提琴图
python scripts/plt/generate_violin_plot.py
```

### 聚类分析
```bash
# 对预测误差进行聚类分析
python scripts/analyze/cluster_analysis.py

# 基于标签的聚类分析
python scripts/analyze/cluster_analysis-pass-tags.py

# 误差阈值聚类分析
python error_threshold_clustering.py
```

### 指标分析
```bash
# 计算尾部10%数据的指标
python scripts/analyze/calculate_metrics_tail10.py

# 基于标签计算指标
python scripts/analyze/calculate_tag_based_metrics.py

# 分析标准偏差与准确率的关系
python scripts/analyze/analyze_std_accuracy.py

# 分析不确定化合物
python scripts/analyze/analyze_uncertain_compounds.py
```

## 开发工具

项目包含多个实用的开发工具：

| 工具 | 功能 |
|------|------|
| [analyze_multiple_logs.py](dev-tools/analyze_multiple_logs.py) | 分析多个日志文件 |
| [analyze_skipped_distribution.py](dev-tools/analyze_skipped_distribution.py) | 分析被跳过样本的分布 |
| [filter_data.py](dev-tools/filter_data.py) | 数据过滤工具 |
| [merge_predictions.py](dev-tools/merge_predictions.py) | 合并预测结果 |
