# C-TDED
High-Resolution Edge Detection Model Based on Dual-Branch Attention Fusion

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview | 概述

C-TDED is a high-performance dual-branch edge detection model that effectively integrates CNN-based branch for fine details with Transformer-based branch for global context.

C-TDED 是一个高性能的双分支边缘检测模型，有效地整合了基于CNN的细节分支和基于Transformer的全局上下文分支。

## Model Architecture | 模型架构

- **Fine Semantic Edge Branch (FSEB)**: Captures high-resolution edge features using modified EfficientNet-B0
- **Global Context Branch (GCB)**: Models global semantic information using lightweight Transformer
- **Memory-Efficient Fusion Module (MEFM)**: Integrates features from both branches

---

- **细粒度语义边缘分支 (FSEB)**：使用改进的EfficientNet-B0捕获高分辨率边缘特征
- **全局上下文分支 (GCB)**：使用轻量级Transformer建模全局语义信息
- **内存高效融合模块 (MEFM)**：整合两个分支的特征

## Installation | 安装

```bash
# Clone the repository | 克隆仓库
git clone https://github.com/Ilk-cq/C-TDED
cd C-TDED

# Install dependencies | 安装依赖
pip install -r requirements.txt
```

## Datasets | 数据集

This model has been evaluated on three benchmark datasets:

本模型在三个基准数据集上进行了评估：

### BSDS500
- **Download | 下载**: [Berkeley Segmentation Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
- **Description | 描述**: 500 natural images with human-annotated boundaries
- **结构**: 500张带有人工标注边界的自然图像

### Multicue
- **Download | 下载**: [Multicue Dataset](https://serre-lab.clps.brown.edu/resource/multicue/)
- **Description | 描述**: Dataset for boundary detection with multiple visual cues
- **结构**: 用于边界检测的多视觉线索数据集

### NYUDv2
- **Download | 下载**: [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- **Description | 描述**: Indoor scene dataset with RGB-D images
- **结构**: 包含RGB-D图像的室内场景数据集

## Usage | 使用方法

### Training | 训练

```python
from training import train_edge_detection_model

# Train on BSDS500 dataset | 在BSDS500数据集上训练
checkpoint_dir = train_edge_detection_model(
    dataset='bsds500',
    data_root='path/to/bsds500',
    epochs=100,
    batch_size=8,
    learning_rate=1e-4,
    img_size=224
)
```

### Inference | 推理

```python
import torch
from model import DualBranchEdgeModel, inference_with_sigmoid

# Load model | 加载模型
model = DualBranchEdgeModel(img_size=224)
model.load_state_dict(torch.load('checkpoint.pth')['model_state_dict'])
model.eval()

# Inference | 推理
with torch.no_grad():
    outputs = inference_with_sigmoid(model, input_image)
    edge_map = outputs['final_edges']
```

## Performance | 性能

| Dataset | ODS | OIS |
|---------|-----|-----|
| BSDS500 | 0.847 | 0.861 |
| Multicue (Edge) | 0.899 | 0.907 |
| Multicue (Boundary) | 0.842 | 0.850 |
| NYUDv2 | 0.761 | 0.776 |

## Key Features | 主要特点

- **High-Resolution Feature Extraction | 高分辨率特征提取**: Modified backbone with reduced initial downsampling
- **Efficient Local Attention | 高效局部注意力**: Window-based attention mechanism (7×7)
- **Structure-Aware Loss | 结构感知损失**: Comprehensive loss function including gradient, continuity, and directional consistency
- **Multi-Scale Supervision | 多尺度监督**: Side outputs at different scales for deep supervision

---

- **高分辨率特征提取**：改进的骨干网络，减少初始下采样
- **高效局部注意力**：基于窗口的注意力机制（7×7）
- **结构感知损失**：包括梯度、连续性和方向一致性的综合损失函数
- **多尺度监督**：不同尺度的侧输出用于深度监督

## File Structure | 文件结构

```
C-TDED/
├── model.py              # Model architecture | 模型架构
├── dataset.py            # Dataset loaders | 数据集加载器
├── edge_losses.py        # Loss functions | 损失函数
├── training.py           # Training script | 训练脚本
├── model_calculate.py    #Calculating model parameters and others
├── requirements.txt      # Dependencies | 依赖项
└── README.md            # This file | 本文件
```

## Requirements | 环境要求

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA >= 11.0 (for GPU support | GPU支持)
- See `requirements.txt` for complete list | 完整列表见 `requirements.txt`

##其他算法详情
HED - Holistically-nested Edge Detection
Full name: Holistically-nested Edge Detection (HED)
GitHub: https://github.com/s9xie/hed
Paper: ICCV 2015

BDCN - Bi-Directional Cascade Network
Full name: Bi-Directional Cascade Network for Perceptual Edge Detection (BDCN)
GitHub: https://github.com/pkuCactus/BDCN
Paper: CVPR 2019

EDTER - Edge Detection with TRansformer
Full name: Edge Detection with Transformer (EDTER)
GitHub: https://github.com/MengyangPu/EDTER
Paper: CVPR 2022

DiffusionEdge - Diffusion Probabilistic Model for Crisp Edge Detection
Full name: DiffusionEdge: Diffusion Probabilistic Model for Crisp Edge Detection
GitHub: https://github.com/GuHuangAI/DiffusionEdge
Paper: AAAI 2024


## License | 许可证

This project is licensed under the MIT License.

本项目采用 MIT 许可证。
