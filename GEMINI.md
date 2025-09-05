# 项目概述

该项目是一个基于DINOv3的下游任务微调框架，专注于**语义分割**和**图像分类**任务。项目采用了先进的DINOv3 Vision Transformer作为特征提取器，结合多种分割架构（DPT、UNet等），实现了高质量的语义分割效果。该项目支持多种数据集格式，包括ADE20K和自定义MRI医学影像数据集。

## 核心特性

*   **多模型架构支持**: DPT (Dense Prediction Transformer) 和 UNet-Full 两种分割架构
*   **医学影像支持**: 专门优化的MRI头部2D分割数据集处理
*   **灵活的配置系统**: YAML配置文件驱动，支持多种实验设置
*   **高级损失函数**: 组合Cross-Entropy和Dice损失，针对分割任务优化
*   **完整的训练推理链**: 从训练到推理的端到端解决方案

## 主要技术栈

*   **Python**: 核心编程语言
*   **PyTorch**: 深度学习框架，模型构建和训练
*   **timm**: 提供预训练模型支持
*   **einops**: 张量操作和维度变换
*   **scikit-learn**: 损失函数计算和评估指标
*   **PyYAML**: 配置文件解析
*   **tqdm**: 训练进度可视化

## 项目架构

该项目采用模块化设计，主要组件如下：

### 核心训练模块
*   `train_segmentor.py`: **主要训练入口**，专门用于分割任务的训练和评估
*   `train_classifier.py`: 图像分类任务的训练脚本
*   `engine.py`: 通用训练引擎，包含优化器、调度器和检查点管理

### 模型架构 (`models/`)
*   `backbone.py`: DINOv3主干网络加载和冻结管理
*   `segmentation_model.py`: DinoV3_DPT模型，基于Dense Prediction Transformer
*   `dinov3_unet_full.py`: **DinoV3_UNet_Full模型**，完整的UNet架构与DINOv3_Adapter
*   `classification_model.py`: 线性分类器模型

### 数据处理 (`data/`)
*   `ADE20Dataset.py`: ADE20K数据集加载器
*   `MRIHead2DDataset.py`: **MRI头部2D分割数据集**专用加载器
*   `augmentations.py`: 数据增强策略
*   `utils.py`: 数据处理辅助函数

### 损失函数 (`loss/`)
*   `losses.py`: 多种损失函数，包括CE_DiceLoss组合损失
*   `lovasz_losses.py`: Lovasz损失函数实现

### 推理模块
*   `inference_segmentor.py`: 分割模型推理和可视化
*   `inference_classifier.py`: 分类模型推理
*   `inference_3d.py`: 3D推理支持

### 配置文件 (`configs/`)
*   `segmentation_mrihead.yaml`: MRI头部分割任务配置
*   `segmentation_ade20k.yaml`: ADE20K分割任务配置
*   `classification_*.yaml`: 分类任务配置文件

# 快速开始

## 1. 环境安装

```bash
cd D:\Proj\dino\downstrean-dinov3
pip install -r requirements.txt
```

核心依赖包括：
- torch, torchvision (深度学习框架)
- timm (预训练模型)
- einops (张量操作)
- PyYAML (配置解析)
- tqdm, scikit-learn (训练辅助)

## 2. 语义分割任务 (主要功能)

### 2.1 MRI头部分割 (推荐)

这是项目的核心应用，专门针对医学影像分割优化：

```bash
# 使用UNet-Full架构进行MRI头部分割
python train_segmentor.py
# 默认使用 configs/segmentation_mrihead.yaml 配置
```

**配置特点**：
- 模型架构：DinoV3_UNet_Full (完整UNet + DINOv3_Adapter)
- 输入尺寸：512x512 (适合医学影像)
- 类别数：2 (背景+前景)
- 优化器：AdamW with Cosine调度
- 损失函数：CE_DiceLoss组合损失

### 2.2 ADE20K通用分割

```bash
# 下载ADE20K数据集
python data/download_ade20k.py

# 修改配置文件中的数据路径
# 编辑 configs/segmentation_ade20k.yaml 中的 root_dir

# 启动训练
python train_segmentor.py --config configs/segmentation_ade20k.yaml
```

### 2.3 模型架构选择

项目支持两种分割架构：

**选项1: DinoV3_UNet_Full (推荐)**
```yaml
model:
  model_type: "unet_full"
  interaction_indexes: [2, 5, 8, 11]
  decoder_channels: [256, 128, 64, 32]
```
- 完整的UNet解码器
- DINOv3_Adapter特征提取
- 更好的细节恢复能力

**选项2: DinoV3_DPT**
```yaml
model:
  model_type: "dpt"
  skip_connection_layers: [2, 5, 8, 11]
```
- Dense Prediction Transformer
- 更轻量级的架构

## 3. 推理和可视化

### 3.1 分割推理

```bash
python inference_segmentor.py \
  --checkpoint output/segmentation/dinov3_vits16_unet_full_mrihead/best_checkpoint.pth \
  --image path/to/your/image.jpg \
  --output result_visualization.png
```

### 3.2 分类推理

```bash
python inference_classifier.py \
  --checkpoint output/classification/model_checkpoint.pth \
  --image path/to/image.jpg
```

## 4. 分类任务 (辅助功能)

```bash
# 下载Imagenette数据集
python data/download_data.py

# 训练分类器
python train_classifier.py --config configs/classification_imagenette.yaml
```

# 开发设计原则

## 核心理念

*   **配置驱动**: YAML配置文件控制所有实验参数，实现代码和配置分离
*   **模块化架构**: 清晰的模块边界，每个组件职责单一且可复用
*   **冻结主干**: DINOv3主干网络始终冻结，只训练任务特定的头部网络
*   **灵活扩展**: 支持多种模型架构和数据集，易于添加新的任务类型

## 关键技术特性

### 模型架构特点
*   **DINOv3_UNet_Full**: 集成DINOv3_Adapter和完整UNet解码器，支持多尺度特征融合
*   **DINOv3_DPT**: 基于Dense Prediction Transformer的轻量级架构
*   **特征投影**: 简化的特征投影模块，替代复杂的FAPM结构
*   **跳跃连接**: UNet风格的跳跃连接，保持空间细节信息

### 训练策略
*   **组合损失**: CE_DiceLoss结合交叉熵和Dice损失，优化分割效果
*   **自适应优化**: AdamW优化器配合余弦退火调度
*   **数据增强**: 随机裁剪、水平翻转等增强策略
*   **mIoU评估**: 使用平均交并比作为主要评估指标

### 代码组织
*   **train_segmentor.py**: 分割任务的主要入口点，集成评估逻辑
*   **engine.py**: 通用训练引擎，处理优化器、调度器和检查点
*   **动态数据集选择**: 根据配置自动选择数据集类型(ADE20K/MRIHead2D)
*   **模型自动构建**: 根据配置自动选择和构建不同的模型架构

## 扩展指南

### 添加新数据集
1. 在`data/`目录创建新的Dataset类
2. 在配置文件中指定`dataset_name`
3. 在训练脚本中添加数据集选择逻辑

### 添加新模型
1. 在`models/`目录实现新的模型类
2. 在配置文件中指定`model_type`
3. 在训练脚本中添加模型构建逻辑

### 自定义损失函数
1. 在`loss/losses.py`中实现新的损失类
2. 在配置文件中指定损失类型
3. 在训练脚本中集成新的损失函数
