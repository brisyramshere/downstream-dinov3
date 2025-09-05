# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目概述

这是一个基于 Meta AI DINOv3 模型的下游任务实现，专注于分类和分割任务。该代码库实现了完整的 DINOv3-UNet 分割架构，以及用于医学图像分析（特别是 MRI 头部图像分割）的专用流水线。

## 开发环境设置

### 依赖安装
```bash
pip install -r requirements.txt
# 核心依赖：torch, torchvision, timm, PyYAML, tqdm, einops, scikit-learn
```

### CUDA 扩展编译
MultiScaleDeformableAttention 需要 CUDA 编译：
```bash
cd models/segmentation_from_dinov3/models/utils/ops
python setup.py build_ext --inplace
```

## 常用命令

### 训练
```bash
# 医学图像分割训练
python train_segmentor.py --config configs/segmentation_mrihead.yaml

# 图像分类训练  
python train_classifier.py --config configs/classification_imagenet.yaml
```

### 推理
```bash
# 单张图像推理
python inference_segmentor.py --checkpoint output/xxx/checkpoint.pth --image path/to/image.jpg

# 3D 医学图像批量推理
python inference_3d.py --checkpoint output/xxx/checkpoint.pth --input_nii input.nii.gz --batch_size 8
```

### 测试 CUDA 扩展
```bash
python test_cuda_extension.py
```

## 核心架构

### 模型架构层次
1. **DINOv3 Backbone** (`models/backbone.py`)
   - 预训练的 DINOv3 视觉 Transformer
   - 支持 vits16, vitb16, vitl16 等变体
   - 参数默认冻结，仅训练下游头

2. **分割模型**
   - **DinoV3_DPT** (`models/segmentation_model.py`): 基于 Dense Prediction Transformer
   - **DinoV3_UNet_Full** (`models/dinov3_unet_full.py`): 完整 DINOv3_Adapter + UNet 架构
     - 包含 DINOv3_Adapter (`models/segmentation_from_dinov3/models/backbone/dinov3_adapter.py`)
     - Spatial Prior Module + MultiScaleDeformableAttention
     - SimpleFeatureProjection 用于通道维度适配

3. **分类模型** (`models/classification_model.py`)

### 数据处理流水线
- **数据集适配器**: `ADE20Dataset`, `MRIHead2DDataset`
- **数据增强**: `data/augmentations.py`
  - 几何变换：RandomRotate, RandomHorizontallyFlip, RandomSizedCrop
  - 光度调整：FixedGamma(0.75) 用于 MRI 图像预处理
- **MRI 专用预处理**: `data/preprocess_mri.py`

### 训练框架
- **训练引擎**: `engine.py` - 标准训练/验证循环
- **损失函数**: `loss/losses.py` - CE_DiceLoss (CrossEntropy + Dice)
- **配置驱动**: YAML 配置文件支持模型类型切换

## 关键实现细节

### 模型类型切换
配置文件中通过 `model_type` 参数控制：
```yaml
model:
  model_type: "unet_full"  # 或 "dpt"
```

### 批处理推理
- `inference_segmentor.py` 的 `run_batch_inference_on_images()` 支持 batch_size=1 到 N
- `inference_3d.py` 实现 3D 体积的分块批处理推理

### 医学图像特殊处理
- **固定 gamma 校正**(0.75): 增强低灰度像素分布
- **训练和推理保持一致**: 预处理管线完全匹配
- **ignore_index=255**: 处理数据增强中的填充区域

### CUDA 内存和版本兼容性
- **avoid in-place operations**: 使用 `.clone()` 避免梯度计算错误
- **torchvision API 适配**: `fillcolor` → `fill`, `resample` → `interpolation`

## 数据组织结构

### 配置文件
- `configs/segmentation_mrihead.yaml`: MRI 分割任务
- `configs/segmentation_ade20k.yaml`: ADE20K 分割
- `configs/classification_*.yaml`: 分类任务

### 检查点和输出
- `output/segmentation/dinov3_vits16_unet_full_mrihead/`: 训练输出
- 检查点包含 model state_dict 和完整 config

### 预训练权重
- DINOv3 预训练权重自动从 torch.hub 加载
- 本地检查点支持：`models/backbone.py` 中的 CHECKPOINT_MAPPING

## 重要注意事项

### PyTorch In-place Operations
参考 `lesson_learn.md` - 避免原地操作导致的梯度计算错误。在 loss 函数中使用 `.clone()` 创建张量副本。

### CUDA 扩展编译
MultiScaleDeformableAttention 依赖 CUDA 编译。确保：
1. CUDA toolkit 版本与 PyTorch 匹配
2. 编译后的 .pyd/.so 文件在正确路径
3. 使用 `test_cuda_extension.py` 验证

### 内存管理
- 3D 推理时合理设置 batch_size（推荐 4-8）
- 大型模型（vitl16, vitg14）需要更多 GPU 内存
- 使用 `torch.no_grad()` 进行推理

### 配置文件继承
YAML 配置支持模型参数动态切换，训练脚本会自动选择相应的模型架构和参数。