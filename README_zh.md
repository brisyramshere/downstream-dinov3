# DINOv3 Downstream Tasks

基于 Meta AI DINOv3 模型的下游任务实现，专注于分类和分割任务。该项目整合了完整的 DINOv3 源码，实现了多种架构变体，支持自然图像分析和医学图像处理。

## 🚀 项目特色

- **完整的 DINOv3 集成**: 包含官方 DINOv3 完整源码，支持 vits16、vitb16、vitl16、vit7b16 等多种预训练模型
- **多架构支持**: 实现了 UNet、DPT、FAPM 等多种分割架构和线性分类器
- **任务多样性**: 支持分类(ImageNette)、自然图像分割(ADE20K)、医学图像分割等多个任务
- **统一接口**: 提供统一的训练、推理框架，支持配置文件驱动的灵活切换

## 📁 项目结构

```
├── dinov3/                    # DINOv3 完整源码
│   ├── models/               # DINOv3 模型实现
│   ├── data/                 # 官方数据加载器
│   ├── eval/                 # 官方评估脚本
│   └── configs/              # DINOv3 官方配置
├── models/                    # 下游任务模型实现
│   ├── backbones.py          # 统一的 DINOv3 backbone 加载器
│   ├── dinov3_unet.py        # DINOv3-UNet 分割模型
│   ├── dinov3_seg_dpt.py     # DINOv3-DPT 分割模型
│   ├── dinov3_unet_fapm.py   # DINOv3-FAPM 高级分割模型
│   └── dinov3_linear_cls.py  # DINOv3 线性分类器
├── data/                      # 数据集加载器
│   ├── Dataset_ADE20k.py     # ADE20K 分割数据集
│   ├── Dataset_Imagenette2.py # ImageNette 分类数据集
│   └── dinov3_transforms.py  # DINOv3 官方数据增强
├── configs/                   # 任务配置文件
│   ├── classification_imagenette.yaml
│   └── segmentation_ade20k.yaml
├── train_classifier.py       # 分类任务训练脚本
├── train_segmentor.py        # 分割任务训练脚本
├── inference_classifier.py   # 分类推理脚本
└── inference_segmentor.py    # 分割推理脚本
```

## 🛠️ 环境安装

### 依赖安装
```bash
pip install -r requirements.txt
```

核心依赖：
- torch, torchvision
- timm, PyYAML, tqdm
- einops, scikit-learn

### 预训练权重下载
下载 DINOv3 官方预训练权重到 `checkpoints/` 目录：

```bash
mkdir checkpoints
# 下载所需的预训练权重文件到此目录
# dinov3_vits16_pretrain_lvd1689m-08c60483.pth
# dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
# dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
# dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
```

## 🎯 支持的任务和数据集

### 分类任务
- **ImageNette2**: 10类图像分类任务
- 特征提取方式：cls token、patch average、或两者结合

### 分割任务
- **ADE20K**: 150类自然场景分割
- **MRI Head 2D**: 医学图像二分类分割
- 架构选择：UNet、DPT、FAPM

## 🚀 快速开始

### 数据准备

**ImageNette2 分类数据集**:
```bash
python data/download_imagenette2.py
```

**ADE20K 分割数据集**:
```bash
python data/download_ade20k.py
```

### 训练

**图像分类训练**:
```bash
python train_classifier.py --config configs/classification_imagenette.yaml
```

**图像分割训练**:
```bash
python train_segmentor.py --config configs/segmentation_ade20k.yaml
```

### 推理

**分类推理**:
```bash
python inference_classifier.py --checkpoint output/xxx/checkpoint.pth --image path/to/image.jpg
```

**分割推理**:
```bash
python inference_segmentor.py --checkpoint output/xxx/checkpoint.pth --image path/to/image.jpg
```

## ⚙️ 配置系统

项目采用 YAML 配置文件系统，支持灵活的模型和训练参数配置。

## 🏗️ 模型架构

### DINOv3 Backbone
- 统一的预训练模型加载接口 (`models/backbones.py`)
- 支持 vits16, vitb16, vitl16, vit7b16 变体
- 本地检查点管理，backbone 参数默认冻结

### 分割模型家族

**DinoV3_UNet** (`models/dinov3_unet.py`)（自创的类unet融合结构，实测分割精度最高，而且参数两也很少）:
- 简洁的 UNet 架构，多层特征融合
- 适用于标准分割任务

**DinoV3_DPT** (`models/dinov3_seg_dpt.py`)（论文：https://arxiv.org/abs/2509.00833v1）:
- Dense Prediction Transformer 架构
- 基于特征投影和融合

**DinoV3_FAPM** (`models/dinov3_unet_fapm.py`)（论文：https://arxiv.org/abs/2508.20909v1）:
- Feature Alignment Pyramid Module
- 支持多尺度分割，适用于复杂场景

### 分类模型

**DinoV3LinearClassifier** (`models/dinov3_linear_cls.py`):
- 线性分类头，支持多种特征提取方式
- 特征来源：cls token、patch average、或两者结合

## 📊 训练流程

1. **配置加载**: 从 YAML 文件加载所有训练参数
2. **数据准备**: 自动选择对应的数据集和数据增强
3. **模型构建**: 根据配置动态选择模型架构
4. **训练循环**: 统一的训练引擎，支持检查点保存和恢复
5. **评估指标**:
   - 分类：Top-1/Top-5 准确率
   - 分割：mIoU (mean Intersection over Union)

## 🔧 推理工具

### 单图像推理
支持单张图像的快速推理，输出可视化结果。

### 批量推理
支持批处理多张图像，提高推理效率。

### 3D 医学图像推理
专门针对医学图像的分块批处理推理工具。

## 🎨 数据增强

### 标准增强
- RandomResizedCrop
- RandomHorizontalFlip
- 标准化 (ImageNet 统计值)

### 任务专用增强
- **MRI 图像**: FixedGamma(0.75) 用于低灰度像素增强
- **自然图像**: 标准 ImageNet 预处理

## 📈 性能优化

- **内存优化**: 推理时使用 `torch.no_grad()` 和 `model.eval()`
- **批处理**: 支持批量推理，提高 GPU 利用率
- **参数冻结**: DINOv3 backbone 参数冻结，只训练任务头
- **混合精度**: 支持自动混合精度训练 (可配置)

## 🔍 模型选择指南

### 选择 Backbone
- **vits16**: 最轻量，适合快速验证和资源受限环境
- **vitb16**: 平衡性能和效率
- **vitl16**: 更好的性能，需要更多计算资源
- **vit7b16**: 最佳性能，需要大量计算资源

### 选择分割架构
- **UNet**: 简洁高效，适合标准分割任务
- **DPT**: 特征投影融合，适合需要精细特征的任务
- **FAPM**: 多尺度金字塔，适合复杂场景分割

## 📝 输出和日志

### 检查点保存
- `checkpoint.pth`: 最新检查点
- `best_checkpoint.pth`: 最佳性能检查点

### 训练日志
- 实时显示训练损失和评估指标
- 支持自定义打印频率
- 自动保存训练配置

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详细信息。

## 🙏 致谢

- Meta AI 团队的 [DINOv3](https://github.com/facebookresearch/dinov3) 项目
- PyTorch 社区和相关开源项目

## 📞 联系方式

如有问题或建议，请通过 [Issues](https://github.com/your-username/downstream-dinov3/issues) 与我们联系。
