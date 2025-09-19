# DINOv3 Downstream Tasks

Read this in Chinese: [README_zh.md](README_zh.md)

Downstream tasks based on Meta AI's DINOv3 model, focusing on classification and segmentation. This project integrates the full DINOv3 source code, implements multiple architectural variants, and supports both natural image analysis and medical image processing.

## 🚀 Highlights

- Complete DINOv3 integration: includes the official DINOv3 source code and supports multiple pretrained models (vits16, vitb16, vitl16, vit7b16)
- Multiple architectures: implementations of UNet, DPT, FAPM segmentation architectures and a linear classifier
- Task diversity: supports classification (ImageNette), natural image segmentation (ADE20K), and medical image segmentation
- Unified interface: a unified training and inference framework, with flexible switching driven by config files

## 📁 Project Structure

```
├── dinov3/                    # Full DINOv3 source code
│   ├── models/               # DINOv3 model implementations
│   ├── data/                 # Official data loaders
│   ├── eval/                 # Official evaluation scripts
│   └── configs/              # Official DINOv3 configs
├── models/                    # Downstream task models
│   ├── backbones.py          # Unified DINOv3 backbone loader
│   ├── dinov3_unet.py        # DINOv3-UNet segmentation model
│   ├── dinov3_seg_dpt.py     # DINOv3-DPT segmentation model
│   ├── dinov3_unet_fapm.py   # DINOv3-FAPM advanced segmentation model
│   └── dinov3_linear_cls.py  # DINOv3 linear classifier
├── data/                      # Dataset loaders
│   ├── Dataset_ADE20k.py     # ADE20K segmentation dataset
│   ├── Dataset_Imagenette2.py # ImageNette classification dataset
│   └── dinov3_transforms.py  # Official DINOv3 transforms
├── configs/                   # Task configs
│   ├── classification_imagenette.yaml
│   └── segmentation_ade20k.yaml
├── train_classifier.py       # Classification training script
├── train_segmentor.py        # Segmentation training script
├── inference_classifier.py   # Classification inference script
└── inference_segmentor.py    # Segmentation inference script
```

## 🛠️ Installation

### Dependencies
```bash
pip install -r requirements.txt
```

Core dependencies:
- torch, torchvision
- timm, PyYAML, tqdm
- einops, scikit-learn

### Pretrained Weights
Download official DINOv3 pretrained weights into the `checkpoints/` directory:

```bash
mkdir checkpoints
# Download the required pretrained weight files into this directory
# dinov3_vits16_pretrain_lvd1689m-08c60483.pth
# dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
# dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
# dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
```

## 🎯 Supported Tasks and Datasets

### Classification
- ImageNette2: 10-class image classification
- Feature extraction: cls token, patch average, or their combination

### Segmentation
- ADE20K: 150-class scene segmentation
- MRI Head 2D: binary medical image segmentation
- Architectures: UNet, DPT, FAPM

## 🚀 Quick Start

### Data Preparation

ImageNette2:
```bash
python data/download_imagenette2.py
```

ADE20K:
```bash
python data/download_ade20k.py
```

### Training

Image classification:
```bash
python train_classifier.py --config configs/classification_imagenette.yaml
```

Image segmentation:
```bash
python train_segmentor.py --config configs/segmentation_ade20k.yaml
```

### Inference

Classification inference:
```bash
python inference_classifier.py --checkpoint output/xxx/checkpoint.pth --image path/to/image.jpg
```

Segmentation inference:
```bash
python inference_segmentor.py --checkpoint output/xxx/checkpoint.pth --image path/to/image.jpg
```

## ⚙️ Configuration System

YAML-based configuration system with flexible model and training hyperparameters.

## 🏗️ Architectures

### DINOv3 Backbone
- Unified pretrained model loading interface (`models/backbones.py`)
- Supports vits16, vitb16, vitl16, vit7b16 variants
- Local checkpoint management; backbone params frozen by default

### Segmentation Families

DinoV3_UNet (`models/dinov3_unet.py`) — a custom UNet-like fusion design that achieved the best segmentation accuracy in our tests with a small parameter count:
- Simple UNet architecture with multi-level feature fusion
- Suited for standard segmentation tasks

DinoV3_DPT (`models/dinov3_seg_dpt.py`) — Paper: https://arxiv.org/abs/2509.00833v1
- Dense Prediction Transformer architecture
- Based on feature projection and fusion

DinoV3_FAPM (`models/dinov3_unet_fapm.py`) — Paper: https://arxiv.org/abs/2508.20909v1
- Feature Alignment Pyramid Module
- Supports multi-scale segmentation for complex scenes

### Classification Model

DinoV3LinearClassifier (`models/dinov3_linear_cls.py`):
- Linear classification head supporting multiple feature extraction modes
- Feature sources: cls token, patch average, or both

## 📊 Training Pipeline

1. Config loading: load all training params from YAML
2. Data preparation: automatically selects dataset and transforms
3. Model building: dynamically choose architecture per config
4. Training loop: unified engine with checkpoint save/restore
5. Metrics:
   - Classification: Top-1/Top-5 accuracy
   - Segmentation: mIoU (mean Intersection over Union)

## 🔧 Inference Tools

### Single Image Inference
Quick inference for a single image with visualized outputs.

### Batch Inference
Batch multiple images to improve throughput.

### 3D Medical Inference
Chunked batch inference utilities tailored for medical images.

## 🎨 Data Augmentation

### Standard
- RandomResizedCrop
- RandomHorizontalFlip
- Normalization (ImageNet stats)

### Task-specific
- MRI: FixedGamma(0.75) for low-intensity enhancement
- Natural images: standard ImageNet preprocessing

## 📈 Performance Tips

- Memory: use `torch.no_grad()` and `model.eval()` during inference
- Batching: support batched inference to utilize GPU
- Frozen params: freeze DINOv3 backbone and train task heads only
- AMP: optional automatic mixed precision training

## 🔍 Model Selection Guide

### Backbone
- vits16: lightest, good for quick validation and constrained environments
- vitb16: balanced performance and efficiency
- vitl16: higher performance, requires more compute
- vit7b16: best performance, requires substantial compute

### Segmentation Architecture
- UNet: simple and efficient for standard tasks
- DPT: projection + fusion for tasks needing fine features
- FAPM: multi-scale pyramid for complex scenes

## 📝 Outputs and Logs

### Checkpoints
- `checkpoint.pth`: latest
- `best_checkpoint.pth`: best

### Training Logs
- Realtime training loss and metrics
- Configurable print frequency
- Auto-save training config

## 🤝 Contributing

1. Fork this repo
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

- Meta AI's [DINOv3](https://github.com/facebookresearch/dinov3)
- The PyTorch community and related open-source projects

## 📞 Contact

For questions or suggestions, please open an [Issue](https://github.com/your-username/downstream-dinov3/issues).
