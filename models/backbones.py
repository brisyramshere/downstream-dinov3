import os
import torch
from pathlib import Path
from dinov3.models.vision_transformer import DinoVisionTransformer
from dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16, dinov3_vitl16, dinov3_vit7b16

BACKBONE_PATH = {
    "dinov3_vits16": "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vitb16": "checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", 
    "dinov3_vitl16": "checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vit7b16": "checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
}

MODEL_FACTORIES = {
    "dinov3_vits16": dinov3_vits16,
    "dinov3_vitb16": dinov3_vitb16,
    "dinov3_vitl16": dinov3_vitl16,
    "dinov3_vit7b16": dinov3_vit7b16,
}

DINOv3_INTERACTION_INDEXES = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [4, 11, 17, 23],
    "dinov3_vit7b16": [9, 19, 29, 39],
}

def load_dinov3_backbone(backbone_name: str) -> DinoVisionTransformer:
    """Load DINOv3 model with local pretrained weights"""
    
    if backbone_name not in BACKBONE_PATH:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    # Get local checkpoint path
    current_dir = Path(__file__).parent.parent
    checkpoint_path = current_dir / BACKBONE_PATH[backbone_name]
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Local checkpoint not found: {checkpoint_path}")
    
    # Create model and load weights
    # model_factory = MODEL_FACTORIES[backbone_name]
    # backbone = model_factory(pretrained=False)
    backbone = torch.hub.load('facebookresearch/dinov3', backbone_name, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    backbone.load_state_dict(state_dict, strict=True)
    
    print(f"Loaded {backbone_name} from: {checkpoint_path}")

        # Freeze all parameters in the backbone
    for param in backbone.parameters():
        param.requires_grad = False
    
    return backbone



