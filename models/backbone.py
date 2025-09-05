import torch
import os
from pathlib import Path

SUPPORTED_BACKBONES = [
    'dinov3_vits16',
    'dinov3_vits14',
    'dinov3_vitb16',
    'dinov3_vitb14',
    'dinov3_vitl16',
    'dinov3_vitl14',
    'dinov3_vitg14',
]

# Mapping from model names to checkpoint filenames
CHECKPOINT_MAPPING = {
    'dinov3_vits16': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
    'dinov3_vits16plus': 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth',
}

def get_dinov3_backbone(model_name: str, pretrained: bool = True, use_local: bool = True, device: str = None):
    """
    Loads a DINOv3 backbone from local checkpoint or torch.hub and freezes its parameters.

    Args:
        model_name (str): The name of the DINOv3 model to load.
                          Must be one of SUPPORTED_BACKBONES.
        pretrained (bool): Whether to load pretrained weights.
        use_local (bool): Whether to use local checkpoint files (default: True).
        device (str): Device to load the model on ('cuda', 'cpu', or None for auto-detect).

    Returns:
        torch.nn.Module: The frozen DINOv3 backbone model.
    """
    if model_name not in SUPPORTED_BACKBONES:
        raise ValueError(f"Model {model_name} is not supported. "
                         f"Supported models are: {SUPPORTED_BACKBONES}")
    
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")

    if use_local and model_name in CHECKPOINT_MAPPING:
        # Construct path to local checkpoint
        current_dir = Path(__file__).parent.parent
        checkpoint_path = current_dir / 'checkpoints' / CHECKPOINT_MAPPING[model_name]
        
        if checkpoint_path.exists():
            print(f"Loading model from local checkpoint: {checkpoint_path}")
            # Load the model architecture from torch.hub without pretrained weights
            backbone = torch.hub.load('facebookresearch/dinov3', model_name, pretrained=False)
            # Load the weights from local checkpoint with specified device
            state_dict = torch.load(checkpoint_path, map_location=device)
            backbone.load_state_dict(state_dict, strict=True)
            print(f"Successfully loaded weights from: {checkpoint_path}")
        else:
            print(f"Local checkpoint not found at {checkpoint_path}, falling back to torch.hub")
            backbone = torch.hub.load('facebookresearch/dinov3', model_name, pretrained=pretrained)
    else:
        # Load the model from torch.hub
        print(f"Loading model from torch.hub: {model_name}")
        backbone = torch.hub.load('facebookresearch/dinov3', model_name, pretrained=pretrained)
    
    # Move model to specified device
    backbone = backbone.to(device)

    # Freeze all parameters in the backbone
    for param in backbone.parameters():
        param.requires_grad = False
    
    # Set the model to evaluation mode
    backbone.eval()

    print(f"Loaded DINOv3 backbone: {model_name}")
    print(f"Backbone parameters frozen: True")
    print(f"Backbone in evaluation mode: True")

    return backbone

if __name__ == '__main__':
    # Example usage:
    try:
        # Test loading from local checkpoint
        print("=" * 60)
        print("Testing local checkpoint loading for ViT-S/16...")
        print("=" * 60)
        model_s16 = get_dinov3_backbone('dinov3_vits16', use_local=True)
        print("\nSuccessfully loaded ViT-S/16 model from local.")
        print(f"Embedding dimension: {model_s16.embed_dim}")

        # Example of forwarding a dummy tensor
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = model_s16.forward_features(dummy_input)
        
        print("\n--- Feature Dictionary Keys ---")
        for key in features.keys():
            if isinstance(features[key], torch.Tensor):
                print(f"- {key}: {features[key].shape}")
            else:
                print(f"- {key}: {features[key]}")
        
        # Test loading from torch.hub (optional)
        print("\n" + "=" * 60)
        print("Testing torch.hub loading (optional)...")
        print("=" * 60)
        # Uncomment to test torch.hub loading
        # model_b14_hub = get_dinov3_backbone('dinov3_vitb14', use_local=False)
        # print("\nSuccessfully loaded ViT-B/14 model from torch.hub.")

    except Exception as e:
        print(f"An error occurred: {e}")
