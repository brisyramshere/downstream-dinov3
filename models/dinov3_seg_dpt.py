import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from models.backbones import DINOv3_INTERACTION_INDEXES

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch

class DPTHead(nn.Module):
    def __init__(
        self, 
        nclass,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024],
    ):
        super(DPTHead, self).__init__()
        
        # Project the concatenated feature maps from the backbone
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        # Refine the projected features
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        # Final convolution to get to the number of classes
        self.scratch.output_conv = nn.Conv2d(features * len(out_channels), nclass, kernel_size=1, stride=1, padding=0)  
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            # Permute and reshape to (B, C, H, W)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            # Project to a new channel dimension
            x = self.projects[i](x)
            out.append(x)
        
        # The features are now in a list `out`
        layer_1, layer_2, layer_3, layer_4 = out
        
        # Refine each projected feature map
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        # Upsample all feature maps to the size of the first one
        target_hw = layer_1_rn.shape[-2:]  
        layer_2_up = F.interpolate(layer_2_rn, size=target_hw, mode="bilinear", align_corners=False)
        layer_3_up = F.interpolate(layer_3_rn, size=target_hw, mode="bilinear", align_corners=False)
        layer_4_up = F.interpolate(layer_4_rn, size=target_hw, mode="bilinear", align_corners=False)
        
        # Concatenate all upsampled features
        fused = torch.cat([layer_1_rn, layer_2_up, layer_3_up, layer_4_up], dim=1)
        
        # Final prediction
        out = self.scratch.output_conv(fused)
        return out

class DinoV3_DPT(nn.Module):
    """
    A DPT (Dense Prediction Transformer) architecture using a DINOv3 Vision Transformer as the encoder.
    """
    def __init__(self, 
                 backbone_name: str = 'dinov3_vits16', 
                 num_classes: int = 2, 
                 features=128):
        super().__init__()

        # --- Encoder ---
        from models.backbones import load_dinov3_backbone
        self.backbone = load_dinov3_backbone(backbone_name)
        self.backbone_embed_dim = self.backbone.embed_dim
        self.skip_connection_layers = DINOv3_INTERACTION_INDEXES[backbone_name]
        
        # The DPT head expects a certain number of channels for the features it processes.
        # These are the output channels of the projection layers in the DPTHead.
        # Let's define them based on the backbone's embedding dimension.
        head_out_channels = [self.backbone_embed_dim // 4] * len(self.skip_connection_layers)

        print(f"Using DINOv3 backbone: {backbone_name}")
        print(f"Extracting features from layers: {self.skip_connection_layers}")

        # --- Decoder ---
        self.head = DPTHead(
            nclass=num_classes, 
            in_channels=self.backbone_embed_dim, 
            features=features, 
            out_channels=head_out_channels
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_h, img_w = x.shape[-2:]
        patch_h, patch_w = img_h // self.backbone.patch_size, img_w // self.backbone.patch_size

        # --- Encoder Forward Pass ---
        with torch.no_grad():
            # Get intermediate features from the backbone
            intermediate_outputs = self.backbone.get_intermediate_layers(
                x, 
                n=self.skip_connection_layers, 
                reshape=False, # Keep as (B, N, C) for the head
                return_class_token=False,
            )

        # --- Decoder Forward Pass ---
        # The head processes the list of features and returns the logits
        logits = self.head(intermediate_outputs, patch_h, patch_w)

        # Upsample logits to the original image size
        return F.interpolate(logits, size=(img_h, img_w), mode='bilinear', align_corners=False)

if __name__ == '__main__':
    print("Testing DinoV3_DPT model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a model (using a small backbone for quick testing)
    # Note: The number of out_channels in DPTHead must match the number of skip_connection_layers.
    # For 'dinov3_vits16', n_blocks is 12, so skip_connection_layers is [3, 6, 9, 11] (4 layers).
    model = DinoV3_DPT(backbone_name='dinov3_vits16', num_classes=150).to(device)
    model.eval()

    # Create a dummy input tensor
    dummy_input = torch.randn(2, 3, 224, 224).to(device)

    # Perform a forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nSuccessfully performed a forward pass.")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {output.shape}")
    assert dummy_input.shape[-2:] == output.shape[-2:]
    assert output.shape[1] == 150
    print("Output shape is correct.")

    # Check trainable parameters
    print("\nTrainable parameters (should only be in the DPT head):")
    total_trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name} (shape: {param.shape})")
            total_trainable_params += param.numel()
    
    print(f"\nTotal trainable parameters: {total_trainable_params}")
    assert total_trainable_params > 0, "There should be trainable parameters in the head."

    # Ensure backbone is frozen
    for name, param in model.backbone.named_parameters():
        assert not param.requires_grad, f"Backbone parameter {name} is not frozen!"
    print("Verified that all backbone parameters are frozen.")