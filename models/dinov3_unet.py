import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import DINOv3_INTERACTION_INDEXES

class ConvBlock(nn.Module):
    """A basic convolutional block with Conv -> BN -> ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class UpConv(nn.Module):
    """An upsampling module with Upsample -> ConvBlock."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))

class DinoV3_UNet(nn.Module):
    """
    A U-Net-like decoder architecture using a DINOv3 Vision Transformer as the encoder.
    """
    def __init__(self, 
                 backbone_name: str = 'dinov3_vits16', 
                 num_classes: int = 2):
        super().__init__()

        # --- Encoder ---
        from models.backbones import load_dinov3_backbone
        self.backbone = load_dinov3_backbone(backbone_name)
        self.backbone_embed_dim = self.backbone.embed_dim
        self.skip_connection_layers = DINOv3_INTERACTION_INDEXES[backbone_name]

        print(f"Using DINOv3 backbone: {backbone_name}")
        print(f"Extracting features from layers: {self.skip_connection_layers}")

        # --- Decoder ---
        embed_dim = self.backbone_embed_dim
        self.upconv1 = UpConv(embed_dim, embed_dim // 2)
        self.upconv2 = UpConv(embed_dim, embed_dim // 2)
        self.upconv3 = UpConv(embed_dim, embed_dim // 2)

        self.upconv4 = UpConv(embed_dim // 2, embed_dim // 4)
        self.upconv5 = UpConv(embed_dim // 2, embed_dim // 4)

        self.upconv6 = UpConv(embed_dim // 4, embed_dim // 8)
        self.upconv7 = UpConv(embed_dim // 8, embed_dim // 16)

        # Skip connection from input
        self.input_skip_conv = ConvBlock(3, embed_dim // 16, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(embed_dim // 16 * 2, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_h, img_w = x.shape[-2:]
        
        # --- Encoder Forward Pass ---
        # Get intermediate features from the backbone
        intermediate_outputs = self.backbone.get_intermediate_layers(
            x, 
            n=self.skip_connection_layers, 
            reshape=False, # Keep as (B, N, C) for the head
            return_class_token=False,
        )

        # Reshape features to [B, C, H, W]
        reshaped_features = []
        patch_h, patch_w = img_h // self.backbone.patch_size, img_w // self.backbone.patch_size
        for feat in intermediate_outputs:
            reshaped_features.append(feat.permute(0, 2, 1).reshape(x.shape[0], -1, patch_h, patch_w))
    
        f2, f5, f8, f11 = reshaped_features

        # --- Decoder ---
        # First level fusion
        f8_11 = self.upconv1(f8 + f11)
        f5_up = self.upconv2(f5)
        f2_up = self.upconv3(f2)

        # Second level fusion
        f5_8_11 = self.upconv4(f8_11 + f5_up)
        f2_up = self.upconv5(f2_up)

        # Third level fusion
        f2_5_8_11 = self.upconv6(f5_8_11 + f2_up)

        # Final upsampling
        out = self.upconv7(f2_5_8_11)
        
        # Skip connection from input
        input_skip = self.input_skip_conv(x)
        
        # Concat with input skip connection
        out = torch.cat([out, input_skip], dim=1)
        
        # Final prediction
        out = self.final_conv(out)
        # out = F.interpolate(out, size=(img_h, img_w), mode='bilinear', align_corners=False)

        return out

if __name__ == '__main__':
    print("Testing DinoV3_DPT model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a model (using a small backbone for quick testing)
    model = DinoV3_UNet(backbone_name='dinov3_vits16', num_classes=10).to(device)
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
    assert output.shape[1] == 10
    print("Output shape is correct.")

    # Check trainable parameters
    print("\nTrainable parameters (should only be in the decoder):")
    total_trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name} (shape: {param.shape})")
            total_trainable_params += param.numel()
    
    print(f"\nTotal trainable parameters: {total_trainable_params}")
    assert total_trainable_params > 0, "There should be trainable parameters in the decoder."

    # Ensure backbone is frozen
    for name, param in model.backbone.named_parameters():
        assert not param.requires_grad, f"Backbone parameter {name} is not frozen!"
    print("Verified that all backbone parameters are frozen.")