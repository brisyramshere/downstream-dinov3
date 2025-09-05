import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from models.backbone import get_dinov3_backbone
from models.segmentation_from_dinov3.models.backbone.dinov3_adapter import DINOv3_Adapter

# ================================================================= #
#                      Helper / Building Blocks                     #
# ================================================================= #

class ConvBlock(nn.Module):
    """A basic convolutional block with Conv -> BN -> ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

# ================================================================= #
#                      Feature Projection Module                   #
# ================================================================= #

class SimpleFeatureProjection(nn.Module):
    """
    Simple feature projection module to convert DINOv3_Adapter outputs
    to appropriate channel dimensions for UNet decoder.
    This is a simplified version without FAPM complexity.
    """
    def __init__(self, in_channels: int, out_channels_list: List[int], bias: bool = False):
        super().__init__()
        
        # Simple 1x1 convolutions for channel dimension adjustment
        self.projections = nn.ModuleList()
        for out_ch in out_channels_list:
            if in_channels != out_ch:
                proj = nn.Sequential(
                    nn.Conv2d(in_channels, out_ch, kernel_size=1, bias=bias),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            else:
                proj = nn.Identity()
            self.projections.append(proj)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        projected_features = []
        for i, x in enumerate(features):
            projected = self.projections[i](x)
            projected_features.append(projected)
        return projected_features

# ================================================================= #
#                      UNet Decoder Components                     #
# ================================================================= #

class DecoderBlock(nn.Module):
    """
    UNet Decoder block.
    Performs up-sampling, concatenation with skip connection, and convolutions.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # The input to the conv_block is the concatenated feature map
        conv_in_channels = out_channels + skip_channels
        self.conv_block = nn.Sequential(
            ConvBlock(conv_in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        # Pad skip connection if spatial dimensions don't match
        if x.shape[-2:] != skip.shape[-2:]:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2])
        
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)

class UNetDecoder(nn.Module):
    """
    The UNet decoder module. It progressively upsamples the feature map
    from the bottleneck, fusing it with skip connections from the encoder.
    """
    def __init__(self, decoder_channels: Tuple[int, ...], skip_channels: Tuple[int, ...]):
        super().__init__()
        
        # For UNet, we need:
        # - decoder input channels = previous decoder output channels  
        # - after concatenation with skip = decoder input + skip channels
        # - decoder output channels = target decoder channels
        
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]  # From previous decoder stage
            skip_ch = skip_channels[len(skip_channels) - 1 - i]  # Skip connection channels (reversed order)
            out_ch = decoder_channels[i + 1]  # Target output channels
            
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))

    def forward(self, bottleneck: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        x = bottleneck
        # Skips are from deepest to shallowest, so we iterate through them in order
        for i, (decoder_block, skip) in enumerate(zip(self.decoder_blocks, skips)):
            x = decoder_block(x, skip)
        return x

# ================================================================= #
#                   Main DinoV3-UNet Model (Full)                  #
# ================================================================= #

class DinoV3_UNet_Full(nn.Module):
    """
    A UNet-style segmentation model using DINOv3_Adapter as the encoder.
    This version uses the complete DINOv3_Adapter for feature extraction,
    with a simple feature projection module instead of FAPM for simplicity.
    """
    def __init__(self,
                 backbone_name: str = 'dinov3_vits16',
                 num_classes: int = 150,
                 interaction_indexes: List[int] = [2, 5, 8, 11],
                 decoder_channels: Tuple[int, ...] = (256, 128, 64, 32),
                 pretrain_size: int = 512,
                 conv_inplane: int = 64,
                 n_points: int = 4,
                 deform_num_heads: int = 6,
                 drop_path_rate: float = 0.3,
                 init_values: float = 0.0,
                 with_cffn: bool = True,
                 cffn_ratio: float = 0.25,
                 deform_ratio: float = 0.5,
                 add_vit_feature: bool = True,
                 use_extra_extractor: bool = True,
                 with_cp: bool = True):
        super().__init__()

        # --- Encoder (DINOv3 Backbone + Adapter) ---
        self.backbone = get_dinov3_backbone(backbone_name, pretrained=True)
        self.backbone_embed_dim = self.backbone.embed_dim
        self.interaction_indexes = interaction_indexes
        
        print(f"Using DINOv3 backbone: {backbone_name}")
        print(f"Backbone embed_dim: {self.backbone_embed_dim}")
        print(f"Interaction indexes: {self.interaction_indexes}")
        
        # Create DINOv3_Adapter
        self.adapter = DINOv3_Adapter(
            backbone=self.backbone,
            interaction_indexes=interaction_indexes,
            pretrain_size=pretrain_size,
            conv_inplane=conv_inplane,
            n_points=n_points,
            deform_num_heads=deform_num_heads,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            deform_ratio=deform_ratio,
            add_vit_feature=add_vit_feature,
            use_extra_extractor=use_extra_extractor,
            with_cp=with_cp
        )

        # --- Feature Projection ---
        # DINOv3_Adapter outputs features with backbone.embed_dim channels
        # We need to project them to decoder input channels
        # Note: skip_channels should match decoder_channels for proper concatenation
        skip_channels = decoder_channels  # Use same channels as decoder, not doubled
        
        self.feature_projection = SimpleFeatureProjection(
            in_channels=self.backbone_embed_dim,
            out_channels_list=list(skip_channels[:-1])  # Exclude the last one (bottleneck)
        )

        # --- Decoder (UNet) ---
        # The bottleneck processes the last feature map from the adapter
        self.bottleneck = ConvBlock(self.backbone_embed_dim, decoder_channels[0])
        
        self.decoder = UNetDecoder(
            decoder_channels=decoder_channels,
            skip_channels=skip_channels[:-1]  # Only skip channels, not including bottleneck
        )
        
        # --- Final Segmentation Head ---
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        
        # --- Encoder Forward Pass (DINOv3_Adapter) ---
        features_dict = self.adapter(x)
        
        # Extract features in order: f1 (highest res) to f4 (lowest res)
        feature_list = [
            features_dict["1"],  # 1/4 scale
            features_dict["2"],  # 1/8 scale  
            features_dict["3"],  # 1/16 scale
            features_dict["4"]   # 1/32 scale (bottleneck)
        ]
        
        # --- Feature Projection ---
        # Project all features except the last one (which goes to bottleneck)
        skip_features = feature_list[:-1]  # f1, f2, f3
        bottleneck_input = feature_list[-1]  # f4
        
        projected_skips = self.feature_projection(skip_features)
        
        # --- Decoder Forward Pass ---
        bottleneck = self.bottleneck(bottleneck_input)
        
        # Reverse skip order for decoder (from deep to shallow)
        decoder_skips = projected_skips[::-1]  # [f3, f2, f1]
        
        decoder_output = self.decoder(bottleneck, decoder_skips)
        
        # --- Final Prediction ---
        logits = self.segmentation_head(decoder_output)
        
        # Upsample logits to the original image size
        return F.interpolate(logits, size=original_size, mode='bilinear', align_corners=True)

# ================================================================= #
#                           Test Main                               #
# ================================================================= #

if __name__ == '__main__':
    print("Testing DinoV3-UNet Full model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Configuration ---
    backbone = 'dinov3_vits16'
    num_classes = 2  # Simple binary segmentation for testing
    interaction_indexes = [2, 5, 8, 11]
    
    print(f"\n=== Model Configuration ===")
    print(f"Backbone: {backbone}")
    print(f"Number of classes: {num_classes}")
    print(f"Interaction indexes: {interaction_indexes}")
    
    try:
        # --- Instantiate Model ---
        print("\n=== Creating Model ===")
        model = DinoV3_UNet_Full(
            backbone_name=backbone,
            num_classes=num_classes,
            interaction_indexes=interaction_indexes
        ).to(device)
        model.eval()
        print("Model created successfully!")

        # --- Create Dummy Input ---
        print("\n=== Testing Forward Pass ===")
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32, device=device)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Input dtype: {dummy_input.dtype}")

        # --- Perform Forward Pass ---
        with torch.no_grad():
            output = model(dummy_input)

        print(f"Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        
        # --- Assertions for Validation ---
        expected_shape = (batch_size, num_classes, 224, 224)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"Output shape validation passed!")

        # Check for NaN or Inf values
        if torch.isnan(output).any():
            print("Warning: Output contains NaN values")
        elif torch.isinf(output).any():
            print("Warning: Output contains Inf values")
        else:
            print("Output values are valid (no NaN/Inf)")

        # --- Check Parameter Statistics ---
        print("\n=== Parameter Statistics ===")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")  
        print(f"Frozen parameters (backbone): {frozen_params / 1e6:.2f}M")
        
        # --- Verify Backbone is Frozen ---
        backbone_frozen = True
        unfrozen_backbone_params = []
        for name, param in model.backbone.named_parameters():
            if param.requires_grad:
                unfrozen_backbone_params.append(name)
                backbone_frozen = False
        
        if backbone_frozen:
            print("All backbone parameters are frozen")
        else:
            print(f"Warning: {len(unfrozen_backbone_params)} backbone parameters are not frozen")
            for name in unfrozen_backbone_params[:5]:  # Show first 5
                print(f"  - {name}")
            if len(unfrozen_backbone_params) > 5:
                print(f"  ... and {len(unfrozen_backbone_params) - 5} more")

        # --- Test Different Input Sizes ---
        print("\n=== Testing Different Input Sizes ===")
        test_sizes = [(256, 256), (384, 384), (512, 512)]
        
        for h, w in test_sizes:
            try:
                test_input = torch.randn(1, 3, h, w, dtype=torch.float32, device=device)
                with torch.no_grad():
                    test_output = model(test_input)
                print(f"Input {h}x{w} -> Output {test_output.shape[-2:]} successful")
            except Exception as e:
                print(f"Input {h}x{w} failed: {str(e)[:50]}...")

        # --- Memory Usage (if CUDA) ---
        if device.type == 'cuda':
            print(f"\n=== GPU Memory Usage ===")
            print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
            print(f"Memory cached: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

        print(f"\nAll tests passed! Model is ready for training/inference.")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        print("\n=== Full Error Traceback ===")
        traceback.print_exc()
        
        print(f"\n=== Debug Info ===")
        print(f"Python version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name()}")
        
        print(f"\nPlease check the error above and fix any issues.")