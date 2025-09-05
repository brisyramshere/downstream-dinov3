import torch
import torch.nn as nn
from .backbone import get_dinov3_backbone

class DinoV3LinearClassifier(nn.Module):
    """
    A full model for linear classification, combining a DINOv3 backbone
    with a trainable linear head.
    """
    def __init__(self, backbone_name: str, num_classes: int, feature_source: str = 'cls_patch_avg'):
        """
        Args:
            backbone_name (str): The name of the DINOv3 backbone to use.
            num_classes (int): The number of output classes for the linear head.
            feature_source (str): The source of features for the classifier.
                                  Options: 'cls', 'patch_avg', 'cls_patch_avg'.
        """
        super().__init__()
        self.feature_source = feature_source

        # Load the frozen backbone
        self.backbone = get_dinov3_backbone(backbone_name, pretrained=True)
        
        # Determine the input dimension for the linear head
        embed_dim = self.backbone.embed_dim
        if feature_source == 'cls':
            self.feature_dim = embed_dim
        elif feature_source == 'patch_avg':
            self.feature_dim = embed_dim
        elif feature_source == 'cls_patch_avg':
            self.feature_dim = embed_dim * 2
        else:
            raise ValueError(f"Unknown feature_source: {feature_source}")

        # Create the linear classifier head
        self.linear_head = nn.Linear(self.feature_dim, num_classes)
        
        # Initialize the weights of the linear head
        self.linear_head.weight.data.normal_(mean=0.0, std=0.01)
        self.linear_head.bias.data.zero_()

        print(f"Created DinoV3LinearClassifier with:")
        print(f"  - Backbone: {backbone_name}")
        print(f"  - Num classes: {num_classes}")
        print(f"  - Feature source: {feature_source} (dim: {self.feature_dim})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the backbone and the linear head.
        
        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The output logits from the linear head.
        """
        # The backbone is frozen and in eval mode, so no_grad() is not strictly
        # necessary, but it's good practice to make it explicit.
        with torch.no_grad():
            features = self.backbone.forward_features(x)
            cls_token = features['x_norm_clstoken']
            patch_tokens = features['x_norm_patchtokens']

        # Extract features based on the specified source
        if self.feature_source == 'cls':
            head_input = cls_token
        elif self.feature_source == 'patch_avg':
            head_input = patch_tokens.mean(dim=1) # Average pooling over patch tokens
        elif self.feature_source == 'cls_patch_avg':
            head_input = torch.cat([
                cls_token,
                patch_tokens.mean(dim=1),
            ], dim=1)
        else:
            # This case is already handled in __init__, but for safety:
            raise ValueError(f"Unknown feature_source: {self.feature_source}")

        # Pass the extracted features through the linear head
        # The head is trainable, so this part is within the autograd context.
        logits = self.linear_head(head_input)
        
        return logits

if __name__ == '__main__':
    # Example usage:
    print("Testing DinoV3LinearClassifier...")
    try:
        # 1. Create the model
        model = DinoV3LinearClassifier(
            backbone_name='dinov3_vits16', 
            num_classes=1000, 
            feature_source='cls_patch_avg'
        )
        model.to('cuda') # Move to CPU for testing without GPU
        model.eval() # Set the whole model to eval mode

        # 2. Create a dummy input
        dummy_input = torch.randn(4, 3, 224, 224) # Batch size of 4

        # 3. Perform a forward pass
        with torch.no_grad():
            logits = model(dummy_input)
        
        print(f"\nSuccessfully performed a forward pass.")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output logits shape: {logits.shape}")
        assert logits.shape == (4, 1000)

        # 4. Check which parameters are trainable
        print("\nTrainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  - {name} (shape: {param.shape})")

    except Exception as e:
        print(f"An error occurred during testing: {e}")
