import torch
import yaml
import argparse
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import torch.nn.functional as F

from models.segmentation_model import DinoV3_DPT
from models.dinov3_unet_full import DinoV3_UNet_Full

def get_args_parser():
    parser = argparse.ArgumentParser('DINOv3 Segmentation Inference', add_help=False)
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to the trained model checkpoint.')
    parser.add_argument('--image', required=True, type=str, help='Path to the input image.')
    parser.add_argument('--output', type=str, default='inference_output.png', help='Path to save the output visualization.')
    return parser

def get_palette(num_classes):
    """Creates a color palette for visualization."""
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    np.random.seed(42) # for reproducibility
    palette[1:] = np.random.randint(0, 255, size=(num_classes - 1, 3))
    return palette

def load_model(checkpoint_path, device):
    """
    Loads a model from a checkpoint file.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'config' not in checkpoint:
        raise ValueError("Config not found in checkpoint. Cannot build model.")
    config = checkpoint['config']

    # Build Model, ensuring all necessary parameters from config are used
    model_type = config['model'].get('model_type', 'dpt')  # Default to DPT if not specified
    
    if model_type == 'unet_full':
        # Use DinoV3_UNet_Full (complete DINOv3_Adapter + UNet)
        model = DinoV3_UNet_Full(
            backbone_name=config['model']['backbone_name'],
            num_classes=config['model']['num_classes'],
            interaction_indexes=config['model'].get('interaction_indexes'),
            decoder_channels=tuple(config['model'].get('decoder_channels')),
        )
    elif model_type == 'dpt':
        # Use original DinoV3_DPT model
        model = DinoV3_DPT(
            backbone_name=config['model']['backbone_name'],
            num_classes=config['model']['num_classes'],
            skip_connection_layers=config['model'].get('skip_connection_layers')
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: 'dpt', 'unet_full'")
    

    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model, config

def run_batch_inference_on_images(model, image_pil_list, config, device):
    """
    Runs batch inference on a list of PIL images and returns prediction masks.
    Handles both single image (batch_size=1) and multi-image batches.
    
    Args:
        model: The loaded segmentation model
        image_pil_list: List of PIL images (can be single image in list)
        config: Model configuration
        device: Computing device
        
    Returns:
        List of prediction masks (numpy arrays)
    """
    # Ensure input is a list
    if not isinstance(image_pil_list, list):
        image_pil_list = [image_pil_list]
    
    # Batch preprocessing
    transform = transforms.Compose([
        transforms.Resize((config['data']['input_size'], config['data']['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(config['data']['mean'], config['data']['std']),
    ])
    
    # Convert PIL images to tensor batch
    image_tensors = []
    original_sizes = []
    
    for image_pil in image_pil_list:
        image_tensor = transform(image_pil)
        image_tensors.append(image_tensor)
        original_sizes.append(image_pil.size[::-1])  # (height, width)
    
    # Stack into batch
    batch_tensor = torch.stack(image_tensors).to(device)
    
    # Batch inference
    with torch.no_grad():
        batch_logits = model(batch_tensor)
        
        # Process each image in the batch
        pred_masks = []
        for i, (logits, original_size) in enumerate(zip(batch_logits, original_sizes)):
            # Upsample logits to match original image size
            logits_upsampled = F.interpolate(
                logits.unsqueeze(0), 
                size=original_size, 
                mode='bilinear', 
                align_corners=False
            )
            pred_mask = torch.argmax(logits_upsampled, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_masks.append(pred_mask)
    
    return pred_masks

# Backward compatibility: single image inference wrapper
def run_inference_on_image(model, image_pil, config, device):
    """
    Wrapper for single image inference using batch function.
    Maintained for backward compatibility.
    """
    return run_batch_inference_on_images(model, [image_pil], config, device)[0]

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load Model and Config ---
    try:
        model, config = load_model(args.checkpoint, device)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    # --- Prepare and Run Inference on Image ---
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        return
    
    original_image = Image.open(args.image).convert('RGB')
    pred_mask = run_inference_on_image(model, original_image, config, device)
    print("Inference complete.")

    # --- Visualize and Save ---
    palette = get_palette(config['model']['num_classes'])
    pred_mask_color = Image.fromarray(palette[pred_mask], 'RGB')

    # Blend original image with the color mask
    blended_image = Image.blend(original_image, pred_mask_color, alpha=0.5)

    # Save the output
    blended_image.save(args.output)
    print(f"Saved visualization to: {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINOv3 Segmentation Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
