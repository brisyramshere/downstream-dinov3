import torch
import yaml
import argparse
from PIL import Image
from torchvision import transforms
import os

from models.dinov3_linear_cls import DinoV3LinearClassifier
from data.Dataset_Imagenette2 import Imagenette2Dataset

def get_args_parser():
    parser = argparse.ArgumentParser('DINOv3 Inference', add_help=False)
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to the trained model checkpoint.')
    parser.add_argument('--image', required=True, type=str, help='Path to the input image.')
    return parser

def main(args):
    # --- Load Checkpoint and Config ---
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']

    # --- Setup Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Build Model ---
    model = DinoV3LinearClassifier(
        backbone_name=config['model']['backbone_name'],
        num_classes=config['model']['num_classes'],
        feature_source=config['model']['feature_source']
    )
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- Prepare Image ---
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        return

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(config['data']['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(config['data']['mean'], config['data']['std']),
    ])
    
    image = Image.open(args.image).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # --- Get Class Names ---
    # Create a dataset instance just to get the class mapping
    dataset_for_classes = Imagenette2Dataset(root_dir=config['data']['root_dir'], split='val')
    class_names = dataset_for_classes.classes

    # --- Perform Inference ---
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class_name = class_names[predicted_idx.item()]
    confidence_score = confidence.item()

    print(f"\n--- Inference Result ---")
    print(f"Image: {os.path.basename(args.image)}")
    print(f"Predicted Class: {predicted_class_name}")
    print(f"Confidence: {confidence_score:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINOv3 Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
