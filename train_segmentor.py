import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import numpy as np

# Import all required components
from data.Dataset_ADE20k import ADE20Dataset
import data.augmentations as augmentations
from loss.losses import CE_DiceLoss, Focal_Dice_Loss
import engine



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ Evaluation functions (moved from engine.py)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = 255):
    """
    Computes the Mean Intersection over Union (mIoU) metric.
    """
    valid_mask = (targets != ignore_index)
    
    iou_list = []
    for cls_id in range(num_classes):
        pred_mask = (preds == cls_id) & valid_mask
        target_mask = (targets == cls_id) & valid_mask
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union > 0:
            iou = intersection / union
            iou_list.append(iou)

    if len(iou_list) == 0:
        return torch.tensor(0.0, device=preds.device)
    
    return torch.stack(iou_list).mean()

@torch.no_grad()
def evaluate_segmentation(model: torch.nn.Module, data_loader: iter, device: torch.device, num_classes: int):
    model.eval()
    
    all_preds = []
    all_targets = []

    progress_bar = tqdm.tqdm(data_loader, desc="Evaluating Segmentation")

    for images, targets in progress_bar:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    miou = compute_miou(all_preds, all_targets, num_classes)
    print(f"Evaluation - Average mIoU: {miou.item():.4f}")
    return {'mIoU': miou.item()}

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ Main training script
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_args_parser():
    parser = argparse.ArgumentParser('DINOv3 Downstream Training for Segmentation', add_help=False)
    parser.add_argument('--config', required=True, type=str, help='Path to the configuration file.')
    parser.add_argument('--eval_only', action='store_true', help='Perform evaluation only.')
    parser.add_argument('--resume', type=str, default='', help='Path to the checkpoint to resume from.')
    return parser

def train_segmentator(config):
    # --- Setup Device ---
    device = torch.device(config['training']['device'])

    # --- Prepare Data with Augmentations ---
    input_size = (config['data']['input_size'])
    

    train_augmentations = augmentations.Compose([
        augmentations.RandomSizedCrop(input_size[0]),   # Random scale + crop (mild scaling)  
        augmentations.RandomRotate(degree=5),           # Small angle rotation (5° max)
        augmentations.RandomHorizontallyFlip(0.5),      # Horizontal flip (anatomically safe)
    ])
    val_augmentations = train_augmentations

    # --- Dynamically select dataset ---
    dataset_name = config['data'].get('dataset_name', 'ade20k') # Default to ade20k if not specified

    if dataset_name == 'ade20k':
        DatasetClass = ADE20Dataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_train = DatasetClass(
        root=config['data']['root_dir'],
        split='training',
        input_size=config['data']['input_size'],
        augmentations=None,
    )
    dataset_val = DatasetClass(
        root=config['data']['root_dir'],
        split='validation',
        input_size=config['data']['input_size'],
        augmentations=None,  # Add validation augmentations
    )

    # --- Build Model ---
    model = engine.get_segmentation_model(config)
    # Use the combined Cross-Entropy and Dice loss from the dedicated loss module
    criterion = Focal_Dice_Loss(ignore_index=config.get('ignore_index', 255))
    # Train the decoder and final layers
    params_to_train = [p for p in model.parameters() if p.requires_grad]

    model.to(device)

    # --- Common Dataloader and Optimizer Setup ---
    data_loader_train = DataLoader(dataset_train, batch_size=config['training']['batch_size'], num_workers=config['training']['num_workers'], shuffle=True, pin_memory=True)
    data_loader_val = DataLoader(dataset_val, batch_size=config['training']['batch_size'], num_workers=config['training']['num_workers'], shuffle=False, pin_memory=True)

    optimizer = engine.create_optimizer(params_to_train, config)
    scheduler = engine.create_scheduler(optimizer, config)

    # --- Output Directory ---
    output_dir = config['logging']['output_dir']
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- Training Loop ---
    print("Starting training...")
    best_metric = 0.0

    for epoch in range(config['training']['epochs']):
        engine.train_one_epoch(model, data_loader_train, optimizer, criterion, device, epoch, config['logging']['print_freq'])
        scheduler.step()
        eval_stats = evaluate_segmentation(model, data_loader_val, device, config['model']['num_classes'])
        current_metric = eval_stats['mIoU']

        # Save checkpoint
        if output_dir:
            engine.save_checkpoint(path=os.path.join(output_dir, 'checkpoint.pth'),model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, config=config)
            if current_metric > best_metric:
                best_metric = current_metric
                engine.save_checkpoint(path=os.path.join(output_dir, 'best_checkpoint.pth'),model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, config=config)
                print(f"Saved new best model with metric: {best_metric:.4f}")

    print("Training finished.")

if __name__ == '__main__':
    # parser = argparse.ArgumentParser('DINOv3 Downstream Training for Segmentation', parents=[get_args_parser()])
    # args = parser.parse_args()
    # config_path = args.config

    config_path = "configs/segmentation_mrihead.yaml"
    # --- Load Configuration ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    train_segmentator(config)
