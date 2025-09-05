import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import numpy as np

# Import all required components
from data.ADE20Dataset import ADE20Dataset
from data.MRIHead2DDataset import MRIHead2DDataset
import data.augmentations as augmentations
from models.segmentation_model import DinoV3_DPT
from models.dinov3_unet_full import DinoV3_UNet_Full  # Add new model
from loss.losses import CE_DiceLoss
import engine

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ Evaluation functions (moved from engine.py)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = 255):
    """
    Computes the Mean Intersection over Union (mIoU) metric.
    """
    preds = torch.argmax(preds, dim=1)
    
    iou_list = []
    for cls_id in range(num_classes):
        pred_mask = (preds == cls_id)
        target_mask = (targets == cls_id)
        
        # Ignore areas where the target is the ignore_index
        if (targets == ignore_index).any():
            pred_mask[targets == ignore_index] = False
            target_mask[targets == ignore_index] = False

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = (intersection + 1e-6) / (union + 1e-6)
        iou_list.append(iou)

    return torch.mean(torch.tensor(iou_list))

@torch.no_grad()
def evaluate_segmentation(model: torch.nn.Module, data_loader: iter, device: torch.device, num_classes: int):
    model.eval()
    
    total_miou = 0.0
    num_batches = 0

    progress_bar = tqdm.tqdm(data_loader, desc="Evaluating Segmentation")

    for images, targets in progress_bar:
        images, targets = images.to(device), targets.to(device)

        outputs = model(images)
        miou = compute_miou(outputs, targets, num_classes)
        
        total_miou += miou.item()
        num_batches += 1

        progress_bar.set_postfix({'mIoU': f'{total_miou / num_batches:.4f}'})

    avg_miou = total_miou / num_batches
    print(f"Evaluation - Average mIoU: {avg_miou:.4f}")
    return {'mIoU': avg_miou}

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
    input_size = (config['data']['input_size'], config['data']['input_size'])
    
    # --- Preprocessing Augmentations (Applied to both training and validation) ---
    # These are fixed transformations to enhance MRI characteristics
    preprocessing_augmentations = augmentations.Compose([
        augmentations.FreeScale(size=input_size),       # Fixed size scaling
        augmentations.FixedGamma(gamma=0.75),           # Fixed gamma to enhance low-intensity pixels
    ])
    
    # --- Training-only Random Augmentations (Geometric transformations) ---
    # These are applied AFTER preprocessing, only during training
    random_augmentations = augmentations.Compose([
        augmentations.RandomSizedCrop(input_size[0]),   # Random scale + crop (mild scaling)  
        augmentations.RandomRotate(degree=5),           # Small angle rotation (5° max)
        augmentations.RandomHorizontallyFlip(0.5),      # Horizontal flip (anatomically safe)
    ])
    
    # --- Combined Training Augmentations ---
    train_augmentations = augmentations.Compose([
        preprocessing_augmentations.augmentations[0],   # FreeScale
        preprocessing_augmentations.augmentations[1],   # FixedGamma
        random_augmentations.augmentations[0],          # RandomSizedCrop
        random_augmentations.augmentations[1],          # RandomRotate  
        random_augmentations.augmentations[2],          # RandomHorizontallyFlip
    ])
    
    # --- Validation Augmentations (Only preprocessing) ---
    val_augmentations = preprocessing_augmentations

    # --- Dynamically select dataset ---
    dataset_name = config['data'].get('dataset_name', 'ade20k') # Default to ade20k if not specified

    if dataset_name == 'mrihead2d':
        DatasetClass = MRIHead2DDataset
    elif dataset_name == 'ade20k':
        DatasetClass = ADE20Dataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_train = DatasetClass(
        root=config['data']['root_dir'],
        split='training',
        img_size=input_size,
        augmentations=train_augmentations,
        mean=config['data']['mean'],
        std=config['data']['std']
    )
    dataset_val = DatasetClass(
        root=config['data']['root_dir'],
        split='validation',
        img_size=input_size,
        augmentations=val_augmentations,  # Add validation augmentations
        mean=config['data']['mean'],
        std=config['data']['std']
    )

    # --- Build Model ---
    model_type = config['model'].get('model_type', 'dpt')  # Default to DPT if not specified
    
    if model_type == 'unet_full':
        # Use DinoV3_UNet_Full (complete DINOv3_Adapter + UNet)
        model = DinoV3_UNet_Full(
            backbone_name=config['model']['backbone_name'],
            num_classes=config['model']['num_classes'],
            interaction_indexes=config['model'].get('interaction_indexes', [2, 5, 8, 11]),
            decoder_channels=tuple(config['model'].get('decoder_channels', [256, 128, 64, 32])),
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
    # Use the combined Cross-Entropy and Dice loss from the dedicated loss module
    criterion = CE_DiceLoss(ignore_index=config.get('ignore_index', 255))
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
        
        if scheduler:
            scheduler.step()

        eval_stats = evaluate_segmentation(model, data_loader_val, device, config['model']['num_classes'])
        current_metric = eval_stats['mIoU']

        # Save checkpoint
        if output_dir:
            engine.save_checkpoint(
                path=os.path.join(output_dir, 'checkpoint.pth'),
                model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, config=config
            )
            if current_metric > best_metric:
                best_metric = current_metric
                engine.save_checkpoint(
                    path=os.path.join(output_dir, 'best_checkpoint.pth'),
                    model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, config=config
                )
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
