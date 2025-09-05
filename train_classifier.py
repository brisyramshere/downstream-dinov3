import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import tqdm

# Import all required components
from data.classification_dataset import ImageNetClassificationDataset
from models.classification_model import DinoV3LinearClassifier
import engine

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ Evaluation function (moved from engine.py)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: iter, 
             criterion: torch.nn.Module, device: torch.device):
    """
    Evaluates the model on a given dataset.
    """
    model.eval()

    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    num_samples = 0

    progress_bar = tqdm.tqdm(data_loader, desc="Evaluating")

    for images, target in progress_bar:
        images, target = images.to(device), target.to(device)

        # Forward pass
        output = model(images)
        loss = criterion(output, target)

        # Calculate accuracy
        batch_size = images.shape[0]
        _, pred = output.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_top1 += correct[0].reshape(-1).float().sum(0, keepdim=True).item()
        correct_top5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
        
        # Update stats
        total_loss += loss.item() * batch_size
        num_samples += batch_size

    avg_loss = total_loss / num_samples
    acc1 = (correct_top1 / num_samples) * 100
    acc5 = (correct_top5 / num_samples) * 100

    print(f"Evaluation - Average Loss: {avg_loss:.4f}, "
          f"Top-1 Accuracy: {acc1:.2f}%, "
          f"Top-5 Accuracy: {acc5:.2f}%")
    
    return {'loss': avg_loss, 'acc1': acc1, 'acc5': acc5}

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ Main training script
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_args_parser():
    parser = argparse.ArgumentParser('DINOv3 Downstream Training for Classification', add_help=False)
    parser.add_argument('--config', required=True, type=str, help='Path to the configuration file.')
    parser.add_argument('--eval_only', action='store_true', help='Perform evaluation only.')
    parser.add_argument('--resume', type=str, default='', help='Path to the checkpoint to resume from.')
    return parser

def main(args):
    # --- Load Configuration ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --- Setup Device ---
    device = torch.device(config['training']['device'])

    # --- Prepare Data ---
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config['data']['input_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config['data']['mean'], config['data']['std']),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(config['data']['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(config['data']['mean'], config['data']['std']),
    ])
    dataset_train = ImageNetClassificationDataset(root_dir=config['data']['root_dir'], split='train', transform=transform_train)
    dataset_val = ImageNetClassificationDataset(root_dir=config['data']['root_dir'], split='val', transform=transform_val)
    
    # --- Build Model ---
    model = DinoV3LinearClassifier(
        backbone_name=config['model']['backbone_name'],
        num_classes=config['model']['num_classes'],
        feature_source=config['model']['feature_source']
    )
    criterion = nn.CrossEntropyLoss()
    params_to_train = model.linear_head.parameters()

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

        eval_stats = evaluate(model, data_loader_val, criterion, device)
        current_metric = eval_stats['acc1']

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
    parser = argparse.ArgumentParser('DINOv3 Downstream Training for Classification', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
