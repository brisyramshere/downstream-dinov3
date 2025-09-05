import torch
import torch.nn as nn
import tqdm
import os

def create_optimizer(params_to_train, config: dict):
    """
    Creates an optimizer based on the configuration.
    """
    optimizer_name = config['optimizer']['name'].lower()
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(params_to_train, lr=config['optimizer']['lr'], momentum=config['optimizer']['momentum'], weight_decay=config['optimizer']['weight_decay'])
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(params_to_train, lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    else:
        raise ValueError(f"Optimizer {config['optimizer']['name']} not supported.")
    return optimizer

def create_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """
    Creates a learning rate scheduler based on the configuration.
    """
    scheduler_name = config['scheduler']['name'].lower()
    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    else:
        scheduler = None
    return scheduler

def save_checkpoint(path: str, model, optimizer, scheduler, epoch, config):
    """
    Saves a checkpoint dictionary to the specified path.
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'config': config,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def train_one_epoch(model: torch.nn.Module, data_loader: iter, 
                    optimizer: torch.optim.Optimizer, 
                    criterion: torch.nn.Module, 
                    device: torch.device, epoch: int, 
                    print_freq: int):
    """
    Trains the model for one epoch.
    """
    model.train()

    total_loss = 0.0
    num_samples = 0

    progress_bar = tqdm.tqdm(data_loader, desc=f"Epoch {epoch}")

    for i, (images, target) in enumerate(progress_bar):
        images, target = images.to(device), target.to(device)

        # Forward pass
        output = model(images)
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update stats
        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        num_samples += batch_size

        if i % print_freq == 0:
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_samples
    print(f"Epoch {epoch} - Average Training Loss: {avg_loss:.4f}")
    return avg_loss