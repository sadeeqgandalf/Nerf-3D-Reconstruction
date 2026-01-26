"""
Main training script for NeRF.

Usage:
    python train.py --config configs/lego_config.yaml
"""

import argparse
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.models import NeRF
from src.data import NeRFDataset
from src.training import NeRFTrainer


def main():
    parser = argparse.ArgumentParser(description='Train NeRF model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to train on (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        config.device = args.device
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = NeRFDataset(
        data_dir=config.data.data_dir,
        split='train',
        image_scale=config.data.image_scale,
        white_bg=config.data.white_bg,
    )
    
    val_dataset = NeRFDataset(
        data_dir=config.data.data_dir,
        split='test',
        image_scale=config.data.image_scale,
        white_bg=config.data.white_bg,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    # Create models
    model_coarse = NeRF(
        num_frequencies_xyz=config.model.num_frequencies_xyz,
        num_frequencies_dir=config.model.num_frequencies_dir,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        skip_connection_layer=config.model.skip_connection_layer,
    )
    
    model_fine = NeRF(
        num_frequencies_xyz=config.model.num_frequencies_xyz,
        num_frequencies_dir=config.model.num_frequencies_dir,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        skip_connection_layer=config.model.skip_connection_layer,
    )
    
    # Create trainer
    trainer = NeRFTrainer(
        model_coarse=model_coarse,
        model_fine=model_fine,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config.training.lr,
        lr_decay=config.training.lr_decay,
        lr_decay_steps=config.training.lr_decay_steps,
        num_rays=config.training.num_rays,
        num_samples_coarse=config.training.num_samples_coarse,
        num_samples_fine=config.training.num_samples_fine,
        white_bg=config.training.white_bg,
        use_mixed_precision=config.training.use_mixed_precision,
        log_dir=config.log_dir,
        checkpoint_dir=config.checkpoint_dir,
        device=device,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed training from checkpoint: {args.resume}")
    
    # Train
    trainer.train(
        num_epochs=config.training.num_epochs,
        save_every=config.training.save_every,
    )


if __name__ == '__main__':
    main()
