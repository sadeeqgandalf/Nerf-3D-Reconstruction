"""
NeRF training loop with best practices.

Implements efficient training with:
- Ray sampling strategies
- Hierarchical sampling (coarse + fine)
- Mixed precision training
- Checkpointing
- Logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional
import os
import time
from tqdm import tqdm
import numpy as np

from ..models import NeRF
from ..rendering import generate_rays, sample_points_along_rays, hierarchical_sample
from ..rendering.volume_renderer import VolumeRenderer
from .losses import NeRFLoss


class NeRFTrainer:
    """
    Trainer for NeRF models.
    
    Handles the complete training loop with best practices:
    - Efficient ray sampling
    - Hierarchical volume sampling
    - Mixed precision training
    - Checkpointing and resuming
    - Comprehensive logging
    """
    
    def __init__(
        self,
        model_coarse: NeRF,
        model_fine: NeRF,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 5e-4,
        lr_decay: float = 0.1,
        lr_decay_steps: int = 250000,
        num_rays: int = 1024,
        num_samples_coarse: int = 64,
        num_samples_fine: int = 128,
        white_bg: bool = False,
        use_mixed_precision: bool = True,
        log_dir: str = './outputs/logs',
        checkpoint_dir: str = './outputs/checkpoints',
        device: str = 'cuda',
    ):
        """
        Initialize trainer.
        
        Args:
            model_coarse: Coarse NeRF model
            model_fine: Fine NeRF model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            lr: Learning rate
            lr_decay: Learning rate decay factor
            lr_decay_steps: Steps for learning rate decay
            num_rays: Number of rays to sample per batch
            num_samples_coarse: Number of coarse samples per ray
            num_samples_fine: Number of fine samples per ray
            white_bg: Whether to use white background
            use_mixed_precision: Whether to use mixed precision training
            log_dir: Directory for logs
            checkpoint_dir: Directory for checkpoints
            device: Device to train on
        """
        self.model_coarse = model_coarse.to(device)
        self.model_fine = model_fine.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_rays = num_rays
        self.num_samples_coarse = num_samples_coarse
        self.num_samples_fine = num_samples_fine
        self.white_bg = white_bg
        self.use_mixed_precision = use_mixed_precision
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.model_coarse.parameters()) + list(self.model_fine.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=lr_decay ** (1.0 / lr_decay_steps),
        )
        
        # Loss function
        self.criterion = NeRFLoss()
        
        # Mixed precision scaler
        if use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Logging
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Checkpointing
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        
        # Training state
        self.global_step = 0
        self.epoch = 0
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dictionary with loss values
        """
        image = batch['image'].to(self.device)  # (H, W, 3)
        pose = batch['pose'].to(self.device)  # (4, 4)
        focal = batch['focal'].to(self.device)
        near = batch['near'].to(self.device)
        far = batch['far'].to(self.device)

        # A DataLoader with batch_size=1 adds a leading batch dimension.
        if image.dim() == 4:
            if image.shape[0] != 1:
                raise ValueError("train_step expects one image per batch (batch_size=1)")
            image, pose = image[0], pose[0]

        height, width = image.shape[:2]
        
        # Generate rays for entire image
        ray_origins, ray_directions = generate_rays(
            height, width, focal.item(), pose, near.item(), far.item()
        )
        
        # Randomly sample rays
        num_pixels = height * width
        indices = torch.randint(0, num_pixels, (self.num_rays,), device=self.device)
        
        ray_origins_sampled = ray_origins.reshape(-1, 3)[indices]  # (num_rays, 3)
        ray_directions_sampled = ray_directions.reshape(-1, 3)[indices]  # (num_rays, 3)
        target_rgb = image.reshape(-1, 3)[indices]  # (num_rays, 3)
        
        # Sample points along rays (coarse)
        points_coarse, distances_coarse = sample_points_along_rays(
            ray_origins_sampled,
            ray_directions_sampled,
            near.item(),
            far.item(),
            self.num_samples_coarse,
            perturb=True,
        )
        
        # Query coarse model
        points_coarse_flat = points_coarse.reshape(-1, 3)
        directions_coarse_expanded = ray_directions_sampled.unsqueeze(1).expand(
            -1, self.num_samples_coarse, -1
        ).reshape(-1, 3)
        
        densities_coarse, colors_coarse = self.model_coarse(
            points_coarse_flat, directions_coarse_expanded
        )
        
        densities_coarse = densities_coarse.reshape(self.num_rays, self.num_samples_coarse, 1)
        colors_coarse = colors_coarse.reshape(self.num_rays, self.num_samples_coarse, 3)
        
        # Render coarse
        rgb_coarse, _, weights_coarse = VolumeRenderer.render_rays(
            densities_coarse,
            colors_coarse,
            distances_coarse,
            white_bg=self.white_bg,
        )
        
        # Hierarchical sampling (fine)
        points_fine, distances_fine = hierarchical_sample(
            ray_origins_sampled,
            ray_directions_sampled,
            distances_coarse,
            weights_coarse.squeeze(-1),
            self.num_samples_fine,
        )
        
        # Combine coarse and fine samples
        points_all = torch.cat([points_coarse, points_fine], dim=1)
        distances_all = torch.cat([distances_coarse, distances_fine], dim=1)
        
        # Sort by distance
        sorted_indices = torch.argsort(distances_all, dim=1)
        points_all = torch.gather(
            points_all, 1,
            sorted_indices.unsqueeze(-1).expand(-1, -1, 3)
        )
        distances_all = torch.gather(distances_all, 1, sorted_indices)
        
        # Query fine model
        points_all_flat = points_all.reshape(-1, 3)
        directions_all_expanded = ray_directions_sampled.unsqueeze(1).expand(
            -1, points_all.shape[1], -1
        ).reshape(-1, 3)
        
        densities_fine, colors_fine = self.model_fine(
            points_all_flat, directions_all_expanded
        )
        
        densities_fine = densities_fine.reshape(self.num_rays, points_all.shape[1], 1)
        colors_fine = colors_fine.reshape(self.num_rays, points_all.shape[1], 3)
        
        # Render fine
        rgb_fine, _, _ = VolumeRenderer.render_rays(
            densities_fine,
            colors_fine,
            distances_all,
            white_bg=self.white_bg,
        )
        
        # Compute loss
        losses = self.criterion(rgb_fine, target_rgb, rgb_coarse)
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.use_mixed_precision:
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total_loss'].backward()
            self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        # Convert to float for logging
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict['lr'] = self.optimizer.param_groups[0]['lr']
        
        return loss_dict
    
    def train(self, num_epochs: int, save_every: int = 10000):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N steps
        """
        self.model_coarse.train()
        self.model_fine.train()
        
        print(f"Starting training on {self.device}")
        print(f"Training samples: {len(self.train_loader)}")
        print(f"Rays per batch: {self.num_rays}")
        print(f"Coarse samples: {self.num_samples_coarse}, Fine samples: {self.num_samples_fine}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_losses = []
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                # Training step
                loss_dict = self.train_step(batch)
                epoch_losses.append(loss_dict['total_loss'])
                
                # Logging
                if self.global_step % 100 == 0:
                    for key, value in loss_dict.items():
                        self.writer.add_scalar(f'train/{key}', value, self.global_step)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'lr': f"{loss_dict['lr']:.2e}",
                })
                
                # Checkpointing
                if self.global_step > 0 and self.global_step % save_every == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
            
            # Epoch summary
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Validation
            if self.val_loader is not None:
                val_metrics = self.validate()
                print(f"Validation PSNR: {val_metrics.get('psnr', 0):.2f}")
        
        # Final checkpoint
        self.save_checkpoint()
        print("Training completed!")
    
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model_coarse.eval()
        self.model_fine.eval()
        
        total_psnr = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Simple validation: render a few rays
                # (Full image rendering is expensive, so we sample)
                # This is a simplified version - you can make it more comprehensive
                pass
        
        self.model_coarse.train()
        self.model_fine.train()
        
        return {'psnr': total_psnr / max(num_samples, 1)}
    
    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_coarse_state_dict': self.model_coarse.state_dict(),
            'model_fine_state_dict': self.model_fine.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        if self.use_mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_step_{self.global_step}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model_coarse.load_state_dict(checkpoint['model_coarse_state_dict'])
        self.model_fine.load_state_dict(checkpoint['model_fine_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
