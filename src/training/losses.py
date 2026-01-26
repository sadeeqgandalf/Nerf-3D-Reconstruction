"""
Loss functions for NeRF training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class NeRFLoss(nn.Module):
    """
    Loss function for NeRF training.
    
    Combines:
        - RGB reconstruction loss (MSE)
        - Optional: Coarse and fine network losses
    """
    
    def __init__(self, lambda_coarse: float = 1.0, lambda_fine: float = 1.0):
        """
        Initialize loss function.
        
        Args:
            lambda_coarse: Weight for coarse network loss
            lambda_fine: Weight for fine network loss
        """
        super().__init__()
        self.lambda_coarse = lambda_coarse
        self.lambda_fine = lambda_fine
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        pred_rgb_coarse: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            pred_rgb: Predicted RGB from fine network (..., 3)
            target_rgb: Ground truth RGB (..., 3)
            pred_rgb_coarse: Predicted RGB from coarse network (optional)
            
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # Fine network loss
        fine_loss = self.mse_loss(pred_rgb, target_rgb)
        losses['fine_loss'] = fine_loss
        
        total_loss = self.lambda_fine * fine_loss
        
        # Coarse network loss (if provided)
        if pred_rgb_coarse is not None:
            coarse_loss = self.mse_loss(pred_rgb_coarse, target_rgb)
            losses['coarse_loss'] = coarse_loss
            total_loss = total_loss + self.lambda_coarse * coarse_loss
        
        losses['total_loss'] = total_loss
        
        return losses
