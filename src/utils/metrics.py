"""
Evaluation metrics for NeRF.

Implements PSNR, SSIM, and other image quality metrics.
"""

import torch
import torch.nn.functional as F
from typing import Optional
import numpy as np


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        pred: Predicted image of shape (..., H, W, 3) or (H, W, 3)
        target: Target image of shape (..., H, W, 3) or (H, W, 3)
        
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((pred - target) ** 2)
    
    if mse == 0:
        return float('inf')
    
    max_pixel = 1.0  # Assuming images in [0, 1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    
    return psnr.item()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    Simplified SSIM implementation. For production, consider using
    pytorch-msssim or similar library.
    
    Args:
        pred: Predicted image of shape (H, W, 3)
        target: Target image of shape (H, W, 3)
        window_size: Size of the Gaussian window
        
    Returns:
        SSIM value in [0, 1]
    """
    # Convert to grayscale for simplicity
    # Full SSIM would compute per channel and average
    pred_gray = torch.mean(pred, dim=-1)
    target_gray = torch.mean(target, dim=-1)
    
    # Simple SSIM approximation
    # Full implementation would use Gaussian window
    mu1 = torch.mean(pred_gray)
    mu2 = torch.mean(target_gray)
    
    sigma1_sq = torch.var(pred_gray)
    sigma2_sq = torch.var(target_gray)
    sigma12 = torch.mean((pred_gray - mu1) * (target_gray - mu2))
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim.item()


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS).
    
    Note: Requires lpips library. Install with: pip install lpips
    
    Args:
        pred: Predicted image of shape (H, W, 3)
        target: Target image of shape (H, W, 3)
        
    Returns:
        LPIPS value (lower is better)
    """
    try:
        import lpips
        
        # Initialize LPIPS model
        loss_fn = lpips.LPIPS(net='alex').to(pred.device)
        
        # Convert to (1, 3, H, W) format
        pred_batch = pred.permute(2, 0, 1).unsqueeze(0)
        target_batch = target.permute(2, 0, 1).unsqueeze(0)
        
        # Compute LPIPS
        with torch.no_grad():
            lpips_value = loss_fn(pred_batch, target_batch)
        
        return lpips_value.item()
    except ImportError:
        print("LPIPS not available. Install with: pip install lpips")
        return 0.0
