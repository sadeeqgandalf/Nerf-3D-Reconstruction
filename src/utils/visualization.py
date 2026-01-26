"""
Visualization utilities for NeRF results.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional
import imageio


def save_image(
    image: torch.Tensor,
    path: str,
    normalize: bool = True,
):
    """
    Save image tensor to file.
    
    Args:
        image: Image tensor of shape (H, W, 3) or (3, H, W)
        path: Output path
        normalize: Whether to normalize to [0, 255]
    """
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    # Handle different shapes
    if image.shape[0] == 3:  # (3, H, W)
        image = image.transpose(1, 2, 0)  # (H, W, 3)
    
    # Normalize to [0, 255]
    if normalize:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Save
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def create_video(
    images: list,
    output_path: str,
    fps: int = 30,
):
    """
    Create video from list of images.
    
    Args:
        images: List of image arrays or tensors
        output_path: Output video path
        fps: Frames per second
    """
    # Convert images to numpy arrays
    frames = []
    for img in images:
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        
        if img.shape[0] == 3:  # (3, H, W)
            img = img.transpose(1, 2, 0)
        
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        frames.append(img)
    
    # Save video
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(output_path, frames, fps=fps)


def visualize_depth(
    depth: torch.Tensor,
    near: float,
    far: float,
) -> torch.Tensor:
    """
    Visualize depth map as color image.
    
    Args:
        depth: Depth map of shape (H, W, 1) or (H, W)
        near: Near plane distance
        far: Far plane distance
        
    Returns:
        Colored depth map of shape (H, W, 3)
    """
    if depth.dim() == 3:
        depth = depth.squeeze(-1)
    
    # Normalize to [0, 1]
    depth_norm = (depth - near) / (far - near + 1e-8)
    depth_norm = torch.clamp(depth_norm, 0, 1)
    
    # Apply colormap (simple jet colormap)
    # Convert to numpy for easier colormap application
    depth_np = depth_norm.detach().cpu().numpy()
    
    # Simple colormap: blue -> green -> red
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    colormap = cm.get_cmap('jet')
    depth_colored = colormap(depth_np)[..., :3]  # Remove alpha
    
    return torch.from_numpy(depth_colored).float()
