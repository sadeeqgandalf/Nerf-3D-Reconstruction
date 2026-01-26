"""
Volume rendering implementation.

Implements the volume rendering equation to convert NeRF predictions
(density and color) into final pixel colors.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class VolumeRenderer:
    """
    Volume renderer for NeRF.
    
    Implements the volume rendering equation:
    C(r) = ∫ T(t) * σ(r(t)) * c(r(t), d) dt
    
    where:
        - T(t) = exp(-∫ σ(r(s)) ds) is the transmittance
        - σ is the volume density
        - c is the RGB color
        - r(t) = o + t*d is the ray
    """
    
    @staticmethod
    def render_rays(
        densities: torch.Tensor,
        colors: torch.Tensor,
        distances: torch.Tensor,
        ray_directions: Optional[torch.Tensor] = None,
        white_bg: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render rays using volume rendering.
        
        Args:
            densities: Volume densities of shape (..., num_samples, 1)
            colors: RGB colors of shape (..., num_samples, 3)
            distances: Distances along rays of shape (..., num_samples)
            ray_directions: Ray directions (optional, for regularization)
            white_bg: Whether to use white background
            
        Returns:
            Tuple of (rgb, depth, weights):
                - rgb: Rendered RGB colors of shape (..., 3)
                - depth: Expected depth of shape (..., 1)
                - weights: Volume rendering weights of shape (..., num_samples)
        """
        # Compute deltas (distance between consecutive samples)
        # distances: (..., num_samples)
        deltas = distances[..., 1:] - distances[..., :-1]
        
        # Add large delta for last sample (infinity)
        infinity = torch.ones_like(deltas[..., :1]) * 1e10
        deltas = torch.cat([deltas, infinity], dim=-1)
        
        # Compute alpha (opacity) from density
        # alpha = 1 - exp(-sigma * delta)
        # densities: (..., num_samples, 1) -> squeeze to (..., num_samples)
        densities = densities.squeeze(-1)
        alphas = 1.0 - torch.exp(-densities * deltas)
        
        # Compute transmittance T(t) = product of (1 - alpha) up to t
        # T[i] = ∏(1 - alpha[j]) for j < i
        transmittance = torch.cumprod(1.0 - alphas + 1e-10, dim=-1)
        transmittance = torch.cat(
            [torch.ones_like(transmittance[..., :1]), transmittance[..., :-1]], 
            dim=-1
        )
        
        # Compute weights: w_i = T_i * alpha_i
        weights = transmittance * alphas
        
        # Render RGB: weighted sum of colors
        rgb = torch.sum(weights.unsqueeze(-1) * colors, dim=-2)
        
        # Render depth: expected distance
        depth = torch.sum(weights * distances, dim=-1, keepdim=True)
        
        # Add white background if specified
        if white_bg:
            # Accumulated opacity
            acc = torch.sum(weights, dim=-1, keepdim=True)
            rgb = rgb + (1.0 - acc) * 1.0
        
        return rgb, depth, weights
    
    @staticmethod
    def render_image(
        model: torch.nn.Module,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        near: float,
        far: float,
        num_samples: int,
        chunk_size: int = 4096,
        white_bg: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render a full image from rays.
        
        Args:
            model: NeRF model
            ray_origins: Ray origins of shape (height, width, 3)
            ray_directions: Ray directions of shape (height, width, 3)
            near: Near plane distance
            far: Far plane distance
            num_samples: Number of samples per ray
            chunk_size: Number of rays to process at once (for memory efficiency)
            white_bg: Whether to use white background
            
        Returns:
            Tuple of (rgb_image, depth_map):
                - rgb_image: Rendered image of shape (height, width, 3)
                - depth_map: Depth map of shape (height, width, 1)
        """
        height, width = ray_origins.shape[:2]
        device = ray_origins.device
        
        # Flatten rays
        ray_origins_flat = ray_origins.reshape(-1, 3)
        ray_directions_flat = ray_directions.reshape(-1, 3)
        num_rays = ray_origins_flat.shape[0]
        
        # Process in chunks to avoid OOM
        rgb_list = []
        depth_list = []
        
        for i in range(0, num_rays, chunk_size):
            chunk_origins = ray_origins_flat[i:i + chunk_size]
            chunk_directions = ray_directions_flat[i:i + chunk_size]
            
            # Sample points along rays
            from .rays import sample_points_along_rays
            points, distances = sample_points_along_rays(
                chunk_origins,
                chunk_directions,
                near,
                far,
                num_samples,
                perturb=False,  # No perturbation during inference
            )
            
            # Flatten for batch processing
            points_flat = points.reshape(-1, 3)
            distances_flat = distances.reshape(-1, num_samples)
            
            # Expand directions for each sample point
            directions_expanded = chunk_directions.unsqueeze(1).expand(-1, num_samples, -1)
            directions_flat = directions_expanded.reshape(-1, 3)
            
            # Query model in chunks
            densities_list = []
            colors_list = []
            
            chunk_points = 8192  # Points per chunk
            for j in range(0, points_flat.shape[0], chunk_points):
                chunk_points_flat = points_flat[j:j + chunk_points]
                chunk_dirs_flat = directions_flat[j:j + chunk_points]
                
                with torch.no_grad():
                    density, color = model(chunk_points_flat, chunk_dirs_flat)
                
                densities_list.append(density)
                colors_list.append(color)
            
            densities = torch.cat(densities_list, dim=0)
            colors = torch.cat(colors_list, dim=0)
            
            # Reshape back
            densities = densities.reshape(-1, num_samples, 1)
            colors = colors.reshape(-1, num_samples, 3)
            distances = distances_flat.unsqueeze(-1)
            
            # Render
            rgb, depth, _ = VolumeRenderer.render_rays(
                densities,
                colors,
                distances.squeeze(-1),
                white_bg=white_bg,
            )
            
            rgb_list.append(rgb)
            depth_list.append(depth)
        
        # Concatenate and reshape
        rgb_image = torch.cat(rgb_list, dim=0).reshape(height, width, 3)
        depth_map = torch.cat(depth_list, dim=0).reshape(height, width, 1)
        
        return rgb_image, depth_map
