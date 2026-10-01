"""
Ray generation and sampling utilities.

Functions to generate camera rays and sample 3D points along rays
for volume rendering.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def generate_rays(
    height: int,
    width: int,
    focal_length: float,
    camera_pose: torch.Tensor,
    near: float = 0.0,
    far: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate camera rays for all pixels in an image.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        focal_length: Camera focal length (in pixels)
        camera_pose: Camera-to-world transformation matrix of shape (4, 4)
        near: Near plane distance
        far: Far plane distance
        
    Returns:
        Tuple of (ray_origins, ray_directions):
            - ray_origins: Tensor of shape (height, width, 3) - camera position in world space
            - ray_directions: Tensor of shape (height, width, 3) - normalized ray directions
    """
    # Accept a pose with a leading batch dimension of 1, as produced by a
    # DataLoader with batch_size=1 or by pose.unsqueeze(0).
    if camera_pose.dim() == 3:
        if camera_pose.shape[0] != 1:
            raise ValueError(
                f"generate_rays expects a single (4, 4) pose, got {tuple(camera_pose.shape)}"
            )
        camera_pose = camera_pose[0]
    device = camera_pose.device
    
    # Create pixel grid
    # i, j are pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # Convert pixel coordinates to camera space
    # Camera coordinate system: x right, y down, z forward
    # Pixel (i, j) maps to camera space (x, y, z=1)
    x = (j - width * 0.5) / focal_length
    y = -(i - height * 0.5) / focal_length  # Negative because y points down
    z = -torch.ones_like(i)  # Negative because z points forward (towards scene)
    
    # Stack into (height, width, 3) tensor
    directions_camera = torch.stack([x, y, z], dim=-1)
    
    # Normalize directions
    directions_camera = F.normalize(directions_camera, dim=-1)
    
    # Transform directions from camera space to world space
    rotation = camera_pose[:3, :3]  # (3, 3)
    translation = camera_pose[:3, 3]  # (3,)
    
    # Apply rotation to directions
    directions_world = directions_camera @ rotation.T  # (height, width, 3)
    
    # Ray origin is camera position in world space
    ray_origins = translation.unsqueeze(0).unsqueeze(0).expand(height, width, -1)
    
    return ray_origins, directions_world


def sample_points_along_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    perturb: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample 3D points along rays using stratified sampling.
    
    Args:
        ray_origins: Ray origins of shape (..., 3)
        ray_directions: Ray directions (normalized) of shape (..., 3)
        near: Near plane distance
        far: Far plane distance
        num_samples: Number of points to sample per ray
        perturb: Whether to add random perturbation (for training)
        
    Returns:
        Tuple of (points, distances):
            - points: Sampled 3D points of shape (..., num_samples, 3)
            - distances: Distances along rays of shape (..., num_samples)
    """
    # Get batch shape (all dimensions except last)
    batch_shape = ray_origins.shape[:-1]
    device = ray_origins.device
    
    # Create stratified bins
    t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
    
    # Add perturbation during training for better sampling
    if perturb:
        # Add random offset to each bin
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
        lower = torch.cat([t_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand(*batch_shape, num_samples, device=device)
        t_vals = lower + (upper - lower) * t_rand
    
    # Convert to actual distances
    distances = near + (far - near) * t_vals
    
    # Expand to match batch shape
    distances = distances.view(*([1] * len(batch_shape)), -1)
    distances = distances.expand(*batch_shape, -1)
    
    # Compute 3D points along rays
    # points = origins + distances * directions
    points = (
        ray_origins.unsqueeze(-2) + 
        distances.unsqueeze(-1) * ray_directions.unsqueeze(-2)
    )
    
    return points, distances


def hierarchical_sample(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    distances: torch.Tensor,
    weights: torch.Tensor,
    num_samples: int,
    perturb: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hierarchical (importance) sampling based on volume rendering weights.
    
    Samples more points in regions with high density (where weights are large).
    Follows ``sample_pdf`` in the reference NeRF implementation: the bins are
    the midpoints between the coarse samples, the PDF uses the weights of the
    interior samples, and new distances are drawn by inverse-transform sampling.
    Weights and the returned distances are detached, so no gradient flows
    through the sample positions.
    
    Args:
        ray_origins: Ray origins of shape (..., 3)
        ray_directions: Ray directions of shape (..., 3)
        distances: Previous (sorted) sample distances of shape (..., num_coarse)
        weights: Volume rendering weights of shape (..., num_coarse)
        num_samples: Number of fine samples to add
        perturb: Draw random uniforms (training) instead of a fixed linspace
        
    Returns:
        Tuple of (points, distances) for fine samples, with shapes
        (..., num_samples, 3) and (..., num_samples)
    """
    if distances.shape[-1] < 3:
        raise ValueError("hierarchical_sample needs at least 3 coarse samples per ray")
    distances = distances.detach()
    weights = weights.detach()

    # Bin edges: midpoints between coarse samples -> (..., num_coarse - 1)
    bins = 0.5 * (distances[..., 1:] + distances[..., :-1])
    # One weight per bin interior: weights of samples 1 .. num_coarse - 2
    bin_weights = weights[..., 1:-1] + 1e-5  # avoid NaNs on empty rays
    pdf = bin_weights / bin_weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (..., num_coarse - 1)

    batch_shape = cdf.shape[:-1]
    device = cdf.device
    if perturb:
        u = torch.rand(*batch_shape, num_samples, device=device)
    else:
        u = torch.linspace(0.0, 1.0, num_samples, device=device).expand(*batch_shape, num_samples)
    u = u.contiguous()

    # Invert the CDF
    indices = torch.searchsorted(cdf.contiguous(), u, right=True)
    below = torch.clamp(indices - 1, min=0)
    above = torch.clamp(indices, max=cdf.shape[-1] - 1)

    cdf_below = torch.gather(cdf, -1, below)
    cdf_above = torch.gather(cdf, -1, above)
    bins_below = torch.gather(bins, -1, below)
    bins_above = torch.gather(bins, -1, above)

    denom = cdf_above - cdf_below
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_below) / denom
    fine_distances = bins_below + t * (bins_above - bins_below)

    # Compute 3D points
    fine_points = (
        ray_origins.unsqueeze(-2) + 
        fine_distances.unsqueeze(-1) * ray_directions.unsqueeze(-2)
    )
    
    return fine_points, fine_distances
