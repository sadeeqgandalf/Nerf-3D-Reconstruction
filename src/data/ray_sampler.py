"""
Ray sampler for step-based NeRF training (paper: sample batch of rays from all pixels in dataset).
"""

from collections import defaultdict
from typing import Any, Dict

import torch

from ..rendering import generate_rays


class RaySampler:
    """
    Samples random rays from the entire training set (all images, all pixels).
    Used for paper-aligned step-based training: each step samples num_rays from the full dataset.
    """

    def __init__(self, dataset: Any, device: torch.device):
        """
        Args:
            dataset: NeRF dataset (BlenderDataset, ColmapDataset, or any with __getitem__
                     returning dict with image, pose, focal, near, far).
            device: Device to place tensors on.
        """
        self.dataset = dataset
        self.device = device
        self._num_images = len(dataset)

    def sample(self, num_rays: int) -> Dict[str, torch.Tensor]:
        """
        Sample num_rays rays uniformly from the set of all pixels in all training images.

        Returns:
            Dict with:
                ray_origins: (num_rays, 3)
                ray_directions: (num_rays, 3)
                target_rgb: (num_rays, 3)
                near: (num_rays,) or scalar
                far: (num_rays,) or scalar
        """
        if self._num_images == 0:
            raise ValueError("Dataset is empty")
        # Sample (image_idx, pixel_idx) with replacement
        image_indices = torch.randint(0, self._num_images, (num_rays,))
        # Group by image to minimize loads
        groups: Dict[int, list] = defaultdict(list)
        for r in range(num_rays):
            groups[int(image_indices[r].item())].append(r)
        ray_origins_list = []
        ray_directions_list = []
        target_rgb_list = []
        near_list = []
        far_list = []
        for img_idx, ray_indices in groups.items():
            sample = self.dataset[img_idx]
            image = sample["image"]
            pose = sample["pose"]
            focal = sample["focal"]
            near = sample["near"]
            far = sample["far"]
            h, w = image.shape[0], image.shape[1]
            num_pixels = h * w
            # Generate all rays for this image
            ray_origins, ray_directions = generate_rays(
                h, w, focal.item(), pose.unsqueeze(0), near.item(), far.item()
            )
            ray_origins = ray_origins.reshape(-1, 3)
            ray_directions = ray_directions.reshape(-1, 3)
            image_flat = image.reshape(-1, 3)
            # Sample pixel indices for this image
            n_from_this = len(ray_indices)
            pixel_indices = torch.randint(0, num_pixels, (n_from_this,))
            ray_origins_list.append(ray_origins[pixel_indices])
            ray_directions_list.append(ray_directions[pixel_indices])
            target_rgb_list.append(image_flat[pixel_indices])
            near_list.append(near.expand(n_from_this))
            far_list.append(far.expand(n_from_this))
        ray_origins = torch.cat(ray_origins_list, dim=0).to(self.device)
        ray_directions = torch.cat(ray_directions_list, dim=0).to(self.device)
        target_rgb = torch.cat(target_rgb_list, dim=0).to(self.device)
        near = torch.cat(near_list, dim=0).to(self.device)
        far = torch.cat(far_list, dim=0).to(self.device)
        return {
            "ray_origins": ray_origins,
            "ray_directions": ray_directions,
            "target_rgb": target_rgb,
            "near": near,
            "far": far,
        }
