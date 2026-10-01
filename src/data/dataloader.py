"""
DataLoader factory with production best practices.

Single place for creating train/val DataLoaders: num_workers, pin_memory,
persistent_workers, prefetch_factor.
"""

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


def create_nerf_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 4,
    prefetch_factor: Optional[int] = 2,
    device: Optional[torch.device] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders with best practices.

    - num_workers=0 when device is CPU to avoid fork issues.
    - pin_memory=True when device is CUDA.
    - persistent_workers=True when num_workers > 0 (avoids respawn each epoch).
    - prefetch_factor when num_workers > 0.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        batch_size: Batch size (typically 1 for NeRF image batches).
        num_workers: Number of loader workers; use 0 for CPU.
        prefetch_factor: Batches to prefetch per worker when num_workers > 0.
        device: Device (used to set pin_memory and num_workers=0 for CPU).

    Returns:
        (train_loader, val_loader)
    """
    is_cuda = device is not None and device.type == "cuda"
    if device is None:
        is_cuda = torch.cuda.is_available()
    use_workers = num_workers > 0 and is_cuda
    if not is_cuda:
        num_workers = 0
        prefetch_factor = None

    common = dict(
        batch_size=batch_size,
        pin_memory=is_cuda,
        num_workers=num_workers,
    )
    if use_workers:
        common["persistent_workers"] = True
        if prefetch_factor is not None:
            common["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader
