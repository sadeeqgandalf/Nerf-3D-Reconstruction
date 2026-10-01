"""
Unified NeRF dataset factory.

Supports:
  - blender: Blender synthetic (transforms_train.json, etc.)
  - colmap: COLMAP reconstruction or exported transforms.json (e.g. from phone images)
"""

from pathlib import Path
from typing import Literal, Union

from .blender import BlenderDataset
from .colmap_loader import ColmapDataset

DatasetType = Literal["blender", "colmap"]


def NeRFDataset(
    data_dir: str,
    split: Literal["train", "test", "val"] = "train",
    image_scale: float = 1.0,
    white_bg: bool = True,
    dataset_type: Union[DatasetType, None] = None,
    **kwargs,
) -> Union[BlenderDataset, ColmapDataset]:
    """
    Create a NeRF dataset. Auto-detects format if dataset_type is None.

    - If transforms_train.json exists -> BlenderDataset
    - Else if transforms.json or sparse/0/ exists -> ColmapDataset
    - Otherwise raises.
    """
    data_path = Path(data_dir)
    if dataset_type == "blender":
        return BlenderDataset(
            data_dir=data_dir,
            split=split,
            image_scale=image_scale,
            white_bg=white_bg,
        )
    if dataset_type == "colmap":
        return ColmapDataset(
            data_dir=data_dir,
            split=split,
            image_scale=image_scale,
            white_bg=white_bg,
            **kwargs,
        )
    # Auto-detect
    if (data_path / "transforms_train.json").exists():
        return BlenderDataset(
            data_dir=data_dir,
            split=split,
            image_scale=image_scale,
            white_bg=white_bg,
        )
    if (data_path / "transforms.json").exists() or (data_path / "sparse" / "0").exists():
        return ColmapDataset(
            data_dir=data_dir,
            split=split,
            image_scale=image_scale,
            white_bg=white_bg,
            **kwargs,
        )
    raise ValueError(
        f"Unknown data format in {data_dir}. "
        "Provide transforms_train.json (Blender) or run scripts/estimate_poses.py for phone images (COLMAP)."
    )
