"""
Blender-format NeRF dataset (synthetic data).

Loads transforms_train.json / transforms_test.json / transforms_val.json
with camera_angle_x and frames (file_path, transform_matrix).
Focal length: focal = 0.5 * width / tan(0.5 * camera_angle_x).
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


Split = Literal["train", "test", "val"]


class BlenderDataset(Dataset):
    """
    Dataset for Blender-rendered NeRF synthetic data.

    Expects:
        data_dir/
          transforms_train.json, transforms_test.json, (transforms_val.json)
          train/, test/, (val/) with images
    """

    def __init__(
        self,
        data_dir: str,
        split: Split = "train",
        image_scale: float = 1.0,
        white_bg: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_scale = image_scale
        self.white_bg = white_bg

        # Map split to possible JSON names (some repos use test, some val)
        json_names = (
            [f"transforms_{split}.json"]
            if (self.data_dir / f"transforms_{split}.json").exists()
            else (
                [f"transforms_{split}.json", "transforms_val.json"]
                if split == "test"
                else [f"transforms_{split}.json"]
            )
        )
        json_path = None
        for name in json_names:
            p = self.data_dir / name
            if p.exists():
                json_path = p
                break
        if json_path is None:
            raise FileNotFoundError(
                f"No transforms found in {self.data_dir} for split {split}. "
                f"Looked for {json_names}. "
                "Ensure transforms_train.json and transforms_test.json exist."
            )

        with open(json_path) as f:
            meta = json.load(f)

        self.camera_angle_x = float(meta["camera_angle_x"])
        self.frames = meta["frames"]

        # Load first image to get size and compute focal
        first_path = self.data_dir / self.frames[0]["file_path"]
        if not first_path.suffix:
            first_path = Path(str(first_path) + ".png")
        if not first_path.exists():
            first_path = self.data_dir / "train" / (Path(self.frames[0]["file_path"]).name + ".png")
        img = Image.open(first_path)
        self._width = img.width
        self._height = img.height
        # focal = 0.5 * width / tan(0.5 * camera_angle_x)
        self.focal_length = 0.5 * self._width / np.tan(0.5 * self.camera_angle_x)

        # NeRF synthetic near/far
        self.near = 2.0
        self.far = 6.0

    def __len__(self) -> int:
        return len(self.frames)

    def get_image_size(self) -> tuple[int, int]:
        return (int(self._height * self.image_scale), int(self._width * self.image_scale))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        frame = self.frames[idx]
        file_path = frame["file_path"]
        path = self.data_dir / file_path
        if not path.suffix:
            path = Path(str(path) + ".png")
        if not path.exists():
            path = self.data_dir / self.split / (Path(file_path).name + ".png")
        if not path.exists():
            path = self.data_dir / "train" / (Path(file_path).name + ".png")

        image = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

        if self.image_scale != 1.0:
            from PIL import Image as PILImage
            h, w = int(image.shape[0] * self.image_scale), int(image.shape[1] * self.image_scale)
            image = np.array(
                PILImage.fromarray((image * 255).astype(np.uint8)).resize((w, h), Image.Resampling.LANCZOS),
                dtype=np.float32,
            ) / 255.0

        if self.white_bg:
            alpha = (image.sum(axis=-1) < 2.99).astype(np.float32)[..., None]
            image = image * alpha + (1.0 - alpha)

        # 4x4 camera-to-world (OpenGL / NeRF convention)
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        pose = torch.from_numpy(c2w)
        focal = torch.tensor(self.focal_length * self.image_scale, dtype=torch.float32)
        near = torch.tensor(self.near, dtype=torch.float32)
        far = torch.tensor(self.far, dtype=torch.float32)

        return {
            "image": torch.from_numpy(image),
            "pose": pose,
            "focal": focal,
            "near": near,
            "far": far,
        }
