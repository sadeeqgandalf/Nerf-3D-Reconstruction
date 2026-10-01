"""
Load NeRF-style data from a COLMAP reconstruction or from our exported transforms.json.

Use scripts/estimate_poses.py to turn a folder of phone images into poses + transforms.json.
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


Split = Literal["train", "test", "val"]


def _parse_camera_to_world_from_colmap(reconstruction, image_id: int, scale: float = 1.0):
    """
    Get 4x4 camera-to-world from COLMAP reconstruction.
    COLMAP uses world-to-camera (R, t). c2w = inv(w2c).
    """
    try:
        import pycolmap
    except ImportError:
        raise ImportError("pycolmap is required for COLMAP data. Install with: pip install pycolmap")

    img = reconstruction.images[image_id]
    cam = reconstruction.cameras[img.camera_id]
    # w2c: rotation 3x3, translation 3x1 (in camera coords)
    R = img.rotmat()
    t = img.tvec
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    c2w = np.linalg.inv(w2c)
    if scale != 1.0:
        c2w[:3, 3] *= scale
    return c2w.astype(np.float32)


def _focal_from_camera(colmap_camera):
    """Focal length in pixels from COLMAP camera model."""
    # SIMPLE_PINHOLE: f, cx, cy
    # PINHOLE: fx, fy, cx, cy
    params = colmap_camera.params
    if colmap_camera.model_name == "SIMPLE_PINHOLE":
        return float(params[0])
    if colmap_camera.model_name == "PINHOLE":
        return (float(params[0]) + float(params[1])) / 2.0
    return float(params[0])


class ColmapDataset(Dataset):
    """
    Dataset from COLMAP reconstruction directory.

    Expects either:
      - data_dir/ with transforms.json (exported by estimate_poses.py), or
      - data_dir/sparse/0/ (cameras.bin, images.bin, points3D.bin) and data_dir/images/
    """

    def __init__(
        self,
        data_dir: str,
        split: Split = "train",
        image_scale: float = 1.0,
        white_bg: bool = False,
        train_split_ratio: float = 0.9,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_scale = image_scale
        self.white_bg = white_bg
        self.train_split_ratio = train_split_ratio

        transforms_json = self.data_dir / "transforms.json"
        if transforms_json.exists():
            self._load_transforms_json(transforms_json)
        else:
            self._load_colmap_native()

    def _load_transforms_json(self, path: Path):
        with open(path) as f:
            meta = json.load(f)
        self.frames = meta["frames"]
        self.focal_length = float(meta.get("focal_length", meta.get("fl_x", 0.0)))
        self._width = int(meta.get("w", 0))
        self._height = int(meta.get("h", 0))
        self.near = float(meta.get("near", 0.1))
        self.far = float(meta.get("far", 10.0))
        if (self._width == 0 or self._height == 0) and self.frames:
            first = self.data_dir / self.frames[0].get("file_path", "images/0.jpg")
            if not first.exists():
                images_dir = self.data_dir / "images"
                if images_dir.exists():
                    first = next(images_dir.iterdir())
                else:
                    first = next(self.data_dir.glob("**/*.jpg"), None) or next(self.data_dir.glob("**/*.png"), None)
                if first is None:
                    raise FileNotFoundError(f"No image found under {self.data_dir}")
            img = Image.open(first)
            self._width = img.width
            self._height = img.height

    def _load_colmap_native(self):
        try:
            import pycolmap
        except ImportError:
            raise ImportError("pycolmap required. pip install pycolmap")
        sparse = self.data_dir / "sparse" / "0"
        if not (sparse / "cameras.bin").exists() and not (sparse / "cameras.txt").exists():
            # Fallback: reconstruction written directly to sparse/ (e.g. older estimate_poses)
            sparse = self.data_dir / "sparse"
        if not (sparse / "cameras.bin").exists() and not (sparse / "cameras.txt").exists():
            raise FileNotFoundError(
                f"COLMAP reconstruction not found under {self.data_dir}. "
                "Run scripts/estimate_poses.py first to generate poses from images."
            )
        recon = pycolmap.Reconstruction(sparse)
        images_dir = self.data_dir / "images"
        self.frames = []
        for image_id, img in recon.images.items():
            rel_path = img.name
            if not (images_dir / rel_path).exists():
                rel_path = Path(rel_path).name
            c2w = _parse_camera_to_world_from_colmap(recon, image_id)
            cam = recon.cameras[img.camera_id]
            focal = _focal_from_camera(cam)
            self.frames.append({
                "file_path": str(images_dir.name + "/" + Path(rel_path).name),
                "transform_matrix": c2w.tolist(),
                "focal": focal,
            })
        if not self.frames:
            raise FileNotFoundError(f"No images in reconstruction at {self.data_dir}")
        self.focal_length = self.frames[0].get("focal") or _focal_from_camera(recon.cameras[recon.images[next(iter(recon.images))].camera_id])
        first_img = images_dir / Path(self.frames[0]["file_path"]).name
        if first_img.exists():
            im = Image.open(first_img)
            self._width = im.width
            self._height = im.height
        else:
            self._width = self._height = 800
        self.near = 0.1
        self.far = 10.0
        return

    def __len__(self) -> int:
        n = len(self.frames)
        if self.split == "train":
            return max(1, int(n * self.train_split_ratio))
        return max(1, n - int(n * self.train_split_ratio))

    def _frame_index(self, idx: int) -> int:
        n = len(self.frames)
        train_n = int(n * self.train_split_ratio)
        if self.split == "train":
            return idx % train_n
        return train_n + (idx % (n - train_n))

    def get_image_size(self) -> tuple[int, int]:
        return (int(self._height * self.image_scale), int(self._width * self.image_scale))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = self._frame_index(idx)
        frame = self.frames[i]
        file_path = frame.get("file_path") or frame.get("filename", "")
        path = self.data_dir / file_path
        if not path.exists():
            path = self.data_dir / "images" / Path(file_path).name
        image = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

        if self.image_scale != 1.0:
            h = int(image.shape[0] * self.image_scale)
            w = int(image.shape[1] * self.image_scale)
            image = np.array(
                Image.fromarray((image * 255).astype(np.uint8)).resize((w, h), Image.Resampling.LANCZOS),
                dtype=np.float32,
            ) / 255.0

        if self.white_bg:
            alpha = (image.sum(axis=-1) < 2.99).astype(np.float32)[..., None]
            image = image * alpha + (1.0 - alpha)

        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        focal = frame.get("focal") or self.focal_length
        if isinstance(focal, (list, tuple)):
            focal = (float(focal[0]) + float(focal[1])) / 2.0
        focal = float(focal) * self.image_scale

        return {
            "image": torch.from_numpy(image),
            "pose": torch.from_numpy(c2w),
            "focal": torch.tensor(focal, dtype=torch.float32),
            "near": torch.tensor(self.near, dtype=torch.float32),
            "far": torch.tensor(self.far, dtype=torch.float32),
        }
