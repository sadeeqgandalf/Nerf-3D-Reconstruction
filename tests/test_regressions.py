"""
Regression tests for correctness fixes (skip connection, rendering exports,
hierarchical sampling, volume-rendering weights, dataset package).
"""

import json
import math

import numpy as np
import pytest
import torch

from src.models import NeRF
from src.rendering import (
    VolumeRenderer,
    generate_rays,
    hierarchical_sample,
    sample_points_along_rays,
)


# --- Model: skip connection -------------------------------------------------

@pytest.mark.parametrize("skip", [1, 4, 7])
def test_skip_connection_forward(skip):
    model = NeRF(hidden_dim=32, num_layers=8, skip_connection_layer=skip)
    assert model.skip_connection_layer == skip
    # The Linear layer at index `skip` takes [h, gamma(x)] as input.
    enc_dim = model.pos_encoding_xyz.output_dim
    assert model.density_layers[2 * skip].in_features == 32 + enc_dim
    density, rgb = model(torch.randn(5, 3), torch.randn(5, 3))
    assert density.shape == (5, 1) and rgb.shape == (5, 3)


def test_skip_connection_uses_input_encoding():
    """Zeroing the skip layer's input-encoding weights must change the output."""
    torch.manual_seed(0)
    model = NeRF(hidden_dim=32)
    xyz, d = torch.randn(8, 3), torch.randn(8, 3)
    _, rgb = model(xyz, d)
    with torch.no_grad():
        model.density_layers[2 * model.skip_connection_layer].weight[:, 32:] = 0.0
    _, rgb2 = model(xyz, d)
    assert not torch.allclose(rgb, rgb2)


def test_invalid_skip_layer_rejected():
    with pytest.raises(ValueError):
        NeRF(num_layers=8, skip_connection_layer=8)


# --- Rendering exports and hierarchical sampling ----------------------------

def test_training_package_imports():
    import src.training  # noqa: F401  (failed with ImportError before the fix)
    from src.rendering import __all__ as exported

    assert "hierarchical_sample" in exported


def _coarse(num_rays=6, n=32, near=2.0, far=6.0):
    origins = torch.zeros(num_rays, 3)
    dirs = torch.tensor([[0.0, 0.0, -1.0]]).expand(num_rays, 3)
    _, z = sample_points_along_rays(origins, dirs, near, far, n, perturb=False)
    return origins, dirs, z


def test_hierarchical_sample_shapes_range_and_detach():
    origins, dirs, z = _coarse()
    weights = torch.rand(z.shape, requires_grad=True)
    points, z_fine = hierarchical_sample(origins, dirs, z, weights, 64)
    assert points.shape == (6, 64, 3) and z_fine.shape == (6, 64)
    assert not z_fine.requires_grad
    assert (z_fine >= z[..., :1]).all() and (z_fine <= z[..., -1:]).all()
    # Points lie on the rays.
    assert torch.allclose(points, origins[:, None] + z_fine[..., None] * dirs[:, None])


def test_hierarchical_sample_monotonic_and_concentrated():
    origins, dirs, z = _coarse()
    weights = torch.zeros_like(z)
    weights[:, 20] = 1.0  # all mass around sample 20
    _, z_fine = hierarchical_sample(origins, dirs, z, weights, 128, perturb=False)
    # Deterministic u is sorted, so the inverse CDF output must be non-decreasing.
    assert (z_fine[..., 1:] >= z_fine[..., :-1] - 1e-6).all()
    lo = 0.5 * (z[0, 19] + z[0, 20])
    hi = 0.5 * (z[0, 20] + z[0, 21])
    inside = ((z_fine >= lo - 1e-4) & (z_fine <= hi + 1e-4)).float().mean()
    assert inside > 0.95


# --- Volume rendering on a known case ---------------------------------------

def test_volume_rendering_weights_known_case():
    z = torch.tensor([[1.0, 2.0, 4.0]])
    sigma = torch.tensor([[[0.5], [1.0], [0.0]]])
    colors = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    rgb, depth, w = VolumeRenderer.render_rays(sigma, colors, z)

    a1 = 1 - math.exp(-0.5 * 1.0)
    a2 = 1 - math.exp(-1.0 * 2.0)
    expected = torch.tensor([[a1, (1 - a1) * a2, 0.0]])
    assert torch.allclose(w, expected, atol=1e-6)
    assert w.sum() <= 1.0 + 1e-6
    assert torch.allclose(rgb, torch.tensor([[a1, (1 - a1) * a2, 0.0]]), atol=1e-6)
    assert torch.allclose(depth, (expected * z).sum(-1, keepdim=True), atol=1e-6)


def test_volume_rendering_white_background():
    z = torch.linspace(2, 6, 16)[None]
    colors = torch.zeros(1, 16, 3)
    empty = torch.zeros(1, 16, 1)
    rgb, _, w = VolumeRenderer.render_rays(empty, colors, z, white_bg=True)
    assert torch.allclose(rgb, torch.ones(1, 3))
    opaque = torch.full((1, 16, 1), 1e4)
    rgb, _, w = VolumeRenderer.render_rays(opaque, colors, z, white_bg=True)
    assert torch.allclose(rgb, torch.zeros(1, 3), atol=1e-5)
    assert abs(float(w.sum()) - 1.0) < 1e-5


# --- Rays --------------------------------------------------------------------

def test_generate_rays_convention_and_batched_pose():
    H = W = 4
    o, d = generate_rays(H, W, 10.0, torch.eye(4))
    # Pixel (H/2, W/2) looks straight down -z (OpenGL/Blender convention).
    assert torch.allclose(d[H // 2, W // 2], torch.tensor([0.0, 0.0, -1.0]))
    # Top row points up (+y), right column points right (+x).
    assert d[0, W // 2, 1] > 0 and d[H // 2, W - 1, 0] > 0
    o2, d2 = generate_rays(H, W, 10.0, torch.eye(4)[None])
    assert torch.allclose(d, d2) and torch.allclose(o, o2)


# --- Dataset package ---------------------------------------------------------

def test_data_package_imports():
    from src.data import BlenderDataset, NeRFDataset, RaySampler  # noqa: F401


def _write_blender_scene(root, size=8):
    from PIL import Image

    (root / "train").mkdir()
    rgba = np.zeros((size, size, 4), dtype=np.uint8)  # transparent black
    rgba[2:6, 2:6] = [255, 0, 0, 255]  # opaque red square
    Image.fromarray(rgba, "RGBA").save(root / "train" / "r_0.png")
    meta = {
        "camera_angle_x": 0.6911112070083618,
        "frames": [{"file_path": "./train/r_0", "transform_matrix": np.eye(4).tolist()}],
    }
    (root / "transforms_train.json").write_text(json.dumps(meta))


def test_blender_dataset_white_background(tmp_path):
    from src.data import NeRFDataset

    _write_blender_scene(tmp_path)
    ds = NeRFDataset(str(tmp_path), split="train", white_bg=True)
    sample = ds[0]
    img = sample["image"]
    assert img.shape == (8, 8, 3)
    assert torch.allclose(img[0, 0], torch.ones(3))  # transparent -> white
    assert torch.allclose(img[3, 3], torch.tensor([1.0, 0.0, 0.0]))
    expected_focal = 0.5 * 8 / math.tan(0.5 * 0.6911112070083618)
    assert abs(float(sample["focal"]) - expected_focal) < 1e-4


def test_ray_sampler_runs(tmp_path):
    from src.data import NeRFDataset, RaySampler

    _write_blender_scene(tmp_path)
    ds = NeRFDataset(str(tmp_path), split="train")
    batch = RaySampler(ds, torch.device("cpu")).sample(16)
    assert batch["ray_origins"].shape == (16, 3)
    assert batch["target_rgb"].shape == (16, 3)


def test_stratified_sampling_perturbed_shapes_and_bins():
    torch.manual_seed(0)
    origins, dirs = torch.zeros(8, 3), torch.tensor([[0.0, 0.0, -1.0]]).expand(8, 3)
    points, z = sample_points_along_rays(origins, dirs, 2.0, 6.0, 16, perturb=True)
    assert points.shape == (8, 16, 3) and z.shape == (8, 16)
    assert (z[..., 1:] >= z[..., :-1]).all()
    assert (z >= 2.0).all() and (z <= 6.0).all()
    # Rays get different jitter.
    assert not torch.allclose(z[0], z[1])


@pytest.mark.parametrize("seed", range(5))
def test_density_not_dead_at_init(seed):
    """With a random negative density bias every sigma was 0 and no gradient flowed."""
    torch.manual_seed(seed)
    model = NeRF(num_frequencies_xyz=6, num_frequencies_dir=2, hidden_dim=64)
    xyz = torch.rand(256, 3) * 2 - 1
    dirs = torch.nn.functional.normalize(torch.randn(256, 3), dim=-1)
    sigma, rgb = model(xyz, dirs)
    assert (sigma > 0).float().mean() > 0.5
    z = torch.linspace(2.0, 6.0, 256)[None]
    out, _, _ = VolumeRenderer.render_rays(sigma[None], rgb[None], z, white_bg=True)
    out.sum().backward()
    assert model.density_head.weight.grad.abs().sum() > 0
