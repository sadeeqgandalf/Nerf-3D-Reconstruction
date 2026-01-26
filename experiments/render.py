"""
Inference script for rendering novel views with trained NeRF model.

Usage:
    python render.py --checkpoint outputs/checkpoints/checkpoint_step_100000.pth --config configs/lego_config.yaml
"""

import argparse
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.models import NeRF
from src.data import NeRFDataset
from src.rendering.volume_renderer import VolumeRenderer
from src.rendering import generate_rays
from src.utils import save_image, compute_psnr, create_video
import numpy as np


def render_novel_view(
    model_fine,
    dataset,
    pose_idx,
    device,
    num_samples=128,
    chunk_size=4096,
    output_path=None,
):
    """Render a single novel view."""
    # Get test sample
    sample = dataset[pose_idx]
    pose = sample['pose'].to(device)
    focal = sample['focal'].to(device)
    near = sample['near'].to(device)
    far = sample['far'].to(device)
    gt_image = sample['image']
    
    height, width = gt_image.shape[:2]
    
    # Generate rays
    ray_origins, ray_directions = generate_rays(
        height, width, focal.item(), pose, near.item(), far.item()
    )
    
    # Render
    model_fine.eval()
    with torch.no_grad():
        rgb_image, depth_map = VolumeRenderer.render_image(
            model_fine,
            ray_origins.to(device),
            ray_directions.to(device),
            near.item(),
            far.item(),
            num_samples=num_samples,
            chunk_size=chunk_size,
            white_bg=dataset.white_bg,
        )
    
    # Compute PSNR
    psnr = compute_psnr(rgb_image.cpu(), gt_image)
    
    # Save if path provided
    if output_path:
        save_image(rgb_image.cpu(), output_path)
        print(f"Saved rendered image to {output_path}")
        print(f"PSNR: {psnr:.2f} dB")
    
    return rgb_image.cpu(), depth_map.cpu(), psnr


def render_360_video(
    model_fine,
    dataset,
    device,
    num_frames=60,
    radius=4.0,
    num_samples=128,
    chunk_size=4096,
    output_path=None,
):
    """Render 360° rotation video."""
    # Get scene center and scale from dataset
    sample = dataset[0]
    height, width = sample['image'].shape[:2]
    focal = sample['focal'].to(device)
    near = sample['near'].to(device)
    far = sample['far'].to(device)
    
    # Generate poses for 360° rotation
    render_poses = []
    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        
        # Camera position on circle
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        y = 0.0
        
        # Look at origin
        forward = -np.array([x, y, z])
        forward = forward / np.linalg.norm(forward)
        
        # Create pose matrix
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        up = np.cross(right, forward)
        
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = [x, y, z]
        
        render_poses.append(pose)
    
    # Render frames
    frames = []
    model_fine.eval()
    
    print(f"Rendering {num_frames} frames for 360° video...")
    for i, pose in enumerate(render_poses):
        pose_tensor = torch.from_numpy(pose).float().to(device)
        
        ray_origins, ray_directions = generate_rays(
            height, width, focal.item(), pose_tensor, near.item(), far.item()
        )
        
        with torch.no_grad():
            rgb_image, _ = VolumeRenderer.render_image(
                model_fine,
                ray_origins.to(device),
                ray_directions.to(device),
                near.item(),
                far.item(),
                num_samples=num_samples,
                chunk_size=chunk_size,
                white_bg=dataset.white_bg,
            )
        
        frames.append(rgb_image.cpu())
        if (i + 1) % 10 == 0:
            print(f"  Rendered {i + 1}/{num_frames} frames")
    
    # Save video
    if output_path:
        create_video(frames, output_path, fps=30)
        print(f"Saved 360° video to {output_path}")
    
    return frames


def main():
    parser = argparse.ArgumentParser(description='Render novel views with trained NeRF')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./outputs/renders', help='Output directory')
    parser.add_argument('--num_views', type=int, default=5, help='Number of novel views to render')
    parser.add_argument('--render_360', action='store_true', help='Render 360° video')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    if args.device:
        config.device = args.device
    
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    test_dataset = NeRFDataset(
        data_dir=config.data.data_dir,
        split='test',
        image_scale=config.data.image_scale,
        white_bg=config.data.white_bg,
    )
    
    # Create model
    model_fine = NeRF(
        num_frequencies_xyz=config.model.num_frequencies_xyz,
        num_frequencies_dir=config.model.num_frequencies_dir,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        skip_connection_layer=config.model.skip_connection_layer,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_fine.load_state_dict(checkpoint['model_fine_state_dict'])
    print(f"Loaded checkpoint from step {checkpoint['global_step']}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Render novel views
    print(f"\nRendering {args.num_views} novel views...")
    psnrs = []
    for i in range(min(args.num_views, len(test_dataset))):
        output_path = Path(args.output_dir) / f'novel_view_{i:03d}.png'
        _, _, psnr = render_novel_view(
            model_fine,
            test_dataset,
            i,
            device,
            output_path=str(output_path),
        )
        psnrs.append(psnr)
    
    print(f"\nAverage PSNR: {np.mean(psnrs):.2f} dB")
    
    # Render 360° video if requested
    if args.render_360:
        output_path = Path(args.output_dir) / '360_rotation.mp4'
        render_360_video(
            model_fine,
            test_dataset,
            device,
            output_path=str(output_path),
        )
    
    print("\nRendering complete!")


if __name__ == '__main__':
    main()
