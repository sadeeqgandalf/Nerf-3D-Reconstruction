# NeRF 3D Reconstruction

PyTorch implementation of NeRF (Neural Radiance Fields) for novel view synthesis. Train on a set of posed images, then render the scene from new camera angles. Paper: https://arxiv.org/abs/2003.08934

Uses positional encoding, an MLP for density and RGB, hierarchical (coarse and fine) volume sampling, and the standard volume rendering equation. Configuration is YAML-based, with type hints, logging, checkpointing, and optional mixed precision.

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Data

The project uses the NeRF synthetic Blender dataset (lego, chair, drums, etc.).

Option A — Nerfstudio (simplest):

```bash
pip install nerfstudio
ns-download-data blender
```

Data is written to ~/.nerfstudio/data/blender/. Copy it into the repo:

```bash
# Linux/Mac
cp -r ~/.nerfstudio/data/blender/* data/nerf_synthetic/

# Windows (adjust path)
xcopy /E /I "%USERPROFILE%\.nerfstudio\data\blender\*" "data\nerf_synthetic\"
```

Option B — Manual:

Download nerf_synthetic from https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4, unzip, and place each scene folder (lego, chair, etc.) under data/nerf_synthetic/. Each scene must contain transforms_train.json, transforms_test.json, and image folders (e.g. train/, test/).

## Run

Smoke test (no GPU or data):

```bash
python scripts/smoke_test.py
```

Runs two training steps on CPU with synthetic data to verify the pipeline. Slow but confirms the setup works.

Train (default scene: lego):

```bash
python experiments/train.py --config configs/lego_config.yaml
```

Resume from a checkpoint:

```bash
python experiments/train.py --config configs/lego_config.yaml --resume outputs/checkpoints/checkpoint_step_100000.pth
```

For CPU-only runs add --device cpu. Expect several hours on a typical GPU (e.g. RTX 3080). View logs with: tensorboard --logdir outputs/logs. Checkpoints are saved every 10k steps. Use image_scale: 0.5 in the config for faster runs at lower resolution.

Notebooks:

- notebooks/01_data_exploration.ipynb — inspect data and camera poses
- notebooks/02_training_analysis.ipynb — training curves
- notebooks/03_results_visualization.ipynb — render novel views, 360-degree videos, PSNR/SSIM

## Project layout

```
src/
  models/nerf.py           NeRF MLP and positional encoding
  rendering/               rays, volume rendering
  data/                    dataset, transforms
  training/                trainer, losses
  utils/                   metrics, visualization
  config/                  YAML config loading
configs/                    e.g. lego_config.yaml
experiments/train.py       training entrypoint
notebooks/                 exploration and visualization
```

Outputs (logs, checkpoints, renders) go to outputs/; this directory is gitignored.

## Config

Main options in configs/lego_config.yaml:

- model: num_frequencies_xyz, hidden_dim, num_layers
- training: lr, num_rays, num_samples_coarse, num_samples_fine, num_epochs
- data: image_scale (1.0 = full resolution, 0.5 = half), white_bg

Reduce num_rays and sample counts if you run out of memory; reduce image_scale for speed.

## Troubleshooting

- OOM: Lower num_rays (e.g. 512), fewer coarse/fine samples, or image_scale: 0.5. Reduce render chunk_size if needed.
- Slow: Use a GPU; CPU is much slower. Half resolution and fewer samples help.
- Poor quality: Check poses and normalization; run the data exploration notebook. Try more epochs or a different learning rate.

## References

- NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (Mildenhall et al.): https://arxiv.org/abs/2003.08934
- Dataset: NeRF synthetic Blender scenes (see Data section for links)

MIT License. Issues and pull requests welcome on GitHub.
