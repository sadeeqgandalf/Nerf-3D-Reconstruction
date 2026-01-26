# NeRF 3D Scene Reconstruction

A production-ready PyTorch implementation of **Neural Radiance Fields (NeRF)** for novel view synthesis and 3D scene reconstruction. This project demonstrates advanced PyTorch engineering practices, efficient data loading, and industry-standard code organization.

## 🎯 Project Overview

NeRF learns a continuous 3D representation of a scene from a set of 2D images with known camera poses. The model can then render photorealistic novel views from any camera angle, effectively creating a 3D scene you can explore.

### Key Features

- ✅ **Complete NeRF Implementation**: Full implementation of the original NeRF paper
- ✅ **Hierarchical Volume Sampling**: Coarse + fine network architecture for efficient training
- ✅ **Production-Ready Code**: Clean architecture, type hints, comprehensive documentation
- ✅ **Efficient PyTorch**: Custom DataLoaders, mixed precision training, optimized rendering
- ✅ **Comprehensive Analysis**: Data exploration notebooks, training visualization, results analysis
- ✅ **Industry Best Practices**: YAML configs, logging, checkpointing, unit tests

## 📁 Project Structure

```
nerf-3d-reconstruction/
├── src/
│   ├── models/              # NeRF model architecture
│   │   └── nerf.py         # MLP-based NeRF model with positional encoding
│   ├── rendering/          # Volume rendering engine
│   │   ├── rays.py         # Ray generation and sampling
│   │   └── volume_renderer.py  # Volume rendering equation
│   ├── data/               # Data loading pipeline
│   │   ├── dataset.py      # NeRF dataset implementation
│   │   └── transforms.py   # Data transformations
│   ├── training/           # Training utilities
│   │   ├── trainer.py      # Training loop with best practices
│   │   └── losses.py       # Loss functions
│   ├── utils/              # Utility functions
│   │   ├── metrics.py      # PSNR, SSIM evaluation
│   │   └── visualization.py  # Image/video saving utilities
│   └── config/             # Configuration management
│       └── config.py       # YAML-based config system
├── configs/                # Training configurations
│   └── lego_config.yaml    # Example config for Lego scene
├── experiments/            # Training scripts
│   └── train.py           # Main training script
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb      # Dataset analysis
│   ├── 02_training_analysis.ipynb     # Training metrics
│   └── 03_results_visualization.ipynb # Results visualization
├── data/                   # Dataset directory (gitignored)
├── outputs/                # Training outputs (gitignored)
│   ├── logs/              # TensorBoard logs
│   ├── checkpoints/       # Model checkpoints
│   └── renders/          # Rendered images/videos
├── requirements.txt
├── setup.py
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd nerf-3d-reconstruction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 2. Download Dataset

The project uses the NeRF synthetic Blender dataset. You have two options to get the data:

#### Option 1: Using Nerfstudio (Recommended - Easiest)

Nerfstudio provides a convenient command-line tool to download the dataset:

```bash
# Install nerfstudio (if not already installed)
pip install nerfstudio

# Download the Blender synthetic dataset
ns-download-data blender
```

**What this gives you:**
- Downloads the complete Blender synthetic dataset (8 scenes: lego, chair, drums, ficus, hotdog, materials, mic, ship)
- Automatically places data in a standard location (usually `~/.nerfstudio/data/`)
- Ready-to-use format with proper directory structure

**After download, move/copy to project directory:**
```bash
# Find where nerfstudio saved the data (usually ~/.nerfstudio/data/blender/)
# Then copy to your project:
cp -r ~/.nerfstudio/data/blender/* data/nerf_synthetic/
# Or on Windows:
xcopy /E /I "C:\Users\YourName\.nerfstudio\data\blender\*" "data\nerf_synthetic\"
```

#### Option 2: Manual Download

Download directly from the original NeRF repository:

1. **Visit the dataset link**: [NeRF Synthetic Dataset on Google Drive](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4)

2. **Download the dataset**:
   - Look for `nerf_synthetic.zip` or individual scene folders
   - The dataset contains 8 synthetic scenes: lego, chair, drums, ficus, hotdog, materials, mic, ship

3. **Extract and organize**:
   ```bash
   # Create data directory
   mkdir -p data/nerf_synthetic
   
   # Extract the downloaded zip file
   # Then move the scene folders to data/nerf_synthetic/
   ```

**Dataset Structure (after setup):**
```
data/nerf_synthetic/
├── lego/
│   ├── transforms_train.json
│   ├── transforms_test.json
│   └── train/ (or images/)  # Training images
│   └── test/                 # Test images
├── chair/
│   ├── transforms_train.json
│   ├── transforms_test.json
│   └── train/
├── drums/
├── ficus/
├── hotdog/
├── materials/
├── mic/
└── ship/
```

**Note**: Each scene folder should contain:
- `transforms_train.json` - Training camera poses and metadata
- `transforms_test.json` - Test camera poses and metadata
- `train/` or `images/` - Training images
- `test/` - Test images (optional, some datasets have separate test folders)

### 3. Explore the Data

Start with the data exploration notebook to understand the dataset:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This notebook will:
- Load and inspect the dataset
- Visualize camera poses in 3D
- Analyze image statistics
- Understand ray generation

### 4. Train the Model

```bash
# Train with default configuration
python experiments/train.py --config configs/lego_config.yaml

# Resume from checkpoint
python experiments/train.py --config configs/lego_config.yaml --resume outputs/checkpoints/checkpoint_step_100000.pth

# Train on CPU (slower, for testing)
python experiments/train.py --config configs/lego_config.yaml --device cpu
```

**Training Tips:**
- Training takes ~2-6 hours on a good GPU (RTX 3080/4090)
- Monitor progress with TensorBoard: `tensorboard --logdir outputs/logs`
- Checkpoints are saved every 10,000 steps by default
- Use `image_scale: 0.5` in config for faster training (half resolution)

### 5. Visualize Results

Use the results visualization notebook:

```bash
jupyter notebook notebooks/03_results_visualization.ipynb
```

This will:
- Load trained model
- Render novel views
- Create 360° rotation videos
- Compute evaluation metrics (PSNR, SSIM)

## 📊 Understanding the Outputs

### What NeRF Produces

1. **Novel View Synthesis**: Render images from camera angles not seen during training
   - Input: 100 training images
   - Output: Any new camera angle you want

2. **360° Videos**: Smooth rotation around the scene
   - Output: MP4 video showing the scene from all angles

3. **Depth Maps**: Understand 3D structure
   - Output: Grayscale images showing distance

4. **3D Mesh** (optional): Extract traditional 3D model
   - Output: .obj or .ply file for Blender/Unity

### Example Results

- **Before Training**: 100 photos of a Lego scene from different angles
- **After Training**: Can render the scene from any angle, create smooth 360° videos

## 🔬 Technical Details

### NeRF Architecture

- **Positional Encoding**: Sinusoidal encoding for 3D coordinates and viewing directions
- **MLP Network**: 8-layer MLP predicts volume density (σ) and RGB color
- **Volume Rendering**: Integrates density and color along rays using the volume rendering equation

### Training Process

1. **Ray Sampling**: Randomly sample rays from training images
2. **Point Sampling**: Sample 3D points along each ray (stratified sampling)
3. **Network Query**: Predict density and color for each point
4. **Volume Rendering**: Integrate predictions into final pixel color
5. **Loss Computation**: MSE loss between rendered and ground truth pixels
6. **Hierarchical Sampling**: Fine network focuses on important regions

### Key Optimizations

- **Mixed Precision Training**: Faster training with FP16
- **Chunked Rendering**: Process rays in chunks to avoid OOM
- **Efficient Ray Sampling**: Only sample necessary rays per batch
- **Hierarchical Sampling**: Coarse network guides fine network sampling

## 📈 Performance Metrics

The model is evaluated using:
- **PSNR** (Peak Signal-to-Noise Ratio): Image quality metric
- **SSIM** (Structural Similarity Index): Perceptual quality metric
- **LPIPS** (optional): Learned perceptual metric

Expected results on NeRF synthetic dataset:
- PSNR: ~30-35 dB (depending on scene)
- Training time: 2-6 hours on RTX 3080/4090
- Inference: ~10-30 seconds per image (depending on resolution)

## 🛠️ Configuration

Edit `configs/lego_config.yaml` to customize training:

```yaml
model:
  num_frequencies_xyz: 10    # Position encoding frequencies
  hidden_dim: 256            # MLP hidden dimension
  num_layers: 8              # Number of MLP layers

training:
  lr: 5e-4                  # Learning rate
  num_rays: 1024            # Rays per batch
  num_samples_coarse: 64    # Coarse samples per ray
  num_samples_fine: 128     # Fine samples per ray
  num_epochs: 100           # Training epochs

data:
  image_scale: 1.0          # Image resolution (1.0 = full, 0.5 = half)
  white_bg: true            # White background for transparent images
```

## 📚 Data Analysis Workflow

1. **01_data_exploration.ipynb**: Understand your dataset
   - Visualize camera positions
   - Analyze image statistics
   - Understand ray generation

2. **02_training_analysis.ipynb**: Monitor training
   - Plot loss curves
   - Analyze learning rate schedule
   - Compare coarse vs fine losses

3. **03_results_visualization.ipynb**: Evaluate results
   - Render novel views
   - Create 360° videos
   - Compute metrics

## 🎓 Learning Resources

### Understanding NeRF

- **Original Paper**: [NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934)
- **Key Concepts**:
  - Volume rendering equation
  - Positional encoding
  - Hierarchical volume sampling

### PyTorch Best Practices Demonstrated

- Custom DataLoader implementation
- Mixed precision training
- Efficient tensor operations
- Memory-efficient rendering
- Configuration management
- Comprehensive logging

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors

- Reduce `num_rays` in config (e.g., 512 instead of 1024)
- Reduce `num_samples_coarse` and `num_samples_fine`
- Use `image_scale: 0.5` for half resolution
- Reduce `chunk_size` in rendering

### Slow Training

- Use GPU (CPU is 10-100x slower)
- Reduce image resolution with `image_scale: 0.5`
- Reduce number of samples per ray
- Use mixed precision training (enabled by default)

### Poor Results

- Ensure camera poses are correct
- Check that images are properly normalized
- Increase training epochs
- Adjust learning rate
- Verify data quality in exploration notebook

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Original NeRF paper by Mildenhall et al.
- NeRF synthetic dataset creators
- PyTorch community for excellent documentation

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ using PyTorch**
