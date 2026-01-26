"""
Helper script to download NeRF synthetic dataset.

Note: The dataset is hosted on Google Drive. You'll need to download it manually
or use gdown if you have the file ID.
"""

import os
import sys
from pathlib import Path

def print_download_instructions():
    """Print instructions for downloading the dataset."""
    print("=" * 70)
    print("NeRF Synthetic Dataset Download Instructions")
    print("=" * 70)
    print()
    print("OPTION 1: Using Nerfstudio (Recommended - Easiest)")
    print("-" * 70)
    print("  1. Install nerfstudio:")
    print("     pip install nerfstudio")
    print()
    print("  2. Download the Blender dataset:")
    print("     ns-download-data blender")
    print()
    print("  3. Copy data to project directory:")
    print("     The data will be in ~/.nerfstudio/data/blender/")
    print("     Copy to: data/nerf_synthetic/")
    print()
    print("     On Linux/Mac:")
    print("     cp -r ~/.nerfstudio/data/blender/* data/nerf_synthetic/")
    print()
    print("     On Windows:")
    print("     xcopy /E /I \"%USERPROFILE%\\.nerfstudio\\data\\blender\\*\" \"data\\nerf_synthetic\\\"")
    print()
    print("OPTION 2: Manual Download")
    print("-" * 70)
    print("  1. Visit the original NeRF dataset repository:")
    print("     https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4")
    print()
    print("  2. Download 'nerf_synthetic.zip' or individual scene folders")
    print()
    print("  3. Extract to: data/nerf_synthetic/")
    print()
    print("Dataset Structure (after setup):")
    print("  data/nerf_synthetic/")
    print("    ├── lego/")
    print("    │   ├── transforms_train.json")
    print("    │   ├── transforms_test.json")
    print("    │   └── train/ (or images/)")
    print("    ├── chair/")
    print("    ├── drums/")
    print("    ├── ficus/")
    print("    ├── hotdog/")
    print("    ├── materials/")
    print("    ├── mic/")
    print("    └── ship/")
    print()
    print("Available scenes:")
    print("  - lego, chair, drums, ficus, hotdog, materials, mic, ship")
    print()
    print("=" * 70)

if __name__ == '__main__':
    # Create data directory
    data_dir = Path('data/nerf_synthetic')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print_download_instructions()
    
    # Check if dataset exists
    if (data_dir / 'lego').exists():
        print("\n✓ Dataset directory exists!")
        print(f"  Location: {data_dir.absolute()}")
    else:
        print(f"\n✗ Dataset not found at: {data_dir.absolute()}")
        print("  Please download the dataset following the instructions above.")
