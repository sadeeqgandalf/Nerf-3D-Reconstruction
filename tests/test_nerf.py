"""
Unit tests for NeRF model.
"""

import torch
import pytest
from src.models import NeRF


def test_nerf_forward():
    """Test NeRF forward pass."""
    model = NeRF()
    
    # Create dummy inputs
    batch_size = 10
    xyz = torch.randn(batch_size, 3)
    view_dir = torch.randn(batch_size, 3)
    view_dir = view_dir / torch.norm(view_dir, dim=-1, keepdim=True)  # Normalize
    
    # Forward pass
    density, rgb = model(xyz, view_dir)
    
    # Check output shapes
    assert density.shape == (batch_size, 1)
    assert rgb.shape == (batch_size, 3)
    
    # Check value ranges
    assert (density >= 0).all(), "Density should be non-negative"
    assert (rgb >= 0).all() and (rgb <= 1).all(), "RGB should be in [0, 1]"


def test_positional_encoding():
    """Test positional encoding."""
    from src.models.nerf import PositionalEncoding
    
    encoding = PositionalEncoding(input_dim=3, num_frequencies=10)
    
    x = torch.randn(5, 3)
    encoded = encoding(x)
    
    # Check output dimension
    expected_dim = 3 * (2 * 10 + 1)  # input_dim * (2 * num_freq + include_input)
    assert encoded.shape == (5, expected_dim)


if __name__ == '__main__':
    pytest.main([__file__])
