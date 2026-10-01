"""
Neural Radiance Field (NeRF) Model Architecture

Implements the MLP-based NeRF model that predicts volume density and RGB color
for any 3D point and viewing direction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PositionalEncoding(nn.Module):
    """
    Positional encoding for input coordinates.
    
    Encodes 3D positions and viewing directions using sinusoidal functions
    to help the network learn high-frequency details.
    
    Args:
        input_dim: Dimension of input (3 for xyz, 3 for viewing direction)
        num_frequencies: Number of frequency bands for encoding
        include_input: Whether to include original input in encoding
    """
    
    def __init__(self, input_dim: int, num_frequencies: int = 10, include_input: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        
        # Create frequency bands: 2^0, 2^1, ..., 2^(L-1)
        self.frequencies = 2.0 ** torch.arange(0, num_frequencies, dtype=torch.float32)
        
        # Output dimension: input_dim * (2 * num_frequencies + include_input)
        self.output_dim = input_dim * (2 * num_frequencies + (1 if include_input else 0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to input.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Encoded tensor of shape (..., output_dim)
        """
        # Expand frequencies to match input device
        frequencies = self.frequencies.to(x.device)
        
        # Reshape for broadcasting: (num_frequencies, 1)
        frequencies = frequencies.view(-1, 1)
        
        # Compute sin and cos encodings: (..., num_frequencies, input_dim)
        x_expanded = x.unsqueeze(-2)  # (..., 1, input_dim)
        x_scaled = x_expanded * frequencies  # (..., num_frequencies, input_dim)
        
        encoded = [torch.sin(x_scaled), torch.cos(x_scaled)]
        
        # Flatten: (..., num_frequencies * 2 * input_dim)
        encoded = torch.cat(encoded, dim=-2)
        encoded = encoded.flatten(start_dim=-2)
        
        # Optionally include original input
        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)
        
        return encoded


class NeRF(nn.Module):
    """
    Neural Radiance Field (NeRF) Model.
    
    Predicts volume density (sigma) and RGB color for any 3D point (x, y, z)
    and viewing direction (theta, phi).
    
    Architecture:
        - Positional encoding for coordinates and directions
        - 8-layer MLP for density prediction
        - 1-layer MLP for color prediction (conditioned on direction)
    """
    
    def __init__(
        self,
        num_frequencies_xyz: int = 10,
        num_frequencies_dir: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_connection_layer: int = 4,
    ):
        """
        Initialize NeRF model.
        
        Args:
            num_frequencies_xyz: Number of frequency bands for position encoding
            num_frequencies_dir: Number of frequency bands for direction encoding
            hidden_dim: Hidden dimension of MLP layers
            num_layers: Number of layers in the density MLP
            skip_connection_layer: Layer index to add skip connection
        """
        super().__init__()

        if not 0 < skip_connection_layer < num_layers:
            raise ValueError(
                f"skip_connection_layer must be in [1, {num_layers - 1}], got {skip_connection_layer}"
            )
        self.num_layers = num_layers
        # Index (0-based) of the Linear layer whose input is [h, gamma(x)].
        self.skip_connection_layer = skip_connection_layer

        # Positional encodings
        self.pos_encoding_xyz = PositionalEncoding(3, num_frequencies_xyz, include_input=True)
        self.pos_encoding_dir = PositionalEncoding(3, num_frequencies_dir, include_input=True)
        
        input_dim_xyz = self.pos_encoding_xyz.output_dim
        input_dim_dir = self.pos_encoding_dir.output_dim
        
        # Density MLP (predicts sigma)
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim_xyz, hidden_dim))
            elif i == skip_connection_layer:
                # Skip connection: concatenate input to this layer
                layers.append(nn.Linear(input_dim_xyz + hidden_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        self.density_layers = nn.ModuleList(layers)
        
        # Density output (sigma)
        self.density_head = nn.Linear(hidden_dim, 1)
        
        # Feature vector for color prediction
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)
        
        # Color MLP (predicts RGB, conditioned on direction)
        self.color_layers = nn.Sequential(
            nn.Linear(hidden_dim + input_dim_dir, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),  # RGB output
            nn.Sigmoid()  # Colors in [0, 1]
        )
    
    def forward(
        self,
        xyz: torch.Tensor,
        view_dir: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through NeRF model.
        
        Args:
            xyz: 3D coordinates of shape (..., 3)
            view_dir: Viewing directions (normalized) of shape (..., 3)
            
        Returns:
            Tuple of (density, rgb):
                - density: Volume density of shape (..., 1)
                - rgb: RGB color of shape (..., 3)
        """
        # Encode positions and directions
        xyz_encoded = self.pos_encoding_xyz(xyz)
        dir_encoded = self.pos_encoding_dir(view_dir)
        
        # Pass through density MLP
        # density_layers alternates [Linear_0, ReLU, Linear_1, ReLU, ...], so the
        # Linear layer with index k lives at position 2 * k. The skip connection
        # concatenates the encoded input only in front of that Linear layer
        # (not in front of the ReLU that follows it).
        skip_position = 2 * self.skip_connection_layer
        x = xyz_encoded
        for i, layer in enumerate(self.density_layers):
            if i == skip_position:
                x = torch.cat([x, xyz_encoded], dim=-1)
            x = layer(x)
        
        # Predict density (sigma)
        density = self.density_head(x)
        density = F.relu(density)  # Ensure non-negative
        
        # Get feature vector for color prediction
        features = self.feature_head(x)
        
        # Concatenate features with encoded direction
        color_input = torch.cat([features, dir_encoded], dim=-1)
        
        # Predict RGB color
        rgb = self.color_layers(color_input)
        
        return density, rgb
