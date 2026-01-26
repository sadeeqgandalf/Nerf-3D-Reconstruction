"""
Configuration management for NeRF training.

Supports loading from YAML files with validation.
"""

import yaml
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """NeRF model configuration."""
    num_frequencies_xyz: int = 10
    num_frequencies_dir: int = 4
    hidden_dim: int = 256
    num_layers: int = 8
    skip_connection_layer: int = 4


@dataclass
class TrainingConfig:
    """Training configuration."""
    lr: float = 5e-4
    lr_decay: float = 0.1
    lr_decay_steps: int = 250000
    num_epochs: int = 100
    num_rays: int = 1024
    num_samples_coarse: int = 64
    num_samples_fine: int = 128
    batch_size: int = 1
    white_bg: bool = False
    use_mixed_precision: bool = True
    save_every: int = 10000


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: str = './data'
    scene_name: str = 'lego'
    image_scale: float = 1.0
    white_bg: bool = False


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    device: str = 'cuda'
    log_dir: str = './outputs/logs'
    checkpoint_dir: str = './outputs/checkpoints'
    output_dir: str = './outputs/renders'
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            device=config_dict.get('device', 'cuda'),
            log_dir=config_dict.get('log_dir', './outputs/logs'),
            checkpoint_dir=config_dict.get('checkpoint_dir', './outputs/checkpoints'),
            output_dir=config_dict.get('output_dir', './outputs/renders'),
        )


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object
    """
    return Config.from_yaml(config_path)
