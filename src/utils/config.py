"""Configuration management utilities"""

import os
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Use default config in project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Get torch device, with fallback logic.
    
    Args:
        device_str: Device string (e.g., 'cuda:0', 'cpu')
        
    Returns:
        torch.device object
    """
    if device_str is None:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    device = torch.device(device_str)
    
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"Warning: CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    
    return device


def ensure_directory(path: str) -> Path:
    """
    Ensure directory exists, create if needed.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert string to torch dtype.
    
    Args:
        dtype_str: String representation (e.g., 'float16', 'float32')
        
    Returns:
        torch.dtype
    """
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    
    return dtype_map.get(dtype_str.lower(), torch.float32)

