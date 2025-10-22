"""Tests for configuration utilities."""

import pytest
import torch
from pathlib import Path
from src.utils.config import load_config, get_device, get_torch_dtype, ensure_directory


def test_load_config():
    """Test config loading."""
    config = load_config()
    
    assert "model" in config
    assert "tensorrt" in config
    assert "inference" in config


def test_get_device():
    """Test device selection."""
    device = get_device("cpu")
    assert device.type == "cpu"
    
    if torch.cuda.is_available():
        device = get_device("cuda:0")
        assert device.type == "cuda"


def test_get_torch_dtype():
    """Test dtype conversion."""
    assert get_torch_dtype("fp32") == torch.float32
    assert get_torch_dtype("float32") == torch.float32
    assert get_torch_dtype("fp16") == torch.float16
    assert get_torch_dtype("float16") == torch.float16


def test_ensure_directory(tmp_path):
    """Test directory creation."""
    test_dir = tmp_path / "test" / "nested" / "dir"
    
    result = ensure_directory(test_dir)
    
    assert result.exists()
    assert result.is_dir()

