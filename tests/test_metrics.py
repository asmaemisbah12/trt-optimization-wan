"""Tests for quality metrics."""

import pytest
import torch
import numpy as np
from src.benchmark.metrics import compute_psnr, compute_ssim


def test_compute_psnr():
    """Test PSNR computation."""
    # Create identical images
    pred = torch.rand(1, 3, 64, 64)
    target = pred.clone()
    
    psnr = compute_psnr(pred, target)
    
    # PSNR should be very high for identical images
    assert psnr > 50.0


def test_compute_psnr_different():
    """Test PSNR with different images."""
    pred = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    
    psnr = compute_psnr(pred, target)
    
    # PSNR should be finite and positive
    assert 0 < psnr < 100


def test_compute_ssim():
    """Test SSIM computation."""
    # Create identical images
    pred = torch.rand(1, 3, 64, 64)
    target = pred.clone()
    
    ssim = compute_ssim(pred, target)
    
    # SSIM should be close to 1.0 for identical images
    assert ssim > 0.99


def test_compute_ssim_different():
    """Test SSIM with different images."""
    pred = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    
    ssim = compute_ssim(pred, target)
    
    # SSIM should be between 0 and 1
    assert 0 <= ssim <= 1.0

