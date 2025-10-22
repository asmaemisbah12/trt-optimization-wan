"""Tests for shape utilities."""

import pytest
import torch
from src.model.shapes import get_latent_shapes, create_dummy_inputs, get_dynamic_axes


def test_get_latent_shapes():
    """Test latent shape calculation."""
    shapes = get_latent_shapes(
        num_frames=16,
        height=720,
        width=1280,
        vae_scale_factor=8,
        latent_channels=4
    )
    
    assert shapes["video_shape"] == (16, 3, 720, 1280)
    assert shapes["latent_shape"] == (4, 16, 90, 160)
    assert shapes["vae_scale_factor"] == 8


def test_create_dummy_inputs_dit():
    """Test dummy input creation for DiT."""
    inputs = create_dummy_inputs(
        component_name="dit",
        batch_size=1,
        num_frames=16,
        height=720,
        width=1280,
        device="cpu",
        dtype=torch.float32
    )
    
    assert "sample" in inputs
    assert "timestep" in inputs
    assert "encoder_hidden_states" in inputs
    
    # Check shapes
    assert inputs["sample"].shape == (1, 4, 16, 90, 160)
    assert inputs["encoder_hidden_states"].shape == (1, 77, 768)


def test_create_dummy_inputs_vae():
    """Test dummy input creation for VAE."""
    inputs = create_dummy_inputs(
        component_name="vae",
        batch_size=1,
        num_frames=16,
        height=720,
        width=1280,
        device="cpu",
        dtype=torch.float32
    )
    
    assert "latent_sample" in inputs
    assert inputs["latent_sample"].shape == (1, 4, 16, 90, 160)


def test_get_dynamic_axes():
    """Test dynamic axes configuration."""
    axes = get_dynamic_axes("dit")
    
    assert "sample" in axes
    assert "encoder_hidden_states" in axes
    
    # Check dynamic dimensions
    assert 0 in axes["sample"]  # batch
    assert 2 in axes["sample"]  # frames

