"""Shape utilities and dummy input generation"""

import torch
from typing import Dict, Tuple, Optional
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_latent_shapes(
    num_frames: int,
    height: int,
    width: int,
    vae_scale_factor: int = 16,  # Wan2.2 VAE uses 16×16 spatial compression
    temporal_compression: int = 4,  # Wan2.2 VAE uses 4× temporal compression
    latent_channels: int = 16  # AutoencoderKLWan uses 16 channels (not 4)
) -> Dict[str, Tuple[int, ...]]:
    """
    Calculate latent space dimensions given video parameters.
    
    Args:
        num_frames: Number of video frames
        height: Video height in pixels
        width: Video width in pixels
        vae_scale_factor: VAE spatial downsampling factor (16 for Wan2.2)
        temporal_compression: VAE temporal compression factor (4 for Wan2.2)
        latent_channels: Number of latent channels (16 for AutoencoderKLWan)
        
    Returns:
        Dictionary with shape information
    """
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    latent_frames = num_frames // temporal_compression
    
    shapes = {
        "video_shape": (num_frames, 3, height, width),
        "latent_shape": (latent_channels, latent_frames, latent_height, latent_width),
        "vae_scale_factor": vae_scale_factor,
        "temporal_compression": temporal_compression,
        "latent_channels": latent_channels,
    }
    
    logger.debug(f"Video shape: {shapes['video_shape']}")
    logger.debug(f"Latent shape: {shapes['latent_shape']}")
    
    return shapes


def create_dummy_inputs(
    component_name: str,
    batch_size: int = 1,
    num_frames: int = 81,  # Wan2.2 default
    height: int = 720,
    width: int = 1280,
    latent_channels: int = 16,  # AutoencoderKLWan uses 16 channels
    seq_length: int = 256,  # Wan uses longer sequences
    hidden_size: int = 2048,  # Adjust based on actual text encoder
    vae_scale_factor: int = 16,  # 16×16 spatial compression
    temporal_compression: int = 4,  # 4× temporal compression
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16  # Wan2.2 uses bfloat16
) -> Dict[str, torch.Tensor]:
    """
    Create dummy inputs for ONNX export and testing.
    
    Args:
        component_name: Component name ('dit', 'transformer', 'vae_encoder', 'vae_decoder')
        batch_size: Batch size
        num_frames: Number of video frames (up to 81 for Wan2.2)
        height: Video height in pixels (720 or 1280)
        width: Video width in pixels (1280 or 720)
        latent_channels: Number of latent channels (16 for AutoencoderKLWan)
        seq_length: Text encoder sequence length
        hidden_size: Text encoder hidden size
        vae_scale_factor: VAE spatial downsampling factor (16)
        temporal_compression: VAE temporal compression (4)
        device: Target device
        dtype: Data type for tensors (bfloat16 for Wan2.2)
        
    Returns:
        Dictionary of dummy input tensors
    """
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    latent_frames = num_frames // temporal_compression
    
    if component_name in ["dit", "transformer", "unet"]:
        # Wan Transformer inputs: noisy latents, timestep, encoder hidden states
        inputs = {
            "sample": torch.randn(
                batch_size, latent_channels, latent_frames, latent_height, latent_width,
                device=device, dtype=dtype
            ),
            "timestep": torch.tensor([999], device=device, dtype=torch.long),
            "encoder_hidden_states": torch.randn(
                batch_size, seq_length, hidden_size,
                device=device, dtype=dtype
            ),
        }
        
        logger.debug(f"DiT/UNet dummy inputs:")
        logger.debug(f"  sample: {inputs['sample'].shape}")
        logger.debug(f"  timestep: {inputs['timestep'].shape}")
        logger.debug(f"  encoder_hidden_states: {inputs['encoder_hidden_states'].shape}")
        
    elif component_name == "vae_encoder":
        # VAE encoder: video frames -> latents
        inputs = {
            "sample": torch.randn(
                batch_size, 3, num_frames, height, width,
                device=device, dtype=dtype
            ),
        }
        
        logger.debug(f"VAE Encoder dummy inputs:")
        logger.debug(f"  sample: {inputs['sample'].shape}")
        
    elif component_name in ["vae_decoder", "vae"]:
        # AutoencoderKLWan decoder: latents -> video frames
        inputs = {
            "latent_sample": torch.randn(
                batch_size, latent_channels, latent_frames, latent_height, latent_width,
                device=device, dtype=dtype
            ),
        }
        
        logger.debug(f"VAE Decoder dummy inputs:")
        logger.debug(f"  latent_sample: {inputs['latent_sample'].shape}")
        
    else:
        raise ValueError(f"Unknown component name: {component_name}")
    
    return inputs


def get_dynamic_axes(component_name: str) -> Dict[str, Dict[int, str]]:
    """
    Get dynamic axes configuration for ONNX export.
    
    Args:
        component_name: Component name
        
    Returns:
        Dictionary mapping input names to dynamic axis specifications
    """
    if component_name in ["dit", "transformer", "unet"]:
        return {
            "sample": {0: "batch", 2: "frames", 3: "height", 4: "width"},
            "encoder_hidden_states": {0: "batch", 1: "seq_len"},
            # timestep is typically fixed
        }
    
    elif component_name == "vae_encoder":
        return {
            "sample": {0: "batch", 2: "frames", 3: "height", 4: "width"},
        }
    
    elif component_name in ["vae_decoder", "vae"]:
        return {
            "latent_sample": {0: "batch", 2: "frames", 3: "height", 4: "width"},
        }
    
    else:
        raise ValueError(f"Unknown component name: {component_name}")

