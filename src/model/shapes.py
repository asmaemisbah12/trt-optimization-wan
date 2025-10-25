"""Shape utilities and dummy input generation"""

import torch
from typing import Dict, Tuple, Optional, Any
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


def detect_model_dimensions(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Automatically detect model input dimensions by inspecting the model structure.
    
    Args:
        model: PyTorch model to inspect
        
    Returns:
        Dictionary with detected dimensions
    """
    detected_dims = {
        "text_encoder_hidden_size": 2048,  # Default fallback
        "text_encoder_seq_length": 256,    # Default fallback
        "vae_latent_channels": 16,         # Default for AutoencoderKLWan
        "vae_scale_factor": 16,            # Default for Wan2.2
        "temporal_compression": 4,         # Default for Wan2.2
    }
    
    try:
        logger.info("Detecting model dimensions...")
        
        # Inspect model structure
        for name, module in model.named_modules():
            # Look for text encoder components
            if hasattr(module, 'config') and hasattr(module.config, 'hidden_size'):
                detected_dims["text_encoder_hidden_size"] = module.config.hidden_size
                logger.info(f"Detected text encoder hidden_size: {module.config.hidden_size}")
            
            if hasattr(module, 'config') and hasattr(module.config, 'max_position_embeddings'):
                detected_dims["text_encoder_seq_length"] = module.config.max_position_embeddings
                logger.info(f"Detected text encoder seq_length: {module.config.max_position_embeddings}")
            
            # Look for VAE components
            if hasattr(module, 'config') and hasattr(module.config, 'latent_channels'):
                detected_dims["vae_latent_channels"] = module.config.latent_channels
                logger.info(f"Detected VAE latent_channels: {module.config.latent_channels}")
            
            if hasattr(module, 'config') and hasattr(module.config, 'scaling_factor'):
                detected_dims["vae_scale_factor"] = module.config.scaling_factor
                logger.info(f"Detected VAE scale_factor: {module.config.scaling_factor}")
            
            # Look for temporal compression in VAE
            if hasattr(module, 'config') and hasattr(module.config, 'temporal_compression'):
                detected_dims["temporal_compression"] = module.config.temporal_compression
                logger.info(f"Detected temporal_compression: {module.config.temporal_compression}")
        
        # Additional inspection for linear layers that might reveal hidden sizes
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Look specifically for the text embedder linear_1 layer (not time embedder)
                if "text_embedder" in name and "linear_1" in name:
                    # Use the INPUT size (4096) as both hidden_size and seq_length
                    # This ensures our dummy input will be [1, 4096, 4096] which matches what the model expects
                    detected_dims["text_encoder_hidden_size"] = module.in_features
                    detected_dims["text_encoder_seq_length"] = module.in_features
                    logger.info(f"Detected text encoder dimensions from {name}: torch.Size([{module.in_features}, {module.out_features}])")
                    logger.info(f"  Input size: {module.in_features}, Output size: {module.out_features}")
                    logger.info(f"Using detected text encoder hidden size: {module.in_features}")
                    logger.info(f"Using detected text encoder seq length: {module.in_features}")
                    break
                # Look for any linear layer with input size exactly 4096 (text embedder input size)
                elif module.in_features == 4096 and "time_embedder" not in name:
                    detected_dims["text_encoder_hidden_size"] = module.in_features
                    logger.info(f"Detected text encoder hidden_size from {name}: {module.in_features}")
                    break
        
        logger.info(f"Final detected dimensions: {detected_dims}")
        
    except Exception as e:
        logger.warning(f"Failed to detect model dimensions: {e}")
        logger.info("Using default dimensions")
    
    return detected_dims


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
    dtype: torch.dtype = torch.bfloat16,  # Wan2.2 uses bfloat16
    model: Optional[torch.nn.Module] = None  # Model for auto-detection
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
        model: Optional model for automatic dimension detection
        
    Returns:
        Dictionary of dummy input tensors
    """
    # Auto-detect dimensions if model is provided
    if model is not None:
        detected_dims = detect_model_dimensions(model)
        # Override with detected values if available
        hidden_size = detected_dims.get("text_encoder_hidden_size", hidden_size)
        seq_length = detected_dims.get("text_encoder_seq_length", seq_length)
        latent_channels = detected_dims.get("vae_latent_channels", latent_channels)
        vae_scale_factor = detected_dims.get("vae_scale_factor", vae_scale_factor)
        temporal_compression = detected_dims.get("temporal_compression", temporal_compression)
        
        logger.info(f"Using auto-detected dimensions:")
        logger.info(f"  hidden_size: {hidden_size}")
        logger.info(f"  seq_length: {seq_length}")
        logger.info(f"  latent_channels: {latent_channels}")
        logger.info(f"  vae_scale_factor: {vae_scale_factor}")
        logger.info(f"  temporal_compression: {temporal_compression}")
    
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

