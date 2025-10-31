"""Shape utilities and dummy input generation"""

import torch
from typing import Dict, Tuple, Optional, Any
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_latent_shapes(
    num_frames: int,
    height: int,
    width: int,
    vae_scale_factor: int = 8,  # WAN 2.2 uses 8× spatial compression (spatial_scale = 8)
    temporal_compression: int = 4,  # WAN 2.2 uses 4× temporal compression (temporal_scale = 4)
    latent_channels: int = 16  # AutoencoderKLWan uses 16 channels (not 4)
) -> Dict[str, Tuple[int, ...]]:
    """
    Calculate latent space dimensions given video parameters.
    
    Args:
        num_frames: Number of video frames
        height: Video height in pixels
        width: Video width in pixels
        vae_scale_factor: VAE spatial downsampling factor (8 for WAN 2.2, spatial_scale = 8)
        temporal_compression: VAE temporal compression factor (4 for WAN 2.2, temporal_scale = 4)
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
        "text_encoder_hidden_size": 4096,  # WAN2.2 uses 4096 (UMT5EncoderModel)
        "text_encoder_seq_length": 1024,   # WAN2.2 supports long prompts
        "vae_latent_channels": 16,         # Default for AutoencoderKLWan
        "vae_scale_factor": 8,             # Default for WAN 2.2 (spatial_scale = 8)
        "temporal_compression": 4,         # Default for WAN 2.2 (temporal_scale = 4)
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
        # Priority: Look for 4096 input size first (WAN2.2 text embedder)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Look for any linear layer with input size exactly 4096 (WAN2.2 text embedder)
                if module.in_features == 4096 and "time_embedder" not in name:
                    detected_dims["text_encoder_hidden_size"] = module.in_features
                    logger.info(f"Detected WAN2.2 text encoder hidden_size from {name}: {module.in_features}")
                    break
                # Look specifically for the text embedder linear_1 layer (not time embedder)
                elif "text_embedder" in name and "linear_1" in name:
                    # Use the INPUT size as hidden_size only
                    # seq_length should remain at detected default (1024 for WAN2.2)
                    detected_dims["text_encoder_hidden_size"] = module.in_features
                    logger.info(f"Detected text encoder dimensions from {name}: torch.Size([{module.in_features}, {module.out_features}])")
                    logger.info(f"  Input size: {module.in_features}, Output size: {module.out_features}")
                    logger.info(f"Using detected text encoder hidden size: {module.in_features}")
                    logger.info(f"Keeping detected seq_length: {detected_dims.get('text_encoder_seq_length', 1024)}")
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
    seq_length: int = 1024,  # WAN2.2 supports long prompts
    hidden_size: int = 4096,  # WAN2.2 uses 4096 (UMT5EncoderModel)
    vae_scale_factor: int = 8,  # WAN 2.2 uses 8× spatial compression (spatial_scale = 8)
    temporal_compression: int = 4,  # WAN 2.2 uses 4× temporal compression (temporal_scale = 4)
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,  # Pure FP16 for ONNX/TensorRT optimization
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
        vae_scale_factor: VAE spatial downsampling factor (8 for WAN 2.2, spatial_scale = 8)
        temporal_compression: VAE temporal compression (4 for WAN 2.2, temporal_scale = 4)
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
    
    if component_name in ["dit", "transformer", "transformer_2", "unet"]:
        # Wan Transformer inputs: noisy latents, timestep, encoder hidden states
        # 
        # CRITICAL: timestep MUST be FP32 for scheduler math stability:
        # - Noise scaling calculations require full precision
        # - Variance computation needs FP32 to avoid rounding errors
        # - Sigma lookups and fractional timestep interpolation are sensitive
        # - FP16 timestep can cause NaNs, unstable noise schedule, or TensorRT failures
        # - Shape must be (batch,) for ONNX/TensorRT compatibility
        inputs = {
            "sample": torch.randn(
                batch_size, latent_channels, latent_frames, latent_height, latent_width,
                device=device, dtype=dtype
            ),
            "timestep": torch.zeros(batch_size, device=device, dtype=torch.float32),  # CRITICAL: FP32 ONLY - batch-dynamic shape
            "encoder_hidden_states": torch.randn(
                batch_size, seq_length, hidden_size,
                device=device, dtype=dtype
            ),
        }
        
        logger.debug(f"DiT/UNet dummy inputs:")
        logger.debug(f"  sample: {inputs['sample'].shape} ({inputs['sample'].dtype})")
        logger.debug(f"  timestep: {inputs['timestep'].shape} ({inputs['timestep'].dtype})")
        logger.debug(f"  encoder_hidden_states: {inputs['encoder_hidden_states'].shape} ({inputs['encoder_hidden_states'].dtype})")
        
        # Validate timestep properties for scheduler compatibility
        # CRITICAL: Timestep MUST be FP32 - scheduler math breaks with FP16/BF16
        assert inputs['timestep'].dtype == torch.float32, f"Timestep must be FP32, got {inputs['timestep'].dtype}"
        assert len(inputs['timestep'].shape) == 1, f"Timestep must be 1D, got shape {inputs['timestep'].shape}"
        logger.info(f"✓ Timestep validation passed: shape={inputs['timestep'].shape}, dtype={inputs['timestep'].dtype}")
        
        # Validate latent channels for WAN2.2 compatibility
        assert inputs["sample"].shape[1] == 16, f"Latent channel mismatch: WAN2.2 requires 16, got {inputs['sample'].shape[1]}"
        logger.info(f"✓ Latent channels validation passed: {inputs['sample'].shape[1]} channels")
        
        # Validate spatial dimensions are divisible by 8 (VAE spatial_scale = 8)
        latent_h, latent_w = inputs["sample"].shape[3], inputs["sample"].shape[4]
        assert latent_h % 8 == 0, f"Latent height must be divisible by 8, got {latent_h}"
        assert latent_w % 8 == 0, f"Latent width must be divisible by 8, got {latent_w}"
        logger.info(f"✓ Spatial dimensions validation passed: {latent_h}×{latent_w}")
        
        # Validate temporal dimension is divisible by 4 (temporal compression)
        latent_frames = inputs["sample"].shape[2]
        assert latent_frames % 4 == 0, f"Latent frames must be divisible by 4, got {latent_frames}"
        logger.info(f"✓ Temporal dimension validation passed: {latent_frames} frames")
        
    elif component_name == "vae_encoder":
        # VAE encoder: video frames -> latents
        # AutoencoderKLWan expects (batch, frames, channels, height, width)
        # VAE should use fp32 for numerical stability
        inputs = {
            "sample": torch.randn(
                batch_size, num_frames, 3, height, width,
                device=device, dtype=torch.float32
            ),
        }
        
        logger.debug(f"VAE Encoder dummy inputs:")
        logger.debug(f"  sample: {inputs['sample'].shape}")
        
        # Validate VAE encoder input shape
        assert inputs["sample"].shape[2] == 3, f"VAE encoder expects 3 channels, got {inputs['sample'].shape[2]}"
        logger.info(f"✓ VAE encoder input validation passed: {inputs['sample'].shape}")
        
    elif component_name in ["vae_decoder", "vae"]:
        # AutoencoderKLWan: can do both encoding and decoding
        # For full VAE export, we'll use decoder inputs (latents -> frames)
        # VAE should use fp32 for numerical stability
        vae_dtype = torch.float32 if component_name in ["vae_decoder", "vae"] else dtype
        
        if component_name == "vae":
            # For full VAE, we need to determine if it's encoding or decoding
            # Let's use decoder inputs as they're more commonly used in inference
            logger.info("Using VAE decoder inputs for full VAE export (latents -> frames)")
            inputs = {
                "latent_sample": torch.randn(
                    batch_size, latent_channels, latent_frames, latent_height, latent_width,
                    device=device, dtype=vae_dtype
                ),
            }
        else:
            # Pure decoder
            inputs = {
                "latent_sample": torch.randn(
                    batch_size, latent_channels, latent_frames, latent_height, latent_width,
                    device=device, dtype=vae_dtype
                ),
            }
        
        logger.debug(f"VAE {'Decoder' if component_name == 'vae_decoder' else 'Full'} dummy inputs:")
        logger.debug(f"  latent_sample: {inputs['latent_sample'].shape}")
        
        # Validate VAE decoder input shape
        assert inputs["latent_sample"].shape[1] == latent_channels, f"VAE decoder expects {latent_channels} latent channels, got {inputs['latent_sample'].shape[1]}"
        logger.info(f"✓ VAE {'decoder' if component_name == 'vae_decoder' else 'full'} input validation passed: {inputs['latent_sample'].shape}")
        
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
    if component_name in ["dit", "transformer", "transformer_2", "unet"]:
        return {
            "sample": {0: "batch", 2: "frames", 3: "height", 4: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "seq_len"},
            # Note: output dynamic axes handled explicitly in export script
        }
    
    elif component_name == "vae_encoder":
        return {
            "sample": {0: "batch", 1: "frames", 3: "height", 4: "width"},
        }
    
    elif component_name in ["vae_decoder", "vae"]:
        return {
            "latent_sample": {0: "batch", 1: "channels", 2: "frames", 3: "height", 4: "width"},
        }
    
    else:
        raise ValueError(f"Unknown component name: {component_name}")


def get_tensorrt_shape_ranges() -> Dict[str, Dict[str, int]]:
    """
    Get TensorRT-optimized min/max shape ranges for dynamic axes.
    
    Returns:
        Dictionary with min/max ranges for each dynamic dimension
    """
    return {
        "batch": {"min": 1, "max": 4},      # Batch size range
        "frames": {"min": 16, "max": 80},    # Frame count range (1-5s at 16 FPS)
        "height": {"min": 512, "max": 720},  # Height range (divisible by 16)
        "width": {"min": 512, "max": 1280}, # Width range (divisible by 16)
        "seq_len": {"min": 77, "max": 1024}, # Sequence length range
    }


def calculate_frames_from_duration(duration_seconds: float, fps: int = 16) -> int:
    """
    Calculate number of frames from duration and FPS, ensuring divisibility by temporal compression.
    
    Args:
        duration_seconds: Video duration in seconds
        fps: Frames per second (default 16 for WAN2.2)
        
    Returns:
        Number of frames (divisible by 4 for temporal compression)
    """
    raw_frames = int(duration_seconds * fps)
    
    # Ensure frames are divisible by 4 (temporal compression factor)
    # Round up to nearest multiple of 4
    frames = ((raw_frames + 3) // 4) * 4
    
    return frames


def get_optimized_dummy_inputs(
    component_name: str,
    batch_size: int = 1,
    num_frames: int = 81,
    height: int = 720,
    width: int = 1280,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    model: Optional[torch.nn.Module] = None
) -> Dict[str, torch.Tensor]:
    """
    Create TensorRT-optimized dummy inputs with proper shape validation.
    
    Args:
        component_name: Component name
        batch_size: Batch size
        num_frames: Number of video frames
        height: Video height in pixels
        width: Video width in pixels
        device: Target device
        dtype: Data type for tensors
        model: Optional model for automatic dimension detection
        
    Returns:
        Dictionary of dummy input tensors optimized for TensorRT
    """
    # Validate input dimensions for TensorRT compatibility
    # WAN 2.2 uses spatial_scale = 8, so height/width should be divisible by 8
    assert height % 8 == 0, f"Height must be divisible by 8, got {height}"
    assert width % 8 == 0, f"Width must be divisible by 8, got {width}"
    assert num_frames % 4 == 0, f"Frames must be divisible by 4, got {num_frames}"
    
    # Get TensorRT shape ranges
    shape_ranges = get_tensorrt_shape_ranges()
    
    # Validate against TensorRT ranges
    assert shape_ranges["height"]["min"] <= height <= shape_ranges["height"]["max"], \
        f"Height {height} outside TensorRT range [{shape_ranges['height']['min']}, {shape_ranges['height']['max']}]"
    assert shape_ranges["width"]["min"] <= width <= shape_ranges["width"]["max"], \
        f"Width {width} outside TensorRT range [{shape_ranges['width']['min']}, {shape_ranges['width']['max']}]"
    assert shape_ranges["frames"]["min"] <= num_frames <= shape_ranges["frames"]["max"], \
        f"Frames {num_frames} outside TensorRT range [{shape_ranges['frames']['min']}, {shape_ranges['frames']['max']}]"
    
    logger = get_logger(__name__)
    logger.info(f"✓ TensorRT shape validation passed: {height}×{width}, {num_frames} frames")
    
    # Create standard dummy inputs
    return create_dummy_inputs(
        component_name=component_name,
        batch_size=batch_size,
        num_frames=num_frames,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
        model=model
    )


def get_duration_based_dummy_inputs(
    component_name: str,
    duration_seconds: float = 3.0,
    fps: int = 16,
    height: int = 512,
    width: int = 512,
    batch_size: int = 1,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    model: Optional[torch.nn.Module] = None
) -> Dict[str, torch.Tensor]:
    """
    Create dummy inputs based on video duration, automatically calculating frames.
    
    Args:
        component_name: Component name
        duration_seconds: Video duration in seconds (1-5s recommended)
        fps: Frames per second (default 16 for WAN2.2)
        height: Video height in pixels (default 512)
        width: Video width in pixels (default 512)
        batch_size: Batch size
        device: Target device
        dtype: Data type for tensors
        model: Optional model for automatic dimension detection
        
    Returns:
        Dictionary of dummy input tensors optimized for TensorRT
        
    Examples:
        # 3-second video at 16 FPS → 48 frames
        inputs = get_duration_based_dummy_inputs("transformer", duration_seconds=3.0)
        
        # 1-second video at 16 FPS → 16 frames  
        inputs = get_duration_based_dummy_inputs("transformer", duration_seconds=1.0)
    """
    logger = get_logger(__name__)
    
    # Calculate frames from duration
    num_frames = calculate_frames_from_duration(duration_seconds, fps)
    
    logger.info(f"📹 Duration: {duration_seconds}s @ {fps} FPS → {num_frames} frames")
    logger.info(f"📐 Resolution: {height}×{width}")
    
    # Validate duration is within reasonable range
    if duration_seconds < 1.0 or duration_seconds > 5.0:
        logger.warning(f"Duration {duration_seconds}s outside recommended range [1-5s]")
    
    # Use optimized dummy inputs with calculated frames
    return get_optimized_dummy_inputs(
        component_name=component_name,
        batch_size=batch_size,
        num_frames=num_frames,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
        model=model
    )

