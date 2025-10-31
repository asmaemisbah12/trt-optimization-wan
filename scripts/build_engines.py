#!/usr/bin/env python3
"""
Build TensorRT engines from ONNX models with multi-profile support.

Usage:
    python scripts/build_engines.py --onnx_path outputs/onnx/dit_fp16.onnx --precision fp16
    python scripts/build_engines.py --onnx_path outputs/onnx/vae_fp32.onnx --precision fp32
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, ensure_directory
from src.utils.logging import setup_logging
from src.tensorrt.engine_builder import build_engine


def parse_profile_config(
    config: dict,
    component: str,
    video_height: int = 512,
    video_width: int = 512,
    vae_scale_factor: int = 8,  # WAN 2.2 uses spatial_scale = 8
    temporal_compression: int = 4,  # WAN 2.2 uses temporal_scale = 4
    seq_length: int = 512,  # ONNX export uses 512 (hard-coded), not 1024
    hidden_size: int = 4096
) -> Optional[Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]]:
    """
    Parse optimization profile from config.
    
    Args:
        config: Configuration dictionary
        component: Component name (e.g., 'dit_fp16', 'vae_fp32', 'text_encoder_fp16')
        video_height: Video height in pixels (default: 512)
        video_width: Video width in pixels (default: 512)
        vae_scale_factor: VAE spatial downsampling factor (default: 8 for WAN 2.2, spatial_scale = 8)
        temporal_compression: VAE temporal compression factor (default: 4 for WAN 2.2, temporal_scale = 4)
        seq_length: Text encoder sequence length (default: 512, matches ONNX export)
        hidden_size: Text encoder hidden size (default: 4096)
        
    Returns:
        Profile shapes dictionary or None if not found
    """
    profiles = config.get("tensorrt", {}).get("profiles", {})
    
    if component not in profiles:
        return None
    
    profile_config = profiles[component]
    
    # WAN2.2 defaults (aligned with WanTransformer3DModel config)
    # These values come from the model config JSON:
    # - in_channels: 16, out_channels: 16 → latent_channels = 16
    # - rope_max_seq_len: 1024 (model supports up to 1024, but ONNX export uses 512)
    # - text_dim: 4096 → hidden_size = 4096
    latent_channels = 16  # AutoencoderKLWan uses 16 channels (from WanTransformer3DModel.in_channels)
    # seq_length and hidden_size are now passed as parameters
    
    # Profile configuration from config file
    # NOTE: ONNX models are exported with batch=1 (hard-coded in attention reshapes)
    # CFG is implemented via sequential passes instead of batch=2
    batch_config = profile_config.get("batch_size", {"min": 1, "opt": 1, "max": 1})
    frames_config = profile_config.get("frames", {"min": 16, "opt": 48, "max": 81})
    height_config = profile_config.get("height", {"min": 512, "opt": video_height, "max": 1280})
    width_config = profile_config.get("width", {"min": 512, "opt": video_width, "max": 1280})
    
    # Calculate latent dimensions from video dimensions (aligned with shapes.py)
    # Latent dimensions are calculated as: height/width // vae_scale_factor, frames // temporal_compression
    latent_height_min = height_config["min"] // vae_scale_factor
    latent_height_opt = height_config["opt"] // vae_scale_factor
    latent_height_max = height_config["max"] // vae_scale_factor
    
    latent_width_min = width_config["min"] // vae_scale_factor
    latent_width_opt = width_config["opt"] // vae_scale_factor
    latent_width_max = width_config["max"] // vae_scale_factor
    
    # Frames are compressed temporally (video frames -> latent frames)
    # Use ceiling division to match create_default_profile() and avoid truncation
    latent_frames_min = max(1, (frames_config["min"] + temporal_compression - 1) // temporal_compression)
    latent_frames_opt = max(1, (frames_config["opt"] + temporal_compression - 1) // temporal_compression)
    latent_frames_max = max(1, (frames_config["max"] + temporal_compression - 1) // temporal_compression)
    
    # Build profile shapes based on component type
    profile_shapes = {}
    
    if "dit" in component or "transformer" in component or "transformer_2" in component or "unet" in component:
        # Wan Transformer (MoE): sample, timestep, encoder_hidden_states
        # CRITICAL: WAN 2.2 parameters
        # - spatial_scale = 8 (latent is 8x smaller spatially)
        # - temporal_scale = 4 (latent is 4x smaller temporally)
        # - latent_channels = 16 (not 4!)
        # Shape: [batch, channels, frames, height, width]
        profile_shapes["sample"] = (
            (batch_config["min"], latent_channels, latent_frames_min, latent_height_min, latent_width_min),
            (batch_config["opt"], latent_channels, latent_frames_opt, latent_height_opt, latent_width_opt),
            (batch_config["max"], latent_channels, latent_frames_max, latent_height_max, latent_width_max),
        )
        
        # Timestep input: CRITICAL - must be FP32 for scheduler math stability
        # Shape: (batch,) - batch-dynamic
        profile_shapes["timestep"] = (
            (batch_config["min"],),
            (batch_config["opt"],),
            (batch_config["max"],),
        )
        
        # Encoder hidden states: [batch, seq_length, hidden_size]
        profile_shapes["encoder_hidden_states"] = (
            (batch_config["min"], seq_length, hidden_size),
            (batch_config["opt"], seq_length, hidden_size),
            (batch_config["max"], seq_length, hidden_size),
        )
    
    elif "vae" in component or "vae_encoder" in component or "vae_decoder" in component:
        # AutoencoderKLWan: latent_sample
        # CRITICAL: WAN 2.2 parameters
        # - spatial_scale = 8 (latent is 8x smaller spatially)
        # - temporal_scale = 4 (latent is 4x smaller temporally)
        # - latent_channels = 16 (not 4!)
        # Shape: [batch, channels, frames, height, width]
        profile_shapes["latent_sample"] = (
            (batch_config["min"], latent_channels, latent_frames_min, latent_height_min, latent_width_min),
            (batch_config["opt"], latent_channels, latent_frames_opt, latent_height_opt, latent_width_opt),
            (batch_config["max"], latent_channels, latent_frames_max, latent_height_max, latent_width_max),
        )
    
    elif "text_encoder" in component:
        # Text encoder profile (UMT5EncoderModel for WAN2.2)
        # NOTE: ONNX export uses seq_len=512 (hard-coded), not 1024
        seq_config = profile_config.get("seq_length", {"min": 77, "opt": 512, "max": 512})
        
        profile_shapes["input_ids"] = (
            (batch_config["min"], seq_config["min"]),
            (batch_config["opt"], seq_config["opt"]),
            (batch_config["max"], seq_config["max"]),
        )
        profile_shapes["attention_mask"] = (
            (batch_config["min"], seq_config["min"]),
            (batch_config["opt"], seq_config["opt"]),
            (batch_config["max"], seq_config["max"]),
        )
    
    return profile_shapes


def create_default_profile(
    component: str,
    min_frames: int = 16,
    opt_frames: int = 48,
    max_frames: int = 81,  # WAN 2.2 default (~5s @ 16 FPS)
    video_height: int = 720,  # WAN 2.2 default 720p
    video_width: int = 1280,  # WAN 2.2 default 720p
    vae_scale_factor: int = 8,  # WAN 2.2 uses spatial_scale = 8
    temporal_compression: int = 4,  # WAN 2.2 uses temporal_scale = 4
    seq_length: int = 512,  # ONNX export uses 512 (hard-coded), not 1024
    hidden_size: int = 4096
) -> Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]:
    """
    Create default optimization profile for components.
    
    Args:
        component: Component name
        min_frames: Minimum video frames
        opt_frames: Optimal video frames  
        max_frames: Maximum video frames
        video_height: Video height in pixels (default: 720 for 720p)
        video_width: Video width in pixels (default: 1280 for 720p)
        vae_scale_factor: VAE spatial downsampling factor (default: 8 for WAN 2.2, spatial_scale = 8)
        temporal_compression: VAE temporal compression factor (default: 4 for WAN 2.2, temporal_scale = 4)
        seq_length: Text encoder sequence length (default: 512, matches ONNX export)
        hidden_size: Text encoder hidden size (default: 4096)
        
    Returns:
        Default profile shapes
    """
    profile_shapes = {}

    if "text_encoder" in component:
        # Text encoder profile (UMT5EncoderModel for WAN2.2)
        # Note: CFG is done via sequential passes, so batch=1 here
        profile_shapes["input_ids"] = (
            (1, 77),      # min: short prompt
            (1, 512),     # opt: medium prompt (matches ONNX export)
            (1, 512),     # max: matches ONNX export seq_len
        )
        profile_shapes["attention_mask"] = (
            (1, 77),
            (1, 512),
            (1, 512),
        )
        return profile_shapes

    # --- everything below here is ONLY for VAE / DiT ---

    # Model config values (from WanTransformer3DModel):
    # - in_channels: 16, out_channels: 16 → latent_channels = 16
    # - text_dim: 4096 → hidden_size = 4096 (passed as param)
    # - rope_max_seq_len: 1024 (model supports up to 1024, but ONNX export uses 512)
    latent_channels = 16  # AutoencoderKLWan uses 16 channels (from WanTransformer3DModel.in_channels)

    latent_height = video_height // vae_scale_factor
    latent_width = video_width // vae_scale_factor

    latent_frames_min = max(1, (min_frames + temporal_compression - 1) // temporal_compression)
    latent_frames_opt = max(1, (opt_frames + temporal_compression - 1) // temporal_compression)
    latent_frames_max = max(1, (max_frames + temporal_compression - 1) // temporal_compression)

    if ("dit" in component or
        "transformer" in component or
        "transformer_2" in component or
        "unet" in component):

        # NOTE: ONNX models are exported with batch=1, seq_len=512 (hard-coded in attention reshapes)
        # CFG is implemented via sequential passes (uncond + cond) instead of batch=2
        # See run_inference.py for the sequential CFG implementation
        profile_shapes["sample"] = (
            (1, latent_channels, latent_frames_min, latent_height, latent_width),
            (1, latent_channels, latent_frames_opt, latent_height, latent_width),
            (1, latent_channels, latent_frames_max, latent_height, latent_width),
        )
        profile_shapes["timestep"] = (
            (1,),  # batch=1 (ONNX was exported with batch=1)
            (1,),
            (1,),
        )
        # ONNX export uses seq_len=512 (hard-coded in attention reshapes), not 1024
        profile_shapes["encoder_hidden_states"] = (
            (1, 512, hidden_size),  # batch=1, seq_len=512 (matches ONNX export)
            (1, 512, hidden_size),
            (1, 512, hidden_size),
        )

        return profile_shapes

    if ("vae" in component or
        "vae_encoder" in component or
        "vae_decoder" in component):

        profile_shapes["latent_sample"] = (
            (1, latent_channels, latent_frames_min, latent_height, latent_width),
            (1, latent_channels, latent_frames_opt, latent_height, latent_width),
            (1, latent_channels, latent_frames_max, latent_height, latent_width),
        )

        return profile_shapes

    # fallback: nothing matched, return empty dict
    return profile_shapes


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engines from ONNX")
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output engine path (default: auto-generate)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="Precision mode"
    )
    parser.add_argument(
        "--workspace_size",
        type=int,
        default=8192,
        help="Max workspace size in MB"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--component",
        type=str,
        default=None,
        help="Component name for profile lookup (auto-detected from filename)"
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=None,
        help="Manual override: minimum frames"
    )
    parser.add_argument(
        "--opt_frames",
        type=int,
        default=None,
        help="Manual override: optimal frames"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Manual override: maximum frames"
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=720,
        help="Video height in pixels (default: 720 for 720p)"
    )
    parser.add_argument(
        "--video_width",
        type=int,
        default=1280,
        help="Video width in pixels (default: 1280 for 720p)"
    )
    parser.add_argument(
        "--vae_scale_factor",
        type=int,
        default=8,
        help="VAE spatial downsampling factor (default: 8 for WAN 2.2, spatial_scale = 8)"
    )
    parser.add_argument(
        "--temporal_compression",
        type=int,
        default=4,
        help="VAE temporal compression factor (default: 4 for Wan2.2)"
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=512,
        help="Text encoder sequence length (default: 512, matches ONNX export. Model supports up to 1024 but ONNX uses 512)"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=4096,
        help="Text encoder hidden size (default: 4096 for WAN2.2, from WanTransformer3DModel.text_dim)"
    )
    
    args = parser.parse_args()
    
    # Track whether user explicitly passed frame overrides (before we mutate args)
    user_overrode_frames = (
        "--min_frames" in sys.argv or
        "--opt_frames" in sys.argv or
        "--max_frames" in sys.argv
    )
    
    # Setup logging
    logger = setup_logging(level="INFO")
    logger.info("Starting TensorRT engine build...")
    logger.info(f"ONNX: {args.onnx_path}")
    logger.info(f"Precision: {args.precision}")
    
    # Validate video dimensions are divisible by vae_scale_factor (skip for text_encoder)
    component_hint = Path(args.onnx_path).stem
    if "text_encoder" not in component_hint:
        if args.video_height % args.vae_scale_factor != 0 or args.video_width % args.vae_scale_factor != 0:
            logger.error(
                f"Video dimensions {args.video_height}×{args.video_width} must be divisible by "
                f"vae_scale_factor {args.vae_scale_factor}."
            )
            return
    
    # Use safe default frames for 1-5s videos at 16 FPS if none provided (skip for text_encoder)
    if "text_encoder" not in component_hint:
        # Set default frames only if user didn't explicitly override them
        # WAN 2.2 default is 81 frames (~5s @ 16 FPS)
        if args.min_frames is None:
            args.min_frames = 16  # 1 second
        if args.opt_frames is None:
            args.opt_frames = 48  # 3 seconds
        if args.max_frames is None:
            args.max_frames = 81  # ~5 seconds (WAN 2.2 default)
        
        # Sanity check: ensure min_frames >= temporal_compression
        if args.min_frames < args.temporal_compression:
            logger.warning(
                f"min_frames ({args.min_frames}) < temporal_compression ({args.temporal_compression}). "
                f"Adjusting min_frames to {args.temporal_compression}."
            )
            args.min_frames = args.temporal_compression
    
    # Load config
    config = load_config(args.config)
    
    # Auto-detect component from filename
    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        logger.error(f"ONNX file not found: {onnx_path}")
        return
    
    component = args.component
    if component is None:
        # Try to detect from filename (e.g., "dit_fp16.onnx" -> "dit_fp16")
        component = onnx_path.stem
        logger.info(f"Auto-detected component: {component}")
    
    # Determine output path
    if args.output_path is None:
        output_dir = Path(config.get("tensorrt", {}).get("output_dir", "outputs/engines"))
        ensure_directory(output_dir)
        args.output_path = output_dir / f"{component}.trt"
    
    # Get optimization profile
    profile_shapes = parse_profile_config(
        config=config,
        component=component,
        video_height=args.video_height,
        video_width=args.video_width,
        vae_scale_factor=args.vae_scale_factor,
        temporal_compression=args.temporal_compression,
        seq_length=args.seq_length,
        hidden_size=args.hidden_size
    )
    
    # If no profile found, create default profile
    if profile_shapes is None:
        logger.info("No profile found in config, creating default profile...")
        # Only log video/vae stuff if this is NOT the text encoder (cos it's meaningless there)
        if "text_encoder" not in component:
            logger.info(f"Using video dimensions: {args.video_height}×{args.video_width}")
            logger.info(
                f"VAE scale factor: {args.vae_scale_factor}, "
                f"temporal compression: {args.temporal_compression}"
            )
        profile_shapes = create_default_profile(
            component=component,
            min_frames=args.min_frames,
            opt_frames=args.opt_frames,
            max_frames=args.max_frames,
            video_height=args.video_height,
            video_width=args.video_width,
            vae_scale_factor=args.vae_scale_factor,
            temporal_compression=args.temporal_compression,
            seq_length=args.seq_length,
            hidden_size=args.hidden_size
        )
    
    # Manual overrides for frames - only if user explicitly passed frame args
    # This preserves profiles from config file unless user explicitly wants to override
    if user_overrode_frames and profile_shapes:
        logger.info(f"User explicitly overrode frames: min={args.min_frames}, opt={args.opt_frames}, max={args.max_frames}")
        logger.info("Recreating profile with user-specified frame ranges...")
        # Recreate the profile with user overrides, preserving other config values
        profile_shapes = create_default_profile(
            component=component,
            min_frames=args.min_frames,
            opt_frames=args.opt_frames or args.min_frames,
            max_frames=args.max_frames or args.min_frames,
            video_height=args.video_height,
            video_width=args.video_width,
            vae_scale_factor=args.vae_scale_factor,
            temporal_compression=args.temporal_compression,
            seq_length=args.seq_length,
            hidden_size=args.hidden_size
        )
    
    logger.info(f"Building engine with profile shapes:")
    if profile_shapes:
        for name, (min_s, opt_s, max_s) in profile_shapes.items():
            logger.info(f"  {name}:")
            logger.info(f"    min: {min_s}")
            logger.info(f"    opt: {opt_s}")
            logger.info(f"    max: {max_s}")
    else:
        logger.warning("No optimization profile found. Using static shapes.")
    
    # Build engine
    try:
        engine_path = build_engine(
            onnx_path=str(args.onnx_path),
            output_path=str(args.output_path),
            precision=args.precision,
            workspace_size=args.workspace_size,
            profile_shapes=profile_shapes,
            strict_types=config.get("tensorrt", {}).get("strict_types", False)  # Allow mixed dtypes (FP32 timestep + FP16 other inputs)
        )
        
        logger.info(f"✓ Engine built successfully!")
        logger.info(f"Saved to: {engine_path}")
        
    except Exception as e:
        logger.error(f"Engine build failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

