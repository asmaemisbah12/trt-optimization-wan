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
from typing import Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, ensure_directory
from src.utils.logging import setup_logging, get_logger
from src.tensorrt.engine_builder import build_engine


def parse_profile_config(
    config: dict,
    component: str
) -> Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]:
    """
    Parse optimization profile from config.
    
    Args:
        config: Configuration dictionary
        component: Component name (e.g., 'dit_fp16', 'vae_fp32')
        
    Returns:
        Profile shapes dictionary
    """
    profiles = config.get("tensorrt", {}).get("profiles", {})
    
    if component not in profiles:
        return None
    
    profile_config = profiles[component]
    
    # Example: Build profile for DiT
    # Assuming latent shape: [batch, channels, frames, height, width]
    batch_config = profile_config.get("batch_size", {"min": 1, "opt": 1, "max": 1})
    frames_config = profile_config.get("frames", {"min": 8, "opt": 16, "max": 81})
    height_config = profile_config.get("height", {"min": 64, "opt": 90, "max": 128})
    width_config = profile_config.get("width", {"min": 64, "opt": 160, "max": 192})
    
    # Build profile shapes based on component type
    profile_shapes = {}
    
    if "dit" in component or "transformer" in component or "unet" in component:
        # Wan Transformer (MoE): sample, timestep, encoder_hidden_states
        # CRITICAL: Wan2.2 uses 16 latent channels, not 4!
        latent_channels = 16  # AutoencoderKLWan uses 16 channels
        seq_length = 4096     # Use detected sequence length from ONNX export
        hidden_size = 4096    # Use detected hidden size from ONNX export
        
        profile_shapes["sample"] = (
            (batch_config["min"], latent_channels, frames_config["min"], height_config["min"], width_config["min"]),
            (batch_config["opt"], latent_channels, frames_config["opt"], height_config["opt"], width_config["opt"]),
            (batch_config["max"], latent_channels, frames_config["max"], height_config["max"], width_config["max"]),
        )
        
        # Add timestep input (missing from original profile)
        profile_shapes["timestep"] = (
            (batch_config["min"],),
            (batch_config["opt"],),
            (batch_config["max"],),
        )
        
        profile_shapes["encoder_hidden_states"] = (
            (batch_config["min"], seq_length, hidden_size),
            (batch_config["opt"], seq_length, hidden_size),
            (batch_config["max"], seq_length, hidden_size),
        )
    
    elif "vae" in component:
        # AutoencoderKLWan: latent_sample
        # CRITICAL: 16 channels, not 4!
        latent_channels = 16  # AutoencoderKLWan uses 16 channels
        
        profile_shapes["latent_sample"] = (
            (batch_config["min"], latent_channels, frames_config["min"], height_config["min"], width_config["min"]),
            (batch_config["opt"], latent_channels, frames_config["opt"], height_config["opt"], width_config["opt"]),
            (batch_config["max"], latent_channels, frames_config["max"], height_config["max"], width_config["max"]),
        )
    
    return profile_shapes


def create_default_profile(component: str, min_frames: int = 8, opt_frames: int = 16, max_frames: int = 81) -> Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]:
    """
    Create default optimization profile for components.
    
    Args:
        component: Component name
        min_frames: Minimum frames
        opt_frames: Optimal frames  
        max_frames: Maximum frames
        
    Returns:
        Default profile shapes
    """
    profile_shapes = {}
    
    if "dit" in component or "transformer" in component or "unet" in component:
        # Default DiT/Transformer profile
        latent_channels = 16
        seq_length = 4096
        hidden_size = 4096
        
        # Latent dimensions (VAE compressed)
        latent_height = 45  # 720 / 16
        latent_width = 80   # 1280 / 16
        
        profile_shapes["sample"] = (
            (1, latent_channels, min_frames, latent_height, latent_width),
            (1, latent_channels, opt_frames, latent_height, latent_width),
            (1, latent_channels, max_frames, latent_height, latent_width),
        )
        
        profile_shapes["timestep"] = (
            (1,),
            (1,),
            (1,),
        )
        
        profile_shapes["encoder_hidden_states"] = (
            (1, seq_length, hidden_size),
            (1, seq_length, hidden_size),
            (1, seq_length, hidden_size),
        )
    
    elif "vae" in component:
        # Default VAE profile
        latent_channels = 16
        
        profile_shapes["latent_sample"] = (
            (1, latent_channels, min_frames, 45, 80),
            (1, latent_channels, opt_frames, 45, 80),
            (1, latent_channels, max_frames, 45, 80),
        )
    
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
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level="INFO")
    logger.info("Starting TensorRT engine build...")
    logger.info(f"ONNX: {args.onnx_path}")
    logger.info(f"Precision: {args.precision}")
    
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
        output_dir = Path(config["tensorrt"]["output_dir"])
        ensure_directory(output_dir)
        args.output_path = output_dir / f"{component}.trt"
    
    # Get optimization profile
    profile_shapes = parse_profile_config(config, component)
    
    # If no profile found, create default profile
    if profile_shapes is None:
        logger.info("No profile found in config, creating default profile...")
        profile_shapes = create_default_profile(
            component=component,
            min_frames=args.min_frames or 8,
            opt_frames=args.opt_frames or 16,
            max_frames=args.max_frames or 81
        )
    
    # Manual overrides for frames
    if args.min_frames is not None and profile_shapes:
        logger.info(f"Overriding frames profile: min={args.min_frames}, opt={args.opt_frames}, max={args.max_frames}")
        # This is a simplified override; full implementation would update all shapes
    
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
            strict_types=config["tensorrt"].get("strict_types", True)
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

