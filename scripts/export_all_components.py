#!/usr/bin/env python3
"""
Comprehensive ONNX export script for WAN2.2 components.

This script exports all WAN2.2 components with the correct precision strategy:
- Transformers: FP16 for speed
- VAE: FP32 for quality
- Timestep: FP32 for scheduler stability

Usage:
    python scripts/export_all_components.py --model_id "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    python scripts/export_all_components.py --components transformer transformer_2 vae
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_logging, get_logger
from src.utils.config import load_config, ensure_directory
from src.model.loader import load_pipeline, get_submodule
from src.model.shapes import create_dummy_inputs, get_dynamic_axes
from src.export.onnx_exporter import export_to_onnx, compare_onnx_outputs


def export_component(
    pipe,
    component_name: str,
    precision: str,
    output_dir: Path,
    config: dict,
    validate: bool = True,
    opset_version: int = 17
):
    """
    Export a single component to ONNX.
    
    Args:
        pipe: Loaded pipeline
        component_name: Component to export
        precision: Precision for export
        output_dir: Output directory
        config: Configuration dict
        validate: Whether to validate outputs
        opset_version: ONNX opset version
    """
    logger = get_logger(__name__)
    
    # Component mapping
    component_map = {
        "dit": "transformer",
        "transformer": "transformer", 
        "transformer_2": "transformer_2",
        "unet": "unet",
        "vae": "vae",
        "vae_encoder": "vae",
        "vae_decoder": "vae",
    }
    
    submodule_name = component_map.get(component_name)
    if submodule_name is None:
        logger.error(f"Unknown component: {component_name}")
        return False
    
    logger.info(f"🔧 Exporting {component_name} ({submodule_name}) with {precision} precision...")
    
    # Get submodule
    model = get_submodule(pipe, submodule_name)
    model.eval()
    
    # Determine dtype based on component and precision
    if component_name in ["vae", "vae_encoder", "vae_decoder"]:
        # VAE always uses FP32 for quality
        dtype = torch.float32
        logger.info("Using FP32 for VAE (quality preservation)")
    else:
        # Transformers use specified precision
        if precision == "fp16":
            dtype = torch.float16
        elif precision == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        logger.info(f"Using {dtype} for {component_name}")
    
    # Create dummy inputs
    logger.info("Creating dummy inputs...")
    dummy_inputs = create_dummy_inputs(
        component_name=component_name,
        num_frames=config.get("num_frames", 81),
        height=config.get("height", 720),
        width=config.get("width", 1280),
        device="cuda:0",
        dtype=dtype,
        model=model
    )
    
    # Get dynamic axes
    dynamic_axes = get_dynamic_axes(component_name)
    
    # Export to ONNX
    output_filename = f"{component_name}_{precision}.onnx"
    output_path = output_dir / output_filename
    
    logger.info(f"Exporting to ONNX: {output_path}")
    logger.info(f"Dynamic axes: {dynamic_axes}")
    
    try:
        export_to_onnx(
            model=model,
            dummy_inputs=dummy_inputs,
            output_path=str(output_path),
            input_names=list(dummy_inputs.keys()),
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False
        )
        
        logger.info(f"✅ {component_name} export successful!")
        
        # Validate if requested
        if validate:
            logger.info(f"🔍 Validating {component_name} ONNX export...")
            match, metrics = compare_onnx_outputs(
                model=model,
                onnx_path=str(output_path),
                dummy_inputs=dummy_inputs,
                input_names=list(dummy_inputs.keys()),
                rtol=1e-3,
                atol=1e-5
            )
            
            if match:
                logger.info(f"✅ {component_name} validation passed!")
            else:
                logger.warning(f"⚠️ {component_name} validation failed: {metrics}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ {component_name} export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Export WAN2.2 components to ONNX")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=["transformer", "transformer_2", "vae"],
        choices=["dit", "transformer", "transformer_2", "unet", "vae", "vae_encoder", "vae_decoder"],
        help="Components to export"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
        help="Precision for transformer components (VAE always uses FP32)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/onnx",
        help="Output directory for ONNX files"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames for dummy input"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Video height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Video width"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate ONNX exports by comparing outputs"
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=17,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--skip_precision_check",
        action="store_true",
        help="Skip precision validation before export"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level="INFO")
    logger.info("🚀 Starting WAN2.2 ONNX export...")
    
    # Load config
    config = load_config(args.config)
    config.update({
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
    })
    
    # Ensure output directory
    output_dir = ensure_directory(args.output_dir)
    
    # Validate precision setup first
    if not args.skip_precision_check:
        logger.info("🔍 Validating precision setup...")
        try:
            from scripts.validate_precision import validate_precision_setup
            if not validate_precision_setup(args.model_id):
                logger.error("❌ Precision validation failed. Use --skip_precision_check to override.")
                return
        except ImportError:
            logger.warning("⚠️ Precision validation script not available, skipping...")
    
    # Load pipeline with mixed precision
    logger.info(f"Loading pipeline: {args.model_id}")
    try:
        from diffusers import AutoencoderKLWan
        
        # Load VAE with FP32
        vae = AutoencoderKLWan.from_pretrained(
            args.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        
        # Load pipeline with FP16 transformers
        pipe = WanPipeline.from_pretrained(
            args.model_id,
            vae=vae,
            torch_dtype=torch.float16,
        )
        
        pipe.to("cuda:0")
        
    except ImportError:
        logger.error("❌ diffusers not available. Install with: pip install diffusers")
        return
    
    # Export components
    success_count = 0
    total_components = len(args.components)
    
    for component in args.components:
        logger.info(f"\n{'='*60}")
        logger.info(f"Exporting component: {component}")
        logger.info(f"{'='*60}")
        
        # Determine precision for this component
        if component in ["vae", "vae_encoder", "vae_decoder"]:
            component_precision = "fp32"  # VAE always FP32
        else:
            component_precision = args.precision
        
        success = export_component(
            pipe=pipe,
            component_name=component,
            precision=component_precision,
            output_dir=output_dir,
            config=config,
            validate=args.validate,
            opset_version=args.opset_version
        )
        
        if success:
            success_count += 1
            logger.info(f"✅ {component} export completed successfully")
        else:
            logger.error(f"❌ {component} export failed")
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPORT SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Successfully exported: {success_count}/{total_components} components")
    
    if success_count == total_components:
        logger.info("🎉 All exports completed successfully!")
        logger.info("🚀 Ready for TensorRT engine building!")
        logger.info(f"📁 ONNX files saved to: {output_dir}")
        
        # List exported files
        logger.info("\n📋 Exported files:")
        for file in output_dir.glob("*.onnx"):
            logger.info(f"  - {file.name}")
    else:
        logger.error(f"❌ {total_components - success_count} exports failed")
        logger.error("Check logs above for error details")


if __name__ == "__main__":
    main()
