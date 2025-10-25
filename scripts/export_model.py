#!/usr/bin/env python3
"""
Export Wan2.2-T2V model components to ONNX format.

Usage:
    python scripts/export_model.py --component dit --precision fp16
    python scripts/export_model.py --component vae --precision fp32
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_torch_dtype, ensure_directory
from src.utils.logging import setup_logging, get_logger
from src.model.loader import load_pipeline, get_submodule
from src.model.shapes import create_dummy_inputs, get_dynamic_axes
from src.export.onnx_exporter import export_to_onnx, compare_onnx_outputs


def main():
    parser = argparse.ArgumentParser(description="Export model components to ONNX")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--component",
        type=str,
        required=True,
        choices=["dit", "transformer", "unet", "vae", "vae_encoder", "vae_decoder"],
        help="Component to export"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",  # Wan2.2 uses bfloat16 by default
        choices=["fp32", "fp16", "bf16", "bfloat16"],
        help="Precision for export (bf16 recommended for Wan2.2)"
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
        default=81,  # Wan2.2 supports up to 81 frames @ 24fps
        help="Number of frames for dummy input (max 81 for Wan2.2)"
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
        help="Validate ONNX export by comparing outputs"
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=17,
        help="ONNX opset version"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level="INFO")
    logger.info("Starting ONNX export...")
    logger.info(f"Component: {args.component}")
    logger.info(f"Precision: {args.precision}")
    
    # Load config
    config = load_config(args.config)
    
    # Ensure output directory
    output_dir = ensure_directory(args.output_dir)
    
    # Load pipeline
    logger.info(f"Loading pipeline: {args.model_id}")
    pipe = load_pipeline(
        model_id=args.model_id,
        cache_dir=config["model"].get("cache_dir"),
        torch_dtype=args.precision,
        vae_dtype="float32" if args.component in ["vae", "vae_decoder", "vae_encoder"] else args.precision,
        device="cuda:0"
    )
    
    # Get submodule
    component_map = {
        "dit": "transformer",
        "transformer": "transformer",
        "unet": "unet",
        "vae": "vae",
        "vae_encoder": "vae",
        "vae_decoder": "vae",
    }
    
    submodule_name = component_map.get(args.component)
    if submodule_name is None:
        logger.error(f"Unknown component: {args.component}")
        return
    
    logger.info(f"Extracting submodule: {submodule_name}")
    model = get_submodule(pipe, submodule_name)
    model.eval()
    
    # Create dummy inputs
    logger.info("Creating dummy inputs...")
    dtype = get_torch_dtype(args.precision)
    dummy_inputs = create_dummy_inputs(
        component_name=args.component,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        device="cuda:0",
        dtype=dtype,
        model=model  # Pass the loaded model for auto-detection
    )
    
    # Get dynamic axes
    dynamic_axes = get_dynamic_axes(args.component)
    
    # Export to ONNX
    output_filename = f"{args.component}_{args.precision}.onnx"
    output_path = output_dir / output_filename
    
    logger.info(f"Exporting to ONNX: {output_path}")
    
    try:
        export_to_onnx(
            model=model,
            dummy_inputs=dummy_inputs,
            output_path=str(output_path),
            input_names=list(dummy_inputs.keys()),
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset_version,
            do_constant_folding=True,
            verbose=False
        )
        
        logger.info("✓ Export successful!")
        
        # Validate if requested
        if args.validate:
            logger.info("Validating ONNX export...")
            match, metrics = compare_onnx_outputs(
                model=model,
                onnx_path=str(output_path),
                dummy_inputs=dummy_inputs,
                input_names=list(dummy_inputs.keys()),
                rtol=1e-3,
                atol=1e-5
            )
            
            if match:
                logger.info("✓ Validation passed! Outputs match within tolerance.")
            else:
                logger.warning("⚠ Validation failed. Output mismatch detected.")
                logger.warning(f"Metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info(f"Done! ONNX model saved to: {output_path}")


if __name__ == "__main__":
    main()

