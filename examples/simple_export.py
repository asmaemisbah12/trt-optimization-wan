#!/usr/bin/env python3
"""
Simple example: Export a model component to ONNX.

This example demonstrates the basic workflow for exporting
the DiT component to ONNX format.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.model.loader import load_pipeline, get_submodule
from src.model.shapes import create_dummy_inputs, get_dynamic_axes
from src.export.onnx_exporter import export_to_onnx


def main():
    # Setup
    logger = setup_logging(level="INFO")
    config = load_config()
    
    logger.info("Simple ONNX Export Example")
    logger.info("="*60)
    
    # Load pipeline
    logger.info("Loading model pipeline...")
    pipe = load_pipeline(
        model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        torch_dtype="bfloat16",  # Wan2.2 uses bfloat16
        vae_dtype="float32",
        device="cuda:0"
    )
    
    # Extract Transformer component (Wan2.2 MoE architecture)
    logger.info("Extracting Transformer component...")
    transformer = get_submodule(pipe, "transformer")
    transformer.eval()
    
    # Create dummy inputs
    logger.info("Creating dummy inputs...")
    dummy_inputs = create_dummy_inputs(
        component_name="transformer",  # Wan uses "transformer"
        num_frames=81,  # Wan2.2 max frames @ 24fps
        height=720,
        width=1280,
        device="cuda:0",
        dtype=torch.bfloat16  # bfloat16 for Wan2.2
    )
    
    # Get dynamic axes
    dynamic_axes = get_dynamic_axes("transformer")
    
    # Export to ONNX
    output_path = "outputs/onnx/transformer_bf16_example.onnx"
    logger.info(f"Exporting to ONNX: {output_path}")
    
    export_to_onnx(
        model=transformer,
        dummy_inputs=dummy_inputs,
        output_path=output_path,
        input_names=list(dummy_inputs.keys()),
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        verbose=False
    )
    
    logger.info("✓ Export complete!")
    logger.info(f"ONNX model saved to: {output_path}")


if __name__ == "__main__":
    import torch
    main()

