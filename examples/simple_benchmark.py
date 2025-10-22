#!/usr/bin/env python3
"""
Simple example: Benchmark model inference.

This example demonstrates how to use the benchmarking framework
to measure inference performance.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_logging
from src.model.loader import load_pipeline
from src.benchmark.benchmark import Benchmark


def main():
    # Setup
    logger = setup_logging(level="INFO")
    
    logger.info("Simple Benchmark Example")
    logger.info("="*60)
    
    # Load pipeline
    logger.info("Loading model pipeline...")
    pipe = load_pipeline(
        model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        torch_dtype="bfloat16",  # Wan2.2 uses bfloat16
        vae_dtype="float32",
        device="cuda:0"
    )
    
    # Define inference function
    def inference_fn(inputs):
        with torch.no_grad():
            output = pipe(
                prompt=inputs["prompt"],
                num_frames=inputs["num_frames"],
                height=inputs["height"],
                width=inputs["width"],
                num_inference_steps=inputs["num_inference_steps"],
            )
        return output
    
    # Create benchmark
    benchmark = Benchmark(
        name="PyTorch_FP16_Baseline",
        device="cuda:0",
        num_warmup_runs=2,
        num_test_runs=5
    )
    
    # Prepare inputs (Wan2.2 specific parameters)
    inputs = {
        "prompt": "A serene lake at sunset",
        "num_frames": 81,  # Wan2.2 max frames
        "height": 720,
        "width": 1280,
        "num_inference_steps": 40,  # Wan2.2 default
        "guidance_scale": 4.0,  # Wan2.2 primary guidance
        # Note: guidance_scale_2 is handled by WanPipeline
    }
    
    # Run benchmark
    logger.info("Running benchmark...")
    result = benchmark.run(
        inference_fn=inference_fn,
        inputs=inputs,
        num_frames=16
    )
    
    # Display results
    print(result)
    
    # Save results
    result.save("outputs/benchmarks/example_result.json")
    logger.info("Results saved to: outputs/benchmarks/example_result.json")


if __name__ == "__main__":
    main()

