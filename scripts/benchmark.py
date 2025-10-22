#!/usr/bin/env python3
"""
Benchmark different inference configurations.

Usage:
    python scripts/benchmark.py --baseline pytorch --engine_dir outputs/engines
"""

import argparse
import sys
from pathlib import Path
import torch
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, ensure_directory
from src.utils.logging import setup_logging, get_logger
from src.model.loader import load_pipeline
from src.benchmark.benchmark import Benchmark, BenchmarkResult


def pytorch_inference_fn(pipe, inputs: Dict[str, Any]):
    """PyTorch baseline inference function."""
    output = pipe(
        prompt=inputs["prompt"],
        num_frames=inputs.get("num_frames", 16),
        height=inputs.get("height", 720),
        width=inputs.get("width", 1280),
        num_inference_steps=inputs.get("num_inference_steps", 50),
        guidance_scale=inputs.get("guidance_scale", 7.5),
    )
    return output


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference configurations")
    parser.add_argument(
        "--baseline",
        type=str,
        default="pytorch",
        choices=["pytorch", "pytorch_mixed", "tensorrt"],
        help="Baseline configuration"
    )
    parser.add_argument(
        "--engine_dir",
        type=str,
        default="outputs/engines",
        help="Directory containing TensorRT engines"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/benchmarks",
        help="Output directory for benchmark results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--num_warmup_runs",
        type=int,
        default=3,
        help="Number of warmup runs"
    )
    parser.add_argument(
        "--num_test_runs",
        type=int,
        default=10,
        help="Number of test runs"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all configurations"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level="INFO")
    logger.info("Starting benchmark...")
    
    # Load config
    config = load_config(args.config)
    
    # Ensure output directory
    output_dir = ensure_directory(args.output_dir)
    
    # Get benchmark configurations
    benchmark_configs = config.get("benchmark", {}).get("test_configurations", [])
    prompts = config.get("benchmark", {}).get("prompts", ["A serene lake at sunset"])
    
    results = []
    
    # Run benchmarks
    if args.baseline == "pytorch" or args.compare:
        logger.info("="*80)
        logger.info("Benchmarking PyTorch FP16 (Baseline)")
        logger.info("="*80)
        
        pipe = load_pipeline(
            model_id=config["model"]["id"],
            torch_dtype="float16",
            vae_dtype="float16",  # Full FP16
            device="cuda:0"
        )
        
        for test_config in benchmark_configs[:1]:  # Run first config for demo
            logger.info(f"\nConfiguration: {test_config['name']}")
            
            benchmark = Benchmark(
                name=f"PyTorch_FP16_{test_config['name']}",
                device="cuda:0",
                num_warmup_runs=args.num_warmup_runs,
                num_test_runs=args.num_test_runs
            )
            
            inputs = {
                "prompt": prompts[0],
                "num_frames": test_config["frames"],
                "height": test_config["height"],
                "width": test_config["width"],
                "num_inference_steps": config["inference"]["num_inference_steps"],
                "guidance_scale": config["inference"]["guidance_scale"],
            }
            
            result = benchmark.run(
                inference_fn=lambda inp: pytorch_inference_fn(pipe, inp),
                inputs=inputs,
                num_frames=test_config["frames"],
                config=test_config
            )
            
            results.append(result)
            print(result)
            
            # Save individual result
            result.save(output_dir / f"{result.name}.json")
        
        # Cleanup
        del pipe
        torch.cuda.empty_cache()
    
    if args.baseline == "pytorch_mixed" or args.compare:
        logger.info("="*80)
        logger.info("Benchmarking PyTorch Mixed (FP16 DiT + FP32 VAE)")
        logger.info("="*80)
        
        pipe = load_pipeline(
            model_id=config["model"]["id"],
            torch_dtype="float16",
            vae_dtype="float32",  # Mixed precision
            device="cuda:0"
        )
        
        for test_config in benchmark_configs[:1]:
            logger.info(f"\nConfiguration: {test_config['name']}")
            
            benchmark = Benchmark(
                name=f"PyTorch_Mixed_{test_config['name']}",
                device="cuda:0",
                num_warmup_runs=args.num_warmup_runs,
                num_test_runs=args.num_test_runs
            )
            
            inputs = {
                "prompt": prompts[0],
                "num_frames": test_config["frames"],
                "height": test_config["height"],
                "width": test_config["width"],
                "num_inference_steps": config["inference"]["num_inference_steps"],
                "guidance_scale": config["inference"]["guidance_scale"],
            }
            
            result = benchmark.run(
                inference_fn=lambda inp: pytorch_inference_fn(pipe, inp),
                inputs=inputs,
                num_frames=test_config["frames"],
                config=test_config
            )
            
            results.append(result)
            print(result)
            
            result.save(output_dir / f"{result.name}.json")
        
        del pipe
        torch.cuda.empty_cache()
    
    if args.baseline == "tensorrt" or args.compare:
        logger.info("="*80)
        logger.info("Benchmarking TensorRT")
        logger.info("="*80)
        
        # TODO: Implement TensorRT benchmark
        logger.warning("TensorRT benchmark not yet implemented")
        logger.info("Please implement TRT inference pipeline first")
    
    # Compare results
    if len(results) > 1:
        logger.info("\n" + "="*80)
        logger.info("COMPARISON")
        logger.info("="*80)
        comparison = Benchmark.compare(results, baseline_name=results[0].name)
        print(comparison)
        
        # Save comparison
        with open(output_dir / "comparison.txt", 'w') as f:
            f.write(comparison)
    
    logger.info(f"\nBenchmark results saved to: {output_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()

