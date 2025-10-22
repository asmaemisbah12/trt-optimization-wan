#!/usr/bin/env python3
"""
Run inference with TensorRT engines.

Usage:
    python scripts/run_inference.py \
        --prompt "A serene lake at sunset" \
        --engine_dir outputs/engines \
        --output_path outputs/videos/output.mp4
"""

import argparse
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.logging import setup_logging, get_logger
from src.model.loader import load_pipeline
from src.tensorrt.runtime import create_execution_context


def main():
    parser = argparse.ArgumentParser(description="Run inference with TensorRT engines")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--engine_dir",
        type=str,
        default="outputs/engines",
        help="Directory containing TensorRT engines"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/videos/output.mp4",
        help="Output video path"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames to generate"
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
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--use_pytorch",
        action="store_true",
        help="Use PyTorch baseline instead of TensorRT"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level="INFO")
    logger.info("Starting inference...")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Frames: {args.num_frames}, Size: {args.width}x{args.height}")
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        logger.info(f"Seed: {args.seed}")
    
    if args.use_pytorch:
        # PyTorch baseline inference
        logger.info("Using PyTorch baseline (no TensorRT)")
        
        pipe = load_pipeline(
            model_id=config["model"]["id"],
            torch_dtype="float16",
            vae_dtype="float32",
            device="cuda:0"
        )
        
        logger.info("Running inference...")
        output = pipe(
            prompt=args.prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
        
        # Save output
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Assuming output is a video tensor or frames
        # TODO: Implement video saving based on actual pipeline output format
        logger.info(f"Output saved to: {output_path}")
        
    else:
        # TensorRT inference
        logger.info("Using TensorRT engines")
        
        engine_dir = Path(args.engine_dir)
        dit_engine = engine_dir / "dit_fp16.trt"
        vae_engine = engine_dir / "vae_fp32.trt"
        
        if not dit_engine.exists():
            logger.error(f"DiT engine not found: {dit_engine}")
            logger.info("Please build engines first with: python scripts/build_engines.py")
            return
        
        # Load TensorRT engines
        logger.info(f"Loading DiT engine: {dit_engine}")
        dit_trt = create_execution_context(str(dit_engine), device="cuda:0")
        
        if vae_engine.exists():
            logger.info(f"Loading VAE engine: {vae_engine}")
            vae_trt = create_execution_context(str(vae_engine), device="cuda:0")
        else:
            logger.warning(f"VAE engine not found: {vae_engine}")
            logger.info("Using PyTorch VAE")
            vae_trt = None
        
        # TODO: Implement full TensorRT inference pipeline
        # This requires integrating TRT engines into the diffusion loop
        logger.info("TensorRT inference pipeline not yet fully implemented")
        logger.info("This is a placeholder for the integration")
        
        # Placeholder for actual implementation
        logger.info("Running inference...")
        # 1. Encode prompt with text encoder (PyTorch)
        # 2. Initialize latent noise
        # 3. Diffusion loop with DiT TRT engine
        # 4. Decode latents with VAE TRT engine
        # 5. Save video
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

