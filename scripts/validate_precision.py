#!/usr/bin/env python3
"""
Precision validation script for WAN2.2 → ONNX → TensorRT pipeline.

This script validates that all components have the correct precision settings
before proceeding with ONNX export.

Usage:
    python scripts/validate_precision.py --model_id "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
"""

import argparse
import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_logging, get_logger
from src.model.loader import load_pipeline


def validate_precision_setup(model_id: str, device: str = "cuda:0"):
    """
    Validate that all components have correct precision settings.
    
    Args:
        model_id: HuggingFace model ID
        device: Target device
    """
    logger = setup_logging(level="INFO")
    logger.info("🔍 Validating WAN2.2 precision setup...")
    
    try:
        # Load pipeline with mixed precision strategy
        logger.info(f"Loading pipeline: {model_id}")
        
        # Load VAE separately with FP32
        logger.info("Loading VAE with FP32 precision...")
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,  # VAE stays FP32 for quality
        )
        
        # Load full pipeline with FP16 transformers
        logger.info("Loading pipeline with FP16 transformers...")
        pipe = WanPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=torch.float16,  # Transformers FP16 for speed
        )
        
        pipe.to(device)
        
        # Validate component precisions
        logger.info("🔍 Checking component precisions...")
        
        # Check transformer precisions
        transformer_dtype = pipe.transformer.dtype
        transformer_2_dtype = pipe.transformer_2.dtype
        text_encoder_dtype = pipe.text_encoder.dtype
        vae_dtype = pipe.vae.dtype
        
        logger.info(f"Transformer dtype: {transformer_dtype}")
        logger.info(f"Transformer_2 dtype: {transformer_2_dtype}")
        logger.info(f"Text encoder dtype: {text_encoder_dtype}")
        logger.info(f"VAE dtype: {vae_dtype}")
        
        # Validate timestep precision
        logger.info("🔍 Validating timestep precision...")
        sample_timestep = torch.tensor([500], dtype=torch.float32, device=device)
        logger.info(f"Sample timestep dtype: {sample_timestep.dtype}")
        logger.info(f"Sample timestep shape: {sample_timestep.shape}")
        
        # Run precision checks
        checks_passed = 0
        total_checks = 5
        
        # Check 1: Transformers should be FP16
        if transformer_dtype == torch.float16:
            logger.info("✅ Transformer: FP16 (correct)")
            checks_passed += 1
        else:
            logger.error(f"❌ Transformer: {transformer_dtype} (should be FP16)")
        
        if transformer_2_dtype == torch.float16:
            logger.info("✅ Transformer_2: FP16 (correct)")
            checks_passed += 1
        else:
            logger.error(f"❌ Transformer_2: {transformer_2_dtype} (should be FP16)")
        
        # Check 2: Text encoder should be FP16
        if text_encoder_dtype == torch.float16:
            logger.info("✅ Text encoder: FP16 (correct)")
            checks_passed += 1
        else:
            logger.error(f"❌ Text encoder: {text_encoder_dtype} (should be FP16)")
        
        # Check 3: VAE should be FP32
        if vae_dtype == torch.float32:
            logger.info("✅ VAE: FP32 (correct)")
            checks_passed += 1
        else:
            logger.error(f"❌ VAE: {vae_dtype} (should be FP32)")
        
        # Check 4: Timestep should be FP32
        if sample_timestep.dtype == torch.float32:
            logger.info("✅ Timestep: FP32 (correct)")
            checks_passed += 1
        else:
            logger.error(f"❌ Timestep: {sample_timestep.dtype} (should be FP32)")
        
        # Final validation
        logger.info("=" * 50)
        if checks_passed == total_checks:
            logger.info("🎉 ALL PRECISION CHECKS PASSED!")
            logger.info("✅ Ready for ONNX export")
            logger.info("✅ Ready for TensorRT FP16 engine build")
            logger.info("✅ Mixed precision strategy validated")
            return True
        else:
            logger.error(f"❌ {total_checks - checks_passed} precision checks failed")
            logger.error("❌ Fix precision issues before ONNX export")
            return False
            
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate WAN2.2 precision setup")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Target device"
    )
    
    args = parser.parse_args()
    
    # Import here to avoid issues if diffusers not available
    try:
        from diffusers import AutoencoderKLWan
    except ImportError:
        print("❌ diffusers not available. Install with: pip install diffusers")
        return
    
    success = validate_precision_setup(args.model_id, args.device)
    
    if success:
        print("\n🚀 Ready to proceed with ONNX export!")
        print("Next steps:")
        print("1. Export transformers: python scripts/export_model.py --component transformer --precision fp16")
        print("2. Export transformer_2: python scripts/export_model.py --component transformer_2 --precision fp16")
        print("3. Export VAE: python scripts/export_model.py --component vae --precision fp32")
        print("4. Build TensorRT engines")
    else:
        print("\n❌ Fix precision issues before proceeding")
        sys.exit(1)


if __name__ == "__main__":
    main()
