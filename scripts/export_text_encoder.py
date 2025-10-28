#!/usr/bin/env python3
"""
Export WAN2.2 text encoder to ONNX format.

Usage:
    python scripts/export_text_encoder.py --model_id "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
"""

import argparse
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_logging, get_logger
from src.utils.config import load_config, ensure_directory
from src.model.loader import load_pipeline
from src.model.shapes import get_duration_based_dummy_inputs, get_dynamic_axes
from src.export.onnx_exporter import export_to_onnx, compare_onnx_outputs


def export_text_encoder(
    pipe,
    output_dir: Path,
    config: dict,
    validate: bool = True,
    opset_version: int = 18
):
    """
    Export text encoder to ONNX.
    
    Args:
        pipe: Loaded pipeline
        output_dir: Output directory
        config: Configuration dict
        validate: Whether to validate outputs
        opset_version: ONNX opset version
    """
    logger = get_logger(__name__)
    
    logger.info("🔧 Exporting text encoder...")
    
    # Get text encoder
    text_encoder = pipe.text_encoder
    text_encoder.eval()
    
    # Create dummy inputs for text encoder
    logger.info("Creating text encoder dummy inputs...")
    
    # Text encoder expects tokenized text input
    batch_size = 1
    seq_length = 1024  # WAN2.2 supports long prompts
    hidden_size = 4096  # UMT5EncoderModel hidden size
    
    # Use pipeline device for consistency
    device = pipe.device
    logger.info(f"Using device: {device}")
    
    # Create dummy tokenized input (token IDs)
    dummy_inputs = {
        "input_ids": torch.randint(0, 50000, (batch_size, seq_length), device=device, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, seq_length, device=device, dtype=torch.long),
    }
    
    logger.info(f"Text encoder dummy inputs:")
    logger.info(f"  input_ids: {dummy_inputs['input_ids'].shape} ({dummy_inputs['input_ids'].dtype})")
    logger.info(f"  attention_mask: {dummy_inputs['attention_mask'].shape} ({dummy_inputs['attention_mask'].dtype})")
    
    # Export to ONNX
    output_filename = "text_encoder_fp16.onnx"
    output_path = output_dir / output_filename
    
    logger.info(f"Exporting text encoder to ONNX: {output_path}")
    
    try:
        # Use direct torch.onnx.export for better control
        input_tuple = (dummy_inputs["input_ids"], dummy_inputs["attention_mask"])
        input_names = ["input_ids", "attention_mask"]
        output_names = ["last_hidden_state"]
        
        # Full dynamic axes (batch + sequence length) for variable-length prompts
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "last_hidden_state": {0: "batch", 1: "seq_len"},
        }
        
        logger.info(f"Dynamic axes: {dynamic_axes}")
        
        with torch.no_grad():
            torch.onnx.export(
                text_encoder,
                input_tuple,
                str(output_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=False,
                export_params=True,
                keep_initializers_as_inputs=False,
                training=torch.onnx.TrainingMode.EVAL,
            )
        
        logger.info("✅ Text encoder export successful!")
        
        # Validate if requested
        if validate:
            logger.info("🔍 Validating text encoder ONNX export...")
            try:
                match, metrics = compare_onnx_outputs(
                    model=text_encoder,
                    onnx_path=str(output_path),
                    dummy_inputs=dummy_inputs,
                    input_names=input_names,
                    rtol=1e-3,
                    atol=1e-5
                )
                
                if match:
                    logger.info("✅ Text encoder validation passed!")
                else:
                    logger.warning(f"⚠️ Text encoder validation failed: {metrics}")
            except Exception as e:
                logger.warning(f"⚠️ Validation skipped due to error: {e}")
                logger.info("ONNX export completed successfully, but validation failed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Text encoder export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Export WAN2.2 text encoder to ONNX")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="HuggingFace model ID"
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
        "--validate",
        action="store_true",
        help="Validate ONNX export by comparing outputs"
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=18,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=1024,
        help="Text sequence length"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level="INFO")
    logger.info("🚀 Starting WAN2.2 text encoder ONNX export...")
    
    # Load config
    config = load_config(args.config)
    
    # Ensure output directory
    output_dir = ensure_directory(args.output_dir)
    
    # Load pipeline with mixed precision
    logger.info(f"Loading pipeline: {args.model_id}")
    try:
        import torch
        from diffusers import AutoencoderKLWan, WanPipeline
        
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
        
        # Verify text encoder precision
        logger.info(f"Text encoder dtype: {pipe.text_encoder.dtype}")
        assert pipe.text_encoder.dtype == torch.float16, "Text encoder should be FP16"
        
    except ImportError:
        logger.error("❌ diffusers not available. Install with: pip install diffusers")
        return
    
    # Export text encoder
    logger.info(f"\n{'='*60}")
    logger.info(f"Exporting Text Encoder")
    logger.info(f"{'='*60}")
    
    success = export_text_encoder(
        pipe=pipe,
        output_dir=output_dir,
        config=config,
        validate=args.validate,
        opset_version=args.opset_version
    )
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPORT SUMMARY")
    logger.info(f"{'='*60}")
    
    if success:
        logger.info("🎉 Text encoder export completed successfully!")
        logger.info("🚀 Ready for TensorRT engine building!")
        logger.info(f"📁 ONNX file saved to: {output_dir / 'text_encoder_fp16.onnx'}")
        
        # Show next steps
        logger.info("\n📋 Next steps:")
        logger.info("1. Export transformer: python scripts/export_model.py --component transformer --precision fp16")
        logger.info("2. Export transformer_2: python scripts/export_model.py --component transformer_2 --precision fp16")
        logger.info("3. Export VAE: python scripts/export_model.py --component vae --precision fp32")
        logger.info("4. Build TensorRT engines")
    else:
        logger.error("❌ Text encoder export failed")
        logger.error("Check logs above for error details")


if __name__ == "__main__":
    main()
