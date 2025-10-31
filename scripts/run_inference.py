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
import numpy as np

# Add project root and local packages to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / ".local_packages"))

from src.utils.logging import setup_logging
from src.tensorrt.runtime import create_execution_context
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKLWan
from transformers import AutoTokenizer


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
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt for classifier-free guidance"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=80,
        help="Number of frames to generate (default: 80, must match engine build max_frames. 80 frames = 20 latent frames)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Video height (default: 512, must match engine build dimensions)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Video width (default: 512, must match engine build dimensions)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=40,
        help="Number of inference steps (default: 40)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale for transformer (default: 4.0)"
    )
    parser.add_argument(
        "--guidance_scale_2",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale for transformer_2 (default: 3.0, used in late/low-noise stages)"
    )
    parser.add_argument(
        "--boundary_ratio",
        type=float,
        default=0.7,
        help="Timestep boundary ratio for switching from transformer to transformer_2 (default: 0.7, meaning last 30%% of steps use transformer_2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--use_pytorch",
        action="store_true",
        help="Use PyTorch baseline instead of TensorRT"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="HuggingFace model ID for non-TRT components"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for model weights"
    )
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level="INFO")
    logger.info("Starting inference...")
    logger.info(f"Prompt: {args.prompt}")
    if args.negative_prompt:
        logger.info(f"Negative prompt: {args.negative_prompt}")
    logger.info(f"Frames: {args.num_frames}, Size: {args.width}x{args.height}")
    logger.info(f"Guidance scales: transformer={args.guidance_scale}, transformer_2={args.guidance_scale_2}")
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        logger.info(f"Seed: {args.seed}")
    
    if args.use_pytorch:
        # PyTorch baseline inference
        logger.info("Using PyTorch baseline (no TensorRT)")
        
        from src.model.loader import load_pipeline
        pipe = load_pipeline(
            model_id=args.model_id,
            cache_dir=args.cache_dir,
            torch_dtype="float16",
            vae_dtype="float32",
            device="cuda:0",
            skip_disk_check=True
        )
        
        logger.info("Running inference...")
        output = pipe(
            prompt=args.prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            guidance_scale_2=args.guidance_scale_2,
            negative_prompt=args.negative_prompt if args.negative_prompt else None,
        )
        
        # Save output
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save video frames
        logger.info(f"Output saved to: {output_path}")
        
    else:
        # Hybrid TensorRT + PyTorch inference
        logger.info("=" * 70)
        logger.info("Hybrid Inference Mode: TensorRT + PyTorch")
        logger.info("=" * 70)
        
        engine_dir = Path(args.engine_dir)
        if not engine_dir.exists():
            logger.error(f"Engine directory does not exist: {engine_dir}")
            logger.error("Please specify a valid engine directory with --engine_dir")
            return
        
        # Check for TensorRT engines (WAN2.2 uses BOTH transformers together)
        text_encoder_engine = engine_dir / "text_encoder_fp16.trt"
        transformer_engine = engine_dir / "transformer_fp16.trt"
        transformer_2_engine = engine_dir / "transformer_2_fp16.trt"
        
        # Check both transformers are available (WAN2.2 requires both)
        if not transformer_engine.exists() or not transformer_2_engine.exists():
            logger.error("WAN2.2 requires BOTH transformer engines!")
            if not transformer_engine.exists():
                logger.error(f"  ✗ Missing: {transformer_engine}")
            if not transformer_2_engine.exists():
                logger.error(f"  ✗ Missing: {transformer_2_engine}")
            logger.info("Please build both engines first")
            return
        
        logger.info("\nLoading components:")
        logger.info("-" * 70)
        
        # 1. Load TensorRT engines
        logger.info("📦 TensorRT Components:")
        
        # Text encoder (required)
        if not text_encoder_engine.exists():
            logger.error(f"  ✗ Text Encoder TRT not found: {text_encoder_engine}")
            logger.error("Please build it first: python scripts/build_engines.py --onnx_path outputs/onnx/text_encoder_fp16.onnx --precision fp16")
            return
        
        logger.info(f"  ✓ Text Encoder: {text_encoder_engine.name} (will unload after encoding)")
        text_encoder_trt = create_execution_context(str(text_encoder_engine), device="cuda:0")
        
        # Both transformers (load after unloading text encoder)
        transformer_path = str(transformer_engine)
        transformer_2_path = str(transformer_2_engine)
        
        # 2. Load PyTorch components from HuggingFace
        logger.info(f"\n📦 PyTorch Components (from {args.model_id}):")
        
        # Load tokenizer
        logger.info("  ✓ Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            subfolder="tokenizer",
            cache_dir=args.cache_dir
        )
        
        # Load scheduler
        logger.info("  ✓ Loading scheduler...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.model_id,
            subfolder="scheduler",
            cache_dir=args.cache_dir
        )
        
        # Load VAE on CPU initially to save GPU memory during denoising
        # We'll move it to GPU after denoising completes and transformers are unloaded
        logger.info("  ✓ Loading VAE on CPU (will move to GPU after denoising)...")
        torch.cuda.empty_cache()  # Clear cache before loading VAE
        
        vae = AutoencoderKLWan.from_pretrained(
            args.model_id,
            subfolder="vae",
            cache_dir=args.cache_dir,
            torch_dtype=torch.float32
        ).to("cpu")  # Load on CPU initially
        
        logger.info("\n" + "=" * 70)
        logger.info("Running Hybrid TensorRT Inference (WAN 2.2 Pipeline)")
        logger.info("=" * 70)
        
        # 1. Tokenize and encode prompts separately (sequential CFG approach)
        logger.info("Step 1: Tokenizing and encoding prompts for sequential classifier-free guidance...")
        logger.info("  NOTE: ONNX models use batch=1, so CFG is done via sequential passes (uncond, then cond)")
        negative_prompt_text = args.negative_prompt if args.negative_prompt else ""
        
        # NOTE: Engines were built with seq_len=1024 and 512x512 video dimensions
        # We'll pad encoder_hidden_states to 1024 if needed to match engine expectations
        max_seq_length = 1024  # Engine expects 1024, not 512
        
        # Tokenize unconditional (negative) prompt
        # NOTE: Tokenize with max_length=512 (text encoder output), then pad to 1024 after encoding
        tok_neg = tokenizer(
            negative_prompt_text,
            return_tensors="pt",
            padding="max_length",
            max_length=512,  # Text encoder works with 512, we'll pad to 1024 after encoding
            truncation=True
        )
        
        # Tokenize conditional (positive) prompt
        tok_pos = tokenizer(
            args.prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=512,  # Text encoder works with 512, we'll pad to 1024 after encoding
            truncation=True
        )
        
        logger.info(f"  Tokenized prompts (max_length=512, will pad to {max_seq_length} after encoding)")
        
        # Encode unconditional prompt with TRT text encoder (batch=1)
        logger.info("  Encoding unconditional prompt with TRT text encoder (batch=1)...")
        text_encoder_inputs_neg = {
            "input_ids": tok_neg.input_ids.to("cuda:0").to(torch.int64),
            "attention_mask": tok_neg.attention_mask.to("cuda:0").to(torch.int64)
        }
        
        with torch.no_grad():
            encoder_outputs_neg = text_encoder_trt.infer(text_encoder_inputs_neg)
            # Text encoder output could be 'last_hidden_state' or 'output'
            if "last_hidden_state" in encoder_outputs_neg:
                encoder_hidden_states_uncond = encoder_outputs_neg["last_hidden_state"]
            elif "output" in encoder_outputs_neg:
                encoder_hidden_states_uncond = encoder_outputs_neg["output"]
            else:
                output_name = list(encoder_outputs_neg.keys())[0]
                encoder_hidden_states_uncond = encoder_outputs_neg[output_name]
                logger.info(f"  Using output '{output_name}' from text encoder")
            
            # Ensure it's a torch tensor on the correct device
            if isinstance(encoder_hidden_states_uncond, np.ndarray):
                encoder_hidden_states_uncond = torch.from_numpy(encoder_hidden_states_uncond).to("cuda:0")
            elif not encoder_hidden_states_uncond.is_cuda:
                encoder_hidden_states_uncond = encoder_hidden_states_uncond.to("cuda:0")
        
        # Encode conditional prompt with TRT text encoder (batch=1)
        logger.info("  Encoding conditional prompt with TRT text encoder (batch=1)...")
        text_encoder_inputs_pos = {
            "input_ids": tok_pos.input_ids.to("cuda:0").to(torch.int64),
            "attention_mask": tok_pos.attention_mask.to("cuda:0").to(torch.int64)
        }
        
        with torch.no_grad():
            encoder_outputs_pos = text_encoder_trt.infer(text_encoder_inputs_pos)
            if "last_hidden_state" in encoder_outputs_pos:
                encoder_hidden_states_cond = encoder_outputs_pos["last_hidden_state"]
            elif "output" in encoder_outputs_pos:
                encoder_hidden_states_cond = encoder_outputs_pos["output"]
            else:
                output_name = list(encoder_outputs_pos.keys())[0]
                encoder_hidden_states_cond = encoder_outputs_pos[output_name]
            
            # Ensure it's a torch tensor on the correct device
            if isinstance(encoder_hidden_states_cond, np.ndarray):
                encoder_hidden_states_cond = torch.from_numpy(encoder_hidden_states_cond).to("cuda:0")
            elif not encoder_hidden_states_cond.is_cuda:
                encoder_hidden_states_cond = encoder_hidden_states_cond.to("cuda:0")
        
        # Pad encoder_hidden_states to match engine expectations (1024 seq_len)
        # If text encoder returned 512, pad to 1024 with zeros
        if encoder_hidden_states_uncond.shape[1] < max_seq_length:
            orig_len = encoder_hidden_states_uncond.shape[1]
            pad_size = max_seq_length - orig_len
            pad_tensor = torch.zeros(1, pad_size, 4096, device=encoder_hidden_states_uncond.device, dtype=encoder_hidden_states_uncond.dtype)
            encoder_hidden_states_uncond = torch.cat([encoder_hidden_states_uncond, pad_tensor], dim=1)
            logger.info(f"  Padded unconditional hidden states from {orig_len} to {max_seq_length}")
        
        if encoder_hidden_states_cond.shape[1] < max_seq_length:
            orig_len = encoder_hidden_states_cond.shape[1]
            pad_size = max_seq_length - orig_len
            pad_tensor = torch.zeros(1, pad_size, 4096, device=encoder_hidden_states_cond.device, dtype=encoder_hidden_states_cond.dtype)
            encoder_hidden_states_cond = torch.cat([encoder_hidden_states_cond, pad_tensor], dim=1)
            logger.info(f"  Padded conditional hidden states from {orig_len} to {max_seq_length}")
        
        logger.info(f"  Unconditional hidden states shape: {encoder_hidden_states_uncond.shape} (expected: [1, {max_seq_length}, 4096])")
        logger.info(f"  Conditional hidden states shape: {encoder_hidden_states_cond.shape} (expected: [1, {max_seq_length}, 4096])")
        
        # Unload text encoder to free ~20GB GPU memory (only needed once)
        del text_encoder_trt
        torch.cuda.empty_cache()
        logger.info("  ✓ Text encoder unloaded to free GPU memory")
        
        # Load first transformer (lazy load transformer_2 later to save GPU memory)
        logger.info("\n  Loading transformer (transformer_2 will be loaded lazily)...")
        transformer_trt = create_execution_context(transformer_path, device="cuda:0")
        # Verify context was created successfully
        if transformer_trt.context is None:
            logger.error("Failed to create execution context for transformer - Out of memory?")
            return
        logger.info(f"  ✓ Transformer loaded")
        
        # transformer_2 will be loaded lazily when needed
        transformer_2_trt = None
        
        # 2. Initialize latents (batch=1 for sequential CFG)
        logger.info("\nStep 2: Initializing latents...")
        
        # WAN 2.2 parameters (from WanTransformer3DModel config and VAE architecture)
        # - spatial_scale = 8 (latent is 8x smaller spatially)
        # - temporal_scale = 4 (latent is 4x smaller temporally)
        # - latent_channels = 16 (from WanTransformer3DModel.in_channels)
        # - text_dim = 4096 (from WanTransformer3DModel.text_dim)
        # - rope_max_seq_len = 1024 (model supports up to 1024, but ONNX export uses 512)
        vae_scale_factor = 8  # WAN 2.2 uses spatial_scale = 8
        temporal_compression = 4  # WAN 2.2 uses temporal_scale = 4
        
        logger.info(f"  VAE scale factor: {vae_scale_factor}× spatial, {temporal_compression}× temporal")
        
        # Validate spatial dimensions are divisible
        if args.height % vae_scale_factor != 0 or args.width % vae_scale_factor != 0:
            logger.error(f"Video dimensions {args.height}×{args.width} must be divisible by {vae_scale_factor}")
            return
        
        latent_channels = 16
        latent_height = args.height // vae_scale_factor  # e.g., 512 / 8 = 64
        latent_width = args.width // vae_scale_factor    # e.g., 512 / 8 = 64
        # Use ceiling division for temporal frames (WAN supports non-divisible frame counts)
        latent_frames = (args.num_frames + temporal_compression - 1) // temporal_compression
        
        # Engine was built with max 20 latent frames (80 video frames)
        # Validate that we don't exceed engine limits
        if latent_frames > 20:
            logger.error(f"Latent frames ({latent_frames}) exceeds engine max (20). Use num_frames <= 80.")
            return
        
        logger.info(f"  Calculated latent dims: {latent_height}×{latent_width}, {latent_frames} frames")
        logger.info(f"  Video frames: {args.num_frames} → Latent frames: {latent_frames} (ceiling division)")
        logger.info(f"  Engine supports 4-20 latent frames (16-80 video frames)")
        
        # Initialize latents with batch=1 (sequential CFG doesn't need batch=2)
        latents = torch.randn(
            1, latent_channels, latent_frames, latent_height, latent_width,
            device="cuda:0",
            dtype=torch.float16
        )
        logger.info(f"  Latent shape: {latents.shape} (batch=1 for sequential CFG)")
        
        # 3. Denoising loop with two-stage transformer switching + sequential CFG
        logger.info(f"\nStep 3: Denoising ({args.num_inference_steps} steps)...")
        logger.info(f"  Two-stage denoising: transformer for early steps, transformer_2 for late steps")
        logger.info(f"  Sequential CFG: two passes per step (uncond, then cond)")
        logger.info(f"  CFG scales: transformer={args.guidance_scale}, transformer_2={args.guidance_scale_2}")

        scheduler.set_timesteps(args.num_inference_steps)
        
        # Calculate boundary timestep for switching transformers
        # boundary_ratio=0.7 means last 30% of steps use transformer_2
        boundary_step_idx = int(args.boundary_ratio * len(scheduler.timesteps))
        boundary_timestep = scheduler.timesteps[boundary_step_idx].item()
        logger.info(f"  Boundary: step {boundary_step_idx}/{len(scheduler.timesteps)} (timestep={boundary_timestep:.2f})")
        logger.info(f"    - Steps 0-{boundary_step_idx-1}: transformer (guidance={args.guidance_scale})")
        logger.info(f"    - Steps {boundary_step_idx}-{len(scheduler.timesteps)-1}: transformer_2 (guidance={args.guidance_scale_2})")

        for i, t in enumerate(scheduler.timesteps):
            is_transformer_2_stage = i >= boundary_step_idx
            
            # Lazy load transformer_2 when we reach the boundary step
            if is_transformer_2_stage and transformer_2_trt is None:
                logger.info(f"\n  Switching to transformer_2: unloading transformer and loading transformer_2...")
                # Unload transformer to free GPU memory
                del transformer_trt
                torch.cuda.empty_cache()
                logger.info(f"  ✓ Transformer unloaded")
                
                # Load transformer_2
                transformer_2_trt = create_execution_context(transformer_2_path, device="cuda:0")
                if transformer_2_trt.context is None:
                    logger.error("Failed to create execution context for transformer_2 - Out of memory?")
                    return
                logger.info(f"  ✓ Transformer_2 loaded")
            
            active_engine = transformer_2_trt if is_transformer_2_stage else transformer_trt
            active_scale = args.guidance_scale_2 if is_transformer_2_stage else args.guidance_scale
            stage_name = "transformer_2" if is_transformer_2_stage else "transformer"
            
            if i % 10 == 0 or i == len(scheduler.timesteps) - 1:
                logger.info(f"  Step {i+1}/{args.num_inference_steps}, timestep: {t.item():.2f}, using {stage_name} (CFG={active_scale})")
            
            # Sequential CFG: Run transformer twice (uncond, then cond) with batch=1
            timestep_tensor = torch.tensor([t.item()], device="cuda:0", dtype=torch.float32)  # batch=1
            
            # First pass: unconditional
            transformer_inputs_uncond = {
                "sample": latents.to(torch.float16),  # [1, C, T, H, W]
                "timestep": timestep_tensor,  # [1]
                "encoder_hidden_states": encoder_hidden_states_uncond.to(torch.float16)  # [1, seq, hidden]
            }
            
            with torch.no_grad():
                transformer_outputs_uncond = active_engine.infer(transformer_inputs_uncond)
                # Get output
                if "output" in transformer_outputs_uncond:
                    noise_pred_uncond = transformer_outputs_uncond["output"]
                else:
                    noise_pred_uncond = list(transformer_outputs_uncond.values())[0]
                
                # Ensure it's a torch tensor on CUDA
                if isinstance(noise_pred_uncond, np.ndarray):
                    noise_pred_uncond = torch.from_numpy(noise_pred_uncond).to("cuda:0")
                elif not noise_pred_uncond.is_cuda:
                    noise_pred_uncond = noise_pred_uncond.to("cuda:0")
                
                # Remove batch dimension if present: [1, C, T, H, W] -> [C, T, H, W]
                if noise_pred_uncond.dim() == 5 and noise_pred_uncond.shape[0] == 1:
                    noise_pred_uncond = noise_pred_uncond.squeeze(0)
            
            # Second pass: conditional
            transformer_inputs_cond = {
                "sample": latents.to(torch.float16),  # [1, C, T, H, W]
                "timestep": timestep_tensor,  # [1]
                "encoder_hidden_states": encoder_hidden_states_cond.to(torch.float16)  # [1, seq, hidden]
            }
            
            with torch.no_grad():
                transformer_outputs_cond = active_engine.infer(transformer_inputs_cond)
                # Get output
                if "output" in transformer_outputs_cond:
                    noise_pred_cond = transformer_outputs_cond["output"]
                else:
                    noise_pred_cond = list(transformer_outputs_cond.values())[0]
                
                # Ensure it's a torch tensor on CUDA
                if isinstance(noise_pred_cond, np.ndarray):
                    noise_pred_cond = torch.from_numpy(noise_pred_cond).to("cuda:0")
                elif not noise_pred_cond.is_cuda:
                    noise_pred_cond = noise_pred_cond.to("cuda:0")
                
                # Remove batch dimension if present: [1, C, T, H, W] -> [C, T, H, W]
                if noise_pred_cond.dim() == 5 and noise_pred_cond.shape[0] == 1:
                    noise_pred_cond = noise_pred_cond.squeeze(0)
            
            # Apply classifier-free guidance: guided_noise = uncond + scale * (cond - uncond)
            # noise_pred_uncond shape: [C, T, H, W]
            # noise_pred_cond shape: [C, T, H, W]
            guided_noise = noise_pred_uncond + active_scale * (noise_pred_cond - noise_pred_uncond)
            # guided_noise shape: [C, T, H, W]
            
            # For scheduler.step(), we need batch dimension
            guided_noise_batch = guided_noise.unsqueeze(0)  # [1, C, T, H, W]
            
            # Scheduler step (latents is already batch=1)
            scheduler_output = scheduler.step(guided_noise_batch, t, latents)
            if hasattr(scheduler_output, 'prev_sample'):
                latents = scheduler_output.prev_sample
            elif hasattr(scheduler_output, 'denoised'):
                latents = scheduler_output.denoised
            else:
                latents = scheduler_output
            
            # Ensure latents stay on CUDA and maintain FP16
            if not latents.is_cuda:
                latents = latents.to("cuda:0")
            if latents.dtype != torch.float16:
                latents = latents.to(torch.float16)
        
        logger.info(f"  Final latents shape after denoising: {latents.shape} (expected: [1, {latent_channels}, {latent_frames}, {latent_height}, {latent_width}])")
        
        # 4. Unload transformers and move VAE to GPU for faster decoding
        logger.info("\nStep 4: Unloading transformers and preparing VAE decode...")
        if transformer_trt is not None:
            del transformer_trt
        if transformer_2_trt is not None:
            del transformer_2_trt
        torch.cuda.empty_cache()
        logger.info("  ✓ Transformers unloaded, moving VAE to GPU...")
        
        # Move VAE to GPU for much faster decoding (transformers are gone now)
        vae = vae.to("cuda:0")
        logger.info("  ✓ VAE moved to GPU for decoding")
        
        logger.info(f"\n  Decoding latents with VAE (on GPU)...")
        logger.info(f"  Input latent shape: {latents.shape}")
        
        with torch.no_grad():
            # VAE expects latents in shape [batch, channels, frames, height, width]
            # Keep latents on GPU and convert to FP32 for VAE (VAE needs FP32)
            latents_for_decode = latents.to(torch.float32)
            logger.info(f"  Decoding latents shape: {latents_for_decode.shape}")
            
            # AutoencoderKLWan.decode() returns a dict with 'sample' key
            vae_output = vae.decode(latents_for_decode)
            if isinstance(vae_output, dict):
                frames = vae_output["sample"]
            else:
                frames = vae_output
        
        logger.info(f"  Decoded frames shape: {frames.shape}")
        logger.info(f"  Expected shape: [1, 3, {args.num_frames}, {args.height}, {args.width}] or [1, {args.num_frames}, 3, {args.height}, {args.width}]")
        
        # 5. Save video
        logger.info("\nStep 5: Saving video...")
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert frames to video format and save
        try:
            import imageio
            
            logger.info(f"  Converting {frames.shape} to video format...")
            
            # Handle different VAE output layouts: [B, C, T, H, W] vs [B, T, C, H, W]
            if frames.dim() != 5:
                raise RuntimeError(f"VAE output is not 5D video tensor, got shape: {frames.shape}")
            
            # Remove batch dimension
            frames_4d = frames[0]  # [C, T, H, W] or [T, C, H, W]
            
            # Detect layout by checking channel dimension position
            # Most common: [C, T, H, W] where C=3
            if frames_4d.shape[0] == 3 and frames_4d.dim() == 4:
                # Layout: [C, T, H, W] -> (T, H, W, C)
                frames_np = frames_4d.permute(1, 2, 3, 0).cpu().float().numpy()
            # Alternative: [T, C, H, W] where C=3
            elif frames_4d.shape[1] == 3 and frames_4d.dim() == 4:
                # Layout: [T, C, H, W] -> (T, H, W, C)
                frames_np = frames_4d.permute(0, 2, 3, 1).cpu().float().numpy()
            else:
                # Try to infer: look for dimension with size 3 (channels)
                channel_dim = None
                for i, size in enumerate(frames_4d.shape):
                    if size == 3:
                        channel_dim = i
                        break
                
                if channel_dim is None:
                    raise RuntimeError(f"Could not find channel dimension (size=3) in frame shape: {frames.shape}")
                
                # Build permutation: move channel dim to last position
                # If shape is [d0, d1, d2, d3] and channel_dim=1, we want [d0, d2, d3, d1]
                dims = list(range(frames_4d.dim()))
                dims.remove(channel_dim)
                dims.append(channel_dim)
                frames_np = frames_4d.permute(*dims).cpu().float().numpy()
            
            logger.info(f"  After permute: {frames_np.shape} (expected: ({args.num_frames}, {args.height}, {args.width}, 3))")
            
            # Normalize from [-1, 1] to [0, 255]
            frames_np = ((frames_np + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
            
            # Save as video using imageio directly
            logger.info(f"  Writing video with {frames_np.shape[0]} frames at 16 FPS...")
            imageio.mimwrite(str(output_path), frames_np, fps=16, quality=8, codec='libx264')
            logger.info(f"  ✓ Video saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"  ✗ Video saving failed: {e}")
            import traceback
            traceback.print_exc()
            logger.info(f"  Frames shape: {frames.shape}")
            logger.info("  Saving frames as numpy array instead...")
            np.save(output_path.with_suffix('.npy'), frames.cpu().numpy())
            logger.info(f"  ✓ Frames saved to: {output_path.with_suffix('.npy')}")
    
    logger.info("\n✓ Done!")


if __name__ == "__main__":
    main()

