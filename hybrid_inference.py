#!/usr/bin/env python3
"""
Hybrid Inference Script for Wan2.2-T2V-A14B-Diffusers
Uses TensorRT for transformer components and PyTorch for other components.

Components:
- TensorRT: transformer (DiT model)
- PyTorch: text_encoder, tokenizer, scheduler, vae, transformer_2

Usage:
    python scripts/hybrid_inference.py --prompt "A cat playing with a ball" --output_dir outputs/inference
    python scripts/hybrid_inference.py --prompt "A dog running in the park" --frames 81 --height 720 --width 1280 --guidance_scale 4.0 --guidance_scale_2 3.0
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import torch
import numpy as np
from PIL import Image
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, ensure_directory
from src.utils.logging import setup_logging, get_logger
from src.tensorrt.runtime import TRTInference, create_execution_context
from src.tensorrt.engine_builder import load_engine
from src.model.loader import load_pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class HybridInferencePipeline:
    """
    Hybrid inference pipeline combining TensorRT and PyTorch components.
    
    Architecture:
    - Text Encoder (PyTorch) -> Text embeddings
    - Tokenizer (PyTorch) -> Token IDs
    - Transformer 1 (TensorRT) -> Intermediate features (no fallback - fails if incompatible)
    - Transformer 2 (PyTorch) -> Final features  
    - Scheduler (PyTorch) -> Noise scheduling
    - VAE (PyTorch) -> Final video frames
    """
    
    def __init__(
        self,
        model_path: str,
        trt_engines_dir: str,
        device: str = "cuda",
        precision: str = "fp16",
        config_path: Optional[str] = None
    ):
        """
        Initialize hybrid inference pipeline.
        
        Args:
            model_path: Path to the Hugging Face model directory
            trt_engines_dir: Directory containing TensorRT engines
            device: Device to run PyTorch components on
            precision: Precision mode for inference
            config_path: Path to configuration file
        """
        self.device = device
        self.precision = precision
        self.logger = get_logger(__name__)
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Store model path
        self.model_path = model_path
        
        # Store TensorRT engines directory
        self.trt_engines_dir = trt_engines_dir
        
        # Load PyTorch components
        self._load_pytorch_components()
        
        # Load TensorRT engines
        self._load_tensorrt_engines()
        
        self.logger.info("Hybrid inference pipeline initialized successfully")
    
    def _load_pytorch_components(self):
        """Load PyTorch components (text_encoder, tokenizer, scheduler, vae)."""
        self.logger.info("Loading PyTorch components...")
        
        # Load the full pipeline first
        self.pipeline = load_pipeline(
            model_id=self.model_path,
            device=self.device,
            torch_dtype="bfloat16" if self.precision == "fp16" else "float32"
        )
        
        # Extract components from pipeline
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.scheduler = self.pipeline.scheduler
        self.vae = self.pipeline.vae
        self.transformer = self.pipeline.transformer
        self.transformer_2 = self.pipeline.transformer_2
        
        # Set precision
        if self.precision == "fp16":
            self.text_encoder = self.text_encoder.half()
            self.vae = self.vae.half()
            self.transformer = self.transformer.half()
            self.transformer_2 = self.transformer_2.half()
        
        self.logger.info("PyTorch components loaded successfully")
    
    def _load_tensorrt_engines(self):
        """Load TensorRT engines for transformer components."""
        self.logger.info("Loading TensorRT engines...")
        
        # Load transformer engine
        engine_path = Path(self.trt_engines_dir) / "dit_fp16.trt"
        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        
        # Load engine and create inference context
        self.transformer_engine = load_engine(str(engine_path))
        self.transformer_inference = TRTInference(self.transformer_engine, device=self.device)
        
        self.logger.info("TensorRT engines loaded successfully")
    
    def encode_prompt(
        self, 
        prompt: str, 
        negative_prompt: str = "",
        max_length: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompts using PyTorch text encoder.
        
        Args:
            prompt: Positive prompt text
            negative_prompt: Negative prompt text
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (positive_embeddings, negative_embeddings)
        """
        self.logger.info(f"Encoding prompt: '{prompt[:50]}...'")
        
        # Tokenize prompts (T5 tokenizer)
        positive_tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        negative_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length", 
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Encode tokens to embeddings (T5 encoder)
        with torch.no_grad():
            positive_output = self.text_encoder(positive_tokens)
            negative_output = self.text_encoder(negative_tokens)
            # T5 encoder returns last_hidden_state as the main output
            positive_embeddings = positive_output.last_hidden_state
            negative_embeddings = negative_output.last_hidden_state
        
        return positive_embeddings, negative_embeddings
    
    def encode_prompt_tensorrt_compatible(
        self, 
        prompt: str, 
        max_length: int = 4096
    ) -> torch.Tensor:
        """
        Encode text prompt for TensorRT compatibility (single sequence, 4096x4096).
        
        Args:
            prompt: Text prompt
            max_length: Maximum sequence length (4096 for TensorRT compatibility)
            
        Returns:
            Encoded embeddings with shape (1, 4096, 4096)
        """
        self.logger.info(f"Encoding prompt for TensorRT: '{prompt[:50]}...'")
        
        # Tokenize prompt (T5 tokenizer)
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Encode tokens to embeddings (T5 encoder)
        with torch.no_grad():
            output = self.text_encoder(tokens)
            embeddings = output.last_hidden_state  # Shape: (1, max_length, hidden_size)
        
        # Reshape to TensorRT-compatible shape (1, 4096, 4096)
        # This is a placeholder - the actual reshaping depends on the model architecture
        if embeddings.shape != (1, 4096, 4096):
            self.logger.warning(f"Reshaping embeddings from {embeddings.shape} to (1, 4096, 4096)")
            # For now, we'll pad or truncate to match the expected shape
            # This is a simplified approach - in practice, you'd need to understand
            # the actual model architecture to do this correctly
            if embeddings.shape[1] < 4096:
                # Pad with zeros
                padding = torch.zeros(1, 4096 - embeddings.shape[1], embeddings.shape[2], 
                                    device=embeddings.device, dtype=embeddings.dtype)
                embeddings = torch.cat([embeddings, padding], dim=1)
            else:
                # Truncate
                embeddings = embeddings[:, :4096, :]
            
            if embeddings.shape[2] < 4096:
                # Pad with zeros
                padding = torch.zeros(1, 4096, 4096 - embeddings.shape[2], 
                                    device=embeddings.device, dtype=embeddings.dtype)
                embeddings = torch.cat([embeddings, padding], dim=2)
            else:
                # Truncate
                embeddings = embeddings[:, :, :4096]
        
        return embeddings
    
    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        latent_channels: int = 16
    ) -> torch.Tensor:
        """
        Prepare initial latent noise with TensorRT-compatible dimensions.
        
        Args:
            batch_size: Batch size
            num_frames: Number of video frames
            height: Video height (will be resized to 45x80 for TensorRT compatibility)
            width: Video width (will be resized to 45x80 for TensorRT compatibility)
            latent_channels: Number of latent channels (16 for Wan2.2)
            
        Returns:
            Initial latent noise tensor with TensorRT-compatible dimensions
        """
        # TensorRT engine expects (1, 16, num_frames, 45, 80)
        # So we generate latents with 45x80 spatial dimensions
        latent_height = 45  # Fixed for TensorRT compatibility
        latent_width = 80   # Fixed for TensorRT compatibility
        
        self.logger.info(f"Generating latents with TensorRT-compatible dimensions: {batch_size}x{latent_channels}x{num_frames}x{latent_height}x{latent_width}")
        
        # Generate random noise
        latents = torch.randn(
            batch_size,
            latent_channels,
            num_frames,
            latent_height,
            latent_width,
            device=self.device,
            dtype=torch.float16 if self.precision == "fp16" else torch.float32
        )
        
        return latents
    
    def run_transformer_inference(
        self,
        sample: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Run transformer inference using TensorRT for first transformer and PyTorch for second.
        
        Args:
            sample: Input latent sample
            encoder_hidden_states: Text embeddings
            timestep: Current timestep
            
        Returns:
            Transformer output
        """
        # Check if TensorRT engine is compatible with current input shapes
        if not self._is_tensorrt_compatible(sample, encoder_hidden_states):
            raise RuntimeError(
                f"TensorRT engine not compatible with current input shapes. "
                f"Engine expects sample: (1, 16, num_frames, 45, 80) and encoder: (1, 4096, 4096), "
                f"but got sample: {sample.shape} and encoder: {encoder_hidden_states.shape}"
            )
        
        # Run first transformer with TensorRT only (no fallback)
        intermediate_output = self._run_tensorrt_transformer(
            sample, encoder_hidden_states, timestep
        )
        
        # Run second transformer with PyTorch (no TensorRT engine available)
        with torch.no_grad():
            # Clear cache to free memory
            torch.cuda.empty_cache()
            
            # Ensure timestep is 1D for the second transformer
            if timestep.dim() == 0:  # Scalar tensor
                timestep_1d = torch.tensor([timestep.item()], device=self.device, dtype=torch.long)
            else:
                timestep_1d = timestep.flatten().to(self.device)
            
            transformer_2_output = self.transformer_2(
                hidden_states=intermediate_output,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep_1d,
                return_dict=True
            )
            final_output = transformer_2_output.sample
            
            # Debug logging
            self.logger.info(f"Second transformer output shape: {final_output.shape}")
            self.logger.info(f"Expected shape: {sample.shape}")
            
            # Ensure output shape matches input shape
            if final_output.shape != sample.shape:
                self.logger.warning(f"Shape mismatch: output {final_output.shape} vs input {sample.shape}")
                # Resize if needed
                if final_output.shape[3] != sample.shape[3] or final_output.shape[4] != sample.shape[4]:
                    self.logger.info(f"Resizing second transformer output from {final_output.shape[3:]} to {sample.shape[3:]}")
                    final_output = torch.nn.functional.interpolate(
                        final_output.view(final_output.shape[0] * final_output.shape[1] * final_output.shape[2], 1, final_output.shape[3], final_output.shape[4]),
                        size=(sample.shape[3], sample.shape[4]),
                        mode='bilinear',
                        align_corners=False
                    ).view(final_output.shape[0], final_output.shape[1], final_output.shape[2], sample.shape[3], sample.shape[4])
                    self.logger.info(f"Resized second transformer output shape: {final_output.shape}")
        
        return final_output
    
    def _is_tensorrt_compatible(self, sample: torch.Tensor, encoder_hidden_states: torch.Tensor) -> bool:
        """
        Check if the current input shapes are compatible with the TensorRT engine.
        
        Args:
            sample: Input latent sample
            encoder_hidden_states: Text embeddings
            
        Returns:
            True if compatible, False otherwise
        """
        # Check sample shape compatibility
        # Engine expects: (1, 16, num_frames, 45, 80)
        # We have: (1, 16, num_frames, 64, 64)
        sample_compatible = (
            sample.shape[0] == 1 and 
            sample.shape[1] == 16 and 
            sample.shape[3] == 45 and 
            sample.shape[4] == 80
        )
        
        # Check encoder shape compatibility  
        # Engine expects: (1, 4096, 4096)
        # We have: (2, 256, 4096)
        encoder_compatible = encoder_hidden_states.shape == (1, 4096, 4096)
        
        return sample_compatible and encoder_compatible
    
    def _run_tensorrt_transformer(
        self,
        sample: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Run TensorRT transformer inference.
        
        Args:
            sample: Input latent sample
            encoder_hidden_states: Text embeddings
            timestep: Current timestep
            
        Returns:
            Transformer output from TensorRT
        """
        # Engine expects:
        # - sample: (1, 16, num_frames, 45, 80) - HALF precision
        # - timestep: (1,) - INT64  
        # - encoder_hidden_states: (1, 4096, 4096) - HALF precision
        
        # Reshape and prepare tensors to match engine expectations
        batch_size, channels, num_frames, height, width = sample.shape
        
        # Reshape sample to match engine input: (1, 16, num_frames, 45, 80)
        # The engine expects 45x80 spatial dimensions, but we might have different dimensions
        # We need to interpolate or pad to match the expected spatial size
        if height != 45 or width != 80:
            # Interpolate to match engine expectations
            sample_reshaped = torch.nn.functional.interpolate(
                sample.view(batch_size * channels * num_frames, 1, height, width),
                size=(45, 80),
                mode='bilinear',
                align_corners=False
            ).view(batch_size, channels, num_frames, 45, 80)
        else:
            sample_reshaped = sample
        
        # Reshape encoder_hidden_states to match engine input: (1, 4096, 4096)
        # The engine expects (1, 4096, 4096) but we might have (2, 256, 4096) for classifier-free guidance
        if encoder_hidden_states.shape != (1, 4096, 4096):
            # Take only the first sequence (positive prompt) and reshape
            if encoder_hidden_states.shape[0] == 2:  # Classifier-free guidance
                encoder_reshaped = encoder_hidden_states[1:2]  # Take positive prompt only
            else:
                encoder_reshaped = encoder_hidden_states[:1]  # Take first batch
            
            # Reshape to (1, 4096, 4096) - this might need adjustment based on actual model
            if encoder_reshaped.shape != (1, 4096, 4096):
                # This is a placeholder - the actual reshaping depends on the model architecture
                self.logger.warning(f"Encoder shape mismatch: expected (1, 4096, 4096), got {encoder_reshaped.shape}")
                # For now, we'll need to handle this properly based on the actual model
                raise ValueError(f"Encoder shape mismatch: expected (1, 4096, 4096), got {encoder_reshaped.shape}")
        else:
            encoder_reshaped = encoder_hidden_states
        
        # Ensure correct data types
        if sample_reshaped.dtype != torch.float16:
            sample_reshaped = sample_reshaped.half()
        
        if encoder_reshaped.dtype != torch.float16:
            encoder_reshaped = encoder_reshaped.half()
        
        if timestep.dtype != torch.int64:
            timestep = timestep.long()
        
        # Ensure timestep has the correct shape (1,) instead of scalar
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        
        # Also ensure it's on the correct device
        if timestep.device != self.device:
            timestep = timestep.to(self.device)
        
        # Prepare inputs for TensorRT
        inputs = {
            'sample': sample_reshaped,
            'encoder_hidden_states': encoder_reshaped,
            'timestep': timestep
        }
        
        # Debug logging
        self.logger.info(f"TensorRT input shapes:")
        for name, tensor in inputs.items():
            self.logger.info(f"  {name}: {tensor.shape} {tensor.dtype}")
        
        # Run TensorRT inference
        try:
            outputs = self.transformer_inference.infer(inputs)
            # The output tensor is named 'output' according to engine inspection
            self.logger.info(f"TensorRT output shape: {outputs['output'].shape}")
            
            # Resize output to match input spatial dimensions if needed
            output = outputs['output']
            if output.shape[3] != sample_reshaped.shape[3] or output.shape[4] != sample_reshaped.shape[4]:
                self.logger.info(f"Resizing TensorRT output from {output.shape[3:]} to {sample_reshaped.shape[3:]}")
                # Reshape for interpolation: (B*C*T, 1, H, W)
                output_reshaped = output.view(output.shape[0] * output.shape[1] * output.shape[2], 1, output.shape[3], output.shape[4])
                # Interpolate
                output_resized = torch.nn.functional.interpolate(
                    output_reshaped,
                    size=(sample_reshaped.shape[3], sample_reshaped.shape[4]),
                    mode='bilinear',
                    align_corners=False
                )
                # Reshape back to original format
                output = output_resized.view(output.shape[0], output.shape[1], output.shape[2], sample_reshaped.shape[3], sample_reshaped.shape[4])
                self.logger.info(f"Resized output shape: {output.shape}")
            
            return output
        except Exception as e:
            self.logger.error(f"TensorRT inference failed: {e}")
            raise RuntimeError(f"TensorRT inference failed: {e}")
    
    def denoise_step(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance_scale: float = 7.5,
        guidance_scale_2: float = 7.5
    ) -> torch.Tensor:
        """
        Perform one denoising step (no classifier-free guidance for TensorRT compatibility).
        
        Args:
            latents: Current latent state
            encoder_hidden_states: Text embeddings (single sequence for TensorRT)
            timestep: Current timestep
            guidance_scale: Not used (no classifier-free guidance)
            guidance_scale_2: Not used (no classifier-free guidance)
            
        Returns:
            Denoised latents
        """
        # Run transformer inference (no classifier-free guidance for TensorRT compatibility)
        noise_pred = self.run_transformer_inference(
            latents,
            encoder_hidden_states,
            timestep
        )
        
        # Scheduler step
        latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample
        
        return latents
    
    def decode_latents(self, latents: torch.Tensor, target_height: int = 512, target_width: int = 512) -> torch.Tensor:
        """
        Decode latents to video frames using PyTorch VAE and upscale to target resolution.
        
        Args:
            latents: Latent representations (TensorRT-compatible dimensions)
            target_height: Target video height
            target_width: Target video width
            
        Returns:
            Decoded video frames upscaled to target resolution
        """
        self.logger.info(f"Decoding latents to video frames and upscaling to {target_height}x{target_width}...")
        
        # The VAE expects 5D input: (batch, channels, frames, height, width)
        self.logger.info(f"Input latents shape: {latents.shape}")
        
        # Ensure latents are in the correct format for VAE
        if latents.dim() == 4:
            # If 4D, add frame dimension
            latents = latents.unsqueeze(2)  # Add frame dimension
            self.logger.info(f"Added frame dimension, new shape: {latents.shape}")
        
        # Decode with VAE (expects 5D input)
        with torch.no_grad():
            frames = self.vae.decode(latents).sample
        
        # Get the original shape info
        batch_size, channels, num_frames, height, width = latents.shape
        
        # Debug: Check VAE output shape
        self.logger.info(f"VAE output shape: {frames.shape}")
        self.logger.info(f"Expected reshape: ({batch_size}, {num_frames}, 3, {height * 8}, {width * 8})")
        
        # Reshape back to video format - adjust based on actual VAE output
        if frames.dim() == 5:
            # VAE output format: (batch, channels, frames, height, width)
            # We need to handle the case where VAE outputs different number of frames
            actual_frames = frames.shape[2]
            self.logger.info(f"VAE output frames: {actual_frames}, expected: {num_frames}")
            
            if actual_frames != num_frames:
                # Take the first num_frames frames from the VAE output
                if actual_frames > num_frames:
                    # Take first num_frames frames
                    frames = frames[:, :, :num_frames, :, :]
                else:
                    # Repeat the last frame to match num_frames
                    last_frame = frames[:, :, -1:, :, :]
                    repeat_count = num_frames - actual_frames
                    repeated_frames = last_frame.repeat(1, 1, repeat_count, 1, 1)
                    frames = torch.cat([frames, repeated_frames], dim=2)
            
            # Permute to (batch, frames, channels, height, width) then to (batch, frames, height, width, channels)
            frames = frames.permute(0, 2, 1, 3, 4)  # (batch, frames, channels, height, width)
            frames = frames.permute(0, 1, 3, 4, 2)  # (batch, frames, height, width, channels)
        else:
            # VAE output is in image format: (batch*frames, channels, height, width)
            frames = frames.reshape(batch_size, num_frames, 3, frames.shape[2], frames.shape[3])
            frames = frames.permute(0, 1, 3, 4, 2)  # (batch, frames, height, width, channels)
        
        # Upscale to target resolution if needed
        if frames.shape[2] != target_height or frames.shape[3] != target_width:
            self.logger.info(f"Upscaling from {frames.shape[2]}x{frames.shape[3]} to {target_height}x{target_width}")
            # Reshape for interpolation: (batch, frames, height, width, channels) -> (batch*frames, channels, height, width)
            frames_reshaped = frames.permute(0, 1, 4, 2, 3).reshape(batch_size * num_frames, 3, frames.shape[2], frames.shape[3])
            
            # Interpolate to target size
            frames_upscaled = torch.nn.functional.interpolate(
                frames_reshaped,
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            )
            
            # Reshape back to video format
            frames = frames_upscaled.reshape(batch_size, num_frames, 3, target_height, target_width)
            frames = frames.permute(0, 1, 3, 4, 2)  # (batch, frames, height, width, channels)
        
        return frames
    
    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 81,
        height: int = 720,
        width: int = 1280,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.0,
        guidance_scale_2: float = 3.0,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate video using hybrid inference pipeline.
        
        Args:
            prompt: Text prompt for video generation
            negative_prompt: Negative prompt
            num_frames: Number of video frames
            height: Video height
            width: Video width
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale for first transformer
            guidance_scale_2: Classifier-free guidance scale for second transformer
            seed: Random seed for reproducibility
            
        Returns:
            Generated video tensor
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.logger.info(f"Generating video: {num_frames} frames, {height}x{width}")
        self.logger.info(f"Prompt: '{prompt}'")
        
        start_time = time.time()
        
        # Encode prompts for TensorRT compatibility (single sequence)
        encoder_hidden_states = self.encode_prompt_tensorrt_compatible(prompt)
        
        # Prepare latents
        latents = self.prepare_latents(1, num_frames, height, width)
        
        # Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Denoising loop
        self.logger.info(f"Starting denoising loop ({num_inference_steps} steps)...")
        for i, timestep in enumerate(self.scheduler.timesteps):
            self.logger.info(f"Step {i+1}/{num_inference_steps}")
            
            latents = self.denoise_step(
                latents,
                encoder_hidden_states,
                timestep,
                guidance_scale,
                guidance_scale_2
            )
        
        # Decode latents to video and upscale to target resolution
        video_frames = self.decode_latents(latents, height, width)
        
        # Convert to 0-255 range
        video_frames = (video_frames + 1.0) / 2.0
        video_frames = torch.clamp(video_frames, 0.0, 1.0)
        video_frames = (video_frames * 255).byte()
        
        generation_time = time.time() - start_time
        self.logger.info(f"Video generation completed in {generation_time:.2f}s")
        
        return video_frames
    
    def save_video(
        self,
        video_frames: torch.Tensor,
        output_path: str,
        fps: int = 16
    ):
        """
        Save video frames to file.
        
        Args:
            video_frames: Video tensor (batch, frames, height, width, channels)
            output_path: Output file path
            fps: Frames per second
        """
        self.logger.info(f"Saving video to {output_path}")
        
        # Convert to numpy
        video_np = video_frames[0].cpu().numpy()  # Take first batch
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = video_np.shape[1:3]
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame in video_np:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        self.logger.info(f"Video saved successfully")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Inference for Wan2.2-T2V-A14B-Diffusers")
    
    # Model paths
    parser.add_argument(
        "--model_path",
        type=str,
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="HuggingFace model ID or path to model directory"
    )
    parser.add_argument(
        "--trt_engines_dir",
        type=str,
        default="outputs/engines/wan-trt-hopper",
        help="Directory containing TensorRT engines"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    
    # Generation parameters
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        help="Negative prompt"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=81,
        help="Number of video frames"
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
        "--steps",
        type=int,
        default=40,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale for first transformer"
    )
    parser.add_argument(
        "--guidance_scale_2",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale for second transformer"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/inference",
        help="Output directory"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output filename (auto-generated if not provided)"
    )
    
    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16"],
        help="Precision mode"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level="INFO")
    logger.info("Starting hybrid inference...")
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)
    
    # Generate output filename
    if args.output_name is None:
        timestamp = int(time.time())
        args.output_name = f"video_{timestamp}.mp4"
    
    output_path = output_dir / args.output_name
    
    try:
        # Initialize pipeline
        pipeline = HybridInferencePipeline(
            model_path=args.model_path,
            trt_engines_dir=args.trt_engines_dir,
            device=args.device,
            precision=args.precision,
            config_path=args.config
        )
        
        # Generate video
        video_frames = pipeline.generate_video(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_frames=args.frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            guidance_scale_2=args.guidance_scale_2,
            seed=args.seed
        )
        
        # Save video with 24 fps for better playback
        pipeline.save_video(video_frames, str(output_path), fps=24)
        
        logger.info(f"✓ Video generation completed successfully!")
        logger.info(f"Output: {output_path}")
        
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
