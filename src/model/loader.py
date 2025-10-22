"""Model loading utilities for Wan2.2-T2V"""

import torch
from typing import Dict, Any, Optional
from diffusers import DiffusionPipeline
from src.utils.logging import get_logger
from src.utils.config import get_torch_dtype

logger = get_logger(__name__)

# Try to import Wan-specific components
try:
    from diffusers import WanPipeline, AutoencoderKLWan
    HAS_WAN_PIPELINE = True
except ImportError:
    logger.warning("WanPipeline not found in diffusers. Using DiffusionPipeline as fallback.")
    WanPipeline = DiffusionPipeline
    AutoencoderKLWan = None
    HAS_WAN_PIPELINE = False


def load_pipeline(
    model_id: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    cache_dir: Optional[str] = None,
    torch_dtype: str = "bfloat16",  # Wan2.2 uses bfloat16 by default
    vae_dtype: str = "float32",
    device: str = "cuda:0",
    **kwargs
) -> DiffusionPipeline:
    """
    Load Wan2.2-T2V pipeline with mixed precision support.
    
    Args:
        model_id: HuggingFace model ID
        cache_dir: Cache directory for model weights
        torch_dtype: Default precision for model (DiT/UNet)
        vae_dtype: Precision for VAE (kept separate for stability)
        device: Target device
        **kwargs: Additional arguments for pipeline loading
        
    Returns:
        Loaded DiffusionPipeline
    """
    logger.info(f"Loading pipeline: {model_id}")
    logger.info(f"Default dtype: {torch_dtype}, VAE dtype: {vae_dtype}")
    
    # Convert dtype strings to torch dtypes
    dtype = get_torch_dtype(torch_dtype)
    vae_dt = get_torch_dtype(vae_dtype)
    
    # Load VAE separately with FP32 for numerical stability
    logger.info("Loading AutoencoderKLWan in FP32...")
    if HAS_WAN_PIPELINE and AutoencoderKLWan is not None:
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            cache_dir=cache_dir,
            torch_dtype=vae_dt
        )
    else:
        vae = None
    
    # Load pipeline with WanPipeline if available
    pipeline_class = WanPipeline if HAS_WAN_PIPELINE else DiffusionPipeline
    logger.info(f"Loading pipeline with {pipeline_class.__name__}...")
    
    pipe = pipeline_class.from_pretrained(
        model_id,
        vae=vae,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        **kwargs
    )
    
    # Move to device
    pipe = pipe.to(device)
    
    logger.info(f"Pipeline loaded successfully on {device}")
    logger.info(f"  Pipeline type: {type(pipe).__name__}")
    logger.info(f"  VAE type: {type(pipe.vae).__name__ if hasattr(pipe, 'vae') else 'N/A'}")
    logger.info(f"  Main dtype: {dtype}")
    logger.info(f"  VAE dtype: {vae_dt}")
    
    return pipe


def inspect_model(pipe: DiffusionPipeline) -> Dict[str, Any]:
    """
    Inspect pipeline components and their properties.
    
    Args:
        pipe: Loaded DiffusionPipeline
        
    Returns:
        Dictionary with component information
    """
    info = {
        "components": [],
        "model_type": type(pipe).__name__,
    }
    
    # Iterate over pipeline components
    for name in pipe.components.keys():
        component = getattr(pipe, name, None)
        if component is None:
            continue
            
        comp_info = {
            "name": name,
            "type": type(component).__name__,
            "dtype": None,
            "device": None,
            "num_parameters": None,
        }
        
        # Get dtype and device if it's a torch module
        if isinstance(component, torch.nn.Module):
            try:
                # Get first parameter to check dtype/device
                first_param = next(component.parameters())
                comp_info["dtype"] = str(first_param.dtype)
                comp_info["device"] = str(first_param.device)
                
                # Count parameters
                num_params = sum(p.numel() for p in component.parameters())
                comp_info["num_parameters"] = f"{num_params:,}"
            except StopIteration:
                pass
        
        info["components"].append(comp_info)
        
        logger.info(f"Component: {name} ({comp_info['type']})")
        if comp_info["dtype"]:
            logger.info(f"  dtype: {comp_info['dtype']}, device: {comp_info['device']}")
        if comp_info["num_parameters"]:
            logger.info(f"  parameters: {comp_info['num_parameters']}")
    
    return info


def get_submodule(
    pipe: DiffusionPipeline,
    component_name: str
) -> torch.nn.Module:
    """
    Extract a specific submodule from the pipeline.
    
    Args:
        pipe: Loaded DiffusionPipeline
        component_name: Name of component (e.g., 'transformer', 'unet', 'vae')
        
    Returns:
        Extracted submodule
    """
    component = getattr(pipe, component_name, None)
    
    if component is None:
        available = list(pipe.components.keys())
        raise ValueError(
            f"Component '{component_name}' not found. "
            f"Available components: {available}"
        )
    
    if not isinstance(component, torch.nn.Module):
        raise TypeError(
            f"Component '{component_name}' is not a torch.nn.Module "
            f"(type: {type(component).__name__})"
        )
    
    logger.info(f"Extracted submodule: {component_name} ({type(component).__name__})")
    
    return component

