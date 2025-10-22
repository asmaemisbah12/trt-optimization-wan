"""Quality metrics for video generation"""

import torch
import numpy as np
from typing import Union, Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_psnr(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: Optional[float] = None
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        pred: Predicted images/videos [B, C, H, W] or [B, C, T, H, W]
        target: Target images/videos
        data_range: Data range (default: max - min)
        
    Returns:
        PSNR value in dB
    """
    # Convert to numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Compute data range
    if data_range is None:
        data_range = max(target.max() - target.min(), pred.max() - pred.min())
    
    # Compute PSNR
    psnr_value = peak_signal_noise_ratio(
        target, pred, data_range=data_range
    )
    
    return float(psnr_value)


def compute_ssim(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: Optional[float] = None,
    multichannel: bool = True
) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        pred: Predicted images/videos
        target: Target images/videos
        data_range: Data range
        multichannel: Whether to treat as multichannel (RGB)
        
    Returns:
        SSIM value [0, 1]
    """
    # Convert to numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Compute data range
    if data_range is None:
        data_range = max(target.max() - target.min(), pred.max() - pred.min())
    
    # Handle batch dimension
    if pred.ndim == 5:  # [B, C, T, H, W]
        # Average over batch and time
        ssim_values = []
        for b in range(pred.shape[0]):
            for t in range(pred.shape[2]):
                # Get frame [C, H, W] -> [H, W, C]
                pred_frame = pred[b, :, t, :, :].transpose(1, 2, 0)
                target_frame = target[b, :, t, :, :].transpose(1, 2, 0)
                
                ssim_val = structural_similarity(
                    target_frame, pred_frame,
                    data_range=data_range,
                    channel_axis=2 if multichannel else None
                )
                ssim_values.append(ssim_val)
        
        return float(np.mean(ssim_values))
    
    elif pred.ndim == 4:  # [B, C, H, W]
        # Average over batch
        ssim_values = []
        for b in range(pred.shape[0]):
            # Get image [C, H, W] -> [H, W, C]
            pred_img = pred[b].transpose(1, 2, 0)
            target_img = target[b].transpose(1, 2, 0)
            
            ssim_val = structural_similarity(
                target_img, pred_img,
                data_range=data_range,
                channel_axis=2 if multichannel else None
            )
            ssim_values.append(ssim_val)
        
        return float(np.mean(ssim_values))
    
    else:
        raise ValueError(f"Unsupported input shape: {pred.shape}")


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "alex",
    device: str = "cuda"
) -> float:
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS).
    
    Args:
        pred: Predicted images/videos [B, C, H, W] or [B, C, T, H, W]
        target: Target images/videos
        net: Network for LPIPS ('alex', 'vgg', 'squeeze')
        device: Compute device
        
    Returns:
        LPIPS distance (lower is better)
    """
    try:
        import lpips
        
        # Initialize LPIPS model
        loss_fn = lpips.LPIPS(net=net).to(device)
        
        # Move to device
        pred = pred.to(device)
        target = target.to(device)
        
        # Handle video format
        if pred.ndim == 5:  # [B, C, T, H, W]
            # Average over time dimension
            lpips_values = []
            for t in range(pred.shape[2]):
                pred_frame = pred[:, :, t, :, :]
                target_frame = target[:, :, t, :, :]
                
                with torch.no_grad():
                    lpips_val = loss_fn(pred_frame, target_frame)
                
                lpips_values.append(lpips_val.mean().item())
            
            return float(np.mean(lpips_values))
        
        else:  # [B, C, H, W]
            with torch.no_grad():
                lpips_val = loss_fn(pred, target)
            
            return float(lpips_val.mean().item())
    
    except ImportError:
        logger.warning("lpips not installed, skipping LPIPS computation")
        return -1.0


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: Optional[float] = None,
    compute_lpips_metric: bool = True
) -> dict:
    """
    Compute all quality metrics.
    
    Args:
        pred: Predicted output
        target: Ground truth
        data_range: Data range for metrics
        compute_lpips_metric: Whether to compute LPIPS
        
    Returns:
        Dictionary of metric values
    """
    metrics = {}
    
    # PSNR
    try:
        metrics["psnr"] = compute_psnr(pred, target, data_range)
    except Exception as e:
        logger.warning(f"Failed to compute PSNR: {e}")
        metrics["psnr"] = -1.0
    
    # SSIM
    try:
        metrics["ssim"] = compute_ssim(pred, target, data_range)
    except Exception as e:
        logger.warning(f"Failed to compute SSIM: {e}")
        metrics["ssim"] = -1.0
    
    # LPIPS
    if compute_lpips_metric:
        try:
            metrics["lpips"] = compute_lpips(pred, target)
        except Exception as e:
            logger.warning(f"Failed to compute LPIPS: {e}")
            metrics["lpips"] = -1.0
    
    return metrics

