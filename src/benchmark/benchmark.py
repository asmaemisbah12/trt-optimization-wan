"""Benchmarking framework for model comparison"""

import time
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import json
from src.utils.logging import get_logger
from .metrics import compute_all_metrics

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    
    name: str
    num_runs: int
    
    # Timing metrics (seconds)
    mean_time: float = 0.0
    std_time: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0
    median_time: float = 0.0
    total_time: float = 0.0
    
    # Throughput
    fps: float = 0.0  # frames per second
    
    # Memory metrics (MB)
    peak_memory: float = 0.0
    allocated_memory: float = 0.0
    
    # Quality metrics
    psnr: float = -1.0
    ssim: float = -1.0
    lpips: float = -1.0
    
    # Additional info
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """String representation."""
        lines = [
            f"\n{'='*60}",
            f"Benchmark Results: {self.name}",
            f"{'='*60}",
            f"Runs: {self.num_runs}",
            f"\nTiming (seconds):",
            f"  Mean:   {self.mean_time:.4f} ± {self.std_time:.4f}",
            f"  Median: {self.median_time:.4f}",
            f"  Min:    {self.min_time:.4f}",
            f"  Max:    {self.max_time:.4f}",
            f"  Total:  {self.total_time:.2f}",
            f"\nThroughput:",
            f"  FPS:    {self.fps:.2f}",
            f"\nMemory (MB):",
            f"  Peak:      {self.peak_memory:.2f}",
            f"  Allocated: {self.allocated_memory:.2f}",
        ]
        
        if self.psnr > 0:
            lines.extend([
                f"\nQuality Metrics:",
                f"  PSNR:  {self.psnr:.2f} dB",
                f"  SSIM:  {self.ssim:.4f}",
                f"  LPIPS: {self.lpips:.4f}",
            ])
        
        lines.append(f"{'='*60}\n")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "num_runs": self.num_runs,
            "timing": {
                "mean": self.mean_time,
                "std": self.std_time,
                "min": self.min_time,
                "max": self.max_time,
                "median": self.median_time,
                "total": self.total_time,
            },
            "throughput": {
                "fps": self.fps,
            },
            "memory": {
                "peak_mb": self.peak_memory,
                "allocated_mb": self.allocated_memory,
            },
            "quality": {
                "psnr": self.psnr,
                "ssim": self.ssim,
                "lpips": self.lpips,
            },
            "metadata": self.metadata,
        }
    
    def save(self, output_path: str) -> None:
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Results saved to: {output_path}")


class Benchmark:
    """
    Benchmarking framework for model performance evaluation.
    """
    
    def __init__(
        self,
        name: str,
        device: str = "cuda:0",
        num_warmup_runs: int = 3,
        num_test_runs: int = 10
    ):
        """
        Initialize benchmark.
        
        Args:
            name: Benchmark name
            device: Target device
            num_warmup_runs: Number of warmup runs
            num_test_runs: Number of test runs
        """
        self.name = name
        self.device = torch.device(device)
        self.num_warmup_runs = num_warmup_runs
        self.num_test_runs = num_test_runs
        
        logger.info(f"Benchmark initialized: {name}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Warmup runs: {num_warmup_runs}")
        logger.info(f"  Test runs: {num_test_runs}")
    
    def run(
        self,
        inference_fn: Callable,
        inputs: Dict[str, Any],
        reference_outputs: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run benchmark on inference function.
        
        Args:
            inference_fn: Function to benchmark (takes inputs dict, returns outputs)
            inputs: Input dictionary for inference
            reference_outputs: Optional reference outputs for quality metrics
            num_frames: Number of frames for FPS calculation
            **kwargs: Additional metadata
            
        Returns:
            BenchmarkResult
        """
        logger.info(f"Running benchmark: {self.name}")
        
        result = BenchmarkResult(
            name=self.name,
            num_runs=self.num_test_runs,
            metadata=kwargs
        )
        
        # Warmup runs
        logger.info(f"Warmup: {self.num_warmup_runs} runs...")
        for i in range(self.num_warmup_runs):
            _ = inference_fn(inputs)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        
        # Test runs
        logger.info(f"Testing: {self.num_test_runs} runs...")
        times = []
        
        for i in range(self.num_test_runs):
            # Reset memory stats
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize(self.device)
            
            # Time inference
            start_time = time.perf_counter()
            outputs = inference_fn(inputs)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            
            # Memory stats
            if self.device.type == "cuda":
                peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                alloc_mem = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                
                result.peak_memory = max(result.peak_memory, peak_mem)
                result.allocated_memory = max(result.allocated_memory, alloc_mem)
            
            logger.debug(f"  Run {i+1}/{self.num_test_runs}: {elapsed:.4f}s")
        
        # Compute timing statistics
        times = np.array(times)
        result.mean_time = float(np.mean(times))
        result.std_time = float(np.std(times))
        result.min_time = float(np.min(times))
        result.max_time = float(np.max(times))
        result.median_time = float(np.median(times))
        result.total_time = float(np.sum(times))
        
        # Compute FPS
        if num_frames:
            result.fps = num_frames / result.mean_time
        
        # Compute quality metrics
        if reference_outputs is not None:
            logger.info("Computing quality metrics...")
            try:
                metrics = compute_all_metrics(outputs, reference_outputs)
                result.psnr = metrics.get("psnr", -1.0)
                result.ssim = metrics.get("ssim", -1.0)
                result.lpips = metrics.get("lpips", -1.0)
            except Exception as e:
                logger.warning(f"Failed to compute quality metrics: {e}")
        
        logger.info(f"Benchmark complete:")
        logger.info(f"  Mean time: {result.mean_time:.4f}s")
        logger.info(f"  FPS: {result.fps:.2f}")
        logger.info(f"  Peak memory: {result.peak_memory:.2f} MB")
        
        return result
    
    @staticmethod
    def compare(
        results: List[BenchmarkResult],
        baseline_name: Optional[str] = None
    ) -> str:
        """
        Compare multiple benchmark results.
        
        Args:
            results: List of benchmark results
            baseline_name: Name of baseline for speedup calculation
            
        Returns:
            Formatted comparison string
        """
        if not results:
            return "No results to compare"
        
        # Find baseline
        baseline = None
        if baseline_name:
            for r in results:
                if r.name == baseline_name:
                    baseline = r
                    break
        
        if baseline is None:
            baseline = results[0]
        
        # Build comparison table
        lines = [
            f"\n{'='*80}",
            f"Benchmark Comparison (Baseline: {baseline.name})",
            f"{'='*80}",
            f"{'Name':<25} {'Time (s)':<12} {'Speedup':<10} {'FPS':<10} {'Memory (MB)':<12}",
            f"{'-'*80}",
        ]
        
        for result in results:
            speedup = baseline.mean_time / result.mean_time if result.mean_time > 0 else 0.0
            
            lines.append(
                f"{result.name:<25} "
                f"{result.mean_time:<12.4f} "
                f"{speedup:<10.2f}x "
                f"{result.fps:<10.2f} "
                f"{result.peak_memory:<12.2f}"
            )
        
        lines.append(f"{'='*80}\n")
        
        return "\n".join(lines)

