"""TensorRT runtime with CUDA graph support"""

import tensorrt as trt
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TRTInference:
    """
    TensorRT inference wrapper with memory management and CUDA graph support.
    """
    
    def __init__(
        self,
        engine: trt.ICudaEngine,
        device: str = "cuda:0",
        enable_cuda_graph: bool = False
    ):
        """
        Initialize TRT inference runtime.
        
        Args:
            engine: TensorRT engine
            device: CUDA device
            enable_cuda_graph: Enable CUDA graph capture
        """
        self.engine = engine
        self.device = torch.device(device)
        self.enable_cuda_graph = enable_cuda_graph
        
        # Create execution context
        try:
            self.context = engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Failed to create execution context - engine.create_execution_context() returned None")
        except Exception as e:
            logger.error(f"Failed to create execution context: {e}")
            self.context = None
            raise
        
        # Binding information
        self.input_names = []
        self.output_names = []
        self.tensor_shapes = {}
        
        # Parse I/O tensors (TensorRT 10.x API)
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            mode = engine.get_tensor_mode(name)
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
            
            self.tensor_shapes[name] = shape
        
        logger.info(f"TRTInference initialized")
        logger.info(f"  Inputs: {self.input_names}")
        logger.info(f"  Outputs: {self.output_names}")
        
        # CUDA graph
        self.cuda_graph = None
        self.cuda_graph_stream = None
        
    def allocate_buffers(
        self,
        input_shapes: Dict[str, Tuple[int, ...]]
    ) -> Dict[str, torch.Tensor]:
        """
        Allocate device buffers for inputs and outputs.
        
        Args:
            input_shapes: Dictionary mapping input names to actual shapes
            
        Returns:
            Dictionary of allocated tensors
        """
        buffers = {}
        
        # Allocate input buffers
        for name in self.input_names:
            if name not in input_shapes:
                raise ValueError(f"Missing shape for input: {name}")
            
            shape = input_shapes[name]
            
            # Set tensor shape for dynamic shapes (TensorRT 10.x API)
            logger.debug(f"Setting input shape for {name}: {shape}")
            self.context.set_input_shape(name, shape)
            
            # Allocate tensor
            dtype = self._trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            buffers[name] = tensor
            
            logger.debug(f"Allocated input buffer: {name} {shape} {dtype}")
        
        # Allocate output buffers
        for name in self.output_names:
            # Get output shape (may depend on input shapes)
            output_shape = self.context.get_tensor_shape(name)
            
            # Handle dynamic dimensions (replace -1 with actual values)
            actual_shape = []
            for i, dim in enumerate(output_shape):
                if dim == -1:
                    # For dynamic dimensions, we need to infer from input shapes
                    # This is a simplified approach - in practice, you'd need to know
                    # the relationship between input and output dynamic dimensions
                    if i == 2:  # Assuming frame dimension
                        # Get frame dimension from sample input
                        sample_shape = input_shapes.get('sample', (1, 16, 16, 45, 80))
                        actual_shape.append(sample_shape[2])  # Use frame dimension from sample
                    else:
                        actual_shape.append(1)  # Default fallback
                else:
                    actual_shape.append(dim)
            
            # Allocate tensor
            dtype = self._trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            tensor = torch.empty(tuple(actual_shape), dtype=dtype, device=self.device)
            buffers[name] = tensor
            
            logger.debug(f"Allocated output buffer: {name} {tuple(actual_shape)} {dtype}")
        
        return buffers
    
    def infer(
        self,
        inputs: Dict[str, torch.Tensor],
        stream: Optional[torch.cuda.Stream] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference.
        
        Args:
            inputs: Dictionary of input tensors
            stream: Optional CUDA stream
            
        Returns:
            Dictionary of output tensors
        """
        # Allocate buffers based on input shapes
        input_shapes = {name: tuple(tensor.shape) for name, tensor in inputs.items()}
        buffers = self.allocate_buffers(input_shapes)
        
        # Copy inputs to buffers
        for name, tensor in inputs.items():
            buffers[name].copy_(tensor)
        
        # Set input tensors (TensorRT 10.x API)
        for name, tensor in inputs.items():
            # Ensure tensor is contiguous and on the correct device
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            self.context.set_tensor_address(name, tensor.data_ptr())
        
        # Set output tensors
        for name in self.output_names:
            # Ensure tensor is contiguous
            if not buffers[name].is_contiguous():
                buffers[name] = buffers[name].contiguous()
            self.context.set_tensor_address(name, buffers[name].data_ptr())
        
        # Execute inference
        if stream is None:
            stream = torch.cuda.current_stream(self.device)
        
        success = self.context.execute_async_v3(stream.cuda_stream)
        
        if not success:
            raise RuntimeError("TensorRT inference failed")
        
        # Synchronize stream
        stream.synchronize()
        
        # Return outputs
        outputs = {name: buffers[name] for name in self.output_names}
        
        return outputs
    
    def capture_cuda_graph(
        self,
        inputs: Dict[str, torch.Tensor],
        warmup_runs: int = 3
    ) -> None:
        """
        Capture CUDA graph for repeated inference.
        
        Args:
            inputs: Sample inputs for graph capture
            warmup_runs: Number of warmup runs before capture
        """
        if not self.enable_cuda_graph:
            logger.warning("CUDA graph not enabled, skipping capture")
            return
        
        logger.info(f"Capturing CUDA graph (warmup: {warmup_runs} runs)...")
        
        # Allocate buffers
        input_shapes = {name: tuple(tensor.shape) for name, tensor in inputs.items()}
        self.graph_buffers = self.allocate_buffers(input_shapes)
        
        # Create dedicated stream
        self.cuda_graph_stream = torch.cuda.Stream(self.device)
        
        # Warmup runs
        with torch.cuda.stream(self.cuda_graph_stream):
            for _ in range(warmup_runs):
                for name, tensor in inputs.items():
                    self.graph_buffers[name].copy_(tensor)
                
                self.context.execute_async_v2(
                    bindings=self.bindings,
                    stream_handle=self.cuda_graph_stream.cuda_stream
                )
        
        self.cuda_graph_stream.synchronize()
        
        # Capture graph
        self.cuda_graph = torch.cuda.CUDAGraph()
        
        with torch.cuda.graph(self.cuda_graph, stream=self.cuda_graph_stream):
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.cuda_graph_stream.cuda_stream
            )
        
        logger.info("CUDA graph captured successfully")
    
    def infer_with_graph(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference using captured CUDA graph.
        
        Args:
            inputs: Dictionary of input tensors (must match capture shapes)
            
        Returns:
            Dictionary of output tensors
        """
        if self.cuda_graph is None:
            raise RuntimeError("CUDA graph not captured. Call capture_cuda_graph() first.")
        
        # Copy inputs to graph buffers
        for name, tensor in inputs.items():
            self.graph_buffers[name].copy_(tensor)
        
        # Replay graph
        self.cuda_graph.replay()
        
        # Synchronize
        self.cuda_graph_stream.synchronize()
        
        # Return outputs
        outputs = {name: self.graph_buffers[name] for name in self.output_names}
        
        return outputs
    
    @staticmethod
    def _trt_dtype_to_torch(trt_dtype: trt.DataType) -> torch.dtype:
        """Convert TensorRT dtype to PyTorch dtype."""
        dtype_map = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
            trt.bool: torch.bool,
        }
        
        return dtype_map.get(trt_dtype, torch.float32)


def create_execution_context(engine_path: str, **kwargs) -> TRTInference:
    """
    Create TRT inference context from engine file (convenience function).
    
    Args:
        engine_path: Path to TensorRT engine
        **kwargs: Additional arguments for TRTInference
        
    Returns:
        TRTInference instance
    """
    from .engine_builder import load_engine
    
    engine = load_engine(engine_path)
    return TRTInference(engine, **kwargs)

