"""TensorRT engine building with multi-profile support"""

import tensorrt as trt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.utils.logging import get_logger
from src.utils.config import ensure_directory

logger = get_logger(__name__)

# TRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class EngineBuilder:
    """
    TensorRT engine builder with multi-profile support.
    """
    
    def __init__(
        self,
        onnx_path: str,
        precision: str = "fp16",
        workspace_size: int = 8192,
        strict_types: bool = True,
        enable_tactic_heuristic: bool = True
    ):
        """
        Initialize engine builder.
        
        Args:
            onnx_path: Path to ONNX model
            precision: Precision mode ('fp32', 'fp16', 'int8')
            workspace_size: Max workspace size in MB
            strict_types: Enforce strict type constraints
            enable_tactic_heuristic: Enable tactic heuristic for faster build
        """
        self.onnx_path = onnx_path
        self.precision = precision.lower()
        self.workspace_size = workspace_size * (1 << 20)  # Convert MB to bytes
        self.strict_types = strict_types
        self.enable_tactic_heuristic = enable_tactic_heuristic
        
        # Initialize TRT builder and network
        self.builder = trt.Builder(TRT_LOGGER)
        self.network = None
        self.config = None
        self.parser = None
        
        logger.info(f"EngineBuilder initialized for {onnx_path}")
        logger.info(f"  Precision: {precision}")
        logger.info(f"  Workspace size: {workspace_size} MB")
    
    def create_network(self) -> trt.INetworkDefinition:
        """Create TensorRT network from ONNX model."""
        logger.info("Creating TensorRT network from ONNX...")
        
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, TRT_LOGGER)
        
        # Parse ONNX model
        with open(self.onnx_path, 'rb') as f:
            if not self.parser.parse(f.read()):
                for error in range(self.parser.num_errors):
                    logger.error(f"ONNX parse error: {self.parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")
        
        logger.info(f"Network created successfully")
        logger.info(f"  Inputs: {self.network.num_inputs}")
        logger.info(f"  Outputs: {self.network.num_outputs}")
        
        return self.network
    
    def create_config(self) -> trt.IBuilderConfig:
        """Create builder configuration."""
        logger.info("Creating builder configuration...")
        
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_size)
        
        # Set precision
        if self.precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                logger.warning("FP16 not supported on this platform, using FP32")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 precision enabled")
        
        elif self.precision == "int8":
            if not self.builder.platform_has_fast_int8:
                logger.warning("INT8 not supported on this platform, using FP16/FP32")
            else:
                self.config.set_flag(trt.BuilderFlag.INT8)
                logger.info("INT8 precision enabled (calibrator required)")
        
        # Strict types (recommended for mixed precision)
        if self.strict_types:
            self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            logger.info("Strict type checking enabled")
        
        # Tactic heuristic for faster builds
        if self.enable_tactic_heuristic:
            self.config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
        
        return self.config
    
    def add_optimization_profile(
        self,
        profile_shapes: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]
    ) -> None:
        """
        Add optimization profile with min/opt/max shapes.
        
        Args:
            profile_shapes: Dict mapping input names to (min_shape, opt_shape, max_shape)
        """
        logger.info("Adding optimization profile...")
        
        profile = self.builder.create_optimization_profile()
        
        for input_idx in range(self.network.num_inputs):
            input_tensor = self.network.get_input(input_idx)
            input_name = input_tensor.name
            
            if input_name in profile_shapes:
                min_shape, opt_shape, max_shape = profile_shapes[input_name]
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                
                logger.info(f"  {input_name}:")
                logger.info(f"    min: {min_shape}")
                logger.info(f"    opt: {opt_shape}")
                logger.info(f"    max: {max_shape}")
            else:
                logger.warning(f"No profile shape specified for input: {input_name}")
        
        self.config.add_optimization_profile(profile)
    
    def build_engine(self) -> trt.ICudaEngine:
        """Build TensorRT engine."""
        logger.info("Building TensorRT engine (this may take several minutes)...")
        
        if self.network is None:
            raise RuntimeError("Network not created. Call create_network() first.")
        
        if self.config is None:
            raise RuntimeError("Config not created. Call create_config() first.")
        
        engine = self.builder.build_serialized_network(self.network, self.config)
        
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        logger.info("Engine built successfully")
        
        return engine
    
    def save_engine(self, engine: trt.ICudaEngine, output_path: str) -> str:
        """
        Save serialized engine to file.
        
        Args:
            engine: Serialized TensorRT engine
            output_path: Output file path
            
        Returns:
            Path to saved engine
        """
        output_path = Path(output_path)
        ensure_directory(output_path.parent)
        
        with open(output_path, 'wb') as f:
            f.write(engine)
        
        logger.info(f"Engine saved to: {output_path}")
        logger.info(f"  Size: {output_path.stat().st_size / (1024**2):.2f} MB")
        
        return str(output_path)


def build_engine(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    workspace_size: int = 8192,
    profile_shapes: Optional[Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]] = None,
    strict_types: bool = True
) -> str:
    """
    Build TensorRT engine from ONNX model (convenience function).
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Output engine file path
        precision: Precision mode ('fp32', 'fp16', 'int8')
        workspace_size: Max workspace size in MB
        profile_shapes: Dict mapping input names to (min_shape, opt_shape, max_shape)
        strict_types: Enforce strict type constraints
        
    Returns:
        Path to built engine
    """
    builder = EngineBuilder(
        onnx_path=onnx_path,
        precision=precision,
        workspace_size=workspace_size,
        strict_types=strict_types
    )
    
    # Create network and config
    builder.create_network()
    builder.create_config()
    
    # Add optimization profile if provided
    if profile_shapes:
        builder.add_optimization_profile(profile_shapes)
    
    # Build and save engine
    engine = builder.build_engine()
    engine_path = builder.save_engine(engine, output_path)
    
    return engine_path


def load_engine(engine_path: str) -> trt.ICudaEngine:
    """
    Load TensorRT engine from file.
    
    Args:
        engine_path: Path to engine file
        
    Returns:
        Deserialized TensorRT engine
    """
    logger.info(f"Loading TensorRT engine: {engine_path}")
    
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    engine = runtime.deserialize_cuda_engine(engine_data)
    
    if engine is None:
        raise RuntimeError(f"Failed to load engine from {engine_path}")
    
    logger.info("Engine loaded successfully")
    
    # Log engine info
    logger.info(f"  Num bindings: {engine.num_bindings}")
    logger.info(f"  Num optimization profiles: {engine.num_optimization_profiles}")
    
    for i in range(engine.num_bindings):
        binding_name = engine.get_binding_name(i)
        binding_shape = engine.get_binding_shape(i)
        is_input = engine.binding_is_input(i)
        logger.info(f"  Binding {i}: {binding_name} - {'input' if is_input else 'output'} - {binding_shape}")
    
    return engine

