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
        
        # Change to ONNX file directory to find external data files
        onnx_path = Path(self.onnx_path)
        original_cwd = Path.cwd()
        
        try:
            # Change to ONNX file directory
            import os
            os.chdir(onnx_path.parent)
            logger.info(f"Changed to ONNX directory: {onnx_path.parent}")
            
            # Parse ONNX model
            with open(onnx_path.name, 'rb') as f:
                if not self.parser.parse(f.read()):
                    for error in range(self.parser.num_errors):
                        logger.error(f"ONNX parse error: {self.parser.get_error(error)}")
                    raise RuntimeError("Failed to parse ONNX model")
        
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
        
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
            try:
                self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)
                logger.info("Strict type checking enabled")
            except AttributeError:
                logger.warning("STRICT_TYPES flag not supported in this TensorRT version, skipping")
        
        # Tactic heuristic for faster builds
        if self.enable_tactic_heuristic:
            self.config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
        
        # Additional flags for better compatibility
        try:
            # Enable more aggressive optimization
            self.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            logger.info("Precision constraints enabled")
        except AttributeError:
            logger.warning("PREFER_PRECISION_CONSTRAINTS not supported, skipping")
        
        try:
            # Allow TensorRT to use more memory for optimization
            self.config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)
            logger.info("Empty algorithms rejection enabled")
        except AttributeError:
            logger.warning("REJECT_EMPTY_ALGORITHMS not supported, skipping")
        
        # Try to enable more permissive settings for unsupported operations
        try:
            # Allow TensorRT to be more flexible with unsupported operations
            self.config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            logger.info("Precision constraints obedience enabled")
        except AttributeError:
            logger.warning("OBEY_PRECISION_CONSTRAINTS not supported, skipping")
        
        # Try to disable some constraints that might help with unsupported operations
        try:
            # Disable precision constraints to be more permissive
            self.config.clear_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            logger.info("Precision constraints disabled for better compatibility")
        except AttributeError:
            logger.warning("Cannot clear PREFER_PRECISION_CONSTRAINTS, skipping")
        
        # Add more aggressive compatibility flags
        try:
            # Allow TensorRT to be more flexible with layer fusion
            self.config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
            logger.info("Timing cache disabled for better compatibility")
        except AttributeError:
            logger.warning("Cannot disable timing cache, skipping")
        
        try:
            # Enable more permissive layer fusion
            self.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            logger.info("Precision constraints re-enabled for compatibility")
        except AttributeError:
            logger.warning("Cannot re-enable precision constraints, skipping")
        
        # Set maximum layers for better compatibility
        try:
            self.config.max_layers = 1000
            logger.info("Max layers set to 1000")
        except AttributeError:
            logger.warning("max_layers not supported, skipping")
        
        # Try to disable problematic optimizations
        try:
            # Disable some optimizations that might cause issues with complex operations
            self.config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
            logger.info("Timing cache disabled for compatibility")
        except AttributeError:
            logger.warning("Cannot disable timing cache, skipping")
        
        # Try to set more permissive layer fusion
        try:
            # Allow more aggressive layer fusion
            self.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            logger.info("Precision constraints enabled for better fusion")
        except AttributeError:
            logger.warning("Cannot set precision constraints, skipping")
        
        # Try to disable layer fusion that might cause issues with complex operations
        try:
            # Disable some layer fusion that might be problematic
            self.config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
            logger.info("Timing cache disabled to avoid fusion issues")
        except AttributeError:
            logger.warning("Cannot disable timing cache, skipping")
        
        # Try to enable more permissive settings
        try:
            # Allow TensorRT to be more flexible with unsupported operations
            self.config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)
            logger.info("Empty algorithms rejection enabled for better compatibility")
        except AttributeError:
            logger.warning("Cannot set empty algorithms rejection, skipping")
        
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
        
        # Store profile shapes for fallback use
        self._profile_shapes = profile_shapes
        
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
        """Build TensorRT engine with fallback strategies."""
        logger.info("Building TensorRT engine (this may take several minutes)...")
        
        if self.network is None:
            raise RuntimeError("Network not created. Call create_network() first.")
        
        if self.config is None:
            raise RuntimeError("Config not created. Call create_config() first.")
        
        # Try building with current configuration
        try:
            engine = self.builder.build_serialized_network(self.network, self.config)
            if engine is not None:
                logger.info("Engine built successfully")
                return engine
        except Exception as e:
            logger.warning(f"Initial build failed: {e}")
        
        # Fallback 1: Try with smaller workspace size (keep FP16)
        logger.info("Trying fallback: Smaller workspace size...")
        try:
            smaller_config = self.builder.create_builder_config()
            smaller_workspace = self.workspace_size // 2  # Half the workspace
            smaller_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, smaller_workspace)
            
            # Set precision
            if self.precision == "fp16":
                smaller_config.set_flag(trt.BuilderFlag.FP16)
            
            # Add optimization profile if it exists
            if hasattr(self, '_profile_shapes') and self._profile_shapes:
                profile = self.builder.create_optimization_profile()
                for input_name, (min_shape, opt_shape, max_shape) in self._profile_shapes.items():
                    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                smaller_config.add_optimization_profile(profile)
            
            engine = self.builder.build_serialized_network(self.network, smaller_config)
            if engine is not None:
                logger.info("Engine built successfully with smaller workspace")
                return engine
        except Exception as e:
            logger.warning(f"Smaller workspace fallback failed: {e}")
        
        # Fallback 2: Try with minimal workspace size
        logger.info("Trying fallback: Minimal workspace size...")
        try:
            minimal_config = self.builder.create_builder_config()
            minimal_workspace = self.workspace_size // 4  # Quarter the workspace
            minimal_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, minimal_workspace)
            
            # Set precision
            if self.precision == "fp16":
                minimal_config.set_flag(trt.BuilderFlag.FP16)
            
            # Add optimization profile if it exists
            if hasattr(self, '_profile_shapes') and self._profile_shapes:
                profile = self.builder.create_optimization_profile()
                for input_name, (min_shape, opt_shape, max_shape) in self._profile_shapes.items():
                    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                minimal_config.add_optimization_profile(profile)
            
            engine = self.builder.build_serialized_network(self.network, minimal_config)
            if engine is not None:
                logger.info("Engine built successfully with minimal workspace")
                return engine
        except Exception as e:
            logger.warning(f"Minimal workspace fallback failed: {e}")
        
        # Fallback 3: Try with even more aggressive TensorRT settings
        logger.info("Trying fallback: Ultra-aggressive TensorRT settings...")
        try:
            ultra_config = self.builder.create_builder_config()
            ultra_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_size // 8)  # Very small workspace
            
            # Set precision
            if self.precision == "fp16":
                ultra_config.set_flag(trt.BuilderFlag.FP16)
            
            # Disable all problematic optimizations
            try:
                ultra_config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
                ultra_config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)
                logger.info("Disabled timing cache and empty algorithms")
            except AttributeError:
                pass
            
            # Add optimization profile if it exists
            if hasattr(self, '_profile_shapes') and self._profile_shapes:
                profile = self.builder.create_optimization_profile()
                for input_name, (min_shape, opt_shape, max_shape) in self._profile_shapes.items():
                    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                ultra_config.add_optimization_profile(profile)
            
            engine = self.builder.build_serialized_network(self.network, ultra_config)
            if engine is not None:
                logger.info("Engine built successfully with ultra-aggressive settings")
                return engine
        except Exception as e:
            logger.warning(f"Ultra-aggressive fallback failed: {e}")
        
        # If all fallbacks fail
        raise RuntimeError("Failed to build TensorRT engine with all fallback strategies")
    
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
    
    # Build and save engine (fallbacks are handled within the class)
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
    
    # Log engine info (TensorRT 10.x API)
    logger.info(f"  Num I/O tensors: {engine.num_io_tensors}")
    logger.info(f"  Num optimization profiles: {engine.num_optimization_profiles}")
    
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_shape = engine.get_tensor_shape(tensor_name)
        tensor_mode = engine.get_tensor_mode(tensor_name)
        tensor_dtype = engine.get_tensor_dtype(tensor_name)
        logger.info(f"  Tensor {i}: {tensor_name} - {tensor_mode} - {tensor_shape} - {tensor_dtype}")
    
    return engine

