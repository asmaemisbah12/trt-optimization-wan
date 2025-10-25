"""ONNX export utilities for model components"""

import torch
import onnx
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from src.utils.logging import get_logger
from src.utils.config import ensure_directory

logger = get_logger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    dummy_inputs: Dict[str, torch.Tensor],
    output_path: str,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 17,
    do_constant_folding: bool = True,
    verbose: bool = False
) -> str:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        dummy_inputs: Dictionary of dummy input tensors
        output_path: Output ONNX file path
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic axis specifications
        opset_version: ONNX opset version
        do_constant_folding: Enable constant folding optimization
        verbose: Enable verbose logging
        
    Returns:
        Path to exported ONNX file
    """
    logger.info(f"Exporting model to ONNX: {output_path}")
    
    # Ensure output directory exists
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    
    # Set model to eval mode
    model.eval()
    
    # Convert dict inputs to tuple for ONNX export
    if input_names is None:
        input_names = list(dummy_inputs.keys())
    
    input_tuple = tuple(dummy_inputs[name] for name in input_names)
    
    # Default output names
    if output_names is None:
        output_names = ["output"]
    
    logger.info(f"Input names: {input_names}")
    logger.info(f"Output names: {output_names}")
    logger.info(f"Dynamic axes: {dynamic_axes}")
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                input_tuple,
                str(output_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=do_constant_folding,
                verbose=verbose,
                # Additional options for better TensorRT compatibility
                export_params=True,
                keep_initializers_as_inputs=False,
                training=torch.onnx.TrainingMode.EVAL,
            )
        
        logger.info(f"Successfully exported to {output_path}")
        
        # Validate the exported model
        validate_onnx(str(output_path))
        
        return str(output_path)
        
    except Exception as e:
        error_msg = str(e)
        if "same reduction dim" in error_msg and "X" in error_msg:
            logger.warning(f"Matrix dimension mismatch detected: {error_msg}")
            logger.info("This suggests the dummy input dimensions don't match the model's expected input sizes.")
            logger.info("Please check the detected dimensions and adjust the dummy input generation.")
        
        logger.error(f"ONNX export failed: {e}")
        raise


def validate_onnx(onnx_path: str) -> bool:
    """
    Validate ONNX model.
    
    Args:
        onnx_path: Path to ONNX file
        
    Returns:
        True if valid
        
    Raises:
        Exception if validation fails
    """
    logger.info(f"Validating ONNX model: {onnx_path}")
    
    try:
        # Load and check model
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        
        # Run shape inference
        inferred_model = onnx.shape_inference.infer_shapes(model)
        
        # Log model info
        logger.info(f"ONNX model valid")
        logger.info(f"  IR version: {model.ir_version}")
        logger.info(f"  Producer: {model.producer_name} {model.producer_version}")
        logger.info(f"  Opset version: {model.opset_import[0].version}")
        
        # Log inputs and outputs
        logger.info(f"  Inputs:")
        for inp in model.graph.input:
            logger.info(f"    {inp.name}: {_get_tensor_shape(inp)}")
        
        logger.info(f"  Outputs:")
        for out in model.graph.output:
            logger.info(f"    {out.name}: {_get_tensor_shape(out)}")
        
        return True
        
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
        raise


def _get_tensor_shape(tensor) -> str:
    """Extract shape information from ONNX tensor."""
    try:
        shape = []
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            elif dim.dim_value:
                shape.append(str(dim.dim_value))
            else:
                shape.append("?")
        return f"[{', '.join(shape)}]"
    except:
        return "unknown"


def optimize_onnx(
    onnx_path: str,
    output_path: Optional[str] = None,
    optimization_level: str = "basic"
) -> str:
    """
    Optimize ONNX model using onnxruntime.
    
    Args:
        onnx_path: Input ONNX file path
        output_path: Output path (defaults to input path with _opt suffix)
        optimization_level: 'basic', 'extended', or 'all'
        
    Returns:
        Path to optimized ONNX file
    """
    try:
        import onnxruntime as ort
        from onnxruntime.transformers import optimizer
        
        logger.info(f"Optimizing ONNX model: {onnx_path}")
        
        if output_path is None:
            p = Path(onnx_path)
            output_path = p.parent / f"{p.stem}_opt{p.suffix}"
        
        # Set optimization level
        opt_level_map = {
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        
        opt_level = opt_level_map.get(optimization_level, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
        
        # Create optimization session options
        sess_options = ort.SessionOptions()
        sess_options.optimized_model_filepath = str(output_path)
        sess_options.graph_optimization_level = opt_level
        
        # Run optimization
        _ = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
        
        logger.info(f"Optimized ONNX saved to: {output_path}")
        
        return str(output_path)
        
    except ImportError:
        logger.warning("onnxruntime not available, skipping optimization")
        return onnx_path
    except Exception as e:
        logger.error(f"ONNX optimization failed: {e}")
        return onnx_path


def compare_onnx_outputs(
    model: torch.nn.Module,
    onnx_path: str,
    dummy_inputs: Dict[str, torch.Tensor],
    input_names: List[str],
    rtol: float = 1e-3,
    atol: float = 1e-5
) -> Tuple[bool, Dict[str, float]]:
    """
    Compare PyTorch model outputs with ONNX model outputs.
    
    Args:
        model: Original PyTorch model
        onnx_path: Path to ONNX model
        dummy_inputs: Dictionary of input tensors
        input_names: Names of inputs
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Tuple of (match, error_metrics)
    """
    try:
        import onnxruntime as ort
        import numpy as np
        
        logger.info("Comparing PyTorch and ONNX outputs...")
        
        # Get PyTorch output
        model.eval()
        with torch.no_grad():
            input_tuple = tuple(dummy_inputs[name] for name in input_names)
            torch_output = model(*input_tuple)
            
            if isinstance(torch_output, tuple):
                torch_output = torch_output[0]
            
            torch_output_np = torch_output.cpu().numpy()
        
        # Get ONNX output
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        onnx_inputs = {name: dummy_inputs[name].cpu().numpy() for name in input_names}
        onnx_outputs = sess.run(None, onnx_inputs)
        onnx_output_np = onnx_outputs[0]
        
        # Compare
        max_diff = np.max(np.abs(torch_output_np - onnx_output_np))
        mean_diff = np.mean(np.abs(torch_output_np - onnx_output_np))
        rel_diff = max_diff / (np.max(np.abs(torch_output_np)) + 1e-8)
        
        match = np.allclose(torch_output_np, onnx_output_np, rtol=rtol, atol=atol)
        
        error_metrics = {
            "max_abs_diff": float(max_diff),
            "mean_abs_diff": float(mean_diff),
            "max_rel_diff": float(rel_diff),
        }
        
        logger.info(f"Output comparison:")
        logger.info(f"  Max abs diff: {max_diff:.6e}")
        logger.info(f"  Mean abs diff: {mean_diff:.6e}")
        logger.info(f"  Max rel diff: {rel_diff:.6e}")
        logger.info(f"  Match (rtol={rtol}, atol={atol}): {match}")
        
        return match, error_metrics
        
    except Exception as e:
        logger.error(f"Output comparison failed: {e}")
        return False, {}

