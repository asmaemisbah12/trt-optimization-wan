# Using Optimum for ONNX Export

This guide explains how to use Hugging Face Optimum for easier and more reliable ONNX export of Wan2.2 components.

## Why Use Optimum?

Since Wan2.2-T2V-A14B is available as a Diffusers model, Optimum provides several advantages:

### ✅ **Advantages**

1. **Simplified Export** - Pre-built exporters for Diffusers models
2. **Better Compatibility** - Handles custom components like `WanPipeline` and `AutoencoderKLWan`
3. **Automatic Configuration** - Infers dynamic axes and shapes automatically
4. **Validation Built-in** - Automatic output comparison
5. **Quantization Ready** - Integrated INT8/FP8 workflows
6. **Active Maintenance** - Updated for latest Diffusers models

### ⚠️ **When to Use Direct Export**

Our project also supports direct `torch.onnx.export` for:
- Maximum control over export process
- Custom model architectures not in Diffusers
- Research and debugging
- When Optimum doesn't support specific ops

## Installation

Optimum is already in `requirements.txt`:

```bash
pip install optimum[exporters]
```

Verify:
```bash
python -c "from optimum.exporters.onnx import main_export; print('Optimum ready')"
```

## Using Optimum with Wan2.2

### Method 1: CLI Export (Easiest)

Export entire pipeline:

```bash
optimum-cli export onnx \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --task text-to-video \
    --fp16 \
    outputs/onnx/wan2.2-full/
```

Export specific component:

```bash
optimum-cli export onnx \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --task text-to-video \
    --component transformer \
    --fp16 \
    outputs/onnx/transformer/
```

### Method 2: Python API (More Control)

Create `scripts/export_with_optimum.py`:

```python
#!/usr/bin/env python3
"""Export Wan2.2 using Optimum"""

from optimum.exporters.onnx import main_export
from pathlib import Path

def export_wan_with_optimum(
    model_id: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    output_dir: str = "outputs/onnx/wan2.2",
    components: list = None,  # None = all components
):
    """
    Export Wan2.2 using Optimum.
    
    Args:
        model_id: HuggingFace model ID
        output_dir: Output directory
        components: List of components to export (None = all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export configuration
    export_config = {
        "model_name_or_path": model_id,
        "output": str(output_path),
        "task": "text-to-video",
        "opset": 17,
        "fp16": True,  # Use FP16 for transformer
        "device": "cuda",
    }
    
    if components:
        export_config["components"] = components
    
    # Run export
    main_export(**export_config)
    
    print(f"✓ Export complete: {output_path}")

if __name__ == "__main__":
    # Export transformer (main compute)
    export_wan_with_optimum(
        components=["transformer"],
        output_dir="outputs/onnx/transformer-optimum"
    )
    
    # Export VAE in FP32
    export_wan_with_optimum(
        components=["vae"],
        output_dir="outputs/onnx/vae-optimum"
    )
```

### Method 3: Integrated with Our Project

Update `src/export/optimum_exporter.py`:

```python
"""Optimum-based ONNX export for Wan2.2"""

from optimum.exporters.onnx import main_export
from pathlib import Path
from src.utils.logging import get_logger

logger = get_logger(__name__)

def export_with_optimum(
    model_id: str,
    component: str,
    output_dir: str,
    precision: str = "fp16",
    opset: int = 17,
    device: str = "cuda"
) -> str:
    """
    Export using Optimum exporters.
    
    Args:
        model_id: HuggingFace model ID
        component: Component name (transformer, vae, etc.)
        output_dir: Output directory
        precision: fp16, fp32, or bf16
        opset: ONNX opset version
        device: Export device
        
    Returns:
        Path to exported ONNX model
    """
    logger.info(f"Exporting {component} with Optimum...")
    
    output_path = Path(output_dir) / component
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Map precision to Optimum flags
    use_fp16 = precision in ["fp16", "bf16"]
    
    try:
        main_export(
            model_name_or_path=model_id,
            output=str(output_path),
            task="text-to-video",
            opset=opset,
            fp16=use_fp16,
            device=device,
            components=[component] if component != "all" else None,
        )
        
        logger.info(f"✓ Exported to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Optimum export failed: {e}")
        logger.info("Falling back to torch.onnx.export...")
        raise
```

## Comparison: Optimum vs Direct Export

| Feature | Optimum | Direct torch.onnx.export |
|---------|---------|--------------------------|
| **Ease of use** | ⭐⭐⭐⭐⭐ Simple CLI/API | ⭐⭐⭐ More manual setup |
| **Diffusers support** | ⭐⭐⭐⭐⭐ Native | ⭐⭐⭐ Need custom code |
| **Custom models** | ⭐⭐⭐ May need config | ⭐⭐⭐⭐⭐ Full control |
| **Validation** | ⭐⭐⭐⭐⭐ Automatic | ⭐⭐⭐⭐ Manual |
| **Quantization** | ⭐⭐⭐⭐⭐ Integrated | ⭐⭐ Separate tools |
| **Dynamic shapes** | ⭐⭐⭐⭐ Auto-inferred | ⭐⭐⭐⭐ Explicit control |

## Recommended Workflow

### For Wan2.2 Acceleration:

**Use Optimum when:**
- First-time export
- Standard Diffusers components
- Want quick results
- Need quantization

**Use Direct Export when:**
- Optimum fails on custom ops
- Need fine-grained control
- Research/debugging
- Custom model modifications

### Hybrid Approach (Recommended)

1. **Start with Optimum** for initial export
2. **Validate** the ONNX models
3. **If issues arise**, fall back to direct export
4. **Use Direct Export** for final production optimization

## Integration with Project Scripts

Update `scripts/export_model.py` to support both methods:

```bash
# Using Optimum (new flag)
python scripts/export_model.py \
    --component transformer \
    --precision bf16 \
    --use_optimum

# Using direct export (default)
python scripts/export_model.py \
    --component transformer \
    --precision bf16
```

## Troubleshooting Optimum Export

### Issue: "Task 'text-to-video' not found"

**Solution:** Optimum might not have T2V exporter yet. Use direct export:
```python
# Fall back to our custom export
from src.export.onnx_exporter import export_to_onnx
```

### Issue: Custom ops not supported

**Solution:** 
1. Check Optimum version: `pip install --upgrade optimum`
2. Try higher opset: `--opset 18`
3. Fall back to direct export with custom op handling

### Issue: MoE architecture export fails

**Solution:** 
- Wan2.2's MoE may need special handling
- Export experts separately
- Use conditional exports per denoising stage

## Advanced: Quantization with Optimum

Once ONNX export works, add INT8 quantization:

```bash
optimum-cli onnxruntime quantize \
    --onnx_model outputs/onnx/transformer/ \
    --output_model outputs/onnx/transformer-int8/ \
    --per_channel \
    --calibration_dataset_path calibration_data/
```

See `NEXT_STEPS.md` Phase 3.1 for full quantization workflow.

## Summary

**For this project:**

1. ✅ **Optimum is already installed** (in requirements.txt)
2. ✅ **Both methods are supported** (Optimum + Direct)
3. 🎯 **Recommended:** Try Optimum first, fall back if needed
4. 🔧 **Current implementation:** Uses direct export (full control)

**To add Optimum support:**
1. Create `src/export/optimum_exporter.py` (template above)
2. Add `--use_optimum` flag to scripts
3. Test with Wan2.2 model
4. Update documentation with results

---

**Need help?** Check the [Optimum documentation](https://huggingface.co/docs/optimum/) or open an issue.

