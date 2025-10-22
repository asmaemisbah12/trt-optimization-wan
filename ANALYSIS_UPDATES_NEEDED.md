# Project Analysis: Files Requiring Updates for Wan2.2

## Critical Issues Found ⚠️

After reviewing the project against the actual Wan2.2 architecture, several files still contain outdated assumptions and need updates.

---

## 1. 🐳 **Dockerfile** - NEEDS UPDATES

### Current Issues:
- Base image `nvcr.io/nvidia/pytorch:24.01-py3` may not support latest Diffusers features
- No specification for torch version (needs >=2.4.0 for bfloat16 support)
- Missing video processing libraries for MP4 export

### Required Changes:
```dockerfile
# Use newer base image
FROM nvcr.io/nvidia/pytorch:24.09-py3  # or later

# Ensure torch >=2.4.0 for bfloat16
RUN pip install --upgrade torch>=2.4.0 torchvision

# Add video libraries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-extra \  # Additional codecs
    x264 \              # H.264 encoder
    libx264-dev
```

**Priority:** HIGH

---

## 2. 📦 **requirements.txt** - NEEDS UPDATES

### Current Issues:
- `torch>=2.0.0` is too old (needs >=2.4.0 for bfloat16)
- `diffusers>=0.27.0` might not have WanPipeline support
- Missing version pins for stability

### Required Changes:
```txt
# Core dependencies (UPDATED)
torch>=2.4.0           # Required for bfloat16 support
torchvision>=0.19.0
diffusers>=0.30.0      # WanPipeline support
transformers>=4.40.0   # Latest for Wan text encoder
accelerate>=0.30.0

# Note: Wan2.2 requires torch>=2.4.0 for native bfloat16
```

**Priority:** CRITICAL

---

## 3. 📜 **scripts/export_model.py** - NEEDS UPDATES

### Current Issues:
- Line 43: Only allows `["fp32", "fp16"]`, missing `"bf16"` and `"bfloat16"`
- Line 62: Default `num_frames=16` (should be 81 for Wan2.2)
- Line 108: Uses old precision for vae_dtype logic

### Required Changes:
```python
# Line 42-46
parser.add_argument(
    "--precision",
    type=str,
    default="bf16",  # Changed from "fp16"
    choices=["fp32", "fp16", "bf16", "bfloat16"],  # Added bf16
    help="Precision for export"
)

# Line 60-64
parser.add_argument(
    "--num_frames",
    type=int,
    default=81,  # Changed from 16 (Wan2.2 max)
    help="Number of frames for dummy input"
)
```

**Priority:** HIGH

---

## 4. 🏗️ **scripts/build_engines.py** - CRITICAL BUGS

### Current Issues:
- Lines 57-59: **Hardcoded `latent_channels = 4`** (should be 16!)
- Lines 58-59: Hardcoded `seq_length = 77` and `hidden_size = 768` (wrong for Wan2.2)
- Line 75: Hardcoded `latent_channels = 4` in VAE section too

### Required Changes:
```python
# Lines 55-71 (transformer section)
if "dit" in component or "transformer" in component or "unet" in component:
    # Wan Transformer: sample, encoder_hidden_states
    latent_channels = 16  # AutoencoderKLWan uses 16 channels!
    seq_length = 256      # Wan uses longer sequences
    hidden_size = 2048    # Adjust based on text encoder
    
    profile_shapes["sample"] = (
        (batch_config["min"], latent_channels, frames_config["min"], height_config["min"], width_config["min"]),
        (batch_config["opt"], latent_channels, frames_config["opt"], height_config["opt"], width_config["opt"]),
        (batch_config["max"], latent_channels, frames_config["max"], height_config["max"], width_config["max"]),
    )

# Lines 73-81 (VAE section)
elif "vae" in component:
    # AutoencoderKLWan: latent_sample
    latent_channels = 16  # Not 4!
    
    profile_shapes["latent_sample"] = (
        # ... same pattern
    )
```

**Priority:** CRITICAL (will cause build failures)

---

## 5. 📝 **examples/simple_export.py** - NEEDS UPDATES

### Current Issues:
- Line 35: Uses `"float16"` (should be `"bfloat16"`)
- Line 49: Uses `num_frames=16` (should be 81)
- Line 53: Uses `torch.float16` (should be `torch.bfloat16`)
- Line 48: Says "DiT component" but Wan uses "transformer"

### Required Changes:
```python
# Line 33-38
pipe = load_pipeline(
    model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    torch_dtype="bfloat16",  # Changed
    vae_dtype="float32",
    device="cuda:0"
)

# Line 40-54
logger.info("Extracting Transformer component...")  # Updated terminology
transformer = get_submodule(pipe, "transformer")
transformer.eval()

dummy_inputs = create_dummy_inputs(
    component_name="transformer",  # Changed from "dit"
    num_frames=81,                 # Changed from 16
    height=720,
    width=1280,
    device="cuda:0",
    dtype=torch.bfloat16          # Changed from float16
)

dynamic_axes = get_dynamic_axes("transformer")
```

**Priority:** MEDIUM

---

## 6. 📖 **README.md** - NEEDS UPDATES

### Current Issues:
- Line 7: Says "PyTorch FP16 baseline" (should be "PyTorch BFloat16")
- Line 15: Says "DiT/UNet: FP16 TensorRT" (should clarify Transformer/MoE)
- Line 22: Mentions "DiT/Transformer" (should explain MoE)
- Line 9: Says "8-81 frames" (should clarify temporal compression)

### Required Changes:
```markdown
## Project Goals

- **Primary**: Reduce end-to-end inference time by ≥1.5× vs PyTorch BFloat16 baseline
- **Secondary**: Maintain perceptual parity (PSNR/SSIM within tolerance)
- **Operational**: Support dynamic shapes (81 frames @ 24fps, 720P-1280P), single-GPU production

## Architecture

### Optimization Strategy

1. **Transformer (MoE)**: FP16 TensorRT engine (converted from BFloat16, maximum speed)
2. **VAE (AutoencoderKLWan)**: FP32 TensorRT engine (16×16×4 compression, numerical stability)
3. **Runtime**: CUDA Graphs, memory reuse, multi-profile engines

### Pipeline Flow

```
Text → Encoder (BF16) → Transformer MoE (FP16 TRT) → Latents [16ch] → VAE Decoder (FP32 TRT) → Video
```

**Note:** Wan2.2 uses a Mixture-of-Experts (MoE) architecture with 2 experts (27B total, 14B active).
```

**Priority:** MEDIUM (documentation clarity)

---

## 7. 📝 **examples/simple_benchmark.py** - NEEDS UPDATES

### Current Issues:
- Uses `"float16"` precision
- Doesn't pass Wan2.2-specific parameters

### Required Changes:
```python
# Update load_pipeline call
pipe = load_pipeline(
    model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    torch_dtype="bfloat16",  # Changed
    vae_dtype="float32",
    device="cuda:0"
)

# Update inputs dict
inputs = {
    "prompt": "A serene lake at sunset",
    "num_frames": 81,  # Changed from 16
    "height": 720,
    "width": 1280,
    "num_inference_steps": 40,  # Wan2.2 default
    "guidance_scale": 4.0,      # Wan2.2 primary
    # Note: guidance_scale_2 handled by WanPipeline
}
```

**Priority:** MEDIUM

---

## 8. 📜 **Makefile** - MINOR UPDATES

### Current Issues:
- Commands reference old defaults (16 frames)

### Required Changes:
```makefile
export-dit:
	python scripts/export_model.py \
		--component transformer \
		--precision bf16 \
		--num_frames 81 \
		--height 720 \
		--width 1280 \
		--validate

build-dit:
	python scripts/build_engines.py \
		--onnx_path outputs/onnx/transformer_bf16.onnx \
		--precision fp16 \
		--workspace_size 8192
```

**Priority:** LOW (convenience)

---

## 9. 📖 **QUICKSTART.md** - NEEDS UPDATES

### Current Issues:
- Still references "DiT" and "FP16"
- Incorrect frame counts and defaults

### Required Changes:
Update all examples to use:
- `--component transformer` (not dit)
- `--precision bf16` (not fp16)
- `--num_frames 81` (not 16)
- `fps=24` (not 16)

**Priority:** MEDIUM

---

## 10. 🧪 **tests/** - NEEDS NEW TESTS

### Missing Tests:
- No tests for WanPipeline loading
- No tests for AutoencoderKLWan
- No tests for bfloat16 conversion
- No tests for 16-channel latents

### Required Additions:
```python
# tests/test_wan_specific.py
def test_wan_pipeline_loading():
    """Test loading WanPipeline correctly"""
    
def test_latent_channels():
    """Test that latent channels = 16"""
    
def test_bfloat16_support():
    """Test bfloat16 dtype handling"""
```

**Priority:** MEDIUM

---

## Summary Table

| File | Priority | Issue | Impact |
|------|----------|-------|--------|
| **build_engines.py** | 🔴 CRITICAL | Hardcoded channels=4 | **Build will fail** |
| **requirements.txt** | 🔴 CRITICAL | torch<2.4.0 | **No bfloat16 support** |
| **Dockerfile** | 🟠 HIGH | Old base image | Deployment issues |
| **export_model.py** | 🟠 HIGH | No bf16 option | Can't export correctly |
| **README.md** | 🟡 MEDIUM | Outdated docs | User confusion |
| **examples/*.py** | 🟡 MEDIUM | Wrong defaults | Poor user experience |
| **QUICKSTART.md** | 🟡 MEDIUM | Old instructions | Incorrect guidance |
| **Makefile** | 🟢 LOW | Old defaults | Minor inconvenience |

---

## Immediate Action Items

### 🔥 Must Fix Before First Use:

1. ✅ **Fix build_engines.py** - Change all `latent_channels = 4` to `16`
2. ✅ **Update requirements.txt** - Pin `torch>=2.4.0`
3. ✅ **Add bf16 to export_model.py** - Support bfloat16 export

### 🎯 Should Fix for Good UX:

4. Update Dockerfile with newer base image
5. Update all examples to use correct defaults
6. Update README.md and QUICKSTART.md
7. Update Makefile commands

### 📝 Nice to Have:

8. Add Wan-specific tests
9. Add architecture diagram
10. Create migration guide from generic T2V

---

## Files Already Updated ✅

- ✅ `src/model/loader.py` - WanPipeline support, bfloat16 default
- ✅ `src/model/shapes.py` - 16 channels, 16×16×4 compression
- ✅ `src/utils/config.py` - bfloat16 dtype mapping
- ✅ `configs/config.yaml` - All Wan2.2 settings
- ✅ `docs/WAN22_ARCHITECTURE.md` - Complete specs
- ✅ `docs/USING_OPTIMUM.md` - Optimum guide
- ✅ `docs/OPTIMUM_RECOMMENDATION.md` - Optimum analysis

---

## Testing Checklist

After updates, test:

- [ ] Load WanPipeline successfully
- [ ] Export transformer to ONNX (bf16)
- [ ] Build TRT engine with 16 channels
- [ ] Verify latent shape: [batch, 16, frames/4, H/16, W/16]
- [ ] Run smoke test inference
- [ ] Benchmark vs PyTorch baseline

---

**Next:** I'll update all critical files now.

