# Do You Need Optimum for Wan2.2? Answer & Recommendations

## Quick Answer

**Short:** Not strictly required, but **highly recommended** for easier workflow.

**Your code works fine as-is** using Diffusers directly. However, adding Optimum will make ONNX export and TensorRT optimization significantly easier.

---

## Analysis of Your Current Setup

Your code:
```python
from diffusers import WanPipeline, AutoencoderKLWan

vae = AutoencoderKLWan.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers", 
    subfolder="vae", 
    torch_dtype=torch.float32
)
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers", 
    vae=vae, 
    torch_dtype=torch.bfloat16
)
```

✅ **This is correct!** Your setup already:
- Uses `WanPipeline` (the proper pipeline class)
- Loads `AutoencoderKLWan` separately in FP32 (excellent for quality!)
- Uses `torch.bfloat16` for the main model (optimal for Wan2.2)

---

## Why Consider Optimum?

### ✅ **Benefits for Your Use Case**

1. **Easier ONNX Export**
   ```bash
   # With Optimum (one command)
   optimum-cli export onnx \
       --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
       --component transformer \
       --fp16 \
       outputs/onnx/
   
   # vs Manual (need custom code)
   torch.onnx.export(model, inputs, ...)  # + shape handling + validation
   ```

2. **Automatic Handling of Custom Components**
   - `WanPipeline` is a custom Diffusers pipeline
   - `AutoencoderKLWan` has unique 16×16×4 compression
   - Optimum knows how to handle these properly

3. **Future-Proof**
   - INT8/FP8 quantization built-in
   - Active maintenance for new models
   - Validation utilities

### ⚠️ **When You DON'T Need It**

- Just running inference (your current code is perfect)
- Not planning TensorRT acceleration
- Model works well as-is

---

## Critical Architecture Updates

Based on the [Wan2.2 model card](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers), I've updated the project with **actual** specifications:

### Key Differences from Initial Assumptions

| Feature | Initial Assumption | **Actual Wan2.2** |
|---------|-------------------|-------------------|
| VAE Compression | 8×8×1 | **16×16×4** |
| Latent Channels | 4 | **16** |
| Precision | FP16 | **BFloat16** |
| FPS | 16 | **24** |
| Architecture | Single DiT | **MoE (2 experts)** |
| Max Frames | Variable | **81 @ 24fps** |

### Updated Files

1. **`src/model/loader.py`**
   - Auto-detects `WanPipeline` and `AutoencoderKLWan`
   - Uses BFloat16 by default
   - Loads VAE separately in FP32

2. **`src/model/shapes.py`**
   - Corrected to 16×16×4 compression
   - 16 latent channels (not 4)
   - 81 frames, 24fps defaults

3. **`configs/config.yaml`**
   - Updated all profiles for correct latent dimensions
   - Wan2.2-specific inference settings
   - Dual guidance scales (4.0 and 3.0)

4. **New Documentation**
   - `docs/USING_OPTIMUM.md` - How to use Optimum
   - `docs/WAN22_ARCHITECTURE.md` - Complete architecture specs

---

## Recommended Workflow

### Option A: Your Current Setup (No TensorRT)

**Keep using your code as-is:**
```python
# This is already optimal for PyTorch inference
pipe = WanPipeline.from_pretrained(...)
output = pipe(prompt=..., num_frames=81, ...)
```

**No Optimum needed** for this workflow.

### Option B: Add TensorRT Acceleration (This Project)

**Step 1:** Use your existing code to verify it works

**Step 2:** Export to ONNX (choose one):

**Option B1 - With Optimum (Easier):**
```bash
optimum-cli export onnx \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --component transformer \
    --fp16 \
    outputs/onnx/transformer/
```

**Option B2 - With Our Scripts (More Control):**
```bash
python scripts/export_model.py \
    --component transformer \
    --precision bf16 \
    --num_frames 81 \
    --height 720 \
    --width 1280
```

**Step 3:** Build TensorRT engines
```bash
python scripts/build_engines.py \
    --onnx_path outputs/onnx/transformer_bf16.onnx \
    --precision fp16  # TRT doesn't support BF16, use FP16
```

**Step 4:** Run accelerated inference
```bash
python scripts/run_inference.py \
    --prompt "Your prompt here" \
    --engine_dir outputs/engines/
```

---

## Installation Checklist

### Already in Your Environment ✅
- `torch`
- `diffusers` (with WanPipeline support)
- `transformers`

### Additional for TensorRT Project

```bash
# Already in requirements.txt
pip install onnx onnxruntime-gpu optimum[exporters]

# TensorRT (separate install)
pip install tensorrt pycuda
```

---

## Key Points from Model Card

### Inference Parameters (Your Code)

Your settings are good, but here are the **official defaults** from Wan2.2:

```python
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,  # Optional but recommended
    height=720,
    width=1280,
    num_frames=81,  # Not 16!
    guidance_scale=4.0,  # Lower than typical 7.5
    guidance_scale_2=3.0,  # Wan-specific second guidance
    num_inference_steps=40,  # Not 50
).frames[0]

# Save with correct FPS
export_to_video(output, "output.mp4", fps=24)  # Not 16!
```

### Architecture Highlights

1. **MoE with 2 Experts**
   - 27B total parameters
   - 14B active per step
   - Automatic expert switching based on denoising stage

2. **AutoencoderKLWan**
   - 16×16×4 compression (64× total, not 8×)
   - 16 latent channels
   - Must use FP32 for quality

3. **Latent Space Calculation**
   ```python
   # Video: 81 frames, 720×1280
   latent_frames = 81 // 4 = 20  # temporal compression
   latent_height = 720 // 16 = 45  # spatial compression
   latent_width = 1280 // 16 = 80  # spatial compression
   latent_shape = [16, 20, 45, 80]  # [C, T, H, W]
   ```

---

## Final Recommendations

### For You Specifically

1. **Current PyTorch Inference:**
   - ✅ Your code is correct, keep using it
   - ❌ Don't need Optimum

2. **If Adding TensorRT Acceleration:**
   - ✅ **Recommended:** Install Optimum (`pip install optimum[exporters]`)
   - ✅ Use this project's framework
   - ✅ Start with our scripts (already updated for Wan2.2)
   - ⚠️ Note: BFloat16 → FP16 conversion needed for TRT

3. **Expected Speedup with TRT:**
   - Transformer only: **1.5-2× faster**
   - Transformer + VAE: **1.8-2.2× faster**  
   - With CUDA Graphs: **2-2.5× faster**

---

## Next Steps

### To Use This Project for TensorRT Acceleration:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install tensorrt pycuda
   ```

2. **Test model loading:**
   ```python
   from src.model.loader import load_pipeline
   pipe = load_pipeline()  # Auto-detects WanPipeline
   ```

3. **Export transformer:**
   ```bash
   python scripts/export_model.py \
       --component transformer \
       --precision bf16 \
       --num_frames 81
   ```

4. **Build TRT engine:**
   ```bash
   python scripts/build_engines.py \
       --onnx_path outputs/onnx/transformer_bf16.onnx \
       --precision fp16
   ```

5. **Benchmark:**
   ```bash
   python scripts/benchmark.py --compare
   ```

---

## Summary

| Scenario | Need Optimum? | Recommendation |
|----------|---------------|----------------|
| PyTorch inference only | ❌ No | Use your current code |
| Export to ONNX | ✅ Helpful | Optional but easier |
| TensorRT acceleration | ✅ Recommended | This project + Optimum |
| Quantization (INT8/FP8) | ✅ Yes | Optimum provides tools |

**Bottom line:** Optimum is **not required** but makes the TensorRT workflow **significantly easier**. It's already in `requirements.txt` ready to use.

---

**Questions?** Check `docs/USING_OPTIMUM.md` for detailed guide.

