# Wan2.2 Architecture Details

Comprehensive technical details about the Wan2.2-T2V-A14B model architecture.

**Source:** [Wan2.2-T2V-A14B-Diffusers on HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)

---

## Model Overview

- **Model Name:** Wan2.2-T2V-A14B
- **Type:** Text-to-Video Diffusion Model (MoE Architecture)
- **Total Parameters:** 27B (Two 14B experts)
- **Active Parameters:** 14B per inference step
- **License:** Apache 2.0
- **Release Date:** July 2025

---

## Key Components

### 1. Mixture-of-Experts (MoE) Transformer

**Architecture:**
- **Two Experts:**
  - **High-Noise Expert:** Activated early in denoising (t > t_moe), focuses on overall layout
  - **Low-Noise Expert:** Activated late in denoising (t < t_moe), refines details
- **Expert Size:** 14B parameters each
- **Switching Criterion:** Signal-to-Noise Ratio (SNR)
  - Threshold: t_moe where SNR = 0.5 × SNR_min
  - High noise → High-noise expert
  - Low noise → Low-noise expert

**Input Format:**
```python
transformer(
    sample,  # Shape: [batch, 16, frames/4, height/16, width/16]
    timestep,  # Current denoising step
    encoder_hidden_states,  # Text embeddings [batch, seq_len, hidden_dim]
)
```

**Precision:** BFloat16 (recommended)

### 2. AutoencoderKLWan (VAE)

**Compression Ratio:** **16×16×4** (T×H×W)
- **Spatial:** 16× in both height and width
- **Temporal:** 4× in frames
- **Total:** 1024× compression (16×16×4)

**Latent Channels:** 16 (not 4 like standard VAE)

**Example:**
```
Input Video:  [3, 81, 720, 1280]  (RGB, 81 frames, 720P)
Latent Space: [16, 20, 45, 80]    (16 channels, 20 frames, 45×80 spatial)
```

**Calculation:**
- Latent Frames: 81 ÷ 4 = 20 (+ 1 for padding)
- Latent Height: 720 ÷ 16 = 45
- Latent Width: 1280 ÷ 16 = 80

**Precision:** Float32 (for numerical stability in reconstruction)

### 3. Text Encoder

- **Model:** Based on UMT5-XXL or similar
- **Sequence Length:** ~256 tokens (longer than typical T2I models)
- **Hidden Size:** ~2048
- **Output:** Text embeddings for conditioning

**Precision:** BFloat16

---

## Technical Specifications

### Supported Resolutions

| Name | Resolution | Latent Size (H×W) | Aspect Ratio |
|------|------------|-------------------|--------------|
| 480P | 480 × 640  | 30 × 40          | 3:4          |
| 720P | 720 × 1280 | 45 × 80          | 9:16         |
| 720P Alt | 1280 × 720 | 80 × 45          | 16:9         |
| 1280P | 1280 × 1280 | 80 × 80         | 1:1          |

**Note:** 720P (720×1280) is the primary recommended resolution.

### Frame Counts

- **Maximum:** 81 frames @ 24fps = 3.4 seconds
- **Typical:** 81 frames (full length)
- **Latent Frames:** 20-21 frames (81 ÷ 4)

### Inference Parameters

**Default Settings:**
```python
num_inference_steps = 40      # Denoising steps
guidance_scale = 4.0          # Primary CFG scale (lower than typical)
guidance_scale_2 = 3.0        # Secondary CFG scale (Wan-specific)
fps = 24                      # Native frame rate (not 16!)
```

**Why Two Guidance Scales?**
- Wan2.2 uses dual guidance for better control
- `guidance_scale`: Primary semantic guidance
- `guidance_scale_2`: Secondary detail guidance

---

## Data Types and Precision

### Recommended Precision Strategy

| Component | Training | Inference | TensorRT Target | Reason |
|-----------|----------|-----------|-----------------|--------|
| Transformer (MoE) | BF16 | BF16 | FP16 | Speed, MoE stability |
| VAE Encoder | FP32 | FP32 | FP32 | Preserve details |
| VAE Decoder | FP32 | FP32 | FP32 | Avoid artifacts |
| Text Encoder | BF16 | BF16 | FP16 | Lightweight |

**Why BFloat16?**
- Better numerical stability than FP16
- Wider dynamic range
- Preferred for large models (14B+ parameters)
- Native support on Ampere/Hopper GPUs

**TensorRT Note:**
- TensorRT doesn't natively support BF16
- Convert BF16 models to FP16 for TRT (acceptable quality loss)
- Keep VAE in FP32 for TRT

---

## Memory Requirements

### Model Sizes (Approximate)

| Component | Parameters | FP32 Size | BF16 Size | FP16 TRT Engine |
|-----------|------------|-----------|-----------|-----------------|
| Transformer (2 experts) | 27B | ~108 GB | ~54 GB | ~27 GB |
| VAE | ~500M | ~2 GB | ~1 GB | ~1 GB (FP32) |
| Text Encoder | ~4B | ~16 GB | ~8 GB | ~4 GB |
| **Total** | ~31B | ~126 GB | ~63 GB | ~32 GB |

### Runtime Memory (Single GPU)

**PyTorch Inference:**
- Model Weights: ~54 GB (BF16)
- Activations: ~10-20 GB
- Peak: **~80 GB VRAM**

**TensorRT Inference (Target):**
- Engine: ~32 GB
- Activations: ~8-12 GB  
- Peak: **~50 GB VRAM** (expected)

**Recommendations:**
- **H100 (80GB):** Can run full model
- **A100 (80GB):** Can run full model
- **A6000 (48GB):** Requires offloading or model parallelism
- **RTX 4090 (24GB):** TI2V-5B model only (separate release)

---

## Pipeline Workflow

### Complete Generation Flow

```
1. Text Input
   ↓
2. Text Encoder (BF16)
   → Text Embeddings [batch, 256, 2048]
   ↓
3. Initialize Noise
   → Latent Noise [batch, 16, 20, 45, 80]
   ↓
4. Diffusion Loop (40 steps)
   ├─ Steps 1-20: High-Noise Expert (14B active)
   │  ├─ Timestep > t_moe
   │  └─ Focus: Overall structure, motion
   │
   └─ Steps 21-40: Low-Noise Expert (14B active)
      ├─ Timestep ≤ t_moe
      └─ Focus: Details, refinement
   ↓
5. Clean Latents
   → [batch, 16, 20, 45, 80]
   ↓
6. VAE Decoder (FP32)
   → Video Frames [batch, 3, 81, 720, 1280]
   ↓
7. Post-Processing
   → Save as MP4 @ 24fps
```

---

## Comparison to Standard Diffusion Models

| Feature | Standard (SD3) | Wan2.2 |
|---------|----------------|--------|
| Architecture | Single DiT | MoE (2 experts) |
| Parameters | 8-10B | 27B (14B active) |
| VAE Compression | 8×8×1 | 16×16×4 |
| Latent Channels | 4-16 | 16 |
| Precision | FP16 | BF16 |
| Video Length | < 3s typical | 3.4s @ 24fps |
| Resolution | Up to 1024P | 720P optimized |

---

## TensorRT Optimization Strategy

### Export Targets

1. **Transformer (Priority 1)**
   - Convert BF16 → FP16 for TRT
   - Build with multiple profiles (frames 20, spatial 45×80)
   - Expected speedup: 1.5-2×

2. **VAE Decoder (Priority 2)**
   - Keep FP32 for quality
   - Single profile (81 frames, 720×1280)
   - Expected speedup: 1.3-1.5×

3. **Text Encoder (Optional)**
   - Lightweight, may not benefit much
   - Consider keeping in PyTorch

### Precision Conversion

```python
# BF16 → FP16 for TRT export
model_bf16 = load_model(torch_dtype=torch.bfloat16)
model_fp16 = model_bf16.to(dtype=torch.float16)

# Export to ONNX
export_to_onnx(model_fp16, precision="fp16")

# Build TRT engine
build_engine(onnx_path, precision="fp16")
```

### Expected Performance

**Target (vs PyTorch BF16 baseline):**
- Transformer Only: 1.5× faster
- Transformer + VAE: 1.8-2.2× faster
- With CUDA Graphs: 2-2.5× faster
- With INT8 (future): 2.5-3.5× faster

---

## Key Differences from Generic T2V Models

1. **MoE Architecture** - Dynamic expert switching based on denoising stage
2. **High Compression VAE** - 16×16×4 instead of 8×8×1
3. **16 Latent Channels** - More than typical 4 channels
4. **Dual Guidance** - Two CFG scales for better control
5. **BFloat16 Native** - Better numerical stability
6. **Longer Sequences** - More detailed text conditioning

---

## References

- **Model Card:** https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers
- **Paper:** arXiv:2503.20314 (Wan: Open and Advanced Large-Scale Video Generative Models)
- **GitHub:** https://github.com/Wan-Video/Wan2.2
- **License:** Apache 2.0

---

**Last Updated:** Based on July 2025 release

For TensorRT acceleration implementation, see `README.md` and `NEXT_STEPS.md`.

