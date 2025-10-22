# Quick Start Guide

Get up and running with Wan TensorRT acceleration in 5 steps.

## Prerequisites

- NVIDIA GPU with Compute Capability >= 7.0 (Volta or newer)
- CUDA 11.8+ or 12.x
- Python 3.8+
- 24GB+ VRAM recommended (for full model)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/wan_trt.git
cd wan_trt
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt
```

### 3. Install TensorRT

Follow NVIDIA's official installation guide for your platform:
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

**Quick install for Ubuntu with pip:**
```bash
pip install tensorrt
pip install pycuda
```

**Verify installation:**
```bash
python -c "import tensorrt; print(tensorrt.__version__)"
```

## Usage

### Step 1: Export Model to ONNX

Export the DiT (Diffusion Transformer) component:

```bash
python scripts/export_model.py \
    --component dit \
    --precision fp16 \
    --num_frames 16 \
    --height 720 \
    --width 1280 \
    --validate
```

Export the VAE decoder:

```bash
python scripts/export_model.py \
    --component vae \
    --precision fp32 \
    --num_frames 16 \
    --height 720 \
    --width 1280 \
    --validate
```

**Expected output:**
- `outputs/onnx/dit_fp16.onnx` (~14GB)
- `outputs/onnx/vae_fp32.onnx` (~500MB)

### Step 2: Build TensorRT Engines

Build the DiT engine:

```bash
python scripts/build_engines.py \
    --onnx_path outputs/onnx/dit_fp16.onnx \
    --precision fp16 \
    --workspace_size 8192
```

Build the VAE engine:

```bash
python scripts/build_engines.py \
    --onnx_path outputs/onnx/vae_fp32.onnx \
    --precision fp32 \
    --workspace_size 4096
```

**Expected output:**
- `outputs/engines/dit_fp16.trt` (~7GB)
- `outputs/engines/vae_fp32.trt` (~250MB)

**Note:** Engine building can take 10-30 minutes per component.

### Step 3: Run Inference

```bash
python scripts/run_inference.py \
    --prompt "A serene lake at sunset with mountains in the background" \
    --num_frames 16 \
    --height 720 \
    --width 1280 \
    --output_path outputs/videos/sunset.mp4
```

**For PyTorch baseline (no TensorRT):**

```bash
python scripts/run_inference.py \
    --prompt "A serene lake at sunset" \
    --use_pytorch \
    --output_path outputs/videos/baseline.mp4
```

### Step 4: Benchmark Performance

```bash
python scripts/benchmark.py \
    --baseline pytorch \
    --num_warmup_runs 3 \
    --num_test_runs 10
```

To compare multiple configurations:

```bash
python scripts/benchmark.py \
    --compare \
    --num_test_runs 5
```

**Expected speedup:** 1.5-2.5× faster than PyTorch FP16 baseline.

### Step 5: Analyze Results

Check the benchmark results:

```bash
cat outputs/benchmarks/comparison.txt
```

View detailed metrics:

```bash
ls outputs/benchmarks/*.json
```

## Configuration

Edit `configs/config.yaml` to customize:

- Model paths and cache directories
- Engine build parameters (workspace size, profiles)
- Runtime settings (CUDA graphs, memory pools)
- Benchmark parameters

**Example: Change optimization profile for longer videos**

```yaml
tensorrt:
  profiles:
    dit_fp16:
      frames:
        min: 8
        opt: 32   # Changed from 16
        max: 81
```

Then rebuild engines.

## Troubleshooting

### Issue: ONNX export fails with "Unsupported operator"

**Solution:** Some custom operators may not be supported. Try:
1. Update `onnx` and `optimum` to latest versions
2. Use a higher opset version: `--opset_version 18`
3. Check if model has custom ops that need plugins

### Issue: TensorRT build fails with "Out of memory"

**Solution:**
1. Reduce workspace size: `--workspace_size 4096`
2. Build on a machine with more VRAM
3. Close other GPU applications

### Issue: Inference is slower than expected

**Solution:**
1. Ensure engines match your input shapes (check optimization profile)
2. Enable CUDA graphs (set in config.yaml)
3. Use the correct precision (FP16 for speed)
4. Warm up the engine before timing

### Issue: Quality degradation with TensorRT

**Solution:**
1. Keep VAE in FP32 (not FP16)
2. Enable strict types in engine builder
3. Compare outputs with `--validate` flag
4. Check PSNR/SSIM metrics in benchmark

## Next Steps

- **Optimize further:** Explore INT8 quantization for additional speedup
- **Multi-GPU:** Implement data parallelism for batch processing
- **Production:** Package engines for deployment
- **Contribute:** See CONTRIBUTING.md for how to help improve the project

## Resources

- [Full Documentation](README.md)
- [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [Wan2.2-T2V Model Card](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
- [Report Issues](https://github.com/yourusername/wan_trt/issues)

---

**Need help?** Open an issue on GitHub or check existing discussions.

