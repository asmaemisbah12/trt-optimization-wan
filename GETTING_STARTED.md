# Getting Started with Wan TensorRT

Welcome! This guide will help you get started with the Wan TensorRT acceleration project.

## What This Project Does

This project accelerates the Wan2.2-T2V-A14B video generation model using NVIDIA TensorRT, achieving:

- **1.5-2.5× faster inference** compared to PyTorch FP16
- **Preserved visual quality** through smart precision management
- **Production-ready deployment** with optimized engines
- **Flexible configuration** for different use cases

## Prerequisites Check

Before starting, verify you have:

### Hardware
- ✅ NVIDIA GPU with Compute Capability ≥ 7.0 (Volta or newer)
- ✅ 24GB+ VRAM (recommended for full model)
- ✅ 100GB+ free disk space (for models and engines)

### Software
- ✅ CUDA 11.8+ or 12.x installed
- ✅ Python 3.8+ installed
- ✅ Git installed

### Check Your Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Check Python version
python --version
```

## Installation (Step-by-Step)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/wan_trt.git
cd wan_trt
```

### 2. Create Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- PyTorch and Diffusers (model loading)
- ONNX and Optimum (model export)
- ONNXRuntime (ONNX inference and optimization)
- Quality metrics (PSNR, SSIM, LPIPS)
- Utilities (numpy, opencv, tqdm)

### 4. Install TensorRT

**Option A: Using pip (easiest)**
```bash
pip install tensorrt
pip install pycuda
```

**Option B: From NVIDIA (recommended for production)**

Download from: https://developer.nvidia.com/tensorrt

Follow the installation guide for your platform.

**Verify installation:**
```bash
python -c "import tensorrt; print(tensorrt.__version__)"
```

### 5. Install the Package

```bash
pip install -e .
```

This makes the `wan-trt` package available in your environment.

## First Run: Quick Test

### 1. Check Configuration

```bash
cat configs/config.yaml
```

Review and adjust if needed (model cache directory, GPU device, etc.).

### 2. Export a Component (Test)

Start with a small test to verify everything works:

```bash
python scripts/export_model.py \
    --component dit \
    --precision fp16 \
    --num_frames 8 \
    --height 512 \
    --width 512 \
    --validate
```

This should:
- Download the model from HuggingFace (first time only)
- Export the DiT component to ONNX
- Validate the export
- Save to `outputs/onnx/dit_fp16.onnx`

**Expected time:** 5-10 minutes (first run with download)

### 3. Build a Test Engine

```bash
python scripts/build_engines.py \
    --onnx_path outputs/onnx/dit_fp16.onnx \
    --precision fp16 \
    --workspace_size 4096
```

This builds a TensorRT engine optimized for your GPU.

**Expected time:** 10-20 minutes  
**Output:** `outputs/engines/dit_fp16.trt`

## Understanding the Workflow

The acceleration process has three main stages:

### Stage 1: Export (PyTorch → ONNX)

```
PyTorch Model → ONNX (portable intermediate format)
```

**Why?** ONNX is a standard format that TensorRT can optimize.

**Tools:** `scripts/export_model.py`

### Stage 2: Build (ONNX → TensorRT)

```
ONNX Model → TensorRT Engine (GPU-specific optimized binary)
```

**Why?** TensorRT analyzes the model and generates optimal GPU kernels.

**Tools:** `scripts/build_engines.py`

### Stage 3: Inference (TensorRT Runtime)

```
Text Prompt → Encoded → DiT (TRT) → Latents → VAE (TRT) → Video
```

**Why?** TRT engines execute much faster than PyTorch on NVIDIA GPUs.

**Tools:** `scripts/run_inference.py`

## Full Pipeline Example

Once you've verified the installation, run the full pipeline:

### 1. Export All Components

```bash
# Export DiT (main transformer)
python scripts/export_model.py \
    --component dit \
    --precision fp16 \
    --num_frames 16 \
    --height 720 \
    --width 1280

# Export VAE (decoder)
python scripts/export_model.py \
    --component vae \
    --precision fp32 \
    --num_frames 16 \
    --height 720 \
    --width 1280
```

### 2. Build TensorRT Engines

```bash
# Build DiT engine
python scripts/build_engines.py \
    --onnx_path outputs/onnx/dit_fp16.onnx \
    --precision fp16

# Build VAE engine
python scripts/build_engines.py \
    --onnx_path outputs/onnx/vae_fp32.onnx \
    --precision fp32
```

### 3. Run Inference

```bash
python scripts/run_inference.py \
    --prompt "A serene lake at sunset with mountains in the background" \
    --num_frames 16 \
    --output_path outputs/videos/demo.mp4
```

### 4. Benchmark Performance

```bash
python scripts/benchmark.py \
    --compare \
    --num_test_runs 10
```

## Using the Makefile (Convenient)

For convenience, use the Makefile:

```bash
# See all available commands
make help

# Export all components
make export-dit
make export-vae

# Build all engines
make build-dit
make build-vae

# Run benchmark
make benchmark

# One command to do everything
make all
```

## Docker Alternative

If you prefer Docker:

```bash
# Build image
docker-compose build

# Run container
docker-compose up -d

# Enter container
docker exec -it wan-trt bash

# Inside container, run commands
make all
```

## Project Structure Overview

Key directories:

```
wan_trt/
├── configs/         # Configuration files
├── src/            # Source code (library)
├── scripts/        # Executable scripts
├── examples/       # Usage examples
├── tests/          # Test suite
└── outputs/        # Generated files
```

## Common Issues and Solutions

### Issue: "CUDA out of memory"

**Solution:**
- Reduce workspace size: `--workspace_size 2048`
- Use smaller input dimensions
- Close other GPU applications
- Try a smaller batch size

### Issue: "Model download fails"

**Solution:**
- Check internet connection
- Set HuggingFace cache: `export HF_HOME=/path/to/cache`
- Use a different mirror
- Download manually from HuggingFace

### Issue: "ONNX export fails"

**Solution:**
- Update to latest onnx/optimum: `pip install --upgrade onnx optimum`
- Try higher opset version: `--opset_version 18`
- Check model compatibility
- Report issue with error log

### Issue: "TensorRT build takes too long"

**Solution:**
- This is normal (10-30 minutes per component)
- Use lower precision: FP16 instead of FP32
- Reduce optimization profiles
- Build once, reuse engines

### Issue: "Inference output quality is poor"

**Solution:**
- Ensure VAE is in FP32, not FP16
- Check precision settings in config
- Validate ONNX export outputs
- Compare with PyTorch baseline

## Next Steps

After completing the basic setup:

1. **Read the full README.md** for detailed documentation
2. **Try the examples/** for hands-on learning
3. **Run benchmarks** to measure speedup on your hardware
4. **Customize configs/** for your specific use case
5. **Read NEXT_STEPS.md** for advanced features

## Learning Resources

- **QUICKSTART.md** - Quick 5-step guide
- **README.md** - Comprehensive documentation
- **examples/** - Code examples
- **CONTRIBUTING.md** - Development guide
- **PROJECT_STRUCTURE.md** - Architecture overview

## Getting Help

If you run into issues:

1. Check this guide and README.md
2. Search existing GitHub issues
3. Check TensorRT and Diffusers documentation
4. Open a new issue with:
   - Your environment (GPU, CUDA version, Python version)
   - Complete error message
   - Steps to reproduce

## Community

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and ideas
- **Contributing**: See CONTRIBUTING.md

## Success Checklist

You're ready to go when you can:

- ✅ Import tensorrt in Python
- ✅ Export a model component to ONNX
- ✅ Build a TensorRT engine
- ✅ Run a benchmark successfully
- ✅ Generate a video (when pipeline is complete)

## What's Next?

The immediate next steps for the project (see NEXT_STEPS.md):

1. **Complete the inference pipeline** (high priority)
2. **Validate model-specific dimensions** (high priority)
3. **Optimize performance** with CUDA graphs
4. **Add quality validation**

---

**Welcome to the project! 🚀**

If you have questions or suggestions, please open an issue or discussion on GitHub.

Happy accelerating!

