# Step-by-Step Deployment Guide: Local → RunPod

Complete guide to build locally on RTX 4060 and deploy to RunPod A100/RTX 5090.

---

## 🎯 Deployment Strategy

1. **Build Docker image** on your local machine (RTX 4060)
2. **Push image** to Docker Hub (or save as file)
3. **Pull/Load image** on RunPod
4. **Export ONNX** on RunPod (needs model weights ~60GB)
5. **Build TRT engines** on RunPod (GPU-specific)
6. **Run inference** and benchmark

---

## 📋 Prerequisites

### Local Machine (RTX 4060)
- ✅ Docker Desktop installed
- ✅ Git installed
- ✅ 100GB+ free disk space
- ✅ Internet connection

### RunPod Account
- ✅ RunPod account created
- ✅ Credit/payment method added
- ✅ Choose: A100 (80GB) or RTX 5090 (32GB)

---

## 🚀 Phase 1: Local Setup & Docker Build

### Step 1.1: Clone and Prepare Project

```bash
# Clone (or you already have it)
cd C:\Users\msira\Downloads\wan_trt

# Verify files are up to date
git status

# Check critical files
cat requirements.txt | grep torch
# Should show: torch>=2.4.0

cat Dockerfile | grep pytorch
# Should show: FROM nvcr.io/nvidia/pytorch:24.09-py3
```

### Step 1.2: Create Docker Hub Account (If Needed)

**Option A: Use Docker Hub (Recommended)**
```bash
# Create account at https://hub.docker.com
# Free account: 1 private repo, unlimited public repos

# Login
docker login
# Enter username and password
```

**Option B: Save as TAR file (Alternative)**
- Good if Docker Hub is blocked or you prefer local transfer
- File will be ~10-15GB

### Step 1.3: Build Docker Image

```bash
# Build the image (takes 20-30 minutes)
docker build -t wan-trt:v1.0 .

# If you get errors, add no-cache:
docker build --no-cache -t wan-trt:v1.0 .

# Verify build
docker images | grep wan-trt
```

**Expected Output:**
```
REPOSITORY   TAG    IMAGE ID       CREATED          SIZE
wan-trt      v1.0   abc123def456   2 minutes ago    15GB
```

### Step 1.4: Test Docker Image Locally (Optional)

```bash
# Quick test - check imports
docker run --rm wan-trt:v1.0 python -c "
import torch
import diffusers
print(f'Torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Diffusers: {diffusers.__version__}')
"

# Expected output:
# Torch: 2.4.0 (or higher)
# CUDA available: True (if GPU passthrough works) or False (OK for now)
# Diffusers: 0.30.0 (or higher)
```

### Step 1.5: Push to Docker Hub

**Option A: Push to Docker Hub**
```bash
# Tag with your Docker Hub username
docker tag wan-trt:v1.0 YOUR_USERNAME/wan-trt:v1.0

# Push to Docker Hub (takes 10-20 minutes)
docker push YOUR_USERNAME/wan-trt:v1.0

# Verify at https://hub.docker.com/r/YOUR_USERNAME/wan-trt
```

**Option B: Save as TAR file**
```bash
# Save image to file (takes 10-15 minutes)
docker save wan-trt:v1.0 -o wan-trt-v1.0.tar

# Compress (optional, saves space)
gzip wan-trt-v1.0.tar

# Result: wan-trt-v1.0.tar.gz (~8-12GB compressed)
```

---

## ☁️ Phase 2: RunPod Setup

### Step 2.1: Create RunPod Instance

1. **Go to RunPod:** https://www.runpod.io/console/pods
2. **Click "Deploy"** or "GPU Pods"
3. **Choose GPU:**
   - **A100 (80GB)** - Best choice, can handle full model
   - **RTX 5090 (32GB)** - Also works, more economical
   - Avoid RTX 4090 (24GB) - too small for full Wan2.2
4. **Select Template:** Start with "RunPod Pytorch" or "NVIDIA PyTorch"
5. **Configure:**
   - **Disk Space:** 200GB minimum (recommend 300GB)
   - **Ports:** Expose port 8888 (Jupyter) and 22 (SSH)
6. **Deploy** and wait for instance to start

### Step 2.2: Connect to RunPod

**Method 1: RunPod Web Terminal (Easiest)**
```bash
# Click "Connect" → "Start Web Terminal"
# Opens in browser
```

**Method 2: SSH (Better for large transfers)**
```bash
# Get SSH command from RunPod
# Will look like: ssh root@<pod-id>.runpod.io -p 12345

# Connect
ssh root@<pod-id>.runpod.io -p <port>
```

**Method 3: Jupyter (For interactive work)**
```bash
# Get Jupyter URL from RunPod dashboard
# Open in browser
```

### Step 2.3: Load Docker Image on RunPod

**Option A: Pull from Docker Hub**
```bash
# On RunPod terminal
docker login
# Enter your credentials

# Pull image
docker pull YOUR_USERNAME/wan-trt:v1.0

# Verify
docker images
```

**Option B: Transfer TAR file**
```bash
# On local machine, upload to RunPod
scp -P <port> wan-trt-v1.0.tar.gz root@<pod-id>.runpod.io:/workspace/

# On RunPod, load image
cd /workspace
gunzip wan-trt-v1.0.tar.gz  # if compressed
docker load -i wan-trt-v1.0.tar

# Verify
docker images
```

---

## 🔧 Phase 3: Model Export on RunPod

### Step 3.1: Start Docker Container

```bash
# On RunPod
docker run --gpus all -it \
    --shm-size=16g \
    -v /workspace/wan_outputs:/workspace/wan_trt/outputs \
    -v /root/.cache:/root/.cache \
    wan-trt:v1.0 bash

# You're now inside the container
cd /workspace/wan_trt
```

**Explanation:**
- `--gpus all` - Access all GPUs
- `--shm-size=16g` - Shared memory for large models
- `-v /workspace/...` - Mount volumes for persistence
- `-v /root/.cache` - Reuse HuggingFace cache

### Step 3.2: Verify GPU Access

```bash
# Inside container
nvidia-smi

# Should show your A100 or RTX 5090

python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

### Step 3.3: Export Transformer to ONNX

```bash
# Export transformer (takes 10-20 minutes, downloads model first time)
python scripts/export_model.py \
    --component transformer \
    --precision bf16 \
    --num_frames 81 \
    --height 720 \
    --width 1280 \
    --validate

# Check output
ls -lh outputs/onnx/
# Should see: transformer_bf16.onnx (~28GB)
```

**If model download is slow:**
```bash
# Pre-download model
python -c "
from src.model.loader import load_pipeline
pipe = load_pipeline()
print('Model downloaded and cached')
"
# Then run export script
```

### Step 3.4: Export VAE to ONNX

```bash
# Export VAE (faster, ~5 minutes)
python scripts/export_model.py \
    --component vae \
    --precision fp32 \
    --num_frames 81 \
    --height 720 \
    --width 1280 \
    --validate

# Check output
ls -lh outputs/onnx/
# Should see: vae_fp32.onnx (~1-2GB)
```

---

## ⚡ Phase 4: Build TensorRT Engines

### Step 4.1: Build Transformer Engine

```bash
# Build transformer TRT engine (takes 30-60 minutes!)
python scripts/build_engines.py \
    --onnx_path outputs/onnx/transformer_bf16.onnx \
    --precision fp16 \
    --component transformer_bf16 \
    --workspace_size 8192

# Monitor progress
# You'll see: "Building TensorRT engine (this may take several minutes)..."
```

**Expected time:**
- A100: ~30-40 minutes
- RTX 5090: ~45-60 minutes

**Output:**
```
outputs/engines/transformer_bf16.trt (~14GB)
```

### Step 4.2: Build VAE Engine

```bash
# Build VAE TRT engine (faster, ~10-15 minutes)
python scripts/build_engines.py \
    --onnx_path outputs/onnx/vae_fp32.onnx \
    --precision fp32 \
    --component vae_fp32 \
    --workspace_size 4096

# Output: outputs/engines/vae_fp32.trt (~500MB-1GB)
```

### Step 4.3: Verify Engines

```bash
# Check engines
ls -lh outputs/engines/

# Should see:
# transformer_bf16.trt  (~14GB)
# vae_fp32.trt         (~500MB-1GB)

# Quick test loading
python -c "
from src.tensorrt.engine_builder import load_engine
engine = load_engine('outputs/engines/transformer_bf16.trt')
print(f'✅ Transformer engine loaded')
print(f'   Bindings: {engine.num_bindings}')
print(f'   Profiles: {engine.num_optimization_profiles}')
"
```

---

## 🧪 Phase 5: Test & Benchmark

### Step 5.1: Run PyTorch Baseline

```bash
# First, run PyTorch baseline for comparison
python scripts/benchmark.py \
    --baseline pytorch \
    --num_warmup_runs 2 \
    --num_test_runs 5

# This will take time (~20-30 minutes per run)
# Results saved to: outputs/benchmarks/
```

### Step 5.2: Run TensorRT Inference (When Pipeline Complete)

```bash
# Once TRT pipeline is integrated (see NEXT_STEPS.md)
python scripts/run_inference.py \
    --prompt "A serene lake at sunset with mountains in the background" \
    --num_frames 81 \
    --output_path outputs/videos/test.mp4

# Or use Makefile
make infer
```

### Step 5.3: Compare Performance

```bash
# Run full comparison
python scripts/benchmark.py \
    --compare \
    --num_warmup_runs 3 \
    --num_test_runs 10

# View results
cat outputs/benchmarks/comparison.txt
```

---

## 📦 Phase 6: Save and Transfer Results

### Step 6.1: Save TRT Engines (for reuse)

```bash
# On RunPod, inside container
cd /workspace/wan_trt

# Create tarball of engines
tar -czf wan_trt_engines_a100.tar.gz outputs/engines/

# Exit container
exit

# On RunPod host, copy to persistent storage
cp /workspace/wan_outputs/wan_trt_engines_a100.tar.gz /workspace/
```

### Step 6.2: Download to Local Machine (Optional)

```bash
# On local machine
scp -P <port> root@<pod-id>.runpod.io:/workspace/wan_trt_engines_a100.tar.gz .

# Or use RunPod's file browser
```

### Step 6.3: Reuse Engines on Same GPU

```bash
# Next time on RunPod A100
docker run --gpus all -it \
    -v /workspace/wan_outputs:/workspace/wan_trt/outputs \
    wan-trt:v1.0 bash

# Engines are already in outputs/engines/ (via volume mount)
# Just run inference!
```

---

## ⚠️ Troubleshooting

### Issue: Docker build fails on Windows

**Solution:**
```bash
# Use WSL2 if available
wsl
cd /mnt/c/Users/msira/Downloads/wan_trt
docker build -t wan-trt:v1.0 .

# Or use Docker Desktop with WSL2 backend
```

### Issue: Out of memory during ONNX export

**Solution:**
```bash
# Use smaller batch or reduce frames temporarily
python scripts/export_model.py \
    --component transformer \
    --precision bf16 \
    --num_frames 16  # Smaller for export, engines handle 81
    --validate
```

### Issue: TRT engine build fails

**Solution:**
```bash
# Check ONNX model first
python -c "
import onnx
model = onnx.load('outputs/onnx/transformer_bf16.onnx')
onnx.checker.check_model(model)
print('ONNX model OK')
"

# Try with less workspace
python scripts/build_engines.py \
    --workspace_size 4096  # Reduced from 8192
    ...
```

### Issue: RunPod keeps stopping/hibernating

**Solution:**
```bash
# Run builds in tmux/screen
tmux new -s build

# Inside tmux, run builds
python scripts/build_engines.py ...

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t build
```

---

## 💰 Cost Estimation

### RunPod Costs (Approximate)

**A100 (80GB):**
- **On-Demand:** ~$1.89/hour
- **Spot:** ~$0.89/hour
- **Total for setup:** ~$5-10 (3-5 hours)

**RTX 5090 (32GB):**
- **On-Demand:** ~$0.69/hour
- **Spot:** ~$0.39/hour
- **Total for setup:** ~$3-6 (3-5 hours)

**Breakdown:**
- ONNX Export: 30-40 min
- TRT Engine Build: 60-90 min
- Testing: 30-60 min
- **Total: 2-3 hours minimum**

**Tips to Save Money:**
- Use **Spot instances** (cheaper but can be interrupted)
- Use **RunPod's file persistence** to avoid rebuilding
- **Stop pod** when not using (engines are saved)

---

## 📋 Complete Command Summary

### On Local Machine:
```bash
# 1. Build Docker image
docker build -t wan-trt:v1.0 .

# 2. Push to Docker Hub
docker tag wan-trt:v1.0 YOUR_USERNAME/wan-trt:v1.0
docker push YOUR_USERNAME/wan-trt:v1.0
```

### On RunPod:
```bash
# 3. Pull and run
docker pull YOUR_USERNAME/wan-trt:v1.0
docker run --gpus all -it --shm-size=16g \
    -v /workspace/wan_outputs:/workspace/wan_trt/outputs \
    -v /root/.cache:/root/.cache \
    wan-trt:v1.0 bash

# 4. Export ONNX
python scripts/export_model.py --component transformer --precision bf16 --num_frames 81 --validate
python scripts/export_model.py --component vae --precision fp32 --num_frames 81 --validate

# 5. Build TRT engines
python scripts/build_engines.py --onnx_path outputs/onnx/transformer_bf16.onnx --precision fp16 --component transformer_bf16
python scripts/build_engines.py --onnx_path outputs/onnx/vae_fp32.onnx --precision fp32 --component vae_fp32

# 6. Benchmark
python scripts/benchmark.py --compare --num_test_runs 10
```

---

## 🎯 Quick Start Checklist

- [ ] Docker Desktop installed locally
- [ ] Docker image built (`docker build -t wan-trt:v1.0 .`)
- [ ] Image pushed to Docker Hub or saved as TAR
- [ ] RunPod account created and funded
- [ ] RunPod instance deployed (A100 or RTX 5090)
- [ ] Docker image pulled/loaded on RunPod
- [ ] Container started with GPU access
- [ ] ONNX models exported (transformer + VAE)
- [ ] TRT engines built (transformer + VAE)
- [ ] Baseline benchmark completed
- [ ] TRT benchmark completed (when pipeline ready)

---

## 📚 Next Steps After Setup

Once engines are built:

1. **Complete TRT Pipeline** (see `NEXT_STEPS.md` Phase 1.1)
   - Integrate engines into full inference pipeline
   - Test video generation end-to-end
   
2. **Optimize Further**
   - Enable CUDA graphs
   - Try INT8 quantization
   
3. **Deploy for Production**
   - Create inference API
   - Set up batch processing
   - Optimize for your use case

---

## 🆘 Getting Help

If you encounter issues:

1. **Check logs:** `docker logs <container-id>`
2. **Check GPU:** `nvidia-smi` inside container
3. **Check disk space:** `df -h`
4. **Review:** `ANALYSIS_UPDATES_NEEDED.md` and `UPDATES_COMPLETED.md`
5. **Open issue** with error logs

---

**Ready to start? Begin with Phase 1, Step 1.1!** 🚀

