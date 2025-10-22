# Quick Start Command Reference

Copy-paste commands for fast deployment. **Read DEPLOYMENT_GUIDE.md for full explanations.**

---

## 🏠 On Your Local Machine (RTX 4060)

### 1. Build Docker Image
```bash
cd C:\Users\msira\Downloads\wan_trt

# Build (takes 20-30 minutes)
docker build -t wan-trt:v1.0 .

# Verify
docker images | grep wan-trt
```

### 2. Push to Docker Hub
```bash
# Login to Docker Hub
docker login

# Tag (replace YOUR_USERNAME with your Docker Hub username)
docker tag wan-trt:v1.0 richdaleai/wan-trt:v1.0

# Push (takes 10-20 minutes)
docker push richdaleai/wan-trt:v1.0
```

**Alternative: Save as file**
```bash
# If Docker Hub doesn't work
docker save wan-trt:v1.0 | gzip > wan-trt-v1.0.tar.gz
# Upload to RunPod manually
```

---

## ☁️ On RunPod (A100 or RTX 5090)

### 3. Pull Docker Image
```bash
# Login
docker login

# Pull (replace YOUR_USERNAME)
docker pull YOUR_USERNAME/wan-trt:v1.0
```

### 4. Start Container
```bash
# Run with GPU access
docker run --gpus all -it \
    --name wan-trt-work \
    --shm-size=16g \
    -v /workspace/wan_outputs:/workspace/wan_trt/outputs \
    -v /root/.cache:/root/.cache \
    YOUR_USERNAME/wan-trt:v1.0 bash

# You're now inside the container
cd /workspace/wan_trt
```

### 5. Verify Setup
```bash
# Check GPU
nvidia-smi

# Check Python environment
python -c "
import torch
from diffusers import WanPipeline, AutoencoderKLWan
print(f'✅ Torch: {torch.__version__}')
print(f'✅ CUDA: {torch.cuda.is_available()}')
print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'✅ WanPipeline available')
"
```

### 6. Export Transformer to ONNX
```bash
# Takes 15-25 minutes (downloads model first time)
python scripts/export_model.py \
    --component transformer \
    --precision bf16 \
    --num_frames 81 \
    --height 720 \
    --width 1280 \
    --validate

# Verify
ls -lh outputs/onnx/transformer_bf16.onnx
# Should be ~28GB
```

### 7. Export VAE to ONNX
```bash
# Takes 5-10 minutes
python scripts/export_model.py \
    --component vae \
    --precision fp32 \
    --num_frames 81 \
    --height 720 \
    --width 1280 \
    --validate

# Verify
ls -lh outputs/onnx/vae_fp32.onnx
# Should be ~1-2GB
```

### 8. Build Transformer TRT Engine
```bash
# Takes 30-60 minutes! (GPU-specific compilation)
python scripts/build_engines.py \
    --onnx_path outputs/onnx/transformer_bf16.onnx \
    --precision fp16 \
    --component transformer_bf16 \
    --workspace_size 8192

# Verify
ls -lh outputs/engines/transformer_bf16.trt
# Should be ~14GB
```

### 9. Build VAE TRT Engine
```bash
# Takes 10-15 minutes
python scripts/build_engines.py \
    --onnx_path outputs/onnx/vae_fp32.onnx \
    --precision fp32 \
    --component vae_fp32 \
    --workspace_size 4096

# Verify
ls -lh outputs/engines/vae_fp32.trt
# Should be ~500MB-1GB
```

### 10. Run Baseline Benchmark
```bash
# Run PyTorch baseline (takes ~30-40 minutes)
python scripts/benchmark.py \
    --baseline pytorch \
    --num_warmup_runs 2 \
    --num_test_runs 5

# Check results
cat outputs/benchmarks/PyTorch_FP16_*.json
```

### 11. Save Your Work
```bash
# Create backup of engines
cd /workspace/wan_trt
tar -czf ../wan_engines_backup.tar.gz outputs/engines/

# Exit container (work is saved in volumes)
exit

# On RunPod host, verify
ls -lh /workspace/wan_engines_backup.tar.gz
```

---

## 🔄 Restarting Work

### Resume on Same RunPod Instance
```bash
# If container stopped
docker start -i wan-trt-work

# Or create new container with saved volumes
docker run --gpus all -it \
    --shm-size=16g \
    -v /workspace/wan_outputs:/workspace/wan_trt/outputs \
    -v /root/.cache:/root/.cache \
    YOUR_USERNAME/wan-trt:v1.0 bash

cd /workspace/wan_trt
# Your engines are still in outputs/engines/
```

### Transfer to Different GPU
```bash
# Engines are GPU-specific! Need to rebuild on new GPU.
# But ONNX models are portable:

# On new RunPod instance:
# 1. Pull Docker image
# 2. Mount volume with ONNX models
# 3. Rebuild TRT engines (steps 8-9)
# ONNX export (steps 6-7) doesn't need to be repeated
```

---

## 🧪 Using Makefile (Convenience)

### All-in-one commands:
```bash
# Export both models
make export-dit export-vae

# Build both engines
make build-dit build-vae

# Run full benchmark
make benchmark
```

---

## 📦 File Sizes to Expect

| File | Size | Time to Create |
|------|------|----------------|
| Docker image | ~15GB | 20-30 min |
| transformer_bf16.onnx | ~28GB | 15-25 min |
| vae_fp32.onnx | ~1-2GB | 5-10 min |
| transformer_bf16.trt | ~14GB | 30-60 min |
| vae_fp32.trt | ~500MB-1GB | 10-15 min |
| **Total** | **~60GB** | **~2-3 hours** |

---

## ⏱️ Total Time Estimate

### First-Time Setup:
- **Local:** 30-40 minutes (build + push Docker)
- **RunPod:** 2-3 hours (download + export + build)
- **Total:** 2.5-4 hours

### Subsequent Runs (engines cached):
- 5-10 minutes (just run inference)

---

## 💰 RunPod Cost Estimate

### A100 (80GB) - Best Performance:
- **Spot:** ~$0.89/hour × 3 hours = **~$2.70**
- **On-Demand:** ~$1.89/hour × 3 hours = **~$5.70**

### RTX 5090 (32GB) - More Economical:
- **Spot:** ~$0.39/hour × 3 hours = **~$1.20**
- **On-Demand:** ~$0.69/hour × 3 hours = **~$2.10**

**Recommendation:** Use Spot instances, but keep terminal open or use tmux!

---

## ⚡ Speed Run (Minimum Commands)

```bash
# === LOCAL (RTX 4060) ===
docker build -t wan-trt:v1.0 .
docker tag wan-trt:v1.0 YOUR_USERNAME/wan-trt:v1.0
docker push YOUR_USERNAME/wan-trt:v1.0

# === RUNPOD (A100/RTX 5090) ===
docker pull YOUR_USERNAME/wan-trt:v1.0
docker run --gpus all -it --shm-size=16g \
    -v /workspace/wan_outputs:/workspace/wan_trt/outputs \
    -v /root/.cache:/root/.cache \
    YOUR_USERNAME/wan-trt:v1.0 bash

cd /workspace/wan_trt

# Export ONNX
make export-dit export-vae

# Build TRT engines (long step!)
make build-dit build-vae

# Benchmark
make benchmark
```

---

## 🆘 Common Issues & Quick Fixes

### Docker build fails
```bash
# Clear cache and retry
docker system prune -a
docker build --no-cache -t wan-trt:v1.0 .
```

### Out of memory on RunPod
```bash
# Check GPU memory
nvidia-smi

# Use smaller frames for initial test
python scripts/export_model.py --num_frames 16 ...
```

### TRT build hangs
```bash
# Check if still running (in another terminal)
nvidia-smi  # Should show ~100% GPU usage

# Or reduce workspace
python scripts/build_engines.py --workspace_size 4096 ...
```

### RunPod connection lost
```bash
# Use tmux to keep sessions alive
tmux new -s build
# Run your commands
# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t build
```

---

## 📝 Progress Tracking

Use this checklist:

```
Local Machine (RTX 4060):
[ ] Docker installed and running
[ ] Project cloned/downloaded
[ ] Docker image built (wan-trt:v1.0)
[ ] Image pushed to Docker Hub

RunPod Setup:
[ ] Account created and funded
[ ] A100 or RTX 5090 instance deployed
[ ] Docker image pulled
[ ] Container started with GPU access
[ ] GPU verification passed (nvidia-smi)

Model Export:
[ ] Transformer exported to ONNX (~28GB)
[ ] VAE exported to ONNX (~1-2GB)
[ ] Both ONNX models validated

TRT Engine Build:
[ ] Transformer TRT engine built (~14GB)
[ ] VAE TRT engine built (~500MB-1GB)
[ ] Both engines tested (load without error)

Testing:
[ ] PyTorch baseline benchmark run
[ ] Results saved and reviewed
[ ] Engines backed up

Next:
[ ] Complete TRT inference pipeline (NEXT_STEPS.md)
[ ] Run TRT benchmark
[ ] Compare PyTorch vs TRT performance
```

---

## 🎯 Success Criteria

You've succeeded when:

✅ Docker image builds without errors  
✅ Container runs on RunPod with GPU access  
✅ Both ONNX models export successfully  
✅ Both TRT engines build without errors  
✅ PyTorch baseline completes a benchmark run  
✅ All files are ~60GB total in outputs/  

**Then you're ready for Phase 2:** Complete the TRT inference pipeline! (See `NEXT_STEPS.md`)

---

**Questions?** Check `DEPLOYMENT_GUIDE.md` for detailed explanations.

**Got this far?** You're 90% done with the hard part! 🎉

