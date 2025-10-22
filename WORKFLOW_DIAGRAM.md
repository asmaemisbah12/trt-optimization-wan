# Visual Workflow: Local RTX 4060 → RunPod A100/5090

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    YOUR LOCAL MACHINE (RTX 4060)                        │
│                                                                         │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐        │
│  │   1. Clone   │ ───▶ │  2. Build    │ ───▶ │  3. Push to  │        │
│  │   Project    │      │    Docker    │      │  Docker Hub  │        │
│  │              │      │    Image     │      │  or Save TAR │        │
│  └──────────────┘      └──────────────┘      └──────────────┘        │
│                              │                       │                 │
│                        15GB image              wan-trt:v1.0           │
│                        20-30 min              YOUR_USERNAME/          │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ Transfer
                                   │ (Docker Hub or Upload)
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    RUNPOD (A100 80GB or RTX 5090 32GB)                  │
│                                                                         │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐        │
│  │   4. Deploy  │ ───▶ │  5. Pull/    │ ───▶ │  6. Start    │        │
│  │   Instance   │      │    Load      │      │  Container   │        │
│  │              │      │    Image     │      │  with GPU    │        │
│  └──────────────┘      └──────────────┘      └──────────────┘        │
│                                                      │                 │
│                                                      ▼                 │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │              INSIDE CONTAINER (GPU Access)                   │     │
│  │                                                              │     │
│  │  ┌──────────────┐              ┌──────────────┐            │     │
│  │  │   7. Export  │              │   8. Export  │            │     │
│  │  │  Transformer │              │     VAE      │            │     │
│  │  │   to ONNX    │              │   to ONNX    │            │     │
│  │  │   (bf16)     │              │   (fp32)     │            │     │
│  │  │  15-25 min   │              │   5-10 min   │            │     │
│  │  │   ~28GB      │              │   ~1-2GB     │            │     │
│  │  └──────────────┘              └──────────────┘            │     │
│  │         │                              │                    │     │
│  │         └──────────────┬───────────────┘                    │     │
│  │                        ▼                                    │     │
│  │  ┌──────────────────────────────────────────────┐          │     │
│  │  │          9. Build TRT Engines                │          │     │
│  │  │                                              │          │     │
│  │  │  ┌──────────────┐    ┌──────────────┐      │          │     │
│  │  │  │ Transformer  │    │     VAE      │      │          │     │
│  │  │  │  TRT (fp16)  │    │  TRT (fp32)  │      │          │     │
│  │  │  │  30-60 min   │    │  10-15 min   │      │          │     │
│  │  │  │   ~14GB      │    │  ~500MB-1GB  │      │          │     │
│  │  │  └──────────────┘    └──────────────┘      │          │     │
│  │  │                                              │          │     │
│  │  │  GPU-Specific Compilation!                  │          │     │
│  │  │  Engines only work on same GPU type         │          │     │
│  │  └──────────────────────────────────────────────┘          │     │
│  │                        │                                    │     │
│  │                        ▼                                    │     │
│  │  ┌──────────────────────────────────────────────┐          │     │
│  │  │          10. Benchmark & Test                │          │     │
│  │  │                                              │          │     │
│  │  │  • PyTorch Baseline                         │          │     │
│  │  │  • TRT Inference (when ready)               │          │     │
│  │  │  • Compare Performance                      │          │     │
│  │  │                                              │          │     │
│  │  │  Expected: 1.5-2.5× speedup                 │          │     │
│  │  └──────────────────────────────────────────────┘          │     │
│  │                        │                                    │     │
│  │                        ▼                                    │     │
│  │  ┌──────────────────────────────────────────────┐          │     │
│  │  │          11. Save Results                    │          │     │
│  │  │                                              │          │     │
│  │  │  • TRT Engines (for reuse)                  │          │     │
│  │  │  • Benchmark Results                        │          │     │
│  │  │  • Generated Videos                         │          │     │
│  │  └──────────────────────────────────────────────┘          │     │
│  │                                                              │     │
│  └─────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Timeline & Costs

### Phase 1: Local (RTX 4060)
```
Time:  30-40 minutes
Cost:  $0 (uses your hardware)
Tasks: • Build Docker image
       • Push to Docker Hub
```

### Phase 2: RunPod (A100 or RTX 5090)
```
Time:  2-3 hours
Cost:  A100: $2.70-5.70 | RTX 5090: $1.20-2.10
Tasks: • Pull image
       • Export ONNX models (2x)
       • Build TRT engines (2x)
       • Run benchmarks
```

### Total
```
Time:  ~3-4 hours
Cost:  $1-6 (depending on GPU choice and spot vs on-demand)
```

---

## File Flow

```
wan_trt/
├── Local Build
│   └── Docker Image (15GB) ──────────────┐
│                                          │
├── Transfer ─────────────────────────────┤
│                                          │
└── RunPod Processing                     │
    ├── Pull/Load Image ◀─────────────────┘
    │
    ├── outputs/onnx/  (Created on RunPod)
    │   ├── transformer_bf16.onnx  (~28GB)
    │   └── vae_fp32.onnx         (~1-2GB)
    │
    ├── outputs/engines/  (Built on RunPod)
    │   ├── transformer_bf16.trt  (~14GB)  ◀── GPU-specific!
    │   └── vae_fp32.trt         (~500MB-1GB)
    │
    ├── outputs/benchmarks/  (Results)
    │   ├── PyTorch_FP16_*.json
    │   ├── TensorRT_*.json
    │   └── comparison.txt
    │
    └── outputs/videos/  (Generated videos)
        └── *.mp4
```

---

## Data Persistence Strategy

### Volumes on RunPod:
```bash
-v /workspace/wan_outputs:/workspace/wan_trt/outputs
   └── Persists between container restarts
   └── Keeps ONNX models + TRT engines + results

-v /root/.cache:/root/.cache
   └── Caches HuggingFace model downloads (~60GB)
   └── Reused across container restarts
```

### What's Reusable:
✅ **ONNX Models** - Portable across GPUs  
✅ **HuggingFace Cache** - Model weights  
❌ **TRT Engines** - GPU-specific, rebuild on different GPU

---

## GPU Comparison

### A100 (80GB) - Best Choice
```
Pros:
✅ 80GB VRAM - plenty for full model
✅ Fastest TRT engine builds (~30-40 min)
✅ Can handle larger batch sizes
✅ Best for development and testing

Cons:
❌ More expensive (~$1.89/hr on-demand)
```

### RTX 5090 (32GB) - Good Budget Option
```
Pros:
✅ 32GB VRAM - sufficient for Wan2.2
✅ Much cheaper (~$0.69/hr on-demand)
✅ Still fast builds (~45-60 min)

Cons:
❌ Less headroom for experiments
❌ Slightly slower than A100
```

### RTX 4090 (24GB) - Not Recommended
```
❌ 24GB VRAM - too tight for full model
❌ Will need model offloading
❌ Slower and more complex
```

---

## Critical Points

### ⚠️ Important Notes:

1. **TRT Engines are GPU-Specific**
   - A100 engines won't work on RTX 5090
   - RTX 5090 engines won't work on A100
   - Need to rebuild if changing GPUs

2. **ONNX Models are Portable**
   - Build once, use anywhere
   - Can export on A100, build engines on 5090
   - But export is fast anyway

3. **First Run Downloads Model**
   - ~60GB Wan2.2 model from HuggingFace
   - Cached in `/root/.cache`
   - Subsequent runs are faster

4. **Disk Space Management**
   ```
   Model cache:        ~60GB
   ONNX models:        ~30GB
   TRT engines:        ~15GB
   Docker image:       ~15GB
   Working space:      ~20GB
   ────────────────────────
   Total needed:      ~140GB (recommend 200GB)
   ```

5. **Cost Optimization**
   - Use **Spot instances** (50% cheaper)
   - Stop pod when not using
   - Use persistent storage for artifacts
   - Consider RTX 5090 for production

---

## Workflow Verification

### Checkpoints to Validate:

```
[ ] Step 1-3 (Local)
    → Docker image exists locally
    → Image on Docker Hub or TAR file created

[ ] Step 4-6 (RunPod Setup)
    → Instance running
    → Container started
    → GPU accessible (nvidia-smi works)

[ ] Step 7-8 (ONNX Export)
    → transformer_bf16.onnx exists (~28GB)
    → vae_fp32.onnx exists (~1-2GB)
    → Both validated without errors

[ ] Step 9 (TRT Build)
    → transformer_bf16.trt exists (~14GB)
    → vae_fp32.trt exists (~500MB-1GB)
    → Both load without errors

[ ] Step 10 (Testing)
    → PyTorch baseline completes
    → Benchmark results saved
    → Performance data collected

[ ] Step 11 (Backup)
    → Engines archived
    → Results saved
    → Ready for deployment
```

---

## Next Phase: Production Deployment

Once engines are built:

```
┌────────────────────────────┐
│  Built TRT Engines Ready   │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Complete TRT Pipeline     │
│  (NEXT_STEPS.md Phase 1)   │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Test Video Generation     │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Optimize & Deploy         │
│  • CUDA Graphs             │
│  • Batch Processing        │
│  • API Server              │
└────────────────────────────┘
```

---

## Quick Reference: Time to Each Milestone

| Milestone | Time from Start | What You Have |
|-----------|-----------------|---------------|
| Docker Built | 30 min | Ready to deploy image |
| RunPod Started | 40 min | Container running |
| ONNX Exported | 1.5 hrs | Portable model files |
| TRT Engines Built | 2.5-3 hrs | GPU-specific engines |
| Benchmarked | 3.5-4 hrs | Performance metrics |
| Production Ready | 4-6 hrs | Full TRT pipeline (after integration) |

---

**Start with:** `QUICK_START_COMMANDS.md` for copy-paste commands  
**Read full details:** `DEPLOYMENT_GUIDE.md`  
**Stuck?** Check troubleshooting sections in both guides

