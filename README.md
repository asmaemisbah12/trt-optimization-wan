# Wan2.2-T2V TensorRT Acceleration Project

Accelerate Wan2.2-T2V-A14B (Hugging Face: Wan-AI/Wan2.2-T2V-A14B-Diffusers) video generation with TensorRT optimization.

## Project Goals

- **Primary**: Reduce end-to-end inference time by ≥1.5× vs PyTorch FP16 baseline
- **Secondary**: Maintain perceptual parity (PSNR/SSIM within tolerance)
- **Operational**: Support dynamic shapes (8-81 frames, up to 1280×720), single-GPU production

## Architecture

### Optimization Strategy

1. **DiT/UNet**: FP16 TensorRT engine (maximum speed)
2. **VAE**: FP32 TensorRT engine (numerical stability)
3. **Runtime**: CUDA Graphs, memory reuse, multi-profile engines

### Pipeline Flow

```
Text/Audio → Encoder (FP16) → DiT/Transformer (FP16 TRT) → Latents → VAE Decoder (FP32 TRT) → Video
```

## Project Structure

```
wan_trt/
├── src/
│   ├── model/              # Model loading and inspection
│   ├── export/             # ONNX export utilities
│   ├── tensorrt/           # TRT engine building and runtime
│   ├── benchmark/          # Benchmarking and validation
│   └── utils/              # Common utilities
├── configs/                # Configuration files
├── scripts/                # Helper scripts
├── outputs/                # Generated engines and results
├── tests/                  # Test files
└── requirements.txt
```

## Quick Start

### 1. Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install TensorRT (follow NVIDIA instructions for your CUDA version)
# https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
```

### 2. Export to ONNX

```bash
python scripts/export_model.py \
    --model_id Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --output_dir outputs/onnx \
    --component dit \
    --precision fp16
```

### 3. Build TensorRT Engines

```bash
python scripts/build_engines.py \
    --onnx_path outputs/onnx/dit_fp16.onnx \
    --output_path outputs/engines/dit_fp16.trt \
    --precision fp16 \
    --min_frames 8 --opt_frames 16 --max_frames 81
```

### 4. Run Inference

```bash
python scripts/run_inference.py \
    --prompt "A beautiful sunset over the ocean" \
    --engine_dir outputs/engines \
    --output_path outputs/videos/output.mp4
```

### 5. Benchmark

```bash
python scripts/benchmark.py \
    --engine_dir outputs/engines \
    --baseline pytorch \
    --num_runs 10
```

## Workflow Phases

### Phase 1: Initial TRT Integration (Current)
- [x] Project setup
- [ ] Export DiT/UNet to FP16 ONNX
- [ ] Build FP16 TRT engine for DiT
- [ ] Integrate TRT runtime with PyTorch VAE (FP32)
- [ ] Smoke test and baseline metrics

### Phase 2: Full TRT Pipeline
- [ ] Export VAE to FP32 ONNX
- [ ] Build FP32 TRT engine for VAE
- [ ] Integrate full TRT pipeline
- [ ] Implement CUDA Graph capture
- [ ] Comprehensive benchmarking

### Phase 3: Advanced Optimization (Optional)
- [ ] Multi-profile engine optimization
- [ ] INT8/FP8 quantization exploration
- [ ] Memory pool and async execution
- [ ] Production deployment packaging

## Benchmarking Metrics

- **Latency**: End-to-end time (s per clip), ms/frame
- **Memory**: Peak VRAM usage
- **Quality**: PSNR, SSIM, LPIPS vs baseline
- **Visual**: Artifact inspection

## Configuration

Edit `configs/config.yaml` for:
- Model paths and cache directories
- Engine build parameters (workspace size, profiles)
- Runtime settings (CUDA graphs, memory pools)
- Benchmark parameters

## Technical Details

### Precision Strategy
- **DiT/UNet**: FP16 for 2-3× speedup with acceptable quality
- **VAE**: FP32 to prevent reconstruction artifacts
- **Boundary**: GPU-side FP16→FP32 cast before VAE

### Dynamic Shape Support
- **Frames**: 8, 16, 32, 81 (common presets)
- **Resolution**: 512×512 to 1280×720 (latent space scaled)
- **Profiles**: min/opt/max for optimal kernel selection

### Non-Calibration Optimizations
- CUDA Graph capture for repeated operations
- Persistent device buffers and memory reuse
- Pre-allocated output tensors
- Kernel warmup and profile selection
- Operator fusion via TRT

## Troubleshooting

### ONNX Export Issues
- Custom ops (rotary embeddings): Use Optimum exporters
- Shape inference failures: Add explicit shape annotations
- Opset version: Use opset 17+ for newer operators

### TensorRT Build Issues
- Unsupported ops: Check TRT plugin support
- Memory errors: Increase workspace size (--workspace_size 8192)
- Parse errors: Validate ONNX with `onnx.checker`

### Runtime Issues
- OOM: Reduce max profile dimensions or use smaller batches
- Numerical drift: Keep sensitive layers in FP32
- Profile selection: Ensure input shapes within engine profiles

## References

- [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Optimum ONNX Export](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model)
- [CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)

## License

See LICENSE file. Model weights subject to original model license.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

