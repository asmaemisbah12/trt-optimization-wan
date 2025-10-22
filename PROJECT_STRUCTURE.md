# Project Structure

Complete overview of the Wan TensorRT acceleration project structure.

## Directory Layout

```
wan_trt/
├── configs/                      # Configuration files
│   └── config.yaml              # Main configuration (model, TRT, benchmark settings)
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── model/                   # Model loading and inspection
│   │   ├── __init__.py
│   │   ├── loader.py           # Pipeline loading, component extraction
│   │   └── shapes.py           # Shape utilities, dummy input generation
│   │
│   ├── export/                  # ONNX export utilities
│   │   ├── __init__.py
│   │   └── onnx_exporter.py    # ONNX export, validation, optimization
│   │
│   ├── tensorrt/                # TensorRT engine building and runtime
│   │   ├── __init__.py
│   │   ├── engine_builder.py   # Engine building with multi-profile support
│   │   └── runtime.py          # TRT inference, CUDA graphs, memory management
│   │
│   ├── benchmark/               # Benchmarking and validation
│   │   ├── __init__.py
│   │   ├── metrics.py          # Quality metrics (PSNR, SSIM, LPIPS)
│   │   └── benchmark.py        # Benchmarking framework, result comparison
│   │
│   └── utils/                   # Common utilities
│       ├── __init__.py
│       ├── config.py           # Config loading, device selection
│       └── logging.py          # Logging setup
│
├── scripts/                     # Main execution scripts
│   ├── export_model.py         # Export components to ONNX
│   ├── build_engines.py        # Build TensorRT engines
│   ├── run_inference.py        # Run inference with TRT or PyTorch
│   └── benchmark.py            # Benchmark different configurations
│
├── examples/                    # Usage examples
│   ├── README.md
│   ├── simple_export.py        # Simple ONNX export example
│   └── simple_benchmark.py     # Simple benchmarking example
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_config.py          # Config utility tests
│   ├── test_shapes.py          # Shape utility tests
│   └── test_metrics.py         # Metric computation tests
│
├── outputs/                     # Generated files (gitignored)
│   ├── onnx/                   # ONNX models
│   ├── engines/                # TensorRT engines
│   ├── videos/                 # Generated videos
│   ├── benchmarks/             # Benchmark results
│   └── logs/                   # Log files
│
├── docs/                        # Documentation (optional, add as needed)
│
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── Makefile                     # Common commands
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Docker Compose configuration
├── .gitignore                   # Git ignore rules
├── .dockerignore               # Docker ignore rules
├── LICENSE                      # MIT License
├── README.md                    # Main documentation
├── QUICKSTART.md               # Quick start guide
├── CONTRIBUTING.md             # Contribution guidelines
└── PROJECT_STRUCTURE.md        # This file
```

## Key Components

### Configuration (`configs/`)

- **config.yaml**: Central configuration file
  - Model settings (ID, cache, precision)
  - ONNX export settings (opset, dynamic axes)
  - TensorRT settings (workspace, profiles)
  - Runtime settings (CUDA graphs, memory)
  - Inference settings (steps, guidance)
  - Benchmark settings (metrics, baselines)

### Source Code (`src/`)

#### Model Module (`src/model/`)

- **loader.py**: Load HF pipeline, extract submodules, inspect components
- **shapes.py**: Calculate latent shapes, create dummy inputs, define dynamic axes

#### Export Module (`src/export/`)

- **onnx_exporter.py**: Export to ONNX, validate models, optimize graphs, compare outputs

#### TensorRT Module (`src/tensorrt/`)

- **engine_builder.py**: Build TRT engines with optimization profiles
- **runtime.py**: TRT inference wrapper with CUDA graph support and buffer management

#### Benchmark Module (`src/benchmark/`)

- **metrics.py**: Quality metrics (PSNR, SSIM, LPIPS)
- **benchmark.py**: Benchmarking framework with timing, memory, and comparison

#### Utils Module (`src/utils/`)

- **config.py**: Configuration management, device selection, directory utilities
- **logging.py**: Logging setup and management

### Scripts (`scripts/`)

Entry points for the main workflows:

1. **export_model.py**: Export pipeline components to ONNX
2. **build_engines.py**: Build TensorRT engines from ONNX
3. **run_inference.py**: Run video generation (TRT or PyTorch)
4. **benchmark.py**: Benchmark and compare configurations

### Examples (`examples/`)

Simple, self-contained examples for learning:

- **simple_export.py**: Basic ONNX export workflow
- **simple_benchmark.py**: Basic benchmarking workflow

### Tests (`tests/`)

Unit tests for core functionality:

- Configuration loading
- Shape calculations
- Metric computations
- ONNX export (integration tests)
- TRT engine building (integration tests)

## Data Flow

### Training/Preparation Phase

```
HuggingFace Model → Load Pipeline → Extract Components → Export ONNX → Build TRT Engines
```

### Inference Phase

```
Text Prompt → Text Encoder (PyTorch) → DiT (TRT FP16) → Latents → VAE Decoder (TRT FP32) → Video
```

### Benchmark Phase

```
Test Configuration → Multiple Runs → Timing + Memory + Quality → Results → Comparison
```

## File Naming Conventions

### ONNX Files

Format: `{component}_{precision}.onnx`

Examples:
- `dit_fp16.onnx`
- `vae_fp32.onnx`
- `unet_fp16.onnx`

### TensorRT Engines

Format: `{component}_{precision}.trt`

Examples:
- `dit_fp16.trt`
- `vae_fp32.trt`

### Benchmark Results

Format: `{configuration}_{test_name}.json`

Examples:
- `PyTorch_FP16_short_720p.json`
- `TensorRT_Mixed_medium_720p.json`
- `comparison.txt` (comparison summary)

## Adding New Components

To add a new component (e.g., audio encoder):

1. **Add shape utilities** in `src/model/shapes.py`:
   - Define input/output shapes
   - Create dummy input function
   - Define dynamic axes

2. **Add export logic** in `scripts/export_model.py`:
   - Add component to choices
   - Handle component extraction
   - Define export parameters

3. **Add build profiles** in `configs/config.yaml`:
   - Define optimization profiles
   - Set precision and workspace

4. **Add runtime integration** in `src/tensorrt/runtime.py`:
   - Integrate into inference pipeline
   - Handle data type conversions
   - Manage execution order

5. **Update tests** in `tests/`:
   - Add test cases for new component
   - Validate shapes and outputs

## Environment Variables

Useful environment variables:

```bash
# CUDA
export CUDA_VISIBLE_DEVICES=0
export CUDA_MODULE_LOADING=LAZY

# HuggingFace
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache

# TensorRT
export TRT_LOGGER_LEVEL=INFO

# Python
export PYTHONUNBUFFERED=1
```

## Output Sizes (Approximate)

Based on Wan2.2-T2V-A14B:

| File | Size | Notes |
|------|------|-------|
| Original Model (HF) | ~50GB | Full pipeline weights |
| DiT ONNX (FP16) | ~14GB | Transformer only |
| VAE ONNX (FP32) | ~500MB | Encoder + Decoder |
| DiT TRT Engine (FP16) | ~7GB | Optimized for GPU |
| VAE TRT Engine (FP32) | ~250MB | Optimized for GPU |
| Generated Video (16 frames, 720p) | ~5MB | H.264 encoded |

## Performance Expectations

Typical speedups with TensorRT (vs PyTorch FP16):

- **DiT only**: 1.3-1.6×
- **DiT + VAE**: 1.5-2.0×
- **With CUDA Graphs**: 1.8-2.5×
- **With INT8 quantization**: 2.5-3.5× (quality dependent)

## Next Steps for Development

Priority areas for expansion:

1. **Complete TRT Integration**: Full pipeline with TRT engines
2. **Quantization**: INT8/FP8 with calibration
3. **Multi-GPU**: Data and model parallelism
4. **Production Packaging**: Deployment artifacts
5. **Advanced Optimizations**: Kernel fusion, custom plugins

See CONTRIBUTING.md for how to contribute to these areas.

