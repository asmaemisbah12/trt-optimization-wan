# Examples

This directory contains simple, self-contained examples demonstrating key features of the Wan TensorRT project.

## Available Examples

### 1. Simple Export (`simple_export.py`)

Demonstrates how to export a model component (DiT) to ONNX format.

```bash
python examples/simple_export.py
```

**What it does:**
- Loads the Wan2.2-T2V pipeline
- Extracts the DiT transformer component
- Creates appropriate dummy inputs
- Exports to ONNX with dynamic axes
- Validates the exported model

**Output:** `outputs/onnx/dit_fp16_example.onnx`

### 2. Simple Benchmark (`simple_benchmark.py`)

Demonstrates how to benchmark model inference performance.

```bash
python examples/simple_benchmark.py
```

**What it does:**
- Loads the PyTorch pipeline
- Defines an inference function
- Runs multiple test iterations with warmup
- Measures timing and memory usage
- Saves results to JSON

**Output:** `outputs/benchmarks/example_result.json`

## Creating Your Own Examples

When creating new examples:

1. Keep examples simple and focused on one concept
2. Include docstrings explaining what the example does
3. Use the project utilities (logging, config, etc.)
4. Save outputs to the `outputs/` directory
5. Add comments to explain non-obvious steps

## Tips

- Run examples from the project root directory
- Ensure your GPU has sufficient VRAM (16GB+ recommended)
- Check `configs/config.yaml` for default settings
- Use `--help` flag with scripts for available options

## Next Steps

After running these examples:

1. Try the full pipeline scripts in `scripts/`
2. Customize configurations in `configs/config.yaml`
3. Explore the source code in `src/`
4. Read the full documentation in `README.md`

