# Makefile for Wan TensorRT project

.PHONY: help install install-dev test format lint clean export-dit export-vae build-dit build-vae benchmark

help:
	@echo "Wan TensorRT - Common Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install dependencies"
	@echo "  make install-dev    Install dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run tests"
	@echo "  make format        Format code with black"
	@echo "  make lint          Run linters"
	@echo "  make clean         Clean generated files"
	@echo ""
	@echo "Model Pipeline:"
	@echo "  make export-dit    Export DiT to ONNX"
	@echo "  make export-vae    Export VAE to ONNX"
	@echo "  make build-dit     Build DiT TRT engine"
	@echo "  make build-vae     Build VAE TRT engine"
	@echo "  make benchmark     Run benchmarks"
	@echo ""
	@echo "All-in-one:"
	@echo "  make all           Export + Build + Benchmark"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

format:
	black src/ scripts/ tests/

lint:
	black src/ scripts/ tests/ --check
	flake8 src/ scripts/ tests/ --max-line-length=100 --ignore=E203,W503

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf outputs/onnx/*.onnx
	rm -rf outputs/engines/*.trt
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

export-dit:
	python scripts/export_model.py \
		--component transformer \
		--precision bf16 \
		--num_frames 81 \
		--height 720 \
		--width 1280 \
		--validate

export-vae:
	python scripts/export_model.py \
		--component vae \
		--precision fp32 \
		--num_frames 16 \
		--height 720 \
		--width 1280 \
		--validate

build-dit:
	python scripts/build_engines.py \
		--onnx_path outputs/onnx/transformer_bf16.onnx \
		--precision fp16 \
		--workspace_size 8192

build-vae:
	python scripts/build_engines.py \
		--onnx_path outputs/onnx/vae_fp32.onnx \
		--precision fp32 \
		--workspace_size 4096

benchmark:
	python scripts/benchmark.py \
		--compare \
		--num_warmup_runs 3 \
		--num_test_runs 10

all: export-dit export-vae build-dit build-vae benchmark

# Inference example
infer:
	python scripts/run_inference.py \
		--prompt "A serene lake at sunset with mountains in the background" \
		--num_frames 81 \
		--output_path outputs/videos/demo.mp4

# PyTorch baseline
baseline:
	python scripts/run_inference.py \
		--prompt "A serene lake at sunset with mountains in the background" \
		--use_pytorch \
		--output_path outputs/videos/baseline.mp4

