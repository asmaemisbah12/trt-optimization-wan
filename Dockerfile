# Dockerfile for Wan TensorRT project
# Based on NVIDIA PyTorch + TensorRT container
# NOTE: Wan2.2 requires torch>=2.4.0 for bfloat16 support

FROM nvcr.io/nvidia/pytorch:24.09-py3

# Set working directory
WORKDIR /workspace/wan_trt

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    ffmpeg \
    libavcodec-extra \
    x264 \
    libx264-dev \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Ensure torch>=2.4.0 for bfloat16 support
RUN pip install --no-cache-dir --upgrade "torch>=2.4.0" "torchvision>=0.19.0"

RUN pip install --no-cache-dir -r requirements.txt

# Install TensorRT Python bindings (if not included in base image)
RUN pip install --no-cache-dir tensorrt pycuda

# Copy project files
COPY . .

# Install package in development mode
RUN pip install -e .

# Create output directories
RUN mkdir -p outputs/onnx outputs/engines outputs/videos outputs/benchmarks outputs/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_MODULE_LOADING=LAZY

# Default command
CMD ["/bin/bash"]

