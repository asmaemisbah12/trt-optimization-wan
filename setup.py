"""Setup script for Wan TensorRT acceleration package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="wan-trt",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="TensorRT acceleration for Wan2.2-T2V-A14B video generation model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wan_trt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "diffusers>=0.27.0",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "onnx>=1.15.0",
        "onnxruntime-gpu>=1.16.0",
        "optimum[exporters]>=1.16.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "opencv-python>=4.8.0",
        "scikit-image>=0.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "quality": [
            "lpips>=0.1.4",
            "pytorch-fid>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "wan-export=scripts.export_model:main",
            "wan-build=scripts.build_engines:main",
            "wan-infer=scripts.run_inference:main",
            "wan-bench=scripts.benchmark:main",
        ],
    },
)

