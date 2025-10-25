#!/usr/bin/env python3
"""
Upload ONNX files to Hugging Face Hub
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import argparse

def upload_onnx_files(repo_id: str, token: str = None):
    """
    Upload ONNX files to Hugging Face Hub
    
    Args:
        repo_id: Repository ID (e.g., "username/onnx")
        token: Hugging Face token (optional, can use environment variable)
    """
    
    # Initialize API
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, exist_ok=True, token=token)
        print(f"✅ Repository {repo_id} is ready")
    except Exception as e:
        print(f"⚠️  Repository creation warning: {e}")
    
    # Path to ONNX files
    onnx_dir = Path("/home/ubuntu/trt-optimization-wan/outputs/onnx")
    
    # Files to upload
    files_to_upload = [
        "dit_fp16.onnx",
        "dit_fp16.onnx.data"
    ]
    
    print(f"📁 Uploading files from: {onnx_dir}")
    print(f"🎯 Target repository: {repo_id}")
    
    # Upload each file
    for filename in files_to_upload:
        file_path = onnx_dir / filename
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            continue
            
        file_size = file_path.stat().st_size / (1024**3)  # GB
        print(f"📤 Uploading {filename} ({file_size:.2f} GB)...")
        
        try:
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=filename,
                repo_id=repo_id,
                token=token,
                commit_message=f"Add {filename} - Wan2.2-T2V ONNX model"
            )
            print(f"✅ Successfully uploaded {filename}")
            
        except Exception as e:
            print(f"❌ Failed to upload {filename}: {e}")
            return False
    
    print(f"🎉 All files uploaded successfully to {repo_id}")
    print(f"🔗 View at: https://huggingface.co/{repo_id}")
    
    return True

def upload_engine_files(repo_id: str, token: str = None):
    """
    Upload TensorRT engine files to Hugging Face Hub
    
    Args:
        repo_id: Repository ID (e.g., "username/engines")
        token: Hugging Face token (optional, can use environment variable)
    """
    
    # Initialize API
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, exist_ok=True, token=token)
        print(f"✅ Repository {repo_id} is ready")
    except Exception as e:
        print(f"⚠️  Repository creation warning: {e}")
    
    # Path to TensorRT engine files
    engines_dir = Path("/home/ubuntu/trt-optimization-wan/outputs/engines")
    
    # Find all .trt files in the engines directory
    trt_files = list(engines_dir.glob("*.trt"))
    
    if not trt_files:
        print(f"❌ No .trt files found in {engines_dir}")
        return False
    
    print(f"📁 Uploading files from: {engines_dir}")
    print(f"🎯 Target repository: {repo_id}")
    print(f"📋 Found {len(trt_files)} TensorRT engine file(s)")
    
    # Upload each file
    for file_path in trt_files:
        filename = file_path.name
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            continue
            
        file_size = file_path.stat().st_size / (1024**3)  # GB
        print(f"📤 Uploading {filename} ({file_size:.2f} GB)...")
        
        try:
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=filename,
                repo_id=repo_id,
                token=token,
                commit_message=f"Add {filename} - Wan2.2-T2V TensorRT engine (FP16, max_frames=32)"
            )
            print(f"✅ Successfully uploaded {filename}")
            
        except Exception as e:
            print(f"❌ Failed to upload {filename}: {e}")
            return False
    
    print(f"🎉 All TensorRT engine files uploaded successfully to {repo_id}")
    print(f"🔗 View at: https://huggingface.co/{repo_id}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Upload files to Hugging Face")
    parser.add_argument("--repo_id", required=True, help="Repository ID (e.g., 'username/onnx' or 'username/engines')")
    parser.add_argument("--token", help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--type", choices=["onnx", "engines"], default="engines", 
                       help="Type of files to upload: 'onnx' for ONNX models, 'engines' for TensorRT engines")
    
    args = parser.parse_args()
    
    # Get token from args or environment
    token = args.token or os.getenv("HF_TOKEN")
    
    if not token:
        print("❌ No Hugging Face token provided!")
        print("Please either:")
        print("1. Use --token argument")
        print("2. Set HF_TOKEN environment variable")
        print("3. Get token from: https://huggingface.co/settings/tokens")
        return
    
    # Upload files based on type
    if args.type == "onnx":
        print("📤 Uploading ONNX files...")
        success = upload_onnx_files(args.repo_id, token)
    else:  # engines
        print("🚀 Uploading TensorRT engine files...")
        success = upload_engine_files(args.repo_id, token)
    
    if success:
        print("\n🎉 Upload completed successfully!")
    else:
        print("\n❌ Upload failed!")

if __name__ == "__main__":
    main()
