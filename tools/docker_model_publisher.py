#!/usr/bin/env python3
"""
Tool to publish vLLM-compatible models from HuggingFace to Docker Hub.

This script downloads a model from HuggingFace and packages it as an OCI artifact
that can be pushed to Docker registries for use with vLLM's --docker-repo option.

Usage:
    python tools/docker_model_publisher.py \
        --model microsoft/DialoGPT-medium \
        --docker-repo myregistry.com/myorg/dialogpt:v1.0 \
        --push

Requirements:
    - skopeo (for OCI operations)
    - huggingface_hub (for downloading models)
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub is required. Install with: pip install huggingface_hub")
    exit(1)


def check_skopeo():
    """Check if skopeo is available."""
    try:
        result = subprocess.run(["skopeo", "--version"], 
                              capture_output=True, check=True, text=True)
        print(f"Found skopeo: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: skopeo is required but not found.")
        print("Install skopeo: https://github.com/containers/skopeo")
        return False


def create_oci_manifest(model_dir: str) -> dict:
    """Create an OCI manifest for the model files."""
    
    # Create a simple tar.gz of the model directory
    tar_path = os.path.join(model_dir, "model.tar.gz")
    
    # Create tar excluding any existing tar files
    subprocess.run([
        "tar", "-czf", tar_path,
        "--exclude=*.tar.gz",
        "-C", model_dir, "."
    ], check=True)
    
    # Get file info
    stat = os.stat(tar_path)
    size = stat.st_size
    
    # Calculate SHA256
    import hashlib
    sha256_hash = hashlib.sha256()
    with open(tar_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    digest = f"sha256:{sha256_hash.hexdigest()}"
    
    # Create layer descriptor
    layer = {
        "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
        "digest": digest,
        "size": size
    }
    
    # Create config (minimal)
    config = {
        "mediaType": "application/vnd.oci.image.config.v1+json",
        "digest": "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        "size": 2
    }
    
    # Create manifest
    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": config,
        "layers": [layer],
        "annotations": {
            "org.opencontainers.image.description": "vLLM compatible model",
            "org.opencontainers.image.title": "vLLM Model",
            "vllm.model.format": "safetensors"
        }
    }
    
    return manifest, tar_path, digest


def create_oci_layout(model_dir: str, oci_dir: str, manifest: dict, 
                      tar_path: str, layer_digest: str):
    """Create OCI layout directory structure."""
    
    # Create directory structure
    blobs_dir = os.path.join(oci_dir, "blobs", "sha256")
    os.makedirs(blobs_dir, exist_ok=True)
    
    # Copy layer file
    layer_hash = layer_digest.split(":")[1]
    layer_dest = os.path.join(blobs_dir, layer_hash)
    shutil.copy2(tar_path, layer_dest)
    
    # Create empty config
    config_content = "{}"
    config_hash = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    config_dest = os.path.join(blobs_dir, config_hash)
    with open(config_dest, "w") as f:
        f.write(config_content)
    
    # Write manifest
    manifest_content = json.dumps(manifest, indent=2)
    manifest_hash = hashlib.sha256(manifest_content.encode()).hexdigest()
    manifest_dest = os.path.join(blobs_dir, manifest_hash)
    with open(manifest_dest, "w") as f:
        f.write(manifest_content)
    
    # Create index
    index = {
        "schemaVersion": 2,
        "manifests": [{
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "digest": f"sha256:{manifest_hash}",
            "size": len(manifest_content.encode())
        }]
    }
    
    # Write index
    with open(os.path.join(oci_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    
    # Write OCI layout version
    layout = {"imageLayoutVersion": "1.0.0"}
    with open(os.path.join(oci_dir, "oci-layout"), "w") as f:
        json.dump(layout, f, indent=2)


def download_model(model_id: str, cache_dir: Optional[str] = None, 
                   revision: Optional[str] = None) -> str:
    """Download model from HuggingFace."""
    print(f"Downloading model {model_id}...")
    
    # Download the model
    model_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=False,
    )
    
    print(f"Model downloaded to: {model_path}")
    return model_path


def push_to_registry(oci_dir: str, docker_repo: str):
    """Push OCI layout to Docker registry using skopeo."""
    
    source = f"oci:{oci_dir}:latest"
    dest = f"docker://{docker_repo}"
    
    print(f"Pushing {source} to {dest}")
    
    try:
        subprocess.run([
            "skopeo", "copy",
            "--retry-times", "3",
            source, dest
        ], check=True)
        
        print(f"Successfully pushed to {docker_repo}")
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to push to registry: {e}")
        raise


def normalize_docker_repo(repo: str) -> str:
    """Normalize Docker repository name."""
    # Add docker.io prefix if no registry specified
    if "/" not in repo or "." not in repo.split("/")[0]:
        if "/" not in repo:
            return f"docker.io/library/{repo}"
        else:
            return f"docker.io/{repo}"
    return repo


def main():
    parser = argparse.ArgumentParser(
        description="Publish vLLM models from HuggingFace to Docker registries"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="HuggingFace model ID (e.g., microsoft/DialoGPT-medium)"
    )
    parser.add_argument(
        "--docker-repo", "-r",
        required=True,
        help="Docker repository to push to (e.g., myregistry.com/org/model:tag)"
    )
    parser.add_argument(
        "--revision",
        help="Model revision/branch to use (default: main)"
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory to cache downloaded models"
    )
    parser.add_argument(
        "--push", "-p",
        action="store_true",
        help="Push to registry after creating OCI layout"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Directory to save OCI layout (default: temporary directory)"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files after completion"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_skopeo():
        return 1
    
    # Normalize repository name
    docker_repo = normalize_docker_repo(args.docker_repo)
    
    temp_dir = None
    try:
        # Create working directory
        if args.output_dir:
            work_dir = os.path.abspath(args.output_dir)
            os.makedirs(work_dir, exist_ok=True)
        else:
            temp_dir = tempfile.mkdtemp(prefix="vllm_docker_publish_")
            work_dir = temp_dir
        
        oci_dir = os.path.join(work_dir, "oci")
        
        # Download model
        model_path = download_model(
            args.model, 
            cache_dir=args.cache_dir,
            revision=args.revision
        )
        
        print("Creating OCI artifact...")
        
        # Create OCI manifest and layout
        import hashlib  # Import here since we use it in create_oci_manifest
        
        manifest, tar_path, layer_digest = create_oci_manifest(model_path)
        create_oci_layout(model_path, oci_dir, manifest, tar_path, layer_digest)
        
        print(f"OCI layout created at: {oci_dir}")
        
        # Push if requested
        if args.push:
            push_to_registry(oci_dir, docker_repo)
        else:
            print(f"OCI layout ready at {oci_dir}")
            print(f"To push manually, run:")
            print(f"  skopeo copy oci:{oci_dir}:latest docker://{docker_repo}")
        
        print("\nTo use with vLLM:")
        print(f"  vllm serve --load-format docker_repo --docker-repo {docker_repo} <model_name>")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    finally:
        # Clean up unless keeping temp files
        if temp_dir and not args.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    return 0


if __name__ == "__main__":
    exit(main())