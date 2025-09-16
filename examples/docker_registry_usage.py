#!/usr/bin/env python3
"""
Example usage demonstration for vLLM's Docker registry model loading feature.

This script shows how to use the new --docker-repo functionality to load models
from Docker registries instead of HuggingFace Hub.
"""

import subprocess
import sys
import tempfile

def example_publish_model():
    """Example: Publishing a model to Docker registry."""
    print("🚀 Example: Publishing a HuggingFace model to Docker registry")
    print("="*60)
    
    # This would be the actual command to publish a model
    publish_cmd = [
        "python", "tools/docker_model_publisher.py",
        "--model", "microsoft/DialoGPT-medium",
        "--docker-repo", "myorg/dialogpt:v1.0",
        "--output-dir", "/tmp/vllm_oci_demo",
        # Note: --push flag omitted for demo (would actually push to registry)
    ]
    
    print("Command that would be used to publish a model:")
    print(" ".join(publish_cmd))
    print()
    
    print("This command would:")
    print("1. Download microsoft/DialoGPT-medium from HuggingFace")
    print("2. Package it as an OCI artifact")
    print("3. Save the OCI layout to /tmp/vllm_oci_demo")
    print("4. With --push flag, it would push to Docker registry")
    print()

def example_load_model():
    """Example: Loading a model from Docker registry."""
    print("🔄 Example: Loading a model from Docker registry")
    print("="*60)
    
    # This would be the actual command to load a model from Docker registry
    load_cmd = [
        "vllm", "serve",
        "--load-format", "docker_repo",
        "--docker-repo", "myorg/dialogpt:v1.0",
        "microsoft/DialoGPT-medium"  # Model name for tokenizer/config
    ]
    
    print("Command that would be used to load a model from Docker registry:")
    print(" ".join(load_cmd))
    print()
    
    print("This command would:")
    print("1. Pull the model OCI artifact from myorg/dialogpt:v1.0")
    print("2. Extract the model weights")
    print("3. Load them into the vLLM engine")
    print("4. Use microsoft/DialoGPT-medium for tokenizer/config metadata")
    print()

def example_private_registry():
    """Example: Using a private registry."""
    print("🔐 Example: Using a private Docker registry")
    print("="*60)
    
    print("1. Login to your private registry:")
    print("   docker login myregistry.com")
    print("   # or")
    print("   skopeo login myregistry.com")
    print()
    
    print("2. Publish model to private registry:")
    publish_cmd = [
        "python", "tools/docker_model_publisher.py",
        "--model", "meta-llama/Llama-2-7b-hf",
        "--docker-repo", "myregistry.com/models/llama:v2.0",
        "--push"
    ]
    print("   " + " ".join(publish_cmd))
    print()
    
    print("3. Load model from private registry:")
    load_cmd = [
        "vllm", "serve",
        "--load-format", "docker_repo",
        "--docker-repo", "myregistry.com/models/llama:v2.0",
        "meta-llama/Llama-2-7b-hf"
    ]
    print("   " + " ".join(load_cmd))
    print()

def example_air_gapped():
    """Example: Air-gapped environment usage."""
    print("✈️  Example: Air-gapped environment deployment")
    print("="*60)
    
    print("For air-gapped environments, you can:")
    print()
    
    print("1. On internet-connected machine, publish models:")
    print("   python tools/docker_model_publisher.py \\")
    print("     --model microsoft/DialoGPT-medium \\")
    print("     --docker-repo localregistry.internal/models/dialogpt:v1.0 \\")
    print("     --push")
    print()
    
    print("2. In air-gapped environment, use local registry:")
    print("   vllm serve \\")
    print("     --load-format docker_repo \\")
    print("     --docker-repo localregistry.internal/models/dialogpt:v1.0 \\")
    print("     microsoft/DialoGPT-medium")
    print()

def show_registry_formats():
    """Show supported Docker registry formats."""
    print("📋 Supported Docker registry formats")
    print("="*60)
    
    formats = [
        ("myorg/model", "Docker Hub (user/repo)"),
        ("myorg/model:v1.0", "Docker Hub with tag"),
        ("docker.io/myorg/model", "Explicit Docker Hub"),
        ("myregistry.com/org/model", "Private registry"),
        ("myregistry.com/org/model:latest", "Private registry with tag"),
        ("localhost:5000/model", "Local registry"),
    ]
    
    print("Repository format examples:")
    for format_str, description in formats:
        print(f"  {format_str:<35} - {description}")
    print()

def show_benefits():
    """Show benefits of using Docker registries for models."""
    print("🌟 Benefits of Docker registry model storage")
    print("="*60)
    
    benefits = [
        "Centralized storage: Use existing Docker infrastructure",
        "Access control: Leverage registry authentication and authorization", 
        "Caching: Docker registry caching for faster downloads",
        "Versioning: Tag-based model versioning (v1.0, v2.0, latest)",
        "Air-gapped environments: Local registry deployment",
        "Bandwidth efficiency: Layer-based deduplication",
        "Enterprise integration: Fits existing Docker workflows",
    ]
    
    for benefit in benefits:
        print(f"• {benefit}")
    print()

def main():
    """Run all examples."""
    print("vLLM Docker Registry Model Loading Examples")
    print("=" * 80)
    print()
    
    show_benefits()
    show_registry_formats() 
    example_publish_model()
    example_load_model()
    example_private_registry()
    example_air_gapped()
    
    print("🔗 Related documentation:")
    print("• docs/deployment/docker.md - Full Docker deployment guide")
    print("• tools/docker_model_publisher.py --help - Publishing tool help")
    print()
    
    print("🛠️  Requirements:")
    print("• skopeo - For OCI artifact operations")
    print("• Docker registry access - For model storage")
    print("• huggingface_hub - For downloading models to publish")

if __name__ == "__main__":
    main()