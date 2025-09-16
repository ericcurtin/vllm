# vLLM Docker Registry Model Loading - Implementation Summary

## Overview

This implementation adds support for loading vLLM-compatible models from Docker registries as OCI (Open Container Initiative) artifacts, following the pattern established by llama.cpp's `--docker-repo` option. This enables users to use Docker registries as an alternative to HuggingFace Hub for model storage and distribution.

## 🚀 Features Implemented

### 1. Core Infrastructure
- **New load format**: Added `docker_repo` to supported load formats alongside `auto`, `safetensors`, etc.
- **CLI integration**: Added `--docker-repo` argument to specify Docker registry repositories  
- **Model loader**: Implemented `DockerRepoModelLoader` class that handles OCI artifact pulling
- **Weight loading**: Support for safetensors, PyTorch binary, and other formats from Docker registries

### 2. Repository Format Support
- **Docker Hub**: `myorg/model:v1.0`, `model` (library images)
- **Private registries**: `myregistry.com/org/model:latest`
- **Explicit formats**: `docker.io/myorg/model`
- **Local registries**: `localhost:5000/model`

### 3. Tools and Utilities
- **Publishing tool**: `tools/docker_model_publisher.py` for publishing HF models to registries
- **Usage examples**: Comprehensive examples in `examples/docker_registry_usage.py`
- **Basic tests**: Functionality validation tests

### 4. Documentation
- **Complete guide**: Enhanced `docs/deployment/docker.md` with registry model loading
- **OCI format specification**: Detailed artifact format documentation
- **Usage examples**: Multiple deployment scenarios covered

## 📁 Files Added/Modified

### Modified Files:
- `vllm/config/load.py` - Added `docker_repo` field to LoadConfig
- `vllm/engine/arg_utils.py` - Added `--docker-repo` CLI argument
- `vllm/model_executor/model_loader/__init__.py` - Registered DockerRepoModelLoader
- `docs/deployment/docker.md` - Added comprehensive Docker registry documentation

### New Files:
- `vllm/model_executor/model_loader/docker_repo_loader.py` - Core loader implementation
- `tools/docker_model_publisher.py` - HuggingFace to Docker registry publishing tool
- `examples/docker_registry_usage.py` - Comprehensive usage examples
- Test files for functionality validation

## 🔧 Technical Implementation

### DockerRepoModelLoader Class
```python
class DockerRepoModelLoader(BaseModelLoader):
    """Model loader that pulls model weights from Docker registries as OCI artifacts."""
    
    def __init__(self, load_config: LoadConfig)
    def download_model(self, model_config: ModelConfig) -> None
    def load_weights(self, model: torch.nn.Module, model_config: ModelConfig) -> None
    def _pull_model_with_skopeo(self, docker_repo: str, dest_dir: str) -> None
    def _extract_model_files(self, oci_dir: str, extract_dir: str) -> None
    def _normalize_docker_repo(self, repo: str) -> str
```

### OCI Artifact Format
```
OCI Artifact
├── Manifest (application/vnd.oci.image.manifest.v1+json)
│   ├── Config (minimal empty JSON)
│   └── Layers
│       └── Model Layer (application/vnd.oci.image.layer.v1.tar+gzip)
│           ├── *.safetensors (preferred)
│           ├── *.bin (PyTorch weights) 
│           ├── *.pt (PyTorch weights)
│           ├── config.json (model configuration)
│           ├── tokenizer.json (tokenizer configuration)
│           └── other model files
```

### Publishing Tool Features
- Downloads models from HuggingFace Hub
- Packages as OCI-compliant artifacts
- Supports multiple registry types (Docker Hub, private, local)
- Handles authentication via existing Docker/skopeo credentials
- Provides both local packaging and direct push capabilities

## 🎯 Usage Examples

### Publishing a Model
```bash
# Download and publish to Docker Hub
python tools/docker_model_publisher.py \
    --model microsoft/DialoGPT-medium \
    --docker-repo myorg/dialogpt:v1.0 \
    --push

# Publish to private registry
python tools/docker_model_publisher.py \
    --model meta-llama/Llama-2-7b-hf \
    --docker-repo myregistry.com/models/llama:v2.0 \
    --push
```

### Loading a Model
```bash
# Load from Docker Hub
vllm serve \
    --load-format docker_repo \
    --docker-repo myorg/dialogpt:v1.0 \
    microsoft/DialoGPT-medium

# Load from private registry
vllm serve \
    --load-format docker_repo \
    --docker-repo myregistry.com/models/llama:v2.0 \
    meta-llama/Llama-2-7b-hf
```

### Air-gapped Environment
```bash
# 1. On internet-connected machine
python tools/docker_model_publisher.py \
    --model microsoft/DialoGPT-medium \
    --docker-repo localregistry.internal/models/dialogpt:v1.0 \
    --push

# 2. In air-gapped environment
vllm serve \
    --load-format docker_repo \
    --docker-repo localregistry.internal/models/dialogpt:v1.0 \
    microsoft/DialoGPT-medium
```

## ✅ Testing and Validation

### Functionality Tests
- ✅ LoadConfig with docker_repo parameter creation and validation
- ✅ DockerRepoModelLoader instantiation and configuration
- ✅ Docker repository name normalization
- ✅ Error handling for missing docker_repo parameter
- ✅ CLI argument parsing integration

### Tool Validation  
- ✅ Publishing tool help and argument parsing
- ✅ Usage examples execution
- ✅ Python syntax validation for all modules

## 🌟 Benefits

### For Users
- **Familiar infrastructure**: Leverage existing Docker registry knowledge
- **Access control**: Use enterprise authentication and authorization
- **Caching**: Benefit from registry caching for faster downloads
- **Versioning**: Tag-based model versioning (v1.0, v2.0, latest)
- **Air-gapped support**: Deploy in disconnected environments

### For Organizations
- **Centralized storage**: Integrate with existing container infrastructure  
- **Bandwidth efficiency**: Layer-based deduplication and caching
- **Compliance**: Use approved registry infrastructure
- **Cost optimization**: Reduce external dependencies on HuggingFace Hub

## 🔧 Requirements

### Runtime Requirements
- **skopeo**: For OCI artifact operations (pulling models)
- **Registry access**: Authentication for private registries
- **Disk space**: Temporary space for model extraction

### Publishing Requirements  
- **huggingface_hub**: For downloading models from HuggingFace
- **skopeo**: For pushing OCI artifacts to registries
- **Registry credentials**: For pushing to target registry

## 🛠️ Installation and Setup

### Installing skopeo
```bash
# Ubuntu/Debian
sudo apt-get install skopeo

# macOS  
brew install skopeo

# From source
# See: https://github.com/containers/skopeo
```

### Registry Authentication
```bash
# Docker Hub
docker login

# Private registry
docker login myregistry.com
# or
skopeo login myregistry.com
```

## 🔮 Future Enhancements

### Potential Improvements
- **Streaming extraction**: Avoid temporary disk usage for large models
- **Multi-layer optimization**: Split models across multiple layers for better caching
- **Registry mirroring**: Automatic fallback between multiple registries
- **Compression optimization**: Better compression algorithms for model data
- **Metadata annotations**: Enhanced OCI manifest annotations for model metadata

### Integration Opportunities
- **vLLM deployment charts**: Kubernetes Helm charts with registry model loading
- **CI/CD integration**: Automated model publishing pipelines
- **Model validation**: Checksum and signature verification
- **Registry management**: Tools for model lifecycle management in registries

## 📈 Impact

This implementation provides vLLM users with:
- **Enterprise-grade model distribution** through established Docker registry infrastructure
- **Reduced dependency** on external services like HuggingFace Hub
- **Enhanced security** through private registry deployment
- **Improved performance** via registry caching and proximity
- **Simplified deployment** in air-gapped and restricted environments

The implementation follows vLLM's existing patterns and maintains backward compatibility while adding powerful new model distribution capabilities.