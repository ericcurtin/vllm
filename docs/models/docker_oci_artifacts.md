# vLLM OCI Artifact Format Specification

## Overview

vLLM supports downloading models from Docker Hub as OCI (Open Container Initiative) artifacts. This allows storing and distributing machine learning models in container registries alongside Docker images, providing a standardized way to share models.

## OCI Artifact Format

vLLM uses the OCI Image Format Specification v1.0.1 to store safetensors models as OCI artifacts. The format follows the standard OCI manifest structure with custom media types for ML model components.

### Media Types

vLLM defines the following custom media types for model artifacts:

- `application/vnd.docker.ai.model.config.v1+json` - Model configuration file
- `application/vnd.docker.ai.safetensors.v1` - Safetensors weight files
- `application/vnd.docker.ai.tokenizer.v1` - Tokenizer files (tokenizer.json, vocab files)
- `application/vnd.docker.ai.license` - License files

### Manifest Structure

```json
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.oci.image.manifest.v1+json",
  "config": {
    "mediaType": "application/vnd.docker.ai.model.config.v1+json",
    "digest": "sha256:d8de73a97e087a13ce44273be79d9179b51024754e6550d29591efc9629c98f5",
    "size": 364
  },
  "layers": [
    {
      "mediaType": "application/vnd.docker.ai.safetensors.v1",
      "digest": "sha256:384a89bd054c0cc1b128d1adb2c6648867e5a84d166bc455c8bda6e4576c2779",
      "size": 91727232,
      "annotations": {
        "org.opencontainers.image.title": "model-00001-of-00002.safetensors"
      }
    },
    {
      "mediaType": "application/vnd.docker.ai.safetensors.v1",
      "digest": "sha256:295b7a8c2b9e4aa0c8c4d8f8a3e8e5d8e1a1b1c1d1e1f1a1b1c1d1e1f1a1b1c",
      "size": 84567890,
      "annotations": {
        "org.opencontainers.image.title": "model-00002-of-00002.safetensors"
      }
    },
    {
      "mediaType": "application/vnd.docker.ai.tokenizer.v1",
      "digest": "sha256:609e2cb599f84aaa41d8ef29d8fdb04d164fab22e8d9292ca34a599d0f56a338",
      "size": 5678,
      "annotations": {
        "org.opencontainers.image.title": "tokenizer.json"
      }
    },
    {
      "mediaType": "application/vnd.docker.ai.license",
      "digest": "sha256:7c2d35c4b2a8e6d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d",
      "size": 12624,
      "annotations": {
        "org.opencontainers.image.title": "LICENSE"
      }
    }
  ],
  "annotations": {
    "org.opencontainers.image.title": "vLLM Model",
    "org.opencontainers.image.description": "Safetensors model for vLLM",
    "vllm.model.format": "safetensors",
    "vllm.model.version": "1.0"
  }
}
```

### Required Components

1. **Config Blob**: Contains model configuration (config.json)
2. **Model Layers**: One or more safetensors files containing model weights
3. **Optional Components**:
   - Tokenizer files (tokenizer.json, vocab files)
   - License files
   - Other metadata

### Annotations

- `org.opencontainers.image.title`: Human-readable file name
- `vllm.model.format`: Must be "safetensors" for vLLM compatibility
- `vllm.model.version`: Format version (currently "1.0")

## Usage

### Downloading Models

Use the `--docker-repo` argument instead of `--model`:

```bash
# Before: Download from Hugging Face
vllm serve --model Qwen/Qwen2.5-3B-Instruct

# After: Download from Docker Hub
vllm serve --docker-repo ai/deepseek-v3
```

### Supported Repositories

The format supports any Docker registry that implements the OCI Distribution Specification:

- Docker Hub: `ai/deepseek-v3`
- Other registries: `registry.example.com/namespace/model`

### Tag Support

Models can be tagged with semantic versions or other identifiers:

```bash
vllm serve --docker-repo ai/deepseek-v3:v1.0
vllm serve --docker-repo ai/deepseek-v3:latest
```

## Features

### Resumable Downloads

vLLM supports resumable downloads for OCI artifacts. If a download is interrupted, it will resume from where it left off rather than starting over.

### Caching

Downloaded models are cached locally in `~/.cache/vllm/docker_models/` by default. The cache directory can be changed using the `--download-dir` argument.

### Integrity Verification

All downloaded blobs are verified against their SHA256 digests to ensure data integrity.

## Creating OCI Model Artifacts

To create an OCI artifact from a safetensors model:

1. Prepare model files (config.json, *.safetensors, tokenizer files)
2. Create OCI manifest with appropriate media types
3. Push to container registry using tools like:
   - `oras` (OCI Registry As Storage)
   - `crane` (part of go-containerregistry)
   - Custom scripts using Docker registry API

Example using `oras`:

```bash
# Create artifact
oras push registry.example.com/models/my-model:v1.0 \
  --config config.json:application/vnd.docker.ai.model.config.v1+json \
  model.safetensors:application/vnd.docker.ai.safetensors.v1 \
  tokenizer.json:application/vnd.docker.ai.tokenizer.v1 \
  LICENSE:application/vnd.docker.ai.license
```

## Comparison with GGUF Format

This format is inspired by the GGUF OCI format used by llama.cpp, but adapted for safetensors models:

**Similarities:**
- Uses OCI Image Format Specification
- Custom media types for ML components
- Supports layer annotations for file naming

**Differences:**
- Uses `safetensors.v1` instead of `gguf.v3` media type
- Supports multiple weight files (sharded models)
- Includes tokenizer as separate layer type
- Designed for transformer models rather than GGUF format

## Limitations

- Currently supports only safetensors format models
- Requires models to follow Hugging Face transformer structure
- Docker registry must support OCI Image Format Specification v1.0.1