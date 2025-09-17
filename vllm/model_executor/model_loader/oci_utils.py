# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for downloading model weights from Docker Hub as OCI artifacts."""

import json
import os
import hashlib
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse
import requests
from tqdm.auto import tqdm

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    # Fallback logger when vLLM is not available (for testing)
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# OCI Media Types for vLLM models
OCI_MEDIA_TYPE_SAFETENSORS = "application/vnd.docker.ai.safetensors.v1"
OCI_MEDIA_TYPE_CONFIG = "application/vnd.docker.ai.model.config.v1+json"
OCI_MEDIA_TYPE_LICENSE = "application/vnd.docker.ai.license"
OCI_MEDIA_TYPE_TOKENIZER = "application/vnd.docker.ai.tokenizer.v1"
OCI_MEDIA_TYPE_MANIFEST = "application/vnd.oci.image.manifest.v1+json"


class OCIArtifactFormat:
    """
    OCI Artifact Format for vLLM safetensors models.
    
    The format follows OCI Image Format Specification v1.0.1 for storing
    machine learning models as OCI artifacts.
    
    Example manifest:
    {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {
            "mediaType": "application/vnd.docker.ai.model.config.v1+json",
            "digest": "sha256:...",
            "size": 1234
        },
        "layers": [
            {
                "mediaType": "application/vnd.docker.ai.safetensors.v1",
                "digest": "sha256:...",
                "size": 91727232,
                "annotations": {
                    "org.opencontainers.image.title": "model-00001-of-00002.safetensors"
                }
            },
            {
                "mediaType": "application/vnd.docker.ai.tokenizer.v1",
                "digest": "sha256:...",
                "size": 5678,
                "annotations": {
                    "org.opencontainers.image.title": "tokenizer.json"
                }
            },
            {
                "mediaType": "application/vnd.docker.ai.license",
                "digest": "sha256:...",
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
    """


class DockerHubOCIClient:
    """Client for downloading OCI artifacts from Docker Hub."""
    
    def __init__(self, registry_url: str = "https://registry-1.docker.io"):
        self.registry_url = registry_url
        self.session = requests.Session()
        # Set User-Agent to identify vLLM
        self.session.headers.update({
            'User-Agent': 'vLLM-OCI-Client/1.0'
        })
        self._token = None
    
    def _get_auth_token(self, repository: str) -> str:
        """Get authentication token from Docker Hub."""
        if self._token:
            return self._token
            
        auth_url = "https://auth.docker.io/token"
        params = {
            "service": "registry.docker.io",
            "scope": f"repository:{repository}:pull"
        }
        
        response = self.session.get(auth_url, params=params)
        response.raise_for_status()
        
        token_data = response.json()
        self._token = token_data["token"]
        return self._token
    
    def _make_request(self, url: str, repository: str, headers: Optional[Dict[str, str]] = None) -> requests.Response:
        """Make authenticated request to registry."""
        token = self._get_auth_token(repository)
        
        request_headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json"
        }
        if headers:
            request_headers.update(headers)
        
        response = self.session.get(url, headers=request_headers)
        response.raise_for_status()
        return response
    
    def get_manifest(self, repository: str, tag: str = "latest") -> Dict[str, Any]:
        """Get OCI manifest for a repository and tag."""
        url = f"{self.registry_url}/v2/{repository}/manifests/{tag}"
        response = self._make_request(url, repository)
        return response.json()
    
    def download_blob(self, repository: str, digest: str, output_path: Path, 
                     resume: bool = True) -> None:
        """Download a blob with resumable downloads."""
        url = f"{self.registry_url}/v2/{repository}/blobs/{digest}"
        
        # Check if partial file exists and get its size
        start_byte = 0
        if resume and output_path.exists():
            start_byte = output_path.stat().st_size
            logger.info(f"Resuming download from byte {start_byte}")
        
        # Make HEAD request to get total size
        token = self._get_auth_token(repository)
        headers = {"Authorization": f"Bearer {token}"}
        
        head_response = self.session.head(url, headers=headers)
        head_response.raise_for_status()
        
        total_size = int(head_response.headers.get('content-length', 0))
        
        # If file is already complete, skip download
        if start_byte == total_size:
            logger.info(f"File {output_path} already complete, skipping download")
            return
        
        # Add Range header for resumable download
        if start_byte > 0:
            headers["Range"] = f"bytes={start_byte}-"
        
        response = self.session.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open file in append mode if resuming, otherwise write mode
        mode = "ab" if start_byte > 0 else "wb"
        
        with open(output_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=start_byte,
                unit='B',
                unit_scale=True,
                desc=f"Downloading {output_path.name}"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Verify file size
        final_size = output_path.stat().st_size
        if final_size != total_size:
            raise RuntimeError(f"Download incomplete. Expected {total_size} bytes, got {final_size}")
        
        logger.info(f"Successfully downloaded {output_path}")


def download_model_from_docker_hub(
    docker_repo: str,
    cache_dir: Optional[str] = None,
    tag: str = "latest",
    allow_patterns: Optional[List[str]] = None,
) -> str:
    """
    Download model from Docker Hub as OCI artifact.
    
    Args:
        docker_repo: Docker repository name (e.g., 'ai/deepseek-v3')
        cache_dir: Directory to store downloaded files
        tag: Docker tag to download (default: 'latest')
        allow_patterns: Patterns to filter which files to download
        
    Returns:
        Path to the downloaded model directory
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/vllm/docker_models")
    
    # Create unique directory for this model
    safe_repo_name = docker_repo.replace("/", "_").replace(":", "_")
    model_dir = Path(cache_dir) / safe_repo_name / tag
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading model from Docker Hub: {docker_repo}:{tag}")
    
    client = DockerHubOCIClient()
    
    # Download manifest
    try:
        manifest = client.get_manifest(docker_repo, tag)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise ValueError(f"Model not found: {docker_repo}:{tag}") from e
        raise
    
    logger.info(f"Found manifest with {len(manifest.get('layers', []))} layers")
    
    # Validate manifest format
    if manifest.get("mediaType") != OCI_MEDIA_TYPE_MANIFEST:
        logger.warning(f"Unexpected manifest media type: {manifest.get('mediaType')}")
    
    # Save manifest for reference
    manifest_path = model_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Download config blob
    config_info = manifest.get("config", {})
    if config_info:
        config_path = model_dir / "config.json"
        client.download_blob(docker_repo, config_info["digest"], config_path)
    
    # Download layer blobs
    downloaded_files = []
    for layer in manifest.get("layers", []):
        media_type = layer.get("mediaType", "")
        digest = layer["digest"]
        
        # Determine filename from annotations or create default
        annotations = layer.get("annotations", {})
        filename = annotations.get("org.opencontainers.image.title")
        
        if not filename:
            # Generate filename based on media type and digest
            if media_type == OCI_MEDIA_TYPE_SAFETENSORS:
                filename = f"model_{digest[7:15]}.safetensors"
            elif media_type == OCI_MEDIA_TYPE_TOKENIZER:
                filename = "tokenizer.json"
            elif media_type == OCI_MEDIA_TYPE_LICENSE:
                filename = "LICENSE"
            else:
                # Generic filename for unknown types
                filename = f"blob_{digest[7:15]}"
        
        # Apply allow_patterns filter if specified
        if allow_patterns:
            import fnmatch
            if not any(fnmatch.fnmatch(filename, pattern) for pattern in allow_patterns):
                logger.info(f"Skipping {filename} (doesn't match allow patterns)")
                continue
        
        blob_path = model_dir / filename
        
        try:
            client.download_blob(docker_repo, digest, blob_path, resume=True)
            downloaded_files.append(str(blob_path))
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            raise
    
    logger.info(f"Successfully downloaded {len(downloaded_files)} files to {model_dir}")
    return str(model_dir)


def verify_oci_artifact_integrity(model_dir: str) -> bool:
    """
    Verify the integrity of downloaded OCI artifact.
    
    Args:
        model_dir: Path to the downloaded model directory
        
    Returns:
        True if verification passes, False otherwise
    """
    model_path = Path(model_dir)
    manifest_path = model_path / "manifest.json"
    
    if not manifest_path.exists():
        logger.error("Manifest file not found")
        return False
    
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        # Verify config file if present
        config_info = manifest.get("config")
        if config_info:
            config_path = model_path / "config.json"
            if not config_path.exists():
                logger.error("Config file missing")
                return False
            
            # Verify config digest
            expected_digest = config_info["digest"]
            if not _verify_file_digest(config_path, expected_digest):
                logger.error("Config file digest mismatch")
                return False
        
        # Verify layer files
        for layer in manifest.get("layers", []):
            digest = layer["digest"]
            annotations = layer.get("annotations", {})
            filename = annotations.get("org.opencontainers.image.title")
            
            if filename:
                file_path = model_path / filename
                if not file_path.exists():
                    logger.error(f"Layer file {filename} missing")
                    return False
                
                if not _verify_file_digest(file_path, digest):
                    logger.error(f"Layer file {filename} digest mismatch")
                    return False
        
        logger.info("OCI artifact integrity verification passed")
        return True
        
    except Exception as e:
        logger.error(f"Integrity verification failed: {e}")
        return False


def _verify_file_digest(file_path: Path, expected_digest: str) -> bool:
    """Verify file digest matches expected value."""
    if not expected_digest.startswith("sha256:"):
        logger.warning(f"Unsupported digest algorithm: {expected_digest}")
        return True  # Skip verification for unsupported algorithms
    
    expected_hash = expected_digest[7:]  # Remove "sha256:" prefix
    
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    actual_hash = sha256_hash.hexdigest()
    return actual_hash == expected_hash