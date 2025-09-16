# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import tempfile
from collections.abc import Generator
from typing import Optional

import torch

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    filter_duplicate_safetensors_files, filter_files_not_needed_for_inference,
    safetensors_weights_iterator, pt_weights_iterator)

logger = init_logger(__name__)


class DockerRepoModelLoader(BaseModelLoader):
    """Model loader that pulls model weights from Docker registries as OCI artifacts."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if not load_config.docker_repo:
            raise ValueError("docker_repo must be specified when using docker_repo load format")
        self.docker_repo = load_config.docker_repo
        self._model_files_cache = None  # Cache the downloaded model files

    def _check_skopeo_available(self) -> bool:
        """Check if skopeo is available in the system."""
        try:
            subprocess.run(["skopeo", "--version"], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _normalize_docker_repo(self, repo: str) -> str:
        """Normalize Docker repository name to include registry if needed."""
        # If no registry specified, default to Docker Hub
        if "/" not in repo or "." not in repo.split("/")[0]:
            if "/" not in repo:
                # Single name, treat as library image
                return f"docker.io/library/{repo}"
            else:
                # user/repo format
                return f"docker.io/{repo}"
        return repo

    def _pull_model_with_skopeo(self, docker_repo: str, dest_dir: str) -> None:
        """Pull model from Docker registry using skopeo."""
        if not self._check_skopeo_available():
            raise RuntimeError(
                "skopeo is required for pulling models from Docker registries. "
                "Please install skopeo: https://github.com/containers/skopeo"
            )

        # Normalize the repository name
        normalized_repo = self._normalize_docker_repo(docker_repo)
        
        # Add docker:// prefix for skopeo
        source = f"docker://{normalized_repo}"
        
        # Use oci: prefix for destination to extract as OCI layout
        dest = f"oci:{dest_dir}:latest"
        
        logger.info(f"Pulling model from {source} to {dest_dir}")
        
        try:
            # Use skopeo to copy the image to local OCI layout
            subprocess.run([
                "skopeo", "copy", 
                "--retry-times", "3",
                source, dest
            ], check=True, capture_output=True, text=True)
            
            logger.info(f"Successfully pulled model from {source}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull model from {source}: {e.stderr}")
            raise RuntimeError(f"Failed to pull model from Docker registry: {e.stderr}")

    def _extract_model_files(self, oci_dir: str, extract_dir: str) -> None:
        """Extract model files from OCI layout to a directory."""
        # Read the OCI index to find the manifest
        index_path = os.path.join(oci_dir, "index.json")
        if not os.path.exists(index_path):
            raise RuntimeError(f"OCI index not found at {index_path}")
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        if not index.get("manifests"):
            raise RuntimeError("No manifests found in OCI index")
        
        # Get the first manifest (should be our image)
        manifest_desc = index["manifests"][0]
        manifest_digest = manifest_desc["digest"]
        
        # Read the manifest
        manifest_path = os.path.join(oci_dir, "blobs", 
                                   manifest_digest.replace(":", "/"))
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Extract each layer
        for layer in manifest.get("layers", []):
            layer_digest = layer["digest"]
            layer_path = os.path.join(oci_dir, "blobs",
                                    layer_digest.replace(":", "/"))
            
            # Extract the layer (should be a tar.gz)
            subprocess.run([
                "tar", "-xzf", layer_path, "-C", extract_dir
            ], check=True)
        
        logger.info(f"Extracted model files to {extract_dir}")

    def _prepare_weights(self, docker_repo: str) -> tuple[str, list[str], bool]:
        """Prepare weights by pulling from Docker registry."""
        # Create a temporary directory for extraction
        temp_dir = tempfile.mkdtemp(prefix="vllm_docker_model_")
        oci_dir = os.path.join(temp_dir, "oci")
        extract_dir = os.path.join(temp_dir, "model")
        
        os.makedirs(oci_dir)
        os.makedirs(extract_dir)
        
        try:
            # Pull the model using skopeo
            self._pull_model_with_skopeo(docker_repo, oci_dir)
            
            # Extract model files
            self._extract_model_files(oci_dir, extract_dir)
            
            # Find weight files
            weight_patterns = ["*.safetensors", "*.bin", "*.pt"]
            weight_files = []
            use_safetensors = False
            
            for pattern in weight_patterns:
                import glob
                files = glob.glob(os.path.join(extract_dir, "**", pattern), recursive=True)
                if files:
                    weight_files.extend(files)
                    if pattern == "*.safetensors":
                        use_safetensors = True
                    break
            
            if not weight_files:
                raise RuntimeError(f"No model weight files found in Docker image {docker_repo}")
            
            if use_safetensors:
                # Filter duplicates for safetensors
                weight_files = filter_duplicate_safetensors_files(
                    weight_files, extract_dir, "model.safetensors.index.json")
            else:
                # Filter unnecessary files
                weight_files = filter_files_not_needed_for_inference(weight_files)
            
            return extract_dir, weight_files, use_safetensors
            
        except Exception:
            # Clean up on failure
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def _get_weights_iterator(
        self, model_name_or_path: str, revision: Optional[str] = None
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get weights iterator for Docker repo format."""
        
        # Use docker_repo from config, not the model_name_or_path
        docker_repo = self.load_config.docker_repo
        
        # Cache the model files so we don't re-download for each call
        if self._model_files_cache is None:
            self._model_files_cache = self._prepare_weights(docker_repo)
        
        hf_folder, hf_weights_files, use_safetensors = self._model_files_cache
        
        if use_safetensors:
            weights_iterator = safetensors_weights_iterator(
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
                self.load_config.safetensors_load_strategy,
            )
        else:
            weights_iterator = pt_weights_iterator(
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
                self.load_config.pt_load_map_location,
            )
        
        yield from weights_iterator

    def download_model(self, model_config: ModelConfig) -> None:
        """Download model from Docker registry."""
        # For docker_repo format, downloading is handled lazily in _prepare_weights
        # This method is called by the base loader but the actual work
        # is done when weights are needed for the first time
        logger.info(f"Docker repo model will be downloaded from {self.docker_repo} when weights are loaded")

    def load_weights(self, model: torch.nn.Module, model_config: ModelConfig) -> None:
        """Load model weights from Docker registry."""
        # Get the weights iterator
        weights_iterator = self._get_weights_iterator(
            model_config.model, model_config.revision
        )
        
        # Load weights into the model using the standard approach
        # This mirrors the pattern used in other loaders
        param_dict = dict(model.named_parameters())
        buffer_dict = dict(model.named_buffers())
        
        for name, loaded_weight in weights_iterator:
            # Handle common prefixes that might be in the weight names
            for prefix in ["model.", "base_model."]:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            
            if name in param_dict:
                param = param_dict[name]
                # Copy the weight data, handling device placement
                param.data.copy_(loaded_weight.to(param.device))
                logger.debug(f"Loaded parameter: {name}")
            elif name in buffer_dict:
                buffer = buffer_dict[name]
                buffer.copy_(loaded_weight.to(buffer.device))
                logger.debug(f"Loaded buffer: {name}")
            else:
                logger.warning(f"Parameter {name} not found in model")
        
        logger.info("Successfully loaded weights from Docker registry")