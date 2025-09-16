#!/usr/bin/env python3
"""
Simple test for docker_repo model loading functionality.
This test validates the basic interface without requiring actual Docker registry access.
"""

import sys
import tempfile
import os

# Add the vllm directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.model_loader.docker_repo_loader import DockerRepoModelLoader
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

def test_load_config_with_docker_repo():
    """Test LoadConfig with docker_repo parameter."""
    try:
        # Test LoadConfig with docker_repo
        config = LoadConfig(
            load_format="docker_repo",
            docker_repo="myorg/test-model:v1.0"
        )
        print("✓ LoadConfig with docker_repo created successfully")
        
        # Verify the fields are set correctly
        assert config.load_format == "docker_repo"
        assert config.docker_repo == "myorg/test-model:v1.0"
        print("✓ LoadConfig fields are correct")
        
        return True
    except Exception as e:
        print(f"✗ LoadConfig test failed: {e}")
        return False

def test_docker_repo_loader_creation():
    """Test DockerRepoModelLoader creation."""
    try:
        # Create a LoadConfig with docker_repo
        config = LoadConfig(
            load_format="docker_repo", 
            docker_repo="myorg/test-model:v1.0"
        )
        
        # Get the model loader
        loader = get_model_loader(config)
        print("✓ Model loader retrieved successfully")
        
        # Verify it's the correct type
        assert isinstance(loader, DockerRepoModelLoader)
        print("✓ DockerRepoModelLoader instance created")
        
        # Verify the docker_repo is set
        assert loader.docker_repo == "myorg/test-model:v1.0"
        print("✓ DockerRepoModelLoader.docker_repo is correct")
        
        return True
    except Exception as e:
        print(f"✗ DockerRepoModelLoader test failed: {e}")
        return False

def test_docker_repo_loader_without_repo():
    """Test that DockerRepoModelLoader fails without docker_repo."""
    try:
        # Create a LoadConfig without docker_repo
        config = LoadConfig(load_format="docker_repo")
        
        # This should fail
        try:
            loader = get_model_loader(config)
            print("✗ Expected failure but got success")
            return False
        except ValueError as e:
            if "docker_repo must be specified" in str(e):
                print("✓ Correctly failed when docker_repo not specified")
                return True
            else:
                print(f"✗ Failed with unexpected error: {e}")
                return False
    except Exception as e:
        print(f"✗ Test failed unexpectedly: {e}")
        return False

def test_docker_repo_normalization():
    """Test Docker repository name normalization."""
    try:
        config = LoadConfig(
            load_format="docker_repo",
            docker_repo="myorg/test-model"
        )
        loader = get_model_loader(config)
        
        # Test normalization
        test_cases = [
            ("myorg/model", "docker.io/myorg/model"),
            ("myregistry.com/org/model:v1.0", "myregistry.com/org/model:v1.0"),
            ("model", "docker.io/library/model"),
        ]
        
        for input_repo, expected in test_cases:
            normalized = loader._normalize_docker_repo(input_repo)
            if normalized == expected:
                print(f"✓ {input_repo} -> {normalized}")
            else:
                print(f"✗ {input_repo} -> {normalized} (expected {expected})")
                return False
        
        print("✓ All Docker repo normalizations correct")
        return True
    except Exception as e:
        print(f"✗ Docker repo normalization test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running docker_repo model loading tests...\n")
    
    tests = [
        test_load_config_with_docker_repo,
        test_docker_repo_loader_creation,
        test_docker_repo_loader_without_repo,
        test_docker_repo_normalization,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\nRunning {test.__name__}:")
        if test():
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())