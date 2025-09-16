#!/usr/bin/env python3
"""
Simple test for just the docker_repo LoadConfig functionality.
This test validates the basic interface without requiring full vLLM imports.
"""

import sys
import os

# Add the vllm directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_load_config_basic():
    """Test basic LoadConfig functionality for docker_repo."""
    try:
        # Import just what we need
        import torch
        from vllm.config.load import LoadConfig
        
        print("✓ LoadConfig import successful")
        
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
        
        # Test without docker_repo
        config2 = LoadConfig(load_format="auto")
        assert config2.docker_repo is None
        print("✓ LoadConfig without docker_repo works correctly")
        
        return True
    except Exception as e:
        print(f"✗ LoadConfig test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_docker_repo_loader_basic():
    """Test basic DockerRepoModelLoader functionality."""
    try:
        import torch
        from vllm.config.load import LoadConfig
        from vllm.model_executor.model_loader.docker_repo_loader import DockerRepoModelLoader
        
        print("✓ DockerRepoModelLoader import successful")
        
        # Test creation with docker_repo
        config = LoadConfig(
            load_format="docker_repo",
            docker_repo="myorg/test-model:v1.0"
        )
        
        loader = DockerRepoModelLoader(config)
        print("✓ DockerRepoModelLoader created successfully")
        
        # Test docker_repo is set
        assert loader.docker_repo == "myorg/test-model:v1.0"
        print("✓ DockerRepoModelLoader.docker_repo is correct")
        
        # Test normalization function
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
        
        # Test failure without docker_repo
        try:
            config_no_repo = LoadConfig(load_format="docker_repo")
            loader_fail = DockerRepoModelLoader(config_no_repo)
            print("✗ Expected failure but got success")
            return False
        except ValueError as e:
            if "docker_repo must be specified" in str(e):
                print("✓ Correctly failed when docker_repo not specified")
            else:
                print(f"✗ Failed with unexpected error: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ DockerRepoModelLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Running basic docker_repo functionality tests...\n")
    
    tests = [
        test_load_config_basic,
        test_docker_repo_loader_basic,
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