"""
Test security fixes for pickle removal
"""
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import importlib.util

# Module constants
THAINLP_ROOT = Path(__file__).parent.parent.parent / "thainlp"
OPTIMIZER_PATH = THAINLP_ROOT / "optimization" / "optimizer.py"

# Direct import to avoid dependencies
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import optimizer module directly
optimizer = import_module_from_path("optimizer", OPTIMIZER_PATH)

MemoryOptimizer = optimizer.MemoryOptimizer
DiskCache = optimizer.DiskCache


def test_memory_optimizer_cache_key_generation():
    """Test that MemoryOptimizer uses secure hash-based cache keys"""
    optimizer = MemoryOptimizer(max_cache_size=10)
    
    # Create a simple function to memoize
    call_count = [0]
    
    @optimizer.memoize
    def add_numbers(a, b):
        call_count[0] += 1
        return a + b
    
    # First call should execute the function
    result1 = add_numbers(1, 2)
    assert result1 == 3
    assert call_count[0] == 1
    
    # Second call with same args should use cache
    result2 = add_numbers(1, 2)
    assert result2 == 3
    assert call_count[0] == 1  # Should not increment
    
    # Different args should execute function again
    result3 = add_numbers(2, 3)
    assert result3 == 5
    assert call_count[0] == 2
    
    print("✓ MemoryOptimizer cache key generation test passed")


def test_disk_cache_json_format():
    """Test that DiskCache uses JSON instead of pickle"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        cache = DiskCache(temp_dir)
        
        # Test setting and getting simple values
        cache.set("test_key", {"value": "test", "number": 42})
        result = cache.get("test_key")
        
        assert result is not None
        assert result["value"] == "test"
        assert result["number"] == 42
        
        # Verify that cache files are JSON, not pickle
        cache_files = list(Path(temp_dir).glob("*.json"))
        assert len(cache_files) > 0, "Cache should create .json files"
        
        # Verify no pickle files are created
        pickle_files = list(Path(temp_dir).glob("*.pkl"))
        assert len(pickle_files) == 0, "No pickle files should be created"
        
        # Test that we can read the JSON file directly
        with open(cache_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert data["value"] == "test"
            assert data["number"] == 42
        
        print("✓ DiskCache JSON format test passed")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_disk_cache_non_serializable():
    """Test that DiskCache handles non-JSON-serializable objects by converting to string"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        cache = DiskCache(temp_dir)
        
        # Try to cache a non-serializable object
        class CustomObject:
            def __str__(self):
                return "CustomObject"
        
        obj = CustomObject()
        # This should convert to string representation
        cache.set("non_serializable", obj)
        
        # Should return the string representation
        result = cache.get("non_serializable")
        assert result is not None
        assert "CustomObject" in str(result)
        
        print("✓ DiskCache non-serializable handling test passed")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_no_pickle_imports():
    """Verify that pickle is not imported in security-critical modules"""
    # Use constant defined at module level
    with open(OPTIMIZER_PATH, 'r') as f:
        source = f.read()
    
    assert "import pickle" not in source, "optimizer.py should not import pickle"
    assert "from pickle" not in source, "optimizer.py should not use pickle"
    
    print("✓ No pickle imports test passed")


if __name__ == "__main__":
    test_memory_optimizer_cache_key_generation()
    test_disk_cache_json_format()
    test_disk_cache_non_serializable()
    test_no_pickle_imports()
    print("\n✓ All security tests passed!")
