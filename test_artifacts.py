#!/usr/bin/env python3
"""Test script to verify artifact loading with BatchFeature objects"""

import torch
from pathlib import Path
from src.interfaces.artifacts import load_artifact

def test_batchfeature_compatibility():
    """Test that our artifact loaders can handle transformers BatchFeature objects"""
    
    # Simulate a BatchFeature-like object
    class MockBatchFeature:
        def __init__(self, data_dict):
            self._data = data_dict
            
        def __getitem__(self, key):
            return self._data[key]
            
        def get(self, key, default=None):
            return self._data.get(key, default)
            
        def __contains__(self, key):
            return key in self._data
            
        @property
        def data(self):
            return self._data
            
        def __iter__(self):
            return iter(self._data)
            
        def keys(self):
            return self._data.keys()
            
        def items(self):
            return self._data.items()
            
        def values(self):
            return self._data.values()

    # Test data
    mock_encoded_data = MockBatchFeature({
        'input_values': torch.randn(1, 16000),
        'attention_mask': torch.ones(1, 16000, dtype=torch.bool)
    })
    
    mock_token_data = MockBatchFeature({
        'input_ids': torch.tensor([[101, 102, 103]]),
        'attention_mask': torch.tensor([[1, 1, 1]]),
        'token_type_ids': torch.tensor([[0, 0, 0]])
    })
    
    # Create temporary files
    temp_encoded_file = Path("/tmp/test_encoded.pt")
    temp_token_file = Path("/tmp/test_token.pt")
    
    try:
        # Save test data
        torch.save(mock_encoded_data, temp_encoded_file)
        torch.save(mock_token_data, temp_token_file)
        
        # Test loading
        print("Testing encoded artifact loading...")
        encoded_artifact = load_artifact('encoded', temp_encoded_file)
        print(f"✓ Encoded artifact loaded successfully: input_values shape = {encoded_artifact.input_values.shape}")
        
        print("Testing token artifact loading...")
        token_artifact = load_artifact('tokens', temp_token_file)
        print(f"✓ Token artifact loaded successfully: input_ids shape = {token_artifact.input_ids.shape}")
        
        print("All tests passed! ✅")
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
        
    finally:
        # Cleanup
        temp_encoded_file.unlink(missing_ok=True)
        temp_token_file.unlink(missing_ok=True)
        
    return True

if __name__ == "__main__":
    test_batchfeature_compatibility()