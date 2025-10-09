"""
Artifacts Interface

Standardizes the terminology and provides loaders for different feature types
used throughout the CULI Scoring project.
"""

from typing import Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch


# ---------------- Artifact Types ----------------

@dataclass
class TokenArtifact:
    """BERT tokenizer output artifact"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    
    @property
    def shape(self) -> torch.Size:
        return self.input_ids.shape


@dataclass
class EncodedArtifact:
    """Wav2Vec2 processor output artifact"""
    input_values: torch.Tensor
    attention_mask: torch.Tensor = None
    
    @property
    def shape(self) -> torch.Size:
        return self.input_values.shape


@dataclass
class LogMelArtifact:
    """Log-Mel spectrogram artifact"""
    spectrogram: torch.Tensor
    
    @property
    def shape(self) -> torch.Size:
        return self.spectrogram.shape


# ---------------- Base Loader ----------------

class ArtifactLoader(ABC):
    """Base class for artifact loaders"""
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> Any:
        """Load artifact from file"""
        pass


# ---------------- Specific Loaders ----------------

class TokenLoader(ArtifactLoader):
    """Loader for token artifacts (BERT tokenizer outputs)"""
    
    def load(self, path: Union[str, Path]) -> TokenArtifact:
        """
        Load token artifact from .pt file
        
        Args:
            path: Path to {fid}_token.pt file
            
        Returns:
            TokenArtifact with input_ids, attention_mask, token_type_ids
        """
        data = torch.load(path, map_location='cpu')
        
        # Handle both dict and transformers.BatchFeature objects
        if not (hasattr(data, '__getitem__') and hasattr(data, 'get')):
            raise ValueError(f"Expected dict-like object from token file, got {type(data)}")
        
        # Convert BatchFeature to dict if needed for consistent access
        if hasattr(data, 'data'):  # BatchFeature has a .data attribute
            data_dict = dict(data)
        else:
            data_dict = data
        
        required_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        missing_keys = [key for key in required_keys if key not in data_dict]
        if missing_keys:
            raise ValueError(f"Missing required keys in token file: {missing_keys}")
        
        return TokenArtifact(
            input_ids=data_dict['input_ids'],
            attention_mask=data_dict['attention_mask'],
            token_type_ids=data_dict['token_type_ids']
        )


class EncodedLoader(ArtifactLoader):
    """Loader for encoded artifacts (Wav2Vec2 processor outputs)"""
    
    def load(self, path: Union[str, Path]) -> EncodedArtifact:
        """
        Load encoded artifact from .pt file
        
        Args:
            path: Path to {fid}_audio.pt file
            
        Returns:
            EncodedArtifact with input_values and optional attention_mask
        """
        data = torch.load(path, map_location='cpu')
        
        # Handle both dict and transformers.BatchFeature objects
        # BatchFeature inherits from dict, so we check for dict-like behavior
        if not (hasattr(data, '__getitem__') and hasattr(data, 'get')):
            raise ValueError(f"Expected dict-like object from encoded file, got {type(data)}")
        
        # Convert BatchFeature to dict if needed for consistent access
        if hasattr(data, 'data'):  # BatchFeature has a .data attribute
            data_dict = dict(data)
        else:
            data_dict = data
        
        if 'input_values' not in data_dict:
            raise ValueError("Missing required key 'input_values' in encoded file")
        
        return EncodedArtifact(
            input_values=data_dict['input_values'],
            attention_mask=data_dict.get('attention_mask', None)
        )


class LogMelLoader(ArtifactLoader):
    """Loader for log-mel artifacts (Log-Mel spectrograms)"""
    
    def load(self, path: Union[str, Path]) -> LogMelArtifact:
        """
        Load log-mel artifact from .pt file
        
        Args:
            path: Path to {fid}_logmel.pt file
            
        Returns:
            LogMelArtifact with spectrogram tensor
        """
        data = torch.load(path, map_location='cpu')
        
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor from logmel file, got {type(data)}")
        
        return LogMelArtifact(spectrogram=data)


# ---------------- Loader Factory ----------------

class ArtifactLoaderFactory:
    """Factory for creating appropriate loaders based on artifact type"""
    
    _loaders = {
        'tokens': TokenLoader(),
        'encoded': EncodedLoader(), 
        'logmel': LogMelLoader(),
    }
    
    @classmethod
    def get_loader(cls, artifact_type: str) -> ArtifactLoader:
        """
        Get loader for specified artifact type
        
        Args:
            artifact_type: One of 'tokens', 'encoded', 'logmel'
            
        Returns:
            Appropriate artifact loader
        """
        if artifact_type not in cls._loaders:
            raise ValueError(f"Unsupported artifact type: {artifact_type}. "
                           f"Supported types: {list(cls._loaders.keys())}")
        
        return cls._loaders[artifact_type]
    
    @classmethod
    def load(cls, artifact_type: str, path: Union[str, Path]) -> Union[TokenArtifact, EncodedArtifact, LogMelArtifact]:
        """
        Load artifact using appropriate loader
        
        Args:
            artifact_type: One of 'tokens', 'encoded', 'logmel'
            path: Path to artifact file
            
        Returns:
            Loaded artifact object
        """
        loader = cls.get_loader(artifact_type)
        return loader.load(path)


# ---------------- Utility Functions ----------------

def load_artifact(artifact_type: str, path: Union[str, Path]) -> Union[TokenArtifact, EncodedArtifact, LogMelArtifact]:
    """
    Convenience function to load any artifact type
    
    Args:
        artifact_type: One of 'tokens', 'encoded', 'logmel'
        path: Path to artifact file
        
    Returns:
        Loaded artifact object
    """
    return ArtifactLoaderFactory.load(artifact_type, path)


def get_supported_artifacts() -> list[str]:
    """Get list of supported artifact types"""
    return list(ArtifactLoaderFactory._loaders.keys())


# ---------------- Constants ----------------

SUPPORTED_ARTIFACTS = ['tokens', 'encoded', 'logmel']
ARTIFACT_EXTENSIONS = {
    'tokens': '_token.pt',
    'encoded': '_audio.pt', 
    'logmel': '_logmel.pt',
}