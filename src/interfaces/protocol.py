"""
Centralized protocol definitions for the CULI Scoring project.

This module contains all the base protocols and abstract interfaces used throughout
the codebase. All modules should import protocols from this centralized location
to ensure consistency and avoid circular imports.

Usage:
    from src.interfaces.protocol import DataModule, ModelModule, BaseTask, etc.
"""

from __future__ import annotations
from typing import Protocol, Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================================
# Data Protocols
# ============================================================================

class DataModule(Protocol):
    """Protocol for data modules that provide train/val/test dataloaders."""
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        ...
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Return validation dataloader."""
        ...
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Return test dataloader."""
        ...


# ============================================================================
# Model Protocols
# ============================================================================

class ModelModule(Protocol):
    """Protocol for model modules that can be used in the training framework."""
    
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass of the model."""
        ...


# ============================================================================
# Task Protocols
# ============================================================================

class BaseTask(Protocol):
    """Protocol that all tasks must implement for training/evaluation."""

    def setup(self, model: ModelModule) -> None:
        """Hook called once before training starts to setup task-specific components."""
        ...

    def training_step(self, batch: Any, model: ModelModule) -> Dict[str, torch.Tensor]:
        """Compute loss and logs for a training batch.
        
        Args:
            batch: Training batch data
            model: Model to train
            
        Returns:
            Dict with keys: {"train/loss": Tensor, "logs": {name: scalar}}
        """
        ...

    def validation_step(self, batch: Any, model: ModelModule) -> Dict[str, torch.Tensor]:
        """Compute metrics/loss for validation batch.
        
        Args:
            batch: Validation batch data
            model: Model to evaluate
            
        Returns:
            Dict with keys: {"val/loss": Tensor, ...}
        """
        ...

    def reduce(self) -> Dict[str, float]:
        """Aggregate metrics across validation epoch and return scalars."""
        ...


# ============================================================================
# Metric Protocols
# ============================================================================

class Metric(Protocol):
    """Protocol for metrics that can be computed incrementally."""
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric state with predictions and targets."""
        ...
    
    def compute(self) -> Dict[str, float]:
        """Compute final metric values from accumulated state."""
        ...
    
    def reset(self) -> None:
        """Reset metric state for next epoch/evaluation."""
        ...


# ============================================================================
# Preprocessing Protocols
# ============================================================================

class BasePipeline(Protocol):
    """Protocol for data preprocessing pipelines."""
    
    def run(self) -> None:
        """Execute the preprocessing pipeline."""
        ...


class BaseDownloader(Protocol):
    """Protocol for model/data downloaders."""
    
    def download(self) -> None:
        """Download and setup resources."""
        ...


# ============================================================================
# Optimizer Protocols
# ============================================================================

class OptimizerFactory(Protocol):
    """Protocol for optimizer factories that create optimizer and scheduler."""
    
    def __call__(self, model: ModelModule, cfg: Any) -> tuple[torch.optim.Optimizer, Optional[Any]]:
        """Create optimizer and optional scheduler for the given model.
        
        Args:
            model: Model to optimize
            cfg: Configuration object
            
        Returns:
            Tuple of (optimizer, scheduler). Scheduler can be None.
        """
        ...


# ============================================================================
# Registry Protocols
# ============================================================================

class Buildable(Protocol):
    """Protocol for objects that can be built from configuration."""
    
    def __call__(self, cfg: Any, **kwargs) -> Any:
        """Build and return an object from configuration.
        
        Args:
            cfg: Configuration object
            **kwargs: Additional keyword arguments
            
        Returns:
            Built object instance
        """
        ...


# ============================================================================
# Batch Interface
# ============================================================================

class BatchProtocol(Protocol):
    """Protocol for batch objects that can be moved between devices."""
    
    inputs: Dict[str, torch.Tensor]
    outputs: Dict[str, torch.Tensor]
    meta: Optional[Dict[str, Any]]
    
    def to(self, device: torch.device) -> "BatchProtocol":
        """Move batch to specified device."""
        ...
    
    def to_device(self, device: torch.device) -> None:
        """Move batch to device in-place (legacy compatibility)."""
        ...


# ============================================================================
# Engine Protocols
# ============================================================================

class Trainer(Protocol):
    """Protocol for training engines."""
    
    def fit(self) -> None:
        """Run the complete training process."""
        ...


class Evaluator(Protocol):
    """Protocol for evaluation engines."""
    
    def run(self) -> Dict[str, float]:
        """Run evaluation and return metrics."""
        ...


# ============================================================================
# Logging Protocols
# ============================================================================

class Logger(Protocol):
    """Protocol for experiment loggers."""
    
    def log_scalars(self, scalars: Dict[str, float], step: int) -> None:
        """Log scalar metrics at a given step."""
        ...
    
    def close(self) -> None:
        """Close and cleanup logger resources."""
        ...


# ============================================================================
# Utility Type Aliases
# ============================================================================

# Common type aliases for better type hints
ConfigDict = Dict[str, Any]
MetricsDict = Dict[str, float] 
LossDict = Dict[str, torch.Tensor]
PathLike = str | Path
